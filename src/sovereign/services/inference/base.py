"""Shared base for native inference-engine managers (llama_cpp, mlx_lm).

One subprocess + HTTP-health + psutil-metrics lifecycle (§2.13), so engines only
implement config parsing, argv generation, and engine-specific pre-flight checks.
Health is defined in config, executed here (§2.7): the bind port and readiness
path come from the entry's ``health_check`` block.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Any, ClassVar, Literal

import psutil

# The runtime HF surface (metadata fetch, estimation, download) is called through
# the module — one seam, so tests patch
# `sovereign.services.inference.hf.<fn>` and every caller sees it. Pure
# helpers are imported by name.
from sovereign.config import ServiceEntry
from sovereign.core.base_config import NativeEngineConfig
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.procmem import _parse_phys_footprint, macos_phys_footprint
from sovereign.core.provisioning import Provisioner
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.core.resources import priority_to_nice
from sovereign.services.inference import hf as hf_models
from sovereign.services.inference.hf import looks_local, parse_model_ref

# Re-exported so existing imports/patches (`sovereign.services.inference.base.
# macos_phys_footprint`, `..._parse_phys_footprint`) keep working after the
# move to `sovereign.core.procmem` — a leaf module shared with the workers.
__all__ = ["macos_phys_footprint", "_parse_phys_footprint"]

# Per-request health probe timeout (seconds) — distinct from the overall boot
# timeout the Orchestrator enforces while polling.
HTTP_TIMEOUT = 2.0
# How long to wait after SIGTERM before escalating to SIGKILL.
STOP_TIMEOUT = 10.0


def check_local_artifact(value: str, *, kind: str, service: str) -> None:
    """Raise if a local-looking path doesn't exist; repo ids pass through."""
    if looks_local(value):
        path = Path(os.path.expanduser(value))
        if not path.exists():
            raise FileNotFoundError(f"{kind} for '{service}' not found: {path}")


class NativeEngineManager(ActivityMixin, Provisioner):
    """Shared lifecycle for a native engine subprocess. Not registered itself.

    Subclasses set ``base_type``, ``config_cls`` and implement ``get_start_args()``;
    they extend ``prepare_environment()`` via ``super()``. Engine toolchains are
    provisioned per-integration (a ``Brewfile`` next to the manager's module +
    ``provisioning_binary``) via the shared :class:`Provisioner` mixin.
    """

    base_type: ClassVar[str]
    config_cls: ClassVar[type[NativeEngineConfig]]
    #: Which HF artifact this engine consumes — an MLX/safetensors *snapshot* or a
    #: single *gguf* file. Drives metadata-based memory estimation and download.
    model_artifact_kind: ClassVar[Literal["snapshot", "gguf"]]
    consumer_kind = ConsumerKind.NATIVE
    #: Optional extra sentence appended to the missing-binary error.
    binary_hint: ClassVar[str] = ""

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config: NativeEngineConfig = self.config_cls.model_validate(entry.config)

        if entry.health_check is None:
            raise ValueError(
                f"{self.base_type} service '{entry.name}' requires a health_check block "
                "(it defines the bind port and readiness path)."
            )
        self.host = self.config.host
        self.port = entry.health_check.port
        self.health_path = entry.health_check.endpoint
        self.priority = entry.priority
        self.memory_override_bytes = entry.memory_bytes

        self.process: subprocess.Popen[bytes] | None = None
        self._log_file: IO[str] | None = None
        # Resolved local paths, populated by prepare_model() before start().
        self.model_path: Path | None = None
        self.draft_model_path: Path | None = None

    # --- engine-specific surface ---
    def get_start_args(self) -> list[str]:
        raise NotImplementedError

    def start_env(self) -> dict[str, str]:
        """Extra environment variables for the engine subprocess.

        Engines override this to pass secrets (e.g. an API key) through the
        environment instead of argv — a command line is world-readable via
        ``ps``; the environment of another user's process is not.
        """
        return {}

    # --- resource estimation (§7) ---
    def estimated_memory_bytes(self) -> int:
        """Model (+ draft) weights plus the engine's extra term, or a declared override.

        Speculative decoding keeps both models in unified memory simultaneously,
        so the draft model's weights always count. For HuggingFace repo ids the
        weight estimate comes from repo metadata; unknown (offline + uncached)
        contributes 0.
        """
        if self.memory_override_bytes is not None:
            return self.memory_override_bytes
        total = self._model_bytes(self.config.model)
        if self.config.draft_model is not None:
            total += self._model_bytes(self.config.draft_model)
        return total + self.extra_memory_bytes()

    def extra_memory_bytes(self) -> int:
        """Engine-specific footprint beyond the weights (bytes) — e.g. llama_cpp's
        KV cache, mlx_lm's hard prompt-cache reservation. Default: nothing."""
        return 0

    def _model_bytes(self, model: str) -> int:
        """Weight-byte estimate for one model ref (local disk, HF cache, or repo
        metadata), for admission control. Unknown (offline+uncached) → 0."""
        ref = parse_model_ref(model)
        return hf_models.estimate_model_bytes(ref, self.model_artifact_kind) or 0

    def estimated_memory_source(self) -> str:
        """Where admission's weight estimate came from (local|cached|hub|unknown).

        Backs :class:`~sovereign.core.base_manager.SupportsEstimateSource` so the
        ``sovereign plan`` SOURCE column can say whether the model is already on
        disk or would be fetched — without the planner reaching into the HF pipeline.
        """
        _, source = hf_models.estimate_model_bytes_with_source(
            parse_model_ref(self.config.model), self.model_artifact_kind
        )
        return source

    # --- pre-download (DOWNLOADING state) ---
    def prepare_model(self) -> None:
        """Resolve the model (and draft model) to local paths, downloading from the
        HuggingFace cache if needed. Runs in the orchestrator's DOWNLOADING state;
        huggingface_hub's own download progress lines are surfaced as activity.
        Local refs resolve in place."""
        self.model_path = hf_models.download_model(
            parse_model_ref(self.config.model),
            self.model_artifact_kind,
            progress=self.set_activity,
        )
        draft = self.config.draft_model
        if draft is not None:
            self.draft_model_path = hf_models.download_model(
                parse_model_ref(draft), self.model_artifact_kind, progress=self.set_activity
            )
        self.clear_activity()

    def resolved_model_path(self) -> str:
        if self.model_path is None:
            raise RuntimeError(f"prepare_model() must run before start for '{self.name}'")
        return str(self.model_path)

    def resolved_draft_model_path(self) -> str:
        if self.draft_model_path is None:
            raise RuntimeError(f"prepare_model() must run before start for '{self.name}'")
        return str(self.draft_model_path)

    # --- wiring ---
    def api_model_name(self) -> str:
        """The string an OpenAI-compatible client sends as ``"model"``.

        ``served_model_name`` overrides when set; otherwise the configured
        ``model`` (local path or HF repo id) is the name clients must send.
        """
        return self.config.served_model_name or self.config.model

    def endpoint(self) -> ResolvedEndpoint:
        """The address consumers reach this engine at (registered when READY)."""
        return ResolvedEndpoint(
            scheme="http", host=self.host, port=self.port, model=self.api_model_name()
        )

    def runtime_handle(self) -> dict | None:
        """A cross-process teardown handle (PID) recorded in state.json for `down`.

        ``create_time`` identifies the *specific* process: PIDs are recycled by
        the OS, so ``down`` verifies it before signalling — a bare PID could
        belong to a stranger process by the time teardown runs.
        """
        if self.process is not None and self.process.poll() is None:
            handle: dict = {"kind": "native", "pid": self.process.pid}
            try:
                handle["create_time"] = psutil.Process(self.process.pid).create_time()
            except psutil.Error:
                pass  # process died between poll() and here; still record the PID
            return handle
        return None

    # --- Lifecycle ---
    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return  # already running

        log_dir = Path(self.config.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.name}.log"
        self._log_file = log_path.open("a")

        extra_env = self.start_env()
        self.process = subprocess.Popen(  # noqa: S603 - argv is constructed, not shell
            self.get_start_args(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **extra_env} if extra_env else None,
        )
        self._apply_priority()

    def _apply_priority(self) -> None:
        """Best-effort QoS: deprioritise lower-priority engines via os.nice (§7)."""
        nice = priority_to_nice(self.priority)
        if nice and self.process is not None:
            try:
                psutil.Process(self.process.pid).nice(nice)
            except (psutil.Error, OSError):
                pass

    def stop(self) -> None:
        proc = self.process
        if proc is not None and proc.poll() is None:
            proc.terminate()  # SIGTERM — let it flush cleanly (§6.4)
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.clear_activity()
        self.process = None

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        url = f"http://{self.host}:{self.port}{self.health_path}"
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:  # noqa: S310 - fixed http scheme
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False

    def get_metrics(self) -> dict[str, Any]:
        proc = self.process
        if proc is None or proc.poll() is not None:
            return {"status": "stopped"}
        try:
            p = psutil.Process(proc.pid)
            with p.oneshot():
                footprint = macos_phys_footprint(proc.pid)
                memory_bytes = footprint if footprint is not None else p.memory_info().rss
                return {
                    "memory_bytes": memory_bytes,
                    "status": "running",
                }
        except psutil.NoSuchProcess:
            return {"status": "stopped"}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        """Shared pre-flight: provision declared deps, then validate binary + model."""
        # Install the engine's own toolchain first (idempotent no-op once present),
        # so a declared service works on a fresh machine without manual setup.
        self.provision()
        binary = self.config.binary
        if shutil.which(binary) is None and not Path(binary).expanduser().is_file():
            message = f"binary '{binary}' not found on PATH for '{self.name}'."
            if self.binary_hint:
                message += f" {self.binary_hint}"
            raise FileNotFoundError(message)
        # A local model path must exist; a HuggingFace repo id is fetched on start.
        check_local_artifact(self.config.model, kind=f"{self.base_type} model", service=self.name)
        # Best-effort metadata prefetch so admission's memoised estimate is warm.
        # A gated/missing repo fails loudly here (in PROVISIONING) with an
        # actionable message; a transient/offline miss returns None and is fine.
        for value in (self.config.model, self.config.draft_model):
            if value and not looks_local(value):
                ref = parse_model_ref(value)
                if ref.repo_id is not None:
                    hf_models.fetch_repo_info(ref.repo_id)

    def adjust_resources(self, memory_limit_bytes: int) -> None:
        """No-op by default; engines override when they can shrink under pressure."""
