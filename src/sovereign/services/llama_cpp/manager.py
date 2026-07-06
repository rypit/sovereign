"""``llama_cpp`` — the first native inference engine (§12, Phase 4).

Runs ``llama-server`` as a native subprocess (bare metal, for real Metal
acceleration — §2.1), health-checks it over HTTP, and reports live metrics via
``psutil``. This is the reference pattern every native engine (and ComfyUI, §2.13)
follows.

Health is defined in config, executed here (§2.7): the bind port and readiness
path come from the entry's ``health_check`` block, and ``is_healthy()`` is what
actually pings it.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psutil

from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.registry import register_service
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.core.resources import priority_to_nice
from sovereign.services.llama_cpp.config import (
    VALID_KV_CACHE_TYPES,
    LlamaCppConfig,
    LlamaPolicy,
)

# Per-request health probe timeout (seconds) — distinct from the overall boot
# timeout the Orchestrator enforces while polling.
_HTTP_TIMEOUT = 2.0
# How long to wait after SIGTERM before escalating to SIGKILL.
_STOP_TIMEOUT = 10.0


@register_service("llama_cpp")
class LlamaCppManager(ActivityMixin):
    """Supervises one native ``llama-server`` process."""

    base_type = "llama_cpp"
    consumer_kind = ConsumerKind.NATIVE

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = LlamaCppConfig.model_validate(entry.config)

        if entry.health_check is None:
            raise ValueError(
                f"llama_cpp service '{entry.name}' requires a health_check block "
                "(it defines the bind port and readiness path)."
            )
        self.host = self.config.host
        self.port = entry.health_check.port
        self.health_path = entry.health_check.endpoint
        self.policy = LlamaPolicy.model_validate(entry.policy or {})
        self.priority = entry.priority
        self.memory_override_gb = entry.memory_gb

        self.process: subprocess.Popen[bytes] | None = None
        self._log_file = None

    def endpoint(self) -> ResolvedEndpoint:
        """The address consumers reach this engine at (registered when READY)."""
        return ResolvedEndpoint(scheme="http", host=self.host, port=self.port)

    def runtime_handle(self) -> dict | None:
        """A cross-process teardown handle (PID) recorded in state.json for `down`."""
        if self.process is not None and self.process.poll() is None:
            return {"kind": "native", "pid": self.process.pid}
        return None

    # --- resource estimation (§7) ---
    def estimated_memory_gb(self) -> float:
        """Estimate resident memory: model file + KV cache (or a declared override)."""
        if self.memory_override_gb is not None:
            return round(self.memory_override_gb, 2)
        model = Path(self.config.model_path).expanduser()
        model_gb = model.stat().st_size / (1024**3) if model.is_file() else 0.0
        return round(model_gb + self.estimated_kv_cache_gb(), 2)

    def estimated_kv_cache_gb(self) -> float:
        """KV cache grows with total context (§7 — slots×context is joint)."""
        ctx = self.config.context_size or 0
        return round(ctx * self.config.kv_bytes_per_token / (1024**3), 2)

    def per_slot_context(self) -> int | None:
        """Context available *per agent* — total context divided across -np slots."""
        if self.config.context_size is None:
            return None
        return self.config.context_size // (self.config.max_parallel or 1)

    # --- flag generation ---
    def get_start_args(self) -> list[str]:
        """Translate the validated config into a ``llama-server`` argv (§7)."""
        model = str(Path(self.config.model_path).expanduser())
        args = [
            self.config.binary,
            "--model",
            model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.config.gpu_layers is not None:
            args += ["-ngl", str(self.config.gpu_layers)]
        if self.config.threads is not None:
            args += ["-t", str(self.config.threads)]
        if self.config.context_size is not None:
            args += ["-c", str(self.config.context_size)]
        if self.config.max_parallel is not None:
            args += ["-np", str(self.config.max_parallel)]
        if self.config.api_key:
            args += ["--api-key", self.config.api_key]

        caching = self.policy.prompt_caching
        if caching and caching.enabled and caching.cache_path:
            cache_dir = str(Path(caching.cache_path).expanduser())
            args += [
                "--slot-save-path",
                cache_dir,
                "--cache-type-k",
                caching.kv_cache_type,
                "--cache-type-v",
                caching.kv_cache_type,
            ]

        args += self.config.extra_args
        return args

    # --- Lifecycle ---
    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return  # already running

        log_dir = Path(self.config.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = (log_dir / f"{self.name}.log").open("a")

        self.process = subprocess.Popen(  # noqa: S603 - argv is constructed, not shell
            self.get_start_args(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
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
            proc.terminate()  # SIGTERM — let it flush caches (§6.4)
            try:
                proc.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.process = None

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        url = f"http://{self.host}:{self.port}{self.health_path}"
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 - fixed http scheme
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
                return {
                    "memory_mb": round(p.memory_info().rss / (1024**2), 2),
                    "cpu_percent": p.cpu_percent(interval=None),
                    "status": "running",
                }
        except psutil.NoSuchProcess:
            return {"status": "stopped"}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        model = Path(self.config.model_path).expanduser()
        if not model.is_file():
            raise FileNotFoundError(
                f"llama_cpp model for '{self.name}' not found: {model}"
            )
        binary = self.config.binary
        if shutil.which(binary) is None and not Path(binary).expanduser().is_file():
            raise FileNotFoundError(
                f"llama-server binary '{binary}' not found on PATH for '{self.name}'."
            )
        self._validate_prompt_caching()

    def _validate_prompt_caching(self) -> None:
        """Pre-flight the prompt-caching policy (§7): dir writable, dtype valid."""
        caching = self.policy.prompt_caching
        if caching is None or not caching.enabled:
            return
        if not caching.cache_path:
            raise ValueError(
                f"prompt_caching is enabled for '{self.name}' but no cache_path is set."
            )
        if caching.kv_cache_type not in VALID_KV_CACHE_TYPES:
            raise ValueError(
                f"invalid kv_cache_type '{caching.kv_cache_type}' for '{self.name}'; "
                f"expected one of {sorted(VALID_KV_CACHE_TYPES)}."
            )
        cache_dir = Path(caching.cache_path).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"prompt cache dir for '{self.name}' is not writable: {cache_dir} ({exc})"
            ) from exc

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """No-op for now; cache-shrinking under pressure is a later refinement."""
