"""``mlx_lm`` — a native MLX inference engine (§12, Phase 11).

Runs ``mlx_lm.server`` (Apple MLX, OpenAI-compatible) as a native subprocess for
real Metal acceleration (§2.1). Mirrors the ``llama_cpp`` native-process pattern
(§2.13) — subprocess + HTTP health + ``psutil`` metrics — adapted to MLX's CLI.

Health is defined in config, executed here (§2.7): the bind port and readiness path
come from the entry's ``health_check`` block (``mlx_lm.server`` serves ``/health``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
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
from sovereign.services.mlx_lm.config import MlxLmConfig

_HTTP_TIMEOUT = 2.0
_STOP_TIMEOUT = 10.0

# Matches tqdm "Fetching N files: XX%|<bar>| done/total" lines from mlx_lm.server.
_FETCH_RE = re.compile(r"Fetching (\d+) files:\s+(\d+)%[^|]*\|[^|]*\|\s*(\d+)/\1")


def _looks_local(model: str) -> bool:
    """Whether ``model`` refers to a local path (vs. a HuggingFace repo id)."""
    return model.startswith(("/", "~", ".")) or Path(os.path.expanduser(model)).exists()


def _local_model_bytes(model: str) -> int:
    """Bytes on disk for a local model path; 0 for a HuggingFace repo id or missing path."""
    p = Path(os.path.expanduser(model))
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if p.is_file():
        return p.stat().st_size
    return 0


@register_service("mlx_lm")
class MlxLmManager(ActivityMixin):
    """Supervises one native ``mlx_lm.server`` process."""

    base_type = "mlx_lm"
    consumer_kind = ConsumerKind.NATIVE

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = MlxLmConfig.model_validate(entry.config)

        if entry.health_check is None:
            raise ValueError(
                f"mlx_lm service '{entry.name}' requires a health_check block "
                "(it defines the bind port and readiness path)."
            )
        self.host = self.config.host
        self.port = entry.health_check.port
        self.health_path = entry.health_check.endpoint
        self.priority = entry.priority
        self.memory_override_gb = entry.memory_gb

        self.process: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._tailer: threading.Thread | None = None
        self._tailer_stop: threading.Event | None = None

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
        """Estimate resident memory from the local model weights (or an override).

        Includes the draft model when speculative decoding is configured — both models
        live in unified memory simultaneously. A declared ``prompt_cache_bytes`` is a
        hard KV-cache reservation that also lives in unified memory. For HuggingFace
        repo ids (not yet downloaded) the footprint is unknown and contributes 0.0
        (admitted).
        """
        if self.memory_override_gb is not None:
            return round(self.memory_override_gb, 2)
        total = _local_model_bytes(self.config.model)
        if self.config.draft_model is not None:
            total += _local_model_bytes(self.config.draft_model)
        if self.config.prompt_cache_bytes is not None:
            total += self.config.prompt_cache_bytes
        return round(total / (1024**3), 2)

    # --- flag generation ---
    def get_start_args(self) -> list[str]:
        """Translate the validated config into an ``mlx_lm.server`` argv."""
        args = [
            self.config.binary,
            "--model",
            os.path.expanduser(self.config.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.config.max_tokens is not None:
            args += ["--max-tokens", str(self.config.max_tokens)]
        if self.config.temp is not None:
            args += ["--temp", str(self.config.temp)]
        if self.config.top_p is not None:
            args += ["--top-p", str(self.config.top_p)]
        if self.config.decode_concurrency is not None:
            args += ["--decode-concurrency", str(self.config.decode_concurrency)]
        if self.config.prompt_cache_size is not None:
            args += ["--prompt-cache-size", str(self.config.prompt_cache_size)]
        if self.config.prompt_cache_bytes is not None:
            args += ["--prompt-cache-bytes", str(self.config.prompt_cache_bytes)]
        if self.config.adapter_path is not None:
            args += ["--adapter-path", os.path.expanduser(self.config.adapter_path)]
        if self.config.draft_model is not None:
            args += ["--draft-model", os.path.expanduser(self.config.draft_model)]
        if self.config.num_draft_tokens is not None:
            args += ["--num-draft-tokens", str(self.config.num_draft_tokens)]
        if self.config.trust_remote_code:
            args += ["--trust-remote-code"]
        args += self.config.extra_args
        return args

    def _tail_log_for_activity(self, log_path: Path, stop: threading.Event) -> None:
        """Tail the log file and surface HuggingFace download progress as activity."""
        try:
            with log_path.open() as fh:
                while not stop.is_set():
                    line = fh.readline()
                    if not line:
                        stop.wait(timeout=0.5)
                        continue
                    for m in _FETCH_RE.finditer(line):
                        total, pct, done = m.group(1), m.group(2), m.group(3)
                        self.set_activity(f"downloading model: {done}/{total} files ({pct}%)")
                        if pct == "100":
                            self.clear_activity()
                            return
        except OSError:
            pass

    # --- Lifecycle ---
    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return  # already running

        log_dir = Path(self.config.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.name}.log"
        self._log_file = log_path.open("a")

        self.process = subprocess.Popen(  # noqa: S603 - argv is constructed, not shell
            self.get_start_args(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._apply_priority()

        stop = threading.Event()
        self._tailer_stop = stop
        self._tailer = threading.Thread(
            target=self._tail_log_for_activity,
            args=(log_path, stop),
            daemon=True,
            name=f"mlx-tailer-{self.name}",
        )
        self._tailer.start()

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
                proc.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._tailer_stop is not None:
            self._tailer_stop.set()
        if self._tailer is not None:
            self._tailer.join(timeout=2.0)
            self._tailer = None
            self._tailer_stop = None
        self.clear_activity()
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
        binary = self.config.binary
        if shutil.which(binary) is None and not Path(binary).expanduser().is_file():
            raise FileNotFoundError(
                f"mlx_lm.server binary '{binary}' not found on PATH for '{self.name}'. "
                "It ships with the `mlx-lm` dependency — run `uv sync`."
            )
        # A local model path must exist; a HuggingFace repo id is fetched on start.
        if _looks_local(self.config.model):
            model = Path(os.path.expanduser(self.config.model))
            if not model.exists():
                raise FileNotFoundError(
                    f"mlx_lm model path for '{self.name}' not found: {model}"
                )
        if self.config.adapter_path is not None:
            adapter = Path(os.path.expanduser(self.config.adapter_path))
            if not adapter.exists():
                raise FileNotFoundError(
                    f"mlx_lm adapter_path for '{self.name}' not found: {adapter}"
                )

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """No-op: MLX's Metal cache limit isn't reachable through the server subprocess."""
