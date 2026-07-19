"""Engine-worker adapter for ComfyUI (launched via comfy-cli).

Follows the ADR 0007 subprocess pattern established by
:mod:`sovereign.workers.llama_cpp_adapter`: this adapter supervises
``comfy … launch`` as a **child process** rather than loading tensors
in-process. Like omlx, ComfyUI exposes no ``/slots``/``/metrics`` scrape
surface (generation progress is a websocket), so there is no
telemetry-translator poll loop — after health the adapter simply supervises
the child (heartbeat and memory events come from ``engine_worker`` itself;
TOK/S/prefill telemetry is an accepted ADR 0006 gap the manager surfaces at
pre-flight).

ComfyUI discovers models from a ``models/`` directory tree instead of taking
a model path, so :func:`prepare_checkpoint_dir` symlinks the one resolved
checkpoint file into a private per-service ``models/checkpoints/`` layout and
writes an ``extra_model_paths.yaml`` pointing ComfyUI at it — one comfyui
instance serves exactly one checkpoint.

Only :func:`build_server_argv` and :func:`prepare_checkpoint_dir` are meant
to be imported elsewhere (pure/near-pure, unit-testable without comfy-cli
installed). This module never imports ``comfy``/``torch`` — it stays
importable on any platform.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.workers.protocol import EventType

if TYPE_CHECKING:
    from sovereign.workers.telemetry import TelemetryClient
    from sovereign.workers.worker_config import WorkerConfig

logger = logging.getLogger("sovereign")

#: engine_kwargs keys consumed here rather than passed through verbatim.
#: ``models_root``/``checkpoint_name`` drive prepare_checkpoint_dir(),
#: ``workspace_dir`` is a comfy-cli flag (before ``launch``), ``output_dir``
#: maps to ComfyUI's ``--output-directory``.
_CONSUMED_KEYS = frozenset({"workspace_dir", "models_root", "checkpoint_name", "output_dir"})

#: Bounded wait for ComfyUI to report healthy before giving up. First boot
#: imports torch and initialises MPS, so this is generous.
_HEALTH_TIMEOUT = 180.0
_HEALTH_POLL_INTERVAL = 0.2
_HTTP_TIMEOUT = 2.0
#: How often the supervision loop checks the child is still alive.
_SUPERVISE_INTERVAL = 0.5
#: How long to wait after SIGTERM before escalating to SIGKILL.
_STOP_TIMEOUT = 10.0


def build_server_argv(
    engine_kwargs: dict[str, Any],
    extra_model_paths: str,
    host: str,
    port: int,
) -> list[str]:
    """Translate Sovereign's engine-agnostic kwargs into a ``comfy`` CLI argv
    (sans program name, starting with comfy-cli's own flags).

    Recognized keys: ``workspace_dir`` -> ``comfy --workspace`` (selects the
    ComfyUI install), ``output_dir`` -> ``--output-directory`` (a ComfyUI
    flag, after the ``--`` separator). ``models_root``/``checkpoint_name``
    are consumed by :func:`prepare_checkpoint_dir` rather than mapped.

    Unknown ``engine_kwargs`` entries pass through the same way as the other
    adapters: kebab-cased ``--flag value`` (bare flag for ``True``, dropped
    for ``False``) appended after the ``--`` separator so ComfyUI's
    ``main.py`` receives them — a flag the installed ComfyUI doesn't know is
    a loud failure at worker boot, not silent misconfiguration.
    """
    argv = ["--skip-prompt"]
    workspace = engine_kwargs.get("workspace_dir")
    if workspace:
        argv += ["--workspace", str(workspace)]
    argv += [
        "launch",
        "--",
        "--listen",
        host,
        "--port",
        str(port),
        "--extra-model-paths-config",
        extra_model_paths,
    ]

    if "output_dir" in engine_kwargs:
        argv += ["--output-directory", str(engine_kwargs["output_dir"])]

    for key, value in engine_kwargs.items():
        if key in _CONSUMED_KEYS:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]

    return argv


def prepare_checkpoint_dir(models_root: str, checkpoint_name: str, model_path: str) -> str:
    """Build the single-checkpoint model layout ComfyUI is pointed at: a
    ``checkpoints/`` directory containing one symlink named ``checkpoint_name``
    pointing at the resolved checkpoint file, plus an ``extra_model_paths.yaml``
    declaring the layout. Idempotent; a stale symlink (model or HF snapshot
    revision changed) is re-pointed. Returns the yaml's path.

    A pre-existing *real* file at the link path is left untouched — the user
    may have materialised the checkpoint there deliberately.
    """
    root = Path(models_root)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    link = checkpoints / checkpoint_name
    target = Path(model_path).resolve()
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target)
    elif not link.exists():
        link.symlink_to(target)

    yaml_path = root / "extra_model_paths.yaml"
    yaml_path.write_text(
        "# Written by Sovereign (workers/comfyui_adapter.py) — do not edit.\n"
        "sovereign:\n"
        f"  base_path: {root}\n"
        "  checkpoints: checkpoints\n"
    )
    return str(yaml_path)


def _wait_for_health(
    url: str,
    process: subprocess.Popen[Any],
    stop_event: threading.Event,
    timeout: float | None = None,
    interval: float | None = None,
) -> bool:
    # Resolved at call time (not bound as a default) so tests can monkeypatch
    # the module-level constants to shrink the wait.
    if timeout is None:
        timeout = _HEALTH_TIMEOUT
    if interval is None:
        interval = _HEALTH_POLL_INTERVAL
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_event.is_set() or process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if stop_event.wait(interval):
            return False
    return False


def supervise(
    process: subprocess.Popen[Any],
    stop_event: threading.Event,
    interval: float = _SUPERVISE_INTERVAL,
) -> bool:
    """Watch the child until ``stop_event`` is set or it exits on its own.
    Returns ``True`` if the child exited unexpectedly (a crash the caller
    should surface), ``False`` on a clean/requested shutdown. Same shape as
    the omlx adapter's supervise loop (no scrape surface to translate)."""
    while not stop_event.is_set():
        if process.poll() is not None:
            return True
        if stop_event.wait(interval):
            return False
    return False


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Launch ``comfy … launch`` as a child process and supervise until shutdown.

    The worker process (this one) never loads tensors — ComfyUI does, in the
    child. ``controller.shutdown_callback`` terminates that child on SIGTERM
    (escalating to SIGKILL after a bounded wait); an unexpected child exit
    raises so ``engine_worker`` surfaces a ``state_change: crashed`` event,
    same as any other adapter failure.
    """
    kwargs = cfg.engine_kwargs
    extra_model_paths = prepare_checkpoint_dir(
        kwargs.get("models_root", str(Path(cfg.model_path).parent / "comfyui-models")),
        kwargs.get("checkpoint_name", cfg.served_model_name or Path(cfg.model_path).name),
        cfg.model_path,
    )

    argv = ["comfy", *build_server_argv(kwargs, extra_model_paths, cfg.host, cfg.port)]

    logger.info("comfyui worker %s: launching comfy launch", cfg.service)
    process = subprocess.Popen(argv)  # noqa: S603 - argv is constructed, not shell

    stop_event = threading.Event()

    def _shutdown() -> None:
        stop_event.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    controller.shutdown_callback = _shutdown

    health_url = f"http://{cfg.host}:{cfg.port}{cfg.health_path}"
    if not _wait_for_health(health_url, process, stop_event):
        if process.poll() is None:
            _shutdown()
        raise RuntimeError(f"ComfyUI for '{cfg.service}' failed to become healthy")

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    if supervise(process, stop_event):
        raise RuntimeError(
            f"ComfyUI for '{cfg.service}' exited unexpectedly (code={process.returncode})"
        )
