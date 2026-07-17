"""Engine-worker adapter for ``omlx serve`` (the oMLX inference server).

Follows the ADR 0007 subprocess pattern established by
:mod:`sovereign.workers.llama_cpp_adapter`: this adapter supervises ``omlx``
as a **child process** rather than loading tensors in-process. Unlike
llama-server, omlx exposes no ``/slots``/``/metrics`` scrape surface, so
there is no telemetry-translator poll loop — after health the adapter simply
supervises the child (heartbeat and memory events come from
``engine_worker`` itself; prefill/tok-s telemetry is an accepted ADR 0006
gap the manager surfaces at pre-flight).

omlx discovers models from a ``--model-dir`` directory layout instead of
taking a model path, so :func:`prepare_model_dir` symlinks the one resolved
snapshot into a private per-service directory under the model name clients
will send — one omlx instance serves exactly one model.

Only :func:`build_server_argv` and :func:`prepare_model_dir` are meant to be
imported elsewhere (pure/near-pure, unit-testable without ``omlx``
installed). This module never imports ``omlx``/``mlx`` — it stays importable
on any platform.
"""

from __future__ import annotations

import logging
import os
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

#: engine_kwargs keys mapped onto an ``omlx serve`` CLI flag with a straight
#: rename. Verified against `omlx serve --help`.
_KWARG_FLAGS: dict[str, str] = {
    "max_concurrent_requests": "--max-concurrent-requests",
    "memory_guard_gb": "--memory-guard-gb",
    "paged_ssd_cache_dir": "--paged-ssd-cache-dir",
}

#: engine_kwargs keys carrying a decimal-GB int that omlx wants as a
#: size string ("8GB").
_GB_KWARG_FLAGS: dict[str, str] = {
    "paged_ssd_cache_max_gb": "--paged-ssd-cache-max-size",
    "hot_cache_gb": "--hot-cache-max-size",
}

#: engine_kwargs keys consumed here rather than passed through verbatim.
#: ``model_dir``/``model_name`` drive prepare_model_dir(), not a flag.
_CONSUMED_KEYS = frozenset({*_KWARG_FLAGS, *_GB_KWARG_FLAGS, "model_dir", "model_name"})

#: Bounded wait for omlx to report healthy before giving up. First boot
#: compiles/loads the model into unified memory, so this is generous.
_HEALTH_TIMEOUT = 120.0
_HEALTH_POLL_INTERVAL = 0.2
_HTTP_TIMEOUT = 2.0
#: How often the supervision loop checks the child is still alive.
_SUPERVISE_INTERVAL = 0.5
#: How long to wait after SIGTERM before escalating to SIGKILL.
_STOP_TIMEOUT = 10.0


def build_server_argv(
    engine_kwargs: dict[str, Any],
    model_dir: str,
    host: str,
    port: int,
) -> list[str]:
    """Translate Sovereign's engine-agnostic kwargs into an ``omlx`` CLI argv
    (sans program name, starting with the ``serve`` subcommand).

    Recognized keys: ``max_concurrent_requests`` ->
    ``--max-concurrent-requests``, ``memory_guard_gb`` ->
    ``--memory-guard-gb``, ``paged_ssd_cache_dir`` ->
    ``--paged-ssd-cache-dir``, ``paged_ssd_cache_max_gb``/``hot_cache_gb`` ->
    ``--paged-ssd-cache-max-size``/``--hot-cache-max-size`` (rendered as
    omlx's ``"<n>GB"`` size strings). ``model_dir``/``model_name`` are
    consumed by :func:`prepare_model_dir` rather than mapped.

    Unknown ``engine_kwargs`` entries pass through the same way as the other
    adapters: kebab-cased ``--flag value`` (bare flag for ``True``, dropped
    for ``False``) — a flag the installed ``omlx`` doesn't know is a loud
    failure at worker boot, not silent misconfiguration.

    The API key is deliberately NOT built here — ``run()`` appends
    ``--api-key`` itself from the ``SOVEREIGN_API_KEY`` environment variable
    so it never appears in a dumped ``WorkerConfig`` JSON (same contract as
    the llama_cpp adapter).
    """
    argv = ["serve", "--model-dir", model_dir, "--host", host, "--port", str(port)]

    for key, flag in _KWARG_FLAGS.items():
        if key in engine_kwargs:
            argv += [flag, str(engine_kwargs[key])]

    for key, flag in _GB_KWARG_FLAGS.items():
        if key in engine_kwargs:
            argv += [flag, f"{engine_kwargs[key]}GB"]

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


def prepare_model_dir(models_root: str, model_name: str, model_path: str) -> str:
    """Build the single-model ``--model-dir`` layout: a symlink named
    ``model_name`` pointing at the resolved snapshot directory. Idempotent;
    a stale symlink (model or HF snapshot revision changed) is re-pointed.
    Returns ``models_root``.

    Nested names still produce nested directories, but the manager always
    passes a flat ``org--name`` (see ``OmlxManager.api_model_name``): omlx
    derives model ids by joining nested subdirectories with ``--``, and a
    nested layout gets double-registered under both its leaf and qualified
    names — flat names make the discovered id equal ``model_name`` verbatim.

    A pre-existing *real* directory at the link path is left untouched — the
    user may have materialised the model there deliberately.
    """
    root = Path(models_root)
    link = root / model_name.lstrip("/")
    target = Path(model_path).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            link.unlink()
            link.symlink_to(target, target_is_directory=True)
    elif not link.exists():
        link.symlink_to(target, target_is_directory=True)
    return str(root)


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
    should surface), ``False`` on a clean/requested shutdown. The
    no-telemetry counterpart of the llama_cpp adapter's ``telemetry_loop``."""
    while not stop_event.is_set():
        if process.poll() is not None:
            return True
        if stop_event.wait(interval):
            return False
    return False


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Launch ``omlx serve`` as a child process and supervise until shutdown.

    The worker process (this one) never loads tensors — omlx does, in the
    child. ``controller.shutdown_callback`` terminates that child on SIGTERM
    (escalating to SIGKILL after a bounded wait); an unexpected child exit
    raises so ``engine_worker`` surfaces a ``state_change: crashed`` event,
    same as any other adapter failure.
    """
    kwargs = cfg.engine_kwargs
    model_dir = prepare_model_dir(
        kwargs.get("model_dir", str(Path(cfg.model_path).parent / "omlx-models")),
        kwargs.get("model_name", cfg.served_model_name or cfg.service),
        cfg.model_path,
    )

    api_key = os.environ.get("SOVEREIGN_API_KEY")
    argv = ["omlx", *build_server_argv(kwargs, model_dir, cfg.host, cfg.port)]
    if api_key:
        argv += ["--api-key", api_key]

    logger.info("omlx worker %s: launching omlx serve", cfg.service)
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
        raise RuntimeError(f"omlx serve for '{cfg.service}' failed to become healthy")

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    if supervise(process, stop_event):
        raise RuntimeError(
            f"omlx serve for '{cfg.service}' exited unexpectedly (code={process.returncode})"
        )
