"""``python -m sovereign.workers.engine_worker --config <path>``: the generic
engine-worker process entrypoint.

Loads a :class:`~sovereign.workers.worker_config.WorkerConfig`, constructs a
:class:`~sovereign.workers.telemetry.TelemetryClient`, and dispatches to the
engine-specific adapter module named ``<package>.<engine>_adapter`` (default
package ``sovereign.workers``, overridable via
``SOVEREIGN_WORKER_ADAPTER_PACKAGE`` so tests can point the real entrypoint at
a fake adapter without touching any engine bindings).

Bindings (``llama_cpp``, ``mlx_lm``, ...) must only ever be imported inside an
adapter's ``run()`` function — nothing this module imports unconditionally may
pull one in, so this entrypoint stays importable (and testable) on any
platform.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import signal
import sys
import threading
import time
import traceback
from types import FrameType, ModuleType

import psutil

from sovereign.core.procmem import macos_phys_footprint
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient
from sovereign.workers.worker_config import WorkerConfig, load_worker_config

logger = logging.getLogger("sovereign")

#: Default package searched for ``<engine>_adapter`` modules; overridable via
#: SOVEREIGN_WORKER_ADAPTER_PACKAGE (a test seam for exercising the real
#: entrypoint against a fake adapter with no bindings installed).
_DEFAULT_ADAPTER_PACKAGE = "sovereign.workers"
_ADAPTER_PACKAGE_ENV = "SOVEREIGN_WORKER_ADAPTER_PACKAGE"

#: Heartbeat/memory-sample cadence (§2 of the plan).
_HEARTBEAT_INTERVAL = 2.0

#: Length of the traceback tail included in a crash's state_change payload.
_CRASH_DETAIL_LINES = 20


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sovereign.workers.engine_worker")
    parser.add_argument("--config", required=True, help="Path to a WorkerConfig JSON file")
    return parser.parse_args(argv)


def _adapter_package() -> str:
    import os

    return os.environ.get(_ADAPTER_PACKAGE_ENV) or _DEFAULT_ADAPTER_PACKAGE


def load_adapter(engine: str) -> ModuleType:
    """Import and return the adapter module for ``engine``.

    Resolves ``<package>.<engine>_adapter`` where ``package`` defaults to
    ``sovereign.workers`` and can be overridden via
    ``SOVEREIGN_WORKER_ADAPTER_PACKAGE``.
    """
    package = _adapter_package()
    module_name = f"{package}.{engine}_adapter"
    return importlib.import_module(module_name)


def _self_memory_bytes() -> int | None:
    """Best-effort resident memory of this process.

    Prefers :func:`macos_phys_footprint` (matches Activity Monitor / ``top``
    on macOS); falls back to psutil RSS everywhere else or on failure.
    """
    pid = psutil.Process().pid
    footprint = macos_phys_footprint(pid)
    if footprint is not None:
        return footprint
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except (psutil.Error, OSError):
        return None


def _heartbeat_loop(telemetry: TelemetryClient, stop: threading.Event) -> None:
    while not stop.is_set():
        telemetry.emit(EventType.HEARTBEAT, {})
        memory_bytes = _self_memory_bytes()
        if memory_bytes is not None:
            telemetry.emit(EventType.MEMORY, {"memory_bytes": memory_bytes})
        if stop.wait(_HEARTBEAT_INTERVAL):
            return


class _ShutdownController:
    """Coordinates a graceful SIGTERM shutdown between the signal handler and
    the adapter's blocking ``run()`` call.

    Adapters are handed this controller's ``stop_event`` (and may register a
    ``shutdown_callback`` for servers with an explicit shutdown hook, e.g.
    ``uvicorn.Server.should_exit``); the entrypoint doesn't assume which
    mechanism a given adapter uses.
    """

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.shutdown_callback: object | None = None

    def request_shutdown(self) -> None:
        self.stop_event.set()
        callback = self.shutdown_callback
        if callback is not None:
            try:
                callback()  # type: ignore[operator]
            except Exception:  # noqa: BLE001 - shutdown must never crash on the way out
                logger.debug("adapter shutdown callback raised", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg: WorkerConfig = load_worker_config(args.config)
    telemetry = TelemetryClient(cfg.telemetry_socket, cfg.service)

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(telemetry, heartbeat_stop),
        name=f"heartbeat-{cfg.service}",
        daemon=True,
    )

    controller = _ShutdownController()

    def _handle_sigterm(signum: int, frame: FrameType | None) -> None:
        telemetry.emit(EventType.STATE_CHANGE, {"state": "stopping"})
        controller.request_shutdown()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    exit_code = 0
    try:
        telemetry.emit(EventType.STATE_CHANGE, {"state": "loading"})
        heartbeat_thread.start()
        adapter = load_adapter(cfg.engine)
        adapter.run(cfg, telemetry, controller)
    except Exception:  # noqa: BLE001 - a crashing adapter must still report+exit cleanly
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-_CRASH_DETAIL_LINES:])
        telemetry.emit(EventType.STATE_CHANGE, {"state": "crashed", "detail": tail})
        logger.error("engine worker %s crashed", cfg.service, exc_info=True)
        exit_code = 1
    finally:
        heartbeat_stop.set()
        # Give the sender thread a brief window to flush the final events
        # (state_change stopping/crashed) before the socket is torn down.
        time.sleep(0.05)
        telemetry.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
