"""Engine-worker adapter for ``mlx_vlm.server``.

Only :func:`build_server_argv` is meant to be imported at module scope
elsewhere (pure, unit-testable without ``mlx_vlm`` installed). :func:`run`
is the adapter entrypoint and the only place in this module allowed to
import ``mlx_vlm`` — always lazily.

No telemetry instrumentation in v1: mlx-vlm's server is an async
FastAPI/uvicorn app with no verified single generate seam to wrap (unlike
``mlx_lm``'s ``ResponseGenerator.generate``), and its ``/metrics`` schema is
unverified, so a llama_cpp-style translator loop would be built blind. The
manager logs the resulting ADR 0006 gap at pre-flight; heartbeat + memory
events come from the generic ``engine_worker``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sovereign.workers.protocol import EventType

if TYPE_CHECKING:
    from sovereign.workers.telemetry import TelemetryClient
    from sovereign.workers.worker_config import WorkerConfig

logger = logging.getLogger("sovereign")


def build_server_argv(
    engine_kwargs: dict[str, Any],
    model_path: str,
    draft_model_path: str | None,
    host: str,
    port: int,
) -> list[str]:
    """Translate Sovereign's engine-agnostic kwargs into an ``mlx_vlm.server``
    CLI argv (sans program name). The adapter's ``run()`` hands this to
    ``mlx_vlm.server.cli.main()`` so the *pinned dependency's own argparse
    defaults* apply (verified against mlx-vlm's server CLI: --max-tokens,
    --prefill-step-size, --vision-cache-size, --kv-bits, --kv-quant-scheme,
    --kv-group-size, --max-kv-size, --quantized-kv-start, --draft-model,
    --draft-kind, --draft-block-size, --adapter-path, --trust-remote-code,
    --enable-thinking, --thinking-budget).

    Boolean kwargs become bare flags when true; everything else becomes
    ``--kebab-case value``. Unknown engine_kwargs pass through the same way
    (escape hatch) — a flag the pinned parser doesn't know is a loud
    argparse error at worker boot, not silent misconfiguration. The API key
    never appears here: it travels via the ``MLX_VLM_SERVER_API_KEY``
    environment variable the server reads natively.
    """
    argv = ["--model", model_path, "--host", host, "--port", str(port)]
    if draft_model_path is not None:
        argv += ["--draft-model", draft_model_path]
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Boot ``mlx_vlm.server`` in-process and serve until shutdown.

    Lazily imports ``mlx_vlm`` — the only place in this module allowed to do
    so. ``main()`` blocks in ``uvicorn.run(...)`` (single worker, no reload,
    so no child process).

    Shutdown: uvicorn installs its own SIGTERM/SIGINT handlers once its event
    loop starts, replacing ``engine_worker``'s — from then on SIGTERM drives
    uvicorn's graceful shutdown and ``main()`` returns cleanly (the worker's
    own ``stopping`` telemetry event is skipped; cosmetic only). The
    ``interrupt_main`` callback below covers the window *before* that
    takeover, where loading a large model can hold the main thread for
    minutes.
    """
    import _thread
    import sys

    import mlx_vlm.server.cli as vlm_cli

    argv = build_server_argv(
        cfg.engine_kwargs, cfg.model_path, cfg.draft_model_path, cfg.host, cfg.port
    )

    controller.shutdown_callback = _thread.interrupt_main

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    # Drive the pinned dependency's own entrypoint (its argparse defaults and
    # app wiring apply verbatim) rather than reconstructing its internals —
    # same contract as the mlx_lm adapter.
    saved_argv = sys.argv
    sys.argv = ["mlx_vlm.server", *argv]
    try:
        vlm_cli.main()
    finally:
        sys.argv = saved_argv
