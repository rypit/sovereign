"""Engine-worker adapter for ``mlx_lm.server``.

Only :func:`build_server_namespace` and :func:`wrap_stream_generate` are meant
to be imported at module scope elsewhere (pure/near-pure, unit-testable
without ``mlx_lm`` installed). :func:`run` is the adapter entrypoint and the
only place in this module allowed to import ``mlx_lm`` — always lazily.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sovereign.workers.protocol import EventType

if TYPE_CHECKING:
    from sovereign.workers.telemetry import TelemetryClient
    from sovereign.workers.worker_config import WorkerConfig

logger = logging.getLogger("sovereign")

#: engine_kwargs keys that map 1:1 onto mlx_lm.server CLI flags (0.31.x:
#: --max-tokens, --temp, --top-p, --adapter-path, --trust-remote-code,
#: --num-draft-tokens, --prompt-cache-size, --prompt-cache-bytes,
#: --decode-concurrency — all verified against the pinned parser in main()).
_PASSTHROUGH_KEYS = (
    "max_tokens",
    "temp",
    "top_p",
    "adapter_path",
    "trust_remote_code",
    "num_draft_tokens",
    "prompt_cache_size",
    "prompt_cache_bytes",
    "decode_concurrency",
)


def build_server_argv(
    engine_kwargs: dict[str, Any],
    model_path: str,
    draft_model_path: str | None,
    host: str,
    port: int,
) -> list[str]:
    """Translate Sovereign's engine-agnostic kwargs into an ``mlx_lm.server``
    CLI argv (sans program name). The adapter's ``run()`` hands this to
    ``mlx_lm.server.main()`` so the *pinned dependency's own argparse
    defaults* apply — the previous approach of reconstructing its namespace
    from a parser helper broke because mlx-lm 0.31.x exposes no such helper.

    Boolean kwargs become bare flags when true; everything else becomes
    ``--kebab-case value``. Unknown engine_kwargs pass through the same way
    (escape hatch) — a flag the pinned parser doesn't know is a loud
    argparse error at worker boot, not silent misconfiguration.
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


def wrap_stream_generate(orig: Any, telemetry: TelemetryClient) -> Any:
    """Wrap ``mlx_lm.server``'s module-global ``stream_generate`` so prefill
    progress and final generation stats get emitted as telemetry.

    ``orig`` only needs to be a callable that accepts a ``prompt_progress_callback``
    keyword (mlx_lm's real signature) and returns/yields ``GenerationResponse``-like
    objects with ``prompt_tokens``/``prompt_tps``/``generation_tokens``/
    ``generation_tps`` attributes — this makes the wrapper testable against a
    bare fake generator function, no ``mlx_lm`` import required. Any mismatch in
    the real API (attributes moved, callback kwarg renamed) is caught and logged;
    the wrapper then falls back to calling ``orig`` unmodified so serving
    continues without stats.
    """

    def wrapped(*args: Any, **kwargs: Any):
        request_id = f"{id(kwargs) ^ int(time.time() * 1000)}"

        def _progress_callback(processed: int, total: int) -> None:
            telemetry.emit(
                EventType.PREFILL_PROGRESS,
                {"request_id": request_id, "processed": processed, "total": total},
            )

        existing_callback = kwargs.get("prompt_progress_callback")

        def _chained(processed: int, total: int) -> None:
            _progress_callback(processed, total)
            if callable(existing_callback):
                existing_callback(processed, total)

        try:
            # mlx_lm.server's handler passes its own prompt_progress_callback;
            # chain ours in front rather than deferring to it, or telemetry
            # would never see prefill progress.
            kwargs["prompt_progress_callback"] = _chained
            generator = orig(*args, **kwargs)
        except TypeError:
            # orig doesn't accept prompt_progress_callback (API mismatch) —
            # fail soft: serve without prefill progress.
            logger.warning(
                "mlx_lm.stream_generate does not accept prompt_progress_callback; "
                "serving without prefill progress",
                exc_info=True,
            )
            if callable(existing_callback):
                kwargs["prompt_progress_callback"] = existing_callback
            else:
                kwargs.pop("prompt_progress_callback", None)
            generator = orig(*args, **kwargs)

        last_response = None
        for response in generator:
            last_response = response
            yield response

        if last_response is not None:
            _emit_generation_stats(last_response, telemetry, request_id)

    return wrapped


def _emit_generation_stats(response: Any, telemetry: TelemetryClient, request_id: str) -> None:
    try:
        payload = {
            "request_id": request_id,
            "prompt_tokens": getattr(response, "prompt_tokens", None),
            "completion_tokens": getattr(response, "generation_tokens", None),
            "prompt_tps": getattr(response, "prompt_tps", None),
            "generation_tps": getattr(response, "generation_tps", None),
        }
        telemetry.emit(EventType.GENERATION_STATS, payload)
    except Exception:  # noqa: BLE001 - fail soft, never break the serving loop
        logger.warning("failed to emit generation_stats", exc_info=True)


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Boot ``mlx_lm.server`` in-process and serve until shutdown.

    Lazily imports ``mlx_lm`` — the only place in this module allowed to do
    so. Version-pinned per the plan; wrapping failures are caught and logged
    so a moved attribute degrades to "serves without stats" rather than
    crashing boot.
    """
    import _thread
    import sys

    import mlx_lm.server as mlx_server

    try:
        original_stream_generate = mlx_server.stream_generate
        mlx_server.stream_generate = wrap_stream_generate(original_stream_generate, telemetry)
    except Exception:  # noqa: BLE001 - fail soft, serve without stats
        logger.warning(
            "could not instrument mlx_lm.server.stream_generate; serving without stats",
            exc_info=True,
        )

    argv = build_server_argv(
        cfg.engine_kwargs, cfg.model_path, cfg.draft_model_path, cfg.host, cfg.port
    )

    # SIGTERM cooperation: mlx's serve loop catches KeyboardInterrupt and
    # shuts the httpd + response generator down cleanly, so interrupting the
    # main thread IS the graceful-stop path.
    controller.shutdown_callback = _thread.interrupt_main

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    # Drive the pinned dependency's own entrypoint (its argparse defaults and
    # ModelProvider/ResponseGenerator wiring apply verbatim) rather than
    # reconstructing its internals — mlx-lm 0.31.x has no stable programmatic
    # construction surface.
    saved_argv = sys.argv
    sys.argv = ["mlx_lm.server", *argv]
    try:
        mlx_server.main()
    finally:
        sys.argv = saved_argv
