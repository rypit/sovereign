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


def wrap_response_generate(orig: Any, telemetry: TelemetryClient) -> Any:
    """Wrap ``mlx_lm.server``'s ``ResponseGenerator.generate`` — the one seam
    both its sequential (``stream_generate``) and batched (``BatchGenerator``)
    paths flow through (batchable models never touch ``stream_generate``, so
    wrapping that alone misses most real traffic).

    ``orig(self, request, generation_args, progress_callback)`` returns
    ``(ctx, response-iterator)``; prefill progress arrives as
    ``(processed, total)`` calls on ``progress_callback``. Since mlx's
    ``Response`` objects carry no throughput fields, the wrapper measures
    timing itself: prefill window = call → first yielded token, generation
    window = first token → stream end. Stats emit from a ``finally`` because
    the HTTP handler abandons the iterator on finish_reason rather than
    exhausting it. Testable with a fake ``orig``; any API mismatch is the
    caller's job to catch (run() patches under a try/except and fails soft).
    """

    def wrapped(
        self: Any,
        request: Any,
        generation_args: Any,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> Any:
        request_id = f"{id(request) ^ int(time.time() * 1000)}"
        t0 = time.monotonic()
        prefill_state: dict[str, Any] = {"processed": 0, "total": None}

        def _chained(processed: int, total: int) -> None:
            prefill_state["processed"] = processed
            prefill_state["total"] = total
            telemetry.emit(
                EventType.PREFILL_PROGRESS,
                {"request_id": request_id, "processed": processed, "total": total},
            )
            if callable(progress_callback):
                progress_callback(processed, total)

        ctx, stream = orig(self, request, generation_args, _chained, **kwargs)

        def _instrumented() -> Any:
            completion_tokens = 0
            t_first: float | None = None
            try:
                for response in stream:
                    if t_first is None:
                        t_first = time.monotonic()
                    completion_tokens += 1
                    yield response
            finally:
                if completion_tokens and t_first is not None:
                    prompt_tokens = prefill_state["processed"] or None
                    prefill_elapsed = max(t_first - t0, 1e-6)
                    generation_elapsed = max(time.monotonic() - t_first, 1e-6)
                    telemetry.emit(
                        EventType.GENERATION_STATS,
                        {
                            "request_id": request_id,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "prompt_tps": (
                                prompt_tokens / prefill_elapsed if prompt_tokens else None
                            ),
                            "generation_tps": completion_tokens / generation_elapsed,
                        },
                    )

        return ctx, _instrumented()

    return wrapped


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
        original_generate = mlx_server.ResponseGenerator.generate
        mlx_server.ResponseGenerator.generate = wrap_response_generate(
            original_generate, telemetry
        )
    except Exception:  # noqa: BLE001 - fail soft, serve without stats
        logger.warning(
            "could not instrument mlx_lm.server ResponseGenerator; serving without stats",
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
