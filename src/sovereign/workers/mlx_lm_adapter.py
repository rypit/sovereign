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

#: engine_kwargs keys that map 1:1 onto mlx_lm.server argparse namespace attrs.
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


def build_server_namespace(
    engine_kwargs: dict[str, Any],
    model_path: str,
    draft_model_path: str | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Merge Sovereign's engine-agnostic kwargs onto mlx_lm.server's real
    argparse defaults (passed in as ``defaults`` — the caller gets these from
    ``mlx_lm.server``'s own parser via ``parser.parse_args([])`` — so this
    function itself never needs to import ``mlx_lm``, and stays testable with
    a fake ``defaults`` dict).

    Returns a plain dict (namespace-shaped); the adapter's ``run()`` turns it
    into whatever object ``mlx_lm.server`` actually expects.
    """
    namespace = dict(defaults)
    namespace["model"] = model_path
    if draft_model_path is not None:
        namespace["draft_model"] = draft_model_path

    for key in _PASSTHROUGH_KEYS:
        if key in engine_kwargs:
            namespace[key] = engine_kwargs[key]

    # Escape hatch: pass through anything else verbatim too.
    for key, value in engine_kwargs.items():
        if key not in _PASSTHROUGH_KEYS:
            namespace[key] = value

    return namespace


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

        try:
            kwargs.setdefault("prompt_progress_callback", _progress_callback)
            generator = orig(*args, **kwargs)
        except TypeError:
            # orig doesn't accept prompt_progress_callback (API mismatch) —
            # fail soft: serve without prefill progress.
            logger.warning(
                "mlx_lm.stream_generate does not accept prompt_progress_callback; "
                "serving without prefill progress",
                exc_info=True,
            )
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
    import argparse

    import mlx_lm.server as mlx_server

    parser = mlx_server.setup_arg_parser()  # verify against pinned mlx-lm version
    defaults = vars(parser.parse_args([]))

    namespace_dict = build_server_namespace(
        cfg.engine_kwargs, cfg.model_path, cfg.draft_model_path, defaults
    )
    namespace_dict["host"] = cfg.host
    namespace_dict["port"] = cfg.port
    namespace = argparse.Namespace(**namespace_dict)

    try:
        original_stream_generate = mlx_server.stream_generate
        mlx_server.stream_generate = wrap_stream_generate(original_stream_generate, telemetry)
    except Exception:  # noqa: BLE001 - fail soft, serve without stats
        logger.warning(
            "could not instrument mlx_lm.server.stream_generate; serving without stats",
            exc_info=True,
        )

    model_provider = mlx_server.ModelProvider(namespace)

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    httpd = mlx_server.run(
        namespace.host,
        namespace.port,
        model_provider,
    )
    # mlx_lm.server.run() is a blocking serve_forever() under the hood in
    # published versions; if it instead returns a server object, cooperate
    # with the shutdown controller so SIGTERM stops it gracefully.
    if httpd is not None and hasattr(httpd, "shutdown"):
        controller.shutdown_callback = httpd.shutdown
        controller.stop_event.wait()
        httpd.shutdown()
