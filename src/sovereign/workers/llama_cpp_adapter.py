"""Engine-worker adapter for llama-cpp-python's built-in OpenAI-compatible server.

Only :func:`build_model_settings` and :func:`instrument_completions` are meant
to be imported at module scope elsewhere (they're pure/near-pure and
unit-testable without the ``llama_cpp`` binding installed). :func:`run` is the
adapter entrypoint the worker calls, and it is the only place in this module
that imports ``llama_cpp`` — always lazily, never at module scope, so this
file stays importable on any platform (including linux CI) even though
``llama_cpp`` is a macOS/arm64-only dependency.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from sovereign.workers.protocol import EventType

if TYPE_CHECKING:
    from sovereign.workers.telemetry import TelemetryClient
    from sovereign.workers.worker_config import WorkerConfig

logger = logging.getLogger("sovereign")

#: name -> GGML_TYPE_* enum value (ggml.h). Kept local rather than imported
#: from llama_cpp so build_model_settings stays importable without the
#: binding: verify against the pinned llama-cpp-python's
#: ``llama_cpp.llama_cpp.GGML_TYPE_*`` constants if bumping the pin.
_KV_CACHE_TYPES: dict[str, int] = {
    "f16": 1,
    "q8_0": 8,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
}

#: engine_kwargs keys mapped onto ModelSettings fields with a straight rename.
_KWARG_RENAMES: dict[str, str] = {
    "gpu_layers": "n_gpu_layers",
    "threads": "n_threads",
    "context_size": "n_ctx",
}

#: engine_kwargs keys consumed here rather than passed through verbatim.
_CONSUMED_KEYS = frozenset({*_KWARG_RENAMES, "kv_cache_type"})


def build_model_settings(
    engine_kwargs: dict[str, Any],
    model_path: str,
    draft_model_path: str | None,
    alias: str | None,
) -> dict[str, Any]:
    """Map Sovereign's engine-agnostic kwargs onto a llama-cpp-python
    ``ModelSettings``-shaped dict (kept as a plain dict so this stays
    importable/testable without ``llama_cpp`` installed).

    Recognized keys: ``gpu_layers`` -> ``n_gpu_layers``, ``threads`` ->
    ``n_threads``, ``context_size`` -> ``n_ctx``, ``kv_cache_type`` (a name
    like ``"q8_0"``) -> ``type_k``/``type_v``. ``alias`` -> ``model_alias``.
    Any other ``engine_kwargs`` entry is passed through verbatim as an escape
    hatch for settings this mapping doesn't know about yet.

    Speculative decoding via a second GGUF is a hard gap in the bindings
    (§3a) — ``draft_model_path`` is accepted for signature symmetry with the
    mlx adapter but is intentionally not wired into the settings; callers
    that need draft-token behavior should use ``num_draft_tokens`` for
    prompt-lookup decoding instead (handled in :func:`run`).
    """
    settings: dict[str, Any] = {"model": model_path}
    if alias is not None:
        settings["model_alias"] = alias

    for src_key, dst_key in _KWARG_RENAMES.items():
        if src_key in engine_kwargs:
            settings[dst_key] = engine_kwargs[src_key]

    kv_cache_type = engine_kwargs.get("kv_cache_type")
    if kv_cache_type is not None:
        ggml_type = _KV_CACHE_TYPES.get(str(kv_cache_type))
        if ggml_type is not None:
            settings["type_k"] = ggml_type
            settings["type_v"] = ggml_type
        else:
            logger.warning(
                "unknown kv_cache_type %r; not passing type_k/type_v to llama_cpp",
                kv_cache_type,
            )

    for key, value in engine_kwargs.items():
        if key not in _CONSUMED_KEYS:
            settings[key] = value

    return settings


def instrument_completions(llama_like: Any, telemetry: TelemetryClient) -> None:
    """Wrap ``llama_like.create_completion``/``create_chat_completion`` so
    prefill/generation telemetry is emitted around each call.

    ``llama_like`` only needs to duck-type ``create_completion`` (and
    optionally ``create_chat_completion``) as callables returning either a
    dict (non-streaming) or an iterator of dicts (streaming) — this makes the
    wrapper unit-testable with a bare fake object, no ``llama_cpp`` import
    required. Failures here must never break serving: any exception while
    instrumenting is logged and the original callable is left in place.
    """
    for attr in ("create_completion", "create_chat_completion"):
        original = getattr(llama_like, attr, None)
        if original is None or not callable(original):
            continue
        try:
            wrapped = _wrap_one(original, telemetry)
            setattr(llama_like, attr, wrapped)
        except Exception:  # noqa: BLE001 - fail soft, serve without stats
            logger.warning("failed to instrument %s; serving without stats", attr, exc_info=True)


def _wrap_one(original: Any, telemetry: TelemetryClient) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        request_id = f"{id(kwargs) ^ int(time.time() * 1000)}"
        prompt = kwargs.get("prompt") or (args[0] if args else None)
        total: int | None = None
        if isinstance(prompt, str):
            # Cheap proxy: llama_cpp isn't asked to tokenize just for a
            # progress estimate. None (indeterminate) is a perfectly valid
            # signal per the wire protocol.
            total = None
        telemetry.emit(
            EventType.PREFILL_PROGRESS,
            {"request_id": request_id, "processed": 0, "total": total},
        )
        result = original(*args, **kwargs)
        if not hasattr(result, "__next__") and not hasattr(result, "__iter__"):
            # Non-streaming: nothing to wrap, but still report basic stats.
            return result
        if isinstance(result, dict):
            return result
        return _wrap_stream(result, telemetry, request_id, total)

    return wrapped


def _wrap_stream(stream: Any, telemetry: TelemetryClient, request_id: str, total: int | None):
    start = time.monotonic()
    first_chunk_seen = False
    completion_tokens = 0
    for chunk in stream:
        completion_tokens += 1
        if not first_chunk_seen:
            first_chunk_seen = True
            elapsed = max(time.monotonic() - start, 1e-6)
            prompt_tps = (total / elapsed) if total else None
            telemetry.emit(
                EventType.PREFILL_PROGRESS,
                {"request_id": request_id, "processed": total or 0, "total": total},
            )
            if prompt_tps is not None:
                _first_chunk_prompt_tps[request_id] = prompt_tps
        yield chunk
    elapsed = max(time.monotonic() - start, 1e-6)
    generation_tps = completion_tokens / elapsed
    telemetry.emit(
        EventType.GENERATION_STATS,
        {
            "request_id": request_id,
            "prompt_tokens": total,
            "completion_tokens": completion_tokens,
            "prompt_tps": _first_chunk_prompt_tps.pop(request_id, None),
            "generation_tps": generation_tps,
        },
    )


#: Small side-table for prompt_tps recorded at first-chunk time, consumed
#: when the stream is exhausted. Keyed by request_id; entries are popped so
#: this never grows unbounded across the worker's lifetime.
_first_chunk_prompt_tps: dict[str, float] = {}


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Boot the llama-cpp-python server in-process and serve until shutdown.

    Lazily imports ``llama_cpp``/``uvicorn``/``fastapi`` — this function is
    the only place in the module allowed to do so.
    """
    import llama_cpp.server.app as llama_app
    import uvicorn
    from llama_cpp.server.settings import ModelSettings, ServerSettings

    max_parallel = cfg.engine_kwargs.get("max_parallel")
    if isinstance(max_parallel, int) and max_parallel > 1:
        logger.warning(
            "llama_cpp worker %s: max_parallel=%d requested but the embedded "
            "server is single-Llama/sequential; requests will queue.",
            cfg.service,
            max_parallel,
        )

    if cfg.draft_model_path:
        raise RuntimeError(
            "llama_cpp engine does not support a second-GGUF draft model for "
            "speculative decoding in this binding; remove draft_model_path or "
            "use num_draft_tokens (prompt-lookup decoding) instead."
        )

    model_settings_dict = build_model_settings(
        cfg.engine_kwargs, cfg.model_path, cfg.draft_model_path, cfg.served_model_name
    )

    num_draft_tokens = cfg.engine_kwargs.get("num_draft_tokens")
    if num_draft_tokens:
        try:
            from llama_cpp import LlamaPromptLookupDecoding

            model_settings_dict["draft_model"] = LlamaPromptLookupDecoding(
                num_pred_tokens=int(num_draft_tokens)
            )
        except Exception:  # noqa: BLE001 - fail soft, serve without speculative decoding
            logger.warning("failed to configure prompt-lookup decoding", exc_info=True)

    model_settings = ModelSettings(**model_settings_dict)

    api_key = os.environ.get("SOVEREIGN_API_KEY")
    server_settings = ServerSettings(host=cfg.host, port=cfg.port, api_key=api_key)

    app = llama_app.create_app(server_settings=server_settings, model_settings=[model_settings])

    @app.get(cfg.health_path)
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    _instrument_app_llama(app, telemetry)

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
    server = uvicorn.Server(config)
    controller.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


def _instrument_app_llama(app: Any, telemetry: TelemetryClient) -> None:
    """Best-effort: reach into the app's held ``Llama`` instance(s) and wrap
    their completion methods. The exact attribute llama_cpp.server.app uses
    to hold model state has moved across versions; this is guarded so a
    mismatch degrades to "serves without stats" rather than crashing boot.
    """
    try:
        from llama_cpp.server.app import get_llama_proxy

        proxy = getattr(app.state, "llama_proxy", None) or get_llama_proxy
        # Different llama-cpp-python versions expose the loaded model(s)
        # differently (a proxy, a dict keyed by model alias, ...). Try the
        # shapes we know about; anything else fails soft.
        if proxy is not None and hasattr(proxy, "_current_model"):
            model = getattr(proxy, "_current_model", None)
            if model is not None:
                instrument_completions(model, telemetry)
    except Exception:  # noqa: BLE001 - instrumentation is best-effort
        logger.warning(
            "could not instrument llama_cpp server model for telemetry; "
            "serving without prefill/generation stats",
            exc_info=True,
        )
