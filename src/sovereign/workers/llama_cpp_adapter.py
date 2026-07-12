"""Engine-worker adapter for llama-cpp-python's built-in OpenAI-compatible server.

Only :func:`build_model_settings`, :func:`instrument_completions`, and
:func:`greedy_draft_tokens` are meant
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

    ``draft_model_path`` is deliberately not wired into the settings dict:
    ``ModelSettings.draft_model`` only accepts the string
    ``"prompt-lookup-decoding"``, so two-model speculation is attached to the
    loaded ``Llama`` instance directly (see :func:`run` /
    :func:`greedy_draft_tokens`) rather than through the server's settings.
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


def greedy_draft_tokens(draft: Any, input_ids: Any, num_pred_tokens: int) -> list[int]:
    """Greedy-decode ``num_pred_tokens`` continuation tokens from a draft model.

    ``draft`` duck-types the small slice of ``llama_cpp.Llama`` this needs —
    ``reset()``, ``eval(tokens: list[int])``, and ``sample() -> int`` — so the
    speculative loop is unit-testable with a deterministic fake and no binding
    installed. The context is reset and the full ``input_ids`` re-evaluated on
    every call: correctness first — the draft model is small, and incremental
    KV reuse across arbitrary candidate acceptance/rejection is llama.cpp's
    problem, not ours to replicate here.
    """
    tokens = [int(t) for t in input_ids]
    draft.reset()
    draft.eval(tokens)
    drafted: list[int] = []
    for _ in range(num_pred_tokens):
        token = int(draft.sample())
        drafted.append(token)
        draft.eval([token])
    return drafted


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

    model_settings_dict = build_model_settings(
        cfg.engine_kwargs, cfg.model_path, cfg.draft_model_path, cfg.served_model_name
    )
    model_settings = ModelSettings(**model_settings_dict)

    api_key = os.environ.get("SOVEREIGN_API_KEY")
    server_settings = ServerSettings(host=cfg.host, port=cfg.port, api_key=api_key)

    app = llama_app.create_app(server_settings=server_settings, model_settings=[model_settings])

    @app.get(cfg.health_path)
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    model = _app_model(app)
    if model is not None:
        _attach_draft_model(model, cfg)
        instrument_completions(model, telemetry)
    else:
        logger.warning(
            "could not reach the llama_cpp server's Llama instance; serving "
            "without prefill/generation stats%s",
            " or speculative decoding" if cfg.draft_model_path else "",
        )

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning")
    server = uvicorn.Server(config)
    controller.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


def _app_model(app: Any) -> Any | None:
    """Best-effort: reach the ``Llama`` instance the server app holds. The
    exact attribute llama_cpp.server.app uses for model state has moved
    across versions; a mismatch returns None so callers degrade gracefully
    (serve without stats/speculation) rather than crashing boot."""
    try:
        import llama_cpp.server.app as llama_app

        # create_app() stores the proxy in the module-global `_llama_proxy`
        # (set_llama_proxy), and LlamaProxy.__init__ eagerly loads the default
        # model into `_current_model` — verified against llama-cpp-python
        # 0.3.33. Both are private; a future rename fails soft to None here.
        proxy = getattr(llama_app, "_llama_proxy", None)
        if proxy is not None:
            return getattr(proxy, "_current_model", None)
    except Exception:  # noqa: BLE001 - accessor is best-effort
        logger.warning("could not reach llama_cpp server model", exc_info=True)
    return None


def _attach_draft_model(model: Any, cfg: WorkerConfig) -> None:
    """Wire speculative decoding onto the loaded ``Llama`` instance.

    With a ``draft_model_path``, loads the draft GGUF as a second ``Llama``
    and attaches Sovereign's own :class:`~llama_cpp.LlamaDraftModel`
    implementation (a greedy loop over :func:`greedy_draft_tokens`) —
    llama-cpp-python itself only ships prompt-lookup decoding, and the
    server's ``ModelSettings.draft_model`` accepts only that string form, so
    object injection has to happen post-construction. Without a draft path
    but with ``num_draft_tokens``, falls back to prompt-lookup decoding.
    Failures degrade to plain decoding with a warning.
    """
    num_pred_tokens = int(cfg.engine_kwargs.get("num_draft_tokens") or 10)
    try:
        if cfg.draft_model_path:
            import numpy as np
            from llama_cpp import Llama
            from llama_cpp.llama_speculative import LlamaDraftModel

            draft_llama = Llama(
                model_path=cfg.draft_model_path,
                n_gpu_layers=-1,
                n_ctx=int(cfg.engine_kwargs.get("context_size") or 0),
                verbose=False,
            )

            class LlamaGgufDraftModel(LlamaDraftModel):
                """Two-model speculation: greedy candidates from a draft GGUF."""

                def __call__(self, input_ids: Any, /, **kwargs: Any) -> Any:
                    drafted = greedy_draft_tokens(draft_llama, input_ids, num_pred_tokens)
                    return np.asarray(drafted, dtype=np.intc)

            model.draft_model = LlamaGgufDraftModel()
            # Llama.__init__ forces logits_all=True whenever a draft model is
            # passed at construction time; replicate that for the post-hoc
            # attachment so candidate verification sees per-token logits.
            if hasattr(model, "_logits_all"):
                model._logits_all = True
            logger.info(
                "llama_cpp worker %s: two-model speculative decoding active "
                "(draft=%s, num_pred_tokens=%d)",
                cfg.service,
                cfg.draft_model_path,
                num_pred_tokens,
            )
        elif cfg.engine_kwargs.get("num_draft_tokens"):
            from llama_cpp import LlamaPromptLookupDecoding

            model.draft_model = LlamaPromptLookupDecoding(num_pred_tokens=num_pred_tokens)
    except Exception:  # noqa: BLE001 - fail soft, serve without speculative decoding
        logger.warning(
            "failed to configure speculative decoding; serving without it", exc_info=True
        )
