"""Pure/near-pure tests for the llama_cpp worker adapter — no ``llama_cpp``
binding required, since ``build_model_settings`` and ``instrument_completions``
never import it at module scope.
"""

from __future__ import annotations

from typing import Any, cast

from sovereign.workers.llama_cpp_adapter import (
    build_model_settings,
    greedy_draft_tokens,
    instrument_completions,
)
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient


def test_build_model_settings_renames_and_alias():
    settings = build_model_settings(
        {"gpu_layers": 20, "threads": 4, "context_size": 4096},
        model_path="/models/foo.gguf",
        draft_model_path=None,
        alias="foo",
    )
    assert settings["model"] == "/models/foo.gguf"
    assert settings["model_alias"] == "foo"
    assert settings["n_gpu_layers"] == 20
    assert settings["n_threads"] == 4
    assert settings["n_ctx"] == 4096


def test_build_model_settings_kv_cache_type_maps_to_ggml_ints():
    settings = build_model_settings(
        {"kv_cache_type": "q8_0"}, model_path="/m.gguf", draft_model_path=None, alias=None
    )
    assert settings["type_k"] == 8
    assert settings["type_v"] == 8


def test_build_model_settings_unknown_kv_cache_type_is_dropped_not_raised():
    settings = build_model_settings(
        {"kv_cache_type": "bogus"}, model_path="/m.gguf", draft_model_path=None, alias=None
    )
    assert "type_k" not in settings
    assert "type_v" not in settings


def test_build_model_settings_passthrough_escape_hatch():
    settings = build_model_settings(
        {"rope_freq_base": 1000000.0}, model_path="/m.gguf", draft_model_path=None, alias=None
    )
    assert settings["rope_freq_base"] == 1000000.0


class _FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []

    def emit(self, event: Any, payload: Any) -> None:
        self.events.append((event, payload))

    def as_client(self) -> TelemetryClient:
        return cast(TelemetryClient, self)


class _FakeLlama:
    def __init__(self, chunks):
        self._chunks = chunks

    def create_completion(self, prompt=None, **kwargs):
        return iter(self._chunks)


def test_instrument_completions_emits_prefill_and_generation_stats():
    telemetry = _FakeTelemetry()
    fake_llama = _FakeLlama([{"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}])
    instrument_completions(fake_llama, telemetry.as_client())

    chunks = list(fake_llama.create_completion(prompt="hello"))
    assert len(chunks) == 2

    event_types = [e for e, _ in telemetry.events]
    assert event_types[0] == EventType.PREFILL_PROGRESS
    assert EventType.GENERATION_STATS in event_types
    gen_stats = next(p for e, p in telemetry.events if e == EventType.GENERATION_STATS)
    assert gen_stats["completion_tokens"] == 2
    assert gen_stats["generation_tps"] > 0


class _FakeDraftLlama:
    """Deterministic draft model: next token = last seen token + 1.

    Records reset()/eval() calls so tests can assert the loop re-primes the
    context from scratch on every speculation round.
    """

    def __init__(self) -> None:
        self.last_token: int | None = None
        self.reset_calls = 0
        self.eval_history: list[list[int]] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.last_token = None

    def eval(self, tokens: list[int]) -> None:
        self.eval_history.append(list(tokens))
        self.last_token = tokens[-1]

    def sample(self) -> int:
        assert self.last_token is not None, "sample() before eval()"
        return self.last_token + 1


def test_greedy_draft_tokens_generates_num_pred_tokens():
    draft = _FakeDraftLlama()
    tokens = greedy_draft_tokens(draft, [5, 6, 7], num_pred_tokens=4)
    # Greedy continuation of the deterministic +1 fake: 8, 9, 10, 11.
    assert tokens == [8, 9, 10, 11]
    # Prompt evaluated whole, then one eval per drafted token.
    assert draft.eval_history[0] == [5, 6, 7]
    assert len(draft.eval_history) == 1 + 4


def test_greedy_draft_tokens_resets_between_calls():
    draft = _FakeDraftLlama()
    first = greedy_draft_tokens(draft, [1], num_pred_tokens=2)
    second = greedy_draft_tokens(draft, [1], num_pred_tokens=2)
    # Identical input yields identical drafts — no state leaks across rounds.
    assert first == second == [2, 3]
    assert draft.reset_calls == 2


def test_greedy_draft_tokens_accepts_numpy_like_input_ids():
    class _IntLike:
        def __init__(self, v: int) -> None:
            self._v = v

        def __int__(self) -> int:
            return self._v

    draft = _FakeDraftLlama()
    tokens = greedy_draft_tokens(draft, [_IntLike(10), _IntLike(11)], num_pred_tokens=1)
    assert tokens == [12]


def test_instrument_completions_is_noop_for_missing_attrs():
    telemetry = _FakeTelemetry()

    class _Bare:
        pass

    # Must not raise even though _Bare has neither completion method.
    instrument_completions(_Bare(), telemetry.as_client())
    assert telemetry.events == []
