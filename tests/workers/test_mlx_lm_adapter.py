"""Pure/near-pure tests for the mlx_lm worker adapter — no ``mlx_lm`` binding
required, since ``build_server_namespace`` and ``wrap_stream_generate`` never
import it at module scope.
"""

from __future__ import annotations

from typing import Any, cast

from sovereign.workers.mlx_lm_adapter import build_server_namespace, wrap_stream_generate
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient


def test_build_server_namespace_merges_defaults_and_overrides():
    defaults = {"max_tokens": 512, "temp": 0.0, "top_p": 1.0, "host": "0.0.0.0", "port": 8080}
    namespace = build_server_namespace(
        {"max_tokens": 1024, "temp": 0.7},
        model_path="/models/foo",
        draft_model_path=None,
        defaults=defaults,
    )
    assert namespace["model"] == "/models/foo"
    assert namespace["max_tokens"] == 1024
    assert namespace["temp"] == 0.7
    assert namespace["top_p"] == 1.0  # untouched default
    assert "draft_model" not in namespace


def test_build_server_namespace_sets_draft_model_path():
    namespace = build_server_namespace(
        {}, model_path="/m", draft_model_path="/draft", defaults={}
    )
    assert namespace["draft_model"] == "/draft"


def test_build_server_namespace_passthrough_escape_hatch():
    namespace = build_server_namespace(
        {"some_future_flag": 42}, model_path="/m", draft_model_path=None, defaults={}
    )
    assert namespace["some_future_flag"] == 42


class _FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []

    def emit(self, event: Any, payload: Any) -> None:
        self.events.append((event, payload))

    def as_client(self) -> TelemetryClient:
        return cast(TelemetryClient, self)


class _FakeGenerationResponse:
    def __init__(self, prompt_tokens, prompt_tps, generation_tokens, generation_tps):
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = prompt_tps
        self.generation_tokens = generation_tokens
        self.generation_tps = generation_tps


def test_wrap_stream_generate_injects_progress_callback_and_emits_stats():
    telemetry = _FakeTelemetry()
    responses = [
        _FakeGenerationResponse(10, 100.0, 1, 0.0),
        _FakeGenerationResponse(10, 100.0, 2, 25.0),
    ]

    def fake_stream_generate(*args, prompt_progress_callback=None, **kwargs):
        if prompt_progress_callback is not None:
            prompt_progress_callback(5, 10)
            prompt_progress_callback(10, 10)
        yield from responses

    wrapped = wrap_stream_generate(fake_stream_generate, telemetry.as_client())
    results = list(wrapped("model", "tokenizer", "prompt"))
    assert results == responses

    event_types = [e for e, _ in telemetry.events]
    assert event_types.count(EventType.PREFILL_PROGRESS) == 2
    assert EventType.GENERATION_STATS in event_types
    gen_stats = next(p for e, p in telemetry.events if e == EventType.GENERATION_STATS)
    assert gen_stats["completion_tokens"] == 2
    assert gen_stats["generation_tps"] == 25.0


def test_wrap_stream_generate_falls_back_when_callback_kwarg_unsupported():
    telemetry = _FakeTelemetry()

    def fake_stream_generate_no_callback(*args, **kwargs):
        assert "prompt_progress_callback" not in kwargs
        yield _FakeGenerationResponse(5, 50.0, 1, 10.0)

    def strict_stream_generate(*args, **kwargs):
        if "prompt_progress_callback" in kwargs:
            raise TypeError("unexpected keyword argument 'prompt_progress_callback'")
        return fake_stream_generate_no_callback(*args, **kwargs)

    wrapped = wrap_stream_generate(strict_stream_generate, telemetry.as_client())
    results = list(wrapped("model", "tokenizer", "prompt"))
    assert len(results) == 1

    event_types = [e for e, _ in telemetry.events]
    assert EventType.PREFILL_PROGRESS not in event_types
    assert EventType.GENERATION_STATS in event_types
