"""Pure/near-pure tests for the mlx_lm worker adapter — no ``mlx_lm`` binding
required, since ``build_server_argv`` and ``wrap_stream_generate`` never
import it at module scope.
"""

from __future__ import annotations

from typing import Any, cast

from sovereign.workers.mlx_lm_adapter import build_server_argv, wrap_stream_generate
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient


def test_build_server_argv_maps_kwargs_to_kebab_flags():
    argv = build_server_argv(
        {"max_tokens": 1024, "temp": 0.7, "num_draft_tokens": 4},
        model_path="/models/foo",
        draft_model_path=None,
        host="127.0.0.1",
        port=9000,
    )
    assert argv[:6] == ["--model", "/models/foo", "--host", "127.0.0.1", "--port", "9000"]
    assert argv[argv.index("--max-tokens") + 1] == "1024"
    assert argv[argv.index("--temp") + 1] == "0.7"
    assert argv[argv.index("--num-draft-tokens") + 1] == "4"
    assert "--draft-model" not in argv


def test_build_server_argv_sets_draft_model_path():
    argv = build_server_argv({}, model_path="/m", draft_model_path="/draft", host="h", port=1)
    assert argv[argv.index("--draft-model") + 1] == "/draft"


def test_build_server_argv_bools_become_bare_flags():
    argv = build_server_argv(
        {"trust_remote_code": True, "use_default_chat_template": False},
        model_path="/m",
        draft_model_path=None,
        host="h",
        port=1,
    )
    assert "--trust-remote-code" in argv
    assert "--use-default-chat-template" not in argv


def test_build_server_argv_passthrough_escape_hatch():
    argv = build_server_argv(
        {"some_future_flag": 42}, model_path="/m", draft_model_path=None, host="h", port=1
    )
    assert argv[argv.index("--some-future-flag") + 1] == "42"


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


def test_wrap_stream_generate_chains_existing_progress_callback():
    # mlx_lm.server's handler always passes its own prompt_progress_callback;
    # the wrapper must emit telemetry AND still invoke the handler's callback.
    telemetry = _FakeTelemetry()
    seen: list[tuple[int, int]] = []

    def fake_stream_generate(*args, **kwargs):
        callback = kwargs["prompt_progress_callback"]
        callback(2, 4)
        yield _FakeGenerationResponse(4, 40.0, 1, 10.0)

    wrapped = wrap_stream_generate(fake_stream_generate, telemetry.as_client())
    list(wrapped("m", "t", "p", prompt_progress_callback=lambda p, t: seen.append((p, t))))

    assert seen == [(2, 4)]
    prefill = [p for e, p in telemetry.events if e == EventType.PREFILL_PROGRESS]
    assert prefill and prefill[0]["processed"] == 2 and prefill[0]["total"] == 4


def test_wrap_stream_generate_emits_stats_when_consumer_breaks_early():
    # mlx_lm.server's handler breaks out of its loop on finish_reason rather
    # than exhausting the generator — stats must still be emitted.
    telemetry = _FakeTelemetry()

    def fake_stream_generate(*args, **kwargs):
        yield _FakeGenerationResponse(5, 50.0, 1, 10.0)
        yield _FakeGenerationResponse(5, 50.0, 2, 20.0)

    wrapped = wrap_stream_generate(fake_stream_generate, telemetry.as_client())
    gen = wrapped("m", "t", "p")
    next(gen)
    gen.close()  # simulates the handler's break (GeneratorExit)

    gen_stats = [p for e, p in telemetry.events if e == EventType.GENERATION_STATS]
    assert len(gen_stats) == 1
    assert gen_stats[0]["generation_tps"] == 10.0
