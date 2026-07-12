"""Pure/near-pure tests for the mlx_lm worker adapter — no ``mlx_lm`` binding
required, since ``build_server_argv`` and ``wrap_stream_generate`` never
import it at module scope.
"""

from __future__ import annotations

from typing import Any, cast

from sovereign.workers.mlx_lm_adapter import build_server_argv, wrap_response_generate
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


def _fake_generate_factory(responses, prefill=(4, 4)):
    """A fake ResponseGenerator.generate: reports prefill via the callback,
    then returns (ctx, iterator) like mlx_lm.server 0.31.x."""

    def fake_generate(self, request, generation_args, progress_callback=None):
        if progress_callback is not None:
            progress_callback(prefill[0] // 2, prefill[1])
            progress_callback(*prefill)
        return "ctx", iter(responses)

    return fake_generate


def test_wrap_response_generate_passes_ctx_and_responses_through():
    telemetry = _FakeTelemetry()
    wrapped = wrap_response_generate(_fake_generate_factory(["r1", "r2"]), telemetry.as_client())
    ctx, stream = wrapped(object(), "request", "args")
    assert ctx == "ctx"
    assert list(stream) == ["r1", "r2"]


def test_wrap_response_generate_emits_prefill_and_generation_stats():
    telemetry = _FakeTelemetry()
    wrapped = wrap_response_generate(_fake_generate_factory(["r1", "r2"]), telemetry.as_client())
    _, stream = wrapped(object(), "request", "args")
    list(stream)

    prefill = [p for e, p in telemetry.events if e == EventType.PREFILL_PROGRESS]
    assert prefill[-1] == {"request_id": prefill[-1]["request_id"], "processed": 4, "total": 4}
    gen_stats = [p for e, p in telemetry.events if e == EventType.GENERATION_STATS]
    assert len(gen_stats) == 1
    assert gen_stats[0]["completion_tokens"] == 2
    assert gen_stats[0]["prompt_tokens"] == 4
    assert gen_stats[0]["generation_tps"] > 0
    assert gen_stats[0]["prompt_tps"] > 0


def test_wrap_response_generate_chains_existing_progress_callback():
    telemetry = _FakeTelemetry()
    seen: list[tuple[int, int]] = []
    wrapped = wrap_response_generate(_fake_generate_factory(["r"]), telemetry.as_client())
    _, stream = wrapped(object(), "request", "args", lambda p, t: seen.append((p, t)))
    list(stream)
    assert seen == [(2, 4), (4, 4)]


def test_wrap_response_generate_emits_stats_when_consumer_breaks_early():
    # mlx_lm.server's HTTP handler abandons the iterator on finish_reason
    # rather than exhausting it — stats must still be emitted via finally.
    telemetry = _FakeTelemetry()
    wrapped = wrap_response_generate(_fake_generate_factory(["r1", "r2"]), telemetry.as_client())
    _, stream = wrapped(object(), "request", "args")
    next(stream)
    stream.close()

    gen_stats = [p for e, p in telemetry.events if e == EventType.GENERATION_STATS]
    assert len(gen_stats) == 1
    assert gen_stats[0]["completion_tokens"] == 1


def test_wrap_response_generate_no_stats_for_empty_stream():
    telemetry = _FakeTelemetry()
    wrapped = wrap_response_generate(_fake_generate_factory([]), telemetry.as_client())
    _, stream = wrapped(object(), "request", "args")
    assert list(stream) == []
    assert EventType.GENERATION_STATS not in [e for e, _ in telemetry.events]
