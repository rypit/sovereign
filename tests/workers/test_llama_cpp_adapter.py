"""Tests for the llama_cpp worker adapter (ADR 0007: ``llama-server``
subprocess, not an embedded binding).

``build_server_argv`` and the telemetry-translator functions are pure/near-
pure and unit-testable with no ``llama-server`` binary installed.
``run()`` is exercised by patching ``subprocess.Popen`` (a fake child) and
``urllib.request.urlopen`` (health + ``/slots``/``/metrics``); the
telemetry-translator is additionally exercised end-to-end over the real UDS
seam (real ``TelemetryHub``/``TelemetryClient``), per CLAUDE.md's testing
conventions.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from sovereign.runtime.telemetry import TelemetryHub, TelemetryStateCache
from sovereign.workers import llama_cpp_adapter as adapter
from sovereign.workers.llama_cpp_adapter import (
    _MetricsDelta,
    _parse_prometheus,
    build_server_argv,
    poll_once,
    telemetry_loop,
)
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient
from sovereign.workers.worker_config import WorkerConfig


# --- build_server_argv ---
def test_build_server_argv_basics():
    argv = build_server_argv(
        {}, model_path="/models/foo.gguf", draft_model_path=None,
        served_model_name=None, host="127.0.0.1", port=9000,
    )
    assert argv[:6] == ["-m", "/models/foo.gguf", "--host", "127.0.0.1", "--port", "9000"]


def test_build_server_argv_always_enables_telemetry_endpoints():
    # /metrics and /slots are off by default in llama-server; the telemetry
    # translator polls both, so build_server_argv must always request them.
    argv = build_server_argv(
        {}, model_path="/m.gguf", draft_model_path=None,
        served_model_name=None, host="h", port=1,
    )
    assert "--metrics" in argv
    assert "--slots" in argv


def test_build_server_argv_maps_resource_kwargs():
    argv = build_server_argv(
        {"gpu_layers": 20, "threads": 4, "context_size": 4096, "max_parallel": 4},
        model_path="/m.gguf", draft_model_path=None, served_model_name=None,
        host="h", port=1,
    )
    assert argv[argv.index("--n-gpu-layers") + 1] == "20"
    assert argv[argv.index("--threads") + 1] == "4"
    assert argv[argv.index("--ctx-size") + 1] == "4096"
    assert argv[argv.index("--parallel") + 1] == "4"  # the max_parallel gap-closing flag


def test_build_server_argv_alias_from_served_model_name():
    argv = build_server_argv(
        {}, model_path="/m.gguf", draft_model_path=None, served_model_name="llama3-70b",
        host="h", port=1,
    )
    assert argv[argv.index("--alias") + 1] == "llama3-70b"


def test_build_server_argv_draft_model_flags():
    argv = build_server_argv(
        {"num_draft_tokens": 8}, model_path="/m.gguf", draft_model_path="/draft.gguf",
        served_model_name=None, host="h", port=1,
    )
    assert argv[argv.index("--model-draft") + 1] == "/draft.gguf"
    assert argv[argv.index("--draft-max") + 1] == "8"


def test_build_server_argv_draft_max_omitted_without_draft_model():
    argv = build_server_argv(
        {"num_draft_tokens": 8}, model_path="/m.gguf", draft_model_path=None,
        served_model_name=None, host="h", port=1,
    )
    assert "--draft-max" not in argv


def test_build_server_argv_kv_cache_type_maps_to_both_flags():
    argv = build_server_argv(
        {"kv_cache_type": "q8_0"}, model_path="/m.gguf", draft_model_path=None,
        served_model_name=None, host="h", port=1,
    )
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"


def test_build_server_argv_passthrough_escape_hatch():
    argv = build_server_argv(
        {"rope_freq_base": 1000000.0}, model_path="/m.gguf", draft_model_path=None,
        served_model_name=None, host="h", port=1,
    )
    assert argv[argv.index("--rope-freq-base") + 1] == "1000000.0"


def test_build_server_argv_bool_kwargs_become_bare_flags():
    argv = build_server_argv(
        {"flash_attn": True, "mlock": False}, model_path="/m.gguf", draft_model_path=None,
        served_model_name=None, host="h", port=1,
    )
    assert "--flash-attn" in argv
    assert "--mlock" not in argv


def test_build_server_argv_api_key_never_included():
    argv = build_server_argv(
        {}, model_path="/m.gguf", draft_model_path=None, served_model_name=None,
        host="h", port=1,
    )
    assert "--api-key" not in argv


# --- telemetry translator: pure parsing ---
def test_parse_prometheus_basic():
    text = """
    # HELP some comment
    llamacpp:tokens_predicted_total 42
    llamacpp:prompt_tokens_total 100
    """
    counters = _parse_prometheus(text)
    assert counters["llamacpp:tokens_predicted_total"] == 42.0
    assert counters["llamacpp:prompt_tokens_total"] == 100.0


def test_parse_prometheus_ignores_malformed_lines():
    assert _parse_prometheus("not a metric line\n\n# comment") == {}


# --- telemetry translator: poll_once over a fake HTTP surface ---
class _FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []

    def emit(self, event: Any, payload: Any) -> None:
        self.events.append((event, payload))

    def as_client(self) -> TelemetryClient:
        return cast(TelemetryClient, self)


def test_poll_once_emits_prefill_for_active_slots(monkeypatch):
    telemetry = _FakeTelemetry()
    slots = [
        {"id": 0, "id_task": 7, "state": 1, "n_past": 50, "n_prompt_tokens": 200},
        {"id": 1, "id_task": 8, "state": "idle"},
    ]

    monkeypatch.setattr(adapter, "_fetch_json", lambda url: slots)
    monkeypatch.setattr(adapter, "_fetch_text", lambda url: None)

    poll_once("http://x", telemetry.as_client(), _MetricsDelta())

    prefill = [p for e, p in telemetry.events if e == EventType.PREFILL_PROGRESS]
    assert prefill == [{"request_id": "7", "processed": 50, "total": 200}]


def test_poll_once_tolerates_total_none():
    telemetry = _FakeTelemetry()
    slots = [{"id_task": 1, "state": 1, "n_past": 5}]  # no n_prompt_tokens

    delta = _MetricsDelta()
    import sovereign.workers.llama_cpp_adapter as mod

    orig_fetch_json = mod._fetch_json
    orig_fetch_text = mod._fetch_text
    try:
        mod._fetch_json = lambda url: slots
        mod._fetch_text = lambda url: None
        poll_once("http://x", telemetry.as_client(), delta)
    finally:
        mod._fetch_json = orig_fetch_json
        mod._fetch_text = orig_fetch_text

    prefill = [p for e, p in telemetry.events if e == EventType.PREFILL_PROGRESS]
    assert prefill[0]["total"] is None


def test_poll_once_emits_generation_stats_from_metrics_deltas(monkeypatch):
    telemetry = _FakeTelemetry()
    monkeypatch.setattr(adapter, "_fetch_json", lambda url: [])

    texts = [
        "llamacpp:tokens_predicted_total 0\nllamacpp:prompt_tokens_total 0\n",
        "llamacpp:tokens_predicted_total 10\nllamacpp:prompt_tokens_total 20\n",
    ]
    it = iter(texts)
    monkeypatch.setattr(adapter, "_fetch_text", lambda url: next(it))

    delta = _MetricsDelta()
    poll_once("http://x", telemetry.as_client(), delta)  # primes prev counters
    delta.prev_ts = time.monotonic() - 1.0  # force a positive dt on the next poll
    poll_once("http://x", telemetry.as_client(), delta)

    gen_stats = [p for e, p in telemetry.events if e == EventType.GENERATION_STATS]
    assert len(gen_stats) == 1
    assert gen_stats[0]["completion_tokens"] == 10
    assert gen_stats[0]["prompt_tokens"] == 20
    assert gen_stats[0]["generation_tps"] > 0
    assert gen_stats[0]["prompt_tps"] > 0


def test_poll_once_never_raises_on_bad_json(monkeypatch):
    monkeypatch.setattr(adapter, "_fetch_json", lambda url: {"not": "a list"})
    monkeypatch.setattr(adapter, "_fetch_text", lambda url: "garbage\n")
    poll_once("http://x", _FakeTelemetry().as_client(), _MetricsDelta())  # must not raise


# --- telemetry translator over the real UDS seam ---
def test_telemetry_translator_lands_in_real_cache(socket_path):
    cache = TelemetryStateCache()
    hub = TelemetryHub(socket_path, cache)
    hub.start()
    try:
        client = TelemetryClient(socket_path, "llama_svc")
        try:
            delta = _MetricsDelta()
            slots = [{"id_task": 3, "state": 1, "n_past": 10, "n_prompt_tokens": 40}]
            adapter._fetch_json = lambda url: slots
            adapter._fetch_text = lambda url: None
            try:
                poll_once("http://x", client, delta)
            finally:
                del adapter._fetch_json
                del adapter._fetch_text

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                snap = cache.snapshot("llama_svc")
                if snap["prefill"]:
                    break
                time.sleep(0.05)
            snap = cache.snapshot("llama_svc")
            assert snap["prefill"] == [{"request_id": "3", "processed": 10, "total": 40}]
        finally:
            client.close()
    finally:
        hub.stop()


# --- telemetry_loop: process-exit detection ---
class _FakeProcess:
    def __init__(self) -> None:
        self._exited = False
        self.returncode: int | None = None

    def poll(self):
        return self.returncode if self._exited else None

    def exit(self, code: int = 1) -> None:
        self._exited = True
        self.returncode = code

    def terminate(self) -> None:
        self.exit(0)

    def kill(self) -> None:
        self.exit(-9)

    def wait(self, timeout=None):
        return self.returncode


def _as_popen(process: _FakeProcess) -> subprocess.Popen[Any]:
    return cast("subprocess.Popen[Any]", process)


def _fake_process() -> subprocess.Popen[Any]:
    return _as_popen(_FakeProcess())


def test_telemetry_loop_stops_cleanly_on_stop_event(monkeypatch):
    monkeypatch.setattr(adapter, "poll_once", lambda *a, **k: None)
    stop_event = threading.Event()
    stop_event.set()
    crashed = telemetry_loop(
        "http://x", _FakeTelemetry().as_client(), stop_event, _fake_process()
    )
    assert crashed is False


def test_telemetry_loop_reports_crash_on_unexpected_exit(monkeypatch):
    monkeypatch.setattr(adapter, "poll_once", lambda *a, **k: None)
    stop_event = threading.Event()
    process = _FakeProcess()
    process.exit(1)
    crashed = telemetry_loop(
        "http://x", _FakeTelemetry().as_client(), stop_event, _as_popen(process)
    )
    assert crashed is True


def test_telemetry_loop_never_raises_on_poll_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter, "poll_once", _boom)
    stop_event = threading.Event()

    def _stop_soon():
        time.sleep(0.05)
        stop_event.set()

    threading.Thread(target=_stop_soon, daemon=True).start()
    crashed = telemetry_loop(
        "http://x", _FakeTelemetry().as_client(), stop_event, _fake_process(), interval=0.01
    )
    assert crashed is False


# --- run(): subprocess supervision ---
class _FakeController:
    def __init__(self) -> None:
        self.shutdown_callback = None


def _cfg(**overrides: Any) -> WorkerConfig:
    cfg = WorkerConfig(
        service="llama_heavy_v1",
        engine="llama_cpp",
        host="127.0.0.1",
        port=11435,
        health_path="/health",
        telemetry_socket="/tmp/does-not-matter.sock",
        model_path="/models/x.gguf",
        engine_kwargs={},
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"[]"


def test_run_waits_for_health_then_emits_serving(monkeypatch):
    process = _FakeProcess()
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: process)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())

    telemetry = _FakeTelemetry()
    controller = _FakeController()

    def _stop_loop_immediately(*a, **k):
        return False  # pretend the poll loop returned via shutdown

    monkeypatch.setattr(adapter, "telemetry_loop", _stop_loop_immediately)

    adapter.run(_cfg(), telemetry.as_client(), controller)

    state_changes = [p for e, p in telemetry.events if e == EventType.STATE_CHANGE]
    assert {"state": "serving"} in state_changes
    assert controller.shutdown_callback is not None


def test_run_raises_when_health_never_succeeds(monkeypatch):
    process = _FakeProcess()
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: process)
    monkeypatch.setattr(adapter, "_HEALTH_TIMEOUT", 0.05)
    monkeypatch.setattr(adapter, "_HEALTH_POLL_INTERVAL", 0.01)

    def boom(url, timeout=None):
        raise adapter.urllib.error.URLError("refused")

    monkeypatch.setattr(adapter.urllib.request, "urlopen", boom)

    telemetry = _FakeTelemetry()
    controller = _FakeController()
    with pytest.raises(RuntimeError, match="failed to become healthy"):
        adapter.run(_cfg(), telemetry.as_client(), controller)


def test_run_raises_on_child_crash(monkeypatch):
    process = _FakeProcess()
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: process)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "telemetry_loop", lambda *a, **k: True)  # crashed

    telemetry = _FakeTelemetry()
    controller = _FakeController()
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        adapter.run(_cfg(), telemetry.as_client(), controller)


def test_run_shutdown_callback_terminates_child(monkeypatch):
    process = _FakeProcess()
    terminated = MagicMock()
    monkeypatch.setattr(process, "terminate", terminated, raising=False)
    monkeypatch.setattr(process, "wait", MagicMock(return_value=0), raising=False)

    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: process)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "telemetry_loop", lambda *a, **k: False)

    telemetry = _FakeTelemetry()
    controller = _FakeController()
    adapter.run(_cfg(), telemetry.as_client(), controller)

    assert controller.shutdown_callback is not None
    controller.shutdown_callback()
    terminated.assert_called_once()


def test_run_passes_api_key_via_env_not_config(monkeypatch):
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return _FakeProcess()

    monkeypatch.setattr(adapter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "telemetry_loop", lambda *a, **k: False)
    monkeypatch.setenv("SOVEREIGN_API_KEY", "s3cr3t")

    adapter.run(_cfg(), _FakeTelemetry().as_client(), _FakeController())
    argv = captured["argv"]
    assert argv[argv.index("--api-key") + 1] == "s3cr3t"
