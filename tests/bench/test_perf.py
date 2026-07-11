"""Bench track (B2): in-house perf prober — attach mode.

The real `httpx` package is not required here: `probe_endpoint` imports it
lazily, so tests inject a fake module tree via `sys.modules`, mirroring the
pattern used for `minisweagent` in `tests/harnesses/test_mini_swe_agent.py`.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from sovereign.bench.perf import (
    PerfError,
    _passes_thresholds,
    _primary_engine,
    make_perf_attach_executor,
    probe_endpoint,
    run_perf_attach_cell,
    summarize,
)
from sovereign.bench.runner import Job
from sovereign.bench.spec import BenchSpec, Thresholds
from sovereign.core.state import write_json


# --- fake httpx ---
class FakeResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeAsyncClient:
    captured: dict = {}
    next_lines: list[str] = []

    def __init__(self, headers=None):
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, timeout=None):
        FakeAsyncClient.captured = {
            "method": method,
            "url": url,
            "json": json,
            "timeout": timeout,
        }
        return FakeResponse(FakeAsyncClient.next_lines)


def _install_fake_httpx(monkeypatch, lines: list[str]) -> None:
    fake = types.ModuleType("httpx")
    fake.AsyncClient = FakeAsyncClient  # type: ignore[attr-defined]
    FakeAsyncClient.next_lines = lines
    monkeypatch.setitem(sys.modules, "httpx", fake)


_SSE_LINES = [
    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
    'data: {"choices":[{"delta":{"content":" world"}}]}',
    'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":7,"prompt_tokens":3}}',
    "data: [DONE]",
]


# --- probe_endpoint / _stream_once ---
def test_probe_endpoint_without_httpx_raises_install_hint(monkeypatch) -> None:
    # A None sys.modules entry forces ImportError even when httpx is really
    # installed (e.g. pulled in transitively by mini-swe-agent).
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match="httpx is not installed"):
        asyncio.run(probe_endpoint("http://127.0.0.1:11435/v1", "llama3-70b", trials=1))


def test_probe_endpoint_measures_tokens_from_usage(monkeypatch) -> None:
    _install_fake_httpx(monkeypatch, _SSE_LINES)
    samples = asyncio.run(probe_endpoint("http://127.0.0.1:11435/v1", "llama3-70b", trials=2))
    assert len(samples) == 2
    for sample in samples:
        assert sample["output_tokens"] == 7  # usage overrides the delta-chunk count
        assert sample["ttft_s"] is not None
        assert sample["tok_s"] is not None


def test_probe_endpoint_hits_chat_completions_url(monkeypatch) -> None:
    _install_fake_httpx(monkeypatch, _SSE_LINES)
    asyncio.run(probe_endpoint("http://127.0.0.1:11435/v1", "llama3-70b", trials=1))
    assert FakeAsyncClient.captured["url"] == "http://127.0.0.1:11435/v1/chat/completions"
    assert FakeAsyncClient.captured["json"]["model"] == "llama3-70b"
    assert FakeAsyncClient.captured["json"]["stream"] is True


# --- summarize ---
def test_summarize_computes_mean_and_stdev() -> None:
    samples = [
        {"ttft_s": 0.1, "total_s": 1.0, "output_tokens": 10, "tok_s": 10.0},
        {"ttft_s": 0.3, "total_s": 2.0, "output_tokens": 10, "tok_s": 5.0},
    ]
    result = summarize(samples)
    assert result["trials"] == 2
    assert result["ttft_ms"]["mean"] == pytest.approx(200.0)
    assert result["tok_s"]["mean"] == pytest.approx(7.5)
    assert result["tok_s"]["stdev"] > 0


def test_summarize_single_trial_has_zero_stdev() -> None:
    result = summarize([{"ttft_s": 0.1, "total_s": 1.0, "output_tokens": 10, "tok_s": 10.0}])
    assert result["tok_s"]["stdev"] == 0.0


def test_summarize_handles_missing_values() -> None:
    result = summarize([{"ttft_s": None, "total_s": 1.0, "output_tokens": 0, "tok_s": None}])
    assert result["ttft_ms"]["mean"] is None
    assert result["tok_s"]["mean"] is None


# --- thresholds ---
def test_passes_thresholds_all_pass() -> None:
    result = {"tok_s": {"mean": 20.0}, "ttft_ms": {"mean": 100.0}}
    thresholds = Thresholds(min_tok_s=10, max_ttft_ms=500, min_headroom_gb=5)
    passed, reasons = _passes_thresholds(result, thresholds, available_gb=10.0)
    assert passed is True
    assert reasons == []


def test_passes_thresholds_reports_failures() -> None:
    result = {"tok_s": {"mean": 2.0}, "ttft_ms": {"mean": 900.0}}
    thresholds = Thresholds(min_tok_s=10, max_ttft_ms=500)
    passed, reasons = _passes_thresholds(result, thresholds, available_gb=None)
    assert passed is False
    assert len(reasons) == 2


def test_passes_thresholds_no_thresholds_always_passes() -> None:
    result = {"tok_s": {"mean": None}, "ttft_ms": {"mean": None}}
    passed, reasons = _passes_thresholds(result, Thresholds(), available_gb=None)
    assert passed is True
    assert reasons == []


# --- _primary_engine ---
def test_primary_engine_picks_first_with_model() -> None:
    manifest = {
        "services": [
            {"name": "docker"},
            {
                "name": "engine",
                "endpoint": {"scheme": "http", "host": "127.0.0.1", "port": 11435, "model": "m"},
            },
        ]
    }
    engine = _primary_engine(manifest)
    assert engine is not None
    assert engine["name"] == "engine"


def test_primary_engine_none_when_no_model_endpoint() -> None:
    manifest = {"services": [{"name": "docker"}]}
    assert _primary_engine(manifest) is None


# --- run_perf_attach_cell / make_perf_attach_executor ---
def _job() -> Job:
    return Job(
        id="stack-_none-_none", cell_key="k", stack="stack.yaml", harness="_none", suite="_none"
    )


def _write_manifest(state_dir, *, with_engine=True) -> None:
    services: list[dict[str, object]] = [{"name": "docker"}]
    if with_engine:
        services.append(
            {
                "name": "engine",
                "endpoint": {
                    "scheme": "http",
                    "host": "127.0.0.1",
                    "port": 11435,
                    "model": "llama3-70b",
                },
                "co_resident": ["docker"],
            }
        )
    write_json(
        state_dir / "manifest.json",
        {
            "variant_hash": "abc123",
            "memory_budget": {"available_gb": 20.0},
            "services": services,
        },
    )


def test_run_perf_attach_cell_no_manifest_raises(tmp_path) -> None:
    with pytest.raises(PerfError, match="no live stack found"):
        asyncio.run(run_perf_attach_cell(_job(), BenchSpec(stacks=["stack.yaml"]), tmp_path))


def test_run_perf_attach_cell_no_engine_raises(tmp_path) -> None:
    _write_manifest(tmp_path, with_engine=False)
    with pytest.raises(PerfError, match="no native engine"):
        asyncio.run(run_perf_attach_cell(_job(), BenchSpec(stacks=["stack.yaml"]), tmp_path))


def test_run_perf_attach_cell_success(tmp_path, monkeypatch) -> None:
    _install_fake_httpx(monkeypatch, _SSE_LINES)
    _write_manifest(tmp_path)
    spec = BenchSpec(stacks=["stack.yaml"], trials=2, thresholds=Thresholds(min_tok_s=1))
    result = asyncio.run(run_perf_attach_cell(_job(), spec, tmp_path))
    assert result["engine"] == "engine"
    assert result["variant_hash"] == "abc123"
    assert result["co_resident"] == ["docker"]
    assert result["gate_passed"] is True


def test_make_perf_attach_executor_runs_end_to_end(tmp_path, monkeypatch) -> None:
    _install_fake_httpx(monkeypatch, _SSE_LINES)
    _write_manifest(tmp_path)
    spec = BenchSpec(stacks=["stack.yaml"], trials=1)
    executor = make_perf_attach_executor(spec, tmp_path)
    result = executor(_job())
    assert result["trials"] == 1
