"""Bench track (B3): clean-room execution — bench owns boot/measure/teardown."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from sovereign.bench.cleanroom import (
    CleanroomError,
    _would_fit,
    make_cleanroom_executor,
    run_cell_cleanroom,
)
from sovereign.bench.runner import Job
from sovereign.bench.spec import BenchSpec
from sovereign.config import load_config
from sovereign.core.registry import _SERVICE_MANAGERS
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint

_BASE_TYPE = "bench_test_fake_engine"


class FakeCleanroomManager:
    consumer_kind = ConsumerKind.NATIVE
    start_calls: list[str] = []
    stop_calls: list[str] = []

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.activity = ()

    def estimated_memory_gb(self) -> float:
        return 1.0

    def prepare_environment(self) -> None:
        pass

    def start(self) -> None:
        FakeCleanroomManager.start_calls.append(self.name)

    def stop(self) -> None:
        FakeCleanroomManager.stop_calls.append(self.name)

    def is_healthy(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"status": "running"}

    def adjust_resources(self, memory_limit_mb: int) -> None:
        pass

    def resolve(self, resolver) -> None:
        pass

    def endpoint(self):
        return ResolvedEndpoint("http", "127.0.0.1", 11435, model="m")

    def runtime_handle(self):
        return {"kind": "native", "pid": 4242}


@pytest.fixture(autouse=True)
def _register_fake_engine(monkeypatch):
    monkeypatch.setitem(_SERVICE_MANAGERS, _BASE_TYPE, FakeCleanroomManager)
    FakeCleanroomManager.start_calls = []
    FakeCleanroomManager.stop_calls = []


def _write_stack(tmp_path, *, max_gb: int = 64, safety_margin_gb: int = 0) -> str:
    path = tmp_path / "stack.yaml"
    path.write_text(
        f"""
version: "1.1"
resources:
  max_unified_memory_gb: {max_gb}
  safety_margin_gb: {safety_margin_gb}
services:
  - name: engine
    base_type: {_BASE_TYPE}
    health_check: {{type: http, endpoint: /health, port: 11435}}
    config: {{}}
"""
    )
    return str(path)


def _job(stack: str) -> Job:
    return Job(id="cell", cell_key="key1", stack=stack, harness="_none", suite="_none")


_SSE_LINES = [
    'data: {"choices":[{"delta":{"content":"hi"}}]}',
    'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":3}}',
    "data: [DONE]",
]


def _install_fake_httpx(monkeypatch) -> None:
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in _SSE_LINES:
                yield line

    class FakeAsyncClient:
        def __init__(self, headers=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *a, **kw):
            return FakeResponse()

    fake = types.ModuleType("httpx")
    fake.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", fake)


# --- _would_fit ---
def test_would_fit_true_when_within_budget(tmp_path) -> None:
    config = load_config(_write_stack(tmp_path, max_gb=64))
    fits, needed, available = _would_fit(config)
    assert fits is True
    assert needed == 1.0


def test_would_fit_false_when_over_budget(tmp_path) -> None:
    config = load_config(_write_stack(tmp_path, max_gb=1, safety_margin_gb=1))
    fits, needed, available = _would_fit(config)
    assert fits is False


# --- run_cell_cleanroom ---
def test_gated_stack_never_boots(tmp_path) -> None:
    stack = _write_stack(tmp_path, max_gb=1, safety_margin_gb=1)
    job = _job(stack)
    spec = BenchSpec(stacks=[stack], trials=1)
    with pytest.raises(CleanroomError, match="gated"):
        asyncio.run(
            run_cell_cleanroom(
                job, spec, tmp_path / "cellstate", bench_dir=tmp_path / "benchmarks"
            )
        )
    assert FakeCleanroomManager.start_calls == []  # never attempted


def test_successful_cell_boots_measures_and_tears_down(tmp_path, monkeypatch) -> None:
    _install_fake_httpx(monkeypatch)
    stack = _write_stack(tmp_path)
    job = _job(stack)
    spec = BenchSpec(stacks=[stack], trials=1)
    result = asyncio.run(
        run_cell_cleanroom(job, spec, tmp_path / "cellstate", bench_dir=tmp_path / "benchmarks")
    )
    assert result["engine"] == "engine"
    assert FakeCleanroomManager.start_calls == ["engine"]
    assert FakeCleanroomManager.stop_calls == ["engine"]
    assert (tmp_path / "cellstate" / "manifest.json").exists()


def test_make_cleanroom_executor_runs_end_to_end(tmp_path, monkeypatch) -> None:
    _install_fake_httpx(monkeypatch)
    stack = _write_stack(tmp_path)
    job = _job(stack)
    spec = BenchSpec(stacks=[stack], trials=1)
    executor = make_cleanroom_executor(spec, tmp_path)
    result = executor(job)
    assert result["trials"] == 1
    assert FakeCleanroomManager.stop_calls == ["engine"]  # torn down
