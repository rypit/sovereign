"""Bench track (B3): the clean-room lockfile — bench can't fight the daemon."""

from __future__ import annotations

import pytest

from sovereign.bench.lock import (
    BenchLockError,
    acquire_bench_lock,
    check_no_live_daemon_stack,
    lock_path,
)
from sovereign.utils.state import read_json, write_json


def test_check_no_live_daemon_stack_passes_without_state(tmp_path) -> None:
    check_no_live_daemon_stack(tmp_path)  # no state.json at all — fine


def test_check_no_live_daemon_stack_passes_when_runtime_empty(tmp_path) -> None:
    write_json(tmp_path / "state.json", {"runtime": {}})
    check_no_live_daemon_stack(tmp_path)  # stopped stack — fine


def test_check_no_live_daemon_stack_raises_when_runtime_populated(tmp_path) -> None:
    write_json(tmp_path / "state.json", {"runtime": {"engine": {"kind": "native", "pid": 1}}})
    with pytest.raises(BenchLockError, match="daemon-managed stack"):
        check_no_live_daemon_stack(tmp_path)


def test_acquire_bench_lock_writes_and_removes_lockfile(tmp_path) -> None:
    path = lock_path(tmp_path)
    assert not path.exists()
    with acquire_bench_lock(tmp_path, "run123"):
        assert path.exists()
        assert read_json(path)["run_id"] == "run123"
    assert not path.exists()


def test_acquire_bench_lock_removes_lockfile_on_exception(tmp_path) -> None:
    path = lock_path(tmp_path)
    with pytest.raises(RuntimeError), acquire_bench_lock(tmp_path, "run123"):
        raise RuntimeError("boom")
    assert not path.exists()


def test_acquire_bench_lock_refuses_concurrent_run(tmp_path) -> None:
    with acquire_bench_lock(tmp_path, "run1"):
        with pytest.raises(BenchLockError, match="another bench run"):
            with acquire_bench_lock(tmp_path, "run2"):
                pass  # pragma: no cover - should never enter


def test_acquire_bench_lock_refuses_when_daemon_stack_up(tmp_path) -> None:
    write_json(tmp_path / "state.json", {"runtime": {"engine": {"kind": "native", "pid": 1}}})
    with pytest.raises(BenchLockError, match="daemon-managed stack"):
        with acquire_bench_lock(tmp_path, "run1"):
            pass  # pragma: no cover - should never enter
