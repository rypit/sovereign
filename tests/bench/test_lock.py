"""Bench track (B3): the clean-room lockfile — bench can't fight the daemon."""

from __future__ import annotations

import pytest

from sovereign.bench.lock import (
    BenchLockError,
    acquire_bench_lock,
    check_no_live_daemon_stack,
    lock_path,
)
from sovereign.core.state import read_json, write_json


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


def test_acquire_bench_lock_clears_stale_lock_of_dead_pid(tmp_path) -> None:
    """A holder that died without cleanup (SIGKILL) must not wedge bench forever."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # reaped: the PID no longer exists
    write_json(lock_path(tmp_path), {"pid": proc.pid, "run_id": "dead-run"})

    with acquire_bench_lock(tmp_path, "run2"):
        assert read_json(lock_path(tmp_path))["run_id"] == "run2"
    assert not lock_path(tmp_path).exists()


def test_acquire_bench_lock_clears_lock_of_recycled_pid(tmp_path) -> None:
    """A live PID with the wrong create_time is a recycled PID — the original
    holder is gone, so the lock is stale."""
    import os

    write_json(
        lock_path(tmp_path),
        {"pid": os.getpid(), "create_time": 1.0, "run_id": "ancient-run"},
    )
    with acquire_bench_lock(tmp_path, "run2"):
        assert read_json(lock_path(tmp_path))["run_id"] == "run2"


def test_acquire_bench_lock_records_holder_identity(tmp_path) -> None:
    import os

    import psutil

    with acquire_bench_lock(tmp_path, "run1"):
        lock = read_json(lock_path(tmp_path))
        assert lock["pid"] == os.getpid()
        assert lock["create_time"] == psutil.Process().create_time()


def test_acquire_bench_lock_loses_reacquisition_race_cleanly(tmp_path, monkeypatch) -> None:
    """Stale lock cleared, but another contender re-creates it first — clean error."""
    import subprocess
    import sys

    from sovereign.bench import lock as lock_mod

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    write_json(lock_path(tmp_path), {"pid": proc.pid, "run_id": "dead-run"})

    real_unlink = type(lock_path(tmp_path)).unlink

    def unlink_then_contender_wins(self, *a, **kw):
        real_unlink(self, *a, **kw)
        if self.name == "bench.lock":
            self.write_text('{"pid": 1, "run_id": "contender"}')  # steals the slot

    monkeypatch.setattr(type(lock_path(tmp_path)), "unlink", unlink_then_contender_wins)
    with pytest.raises(BenchLockError, match="another bench run"):
        with lock_mod.acquire_bench_lock(tmp_path, "run2"):
            pass  # pragma: no cover - should never enter


def test_acquire_bench_lock_refuses_when_daemon_stack_up(tmp_path) -> None:
    write_json(tmp_path / "state.json", {"runtime": {"engine": {"kind": "native", "pid": 1}}})
    with pytest.raises(BenchLockError, match="daemon-managed stack"):
        with acquire_bench_lock(tmp_path, "run1"):
            pass  # pragma: no cover - should never enter
