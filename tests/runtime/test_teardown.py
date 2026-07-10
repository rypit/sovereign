"""runtime/teardown.py — cross-process stop with PID-identity verification.

`sovereign down` signals PIDs recorded in state.json, possibly long after the
stack died — the PID may have been recycled to a stranger process. The handle
records the process create_time and teardown must verify it before terminate.
"""

from __future__ import annotations

import subprocess
import sys

import psutil
import pytest

from sovereign.runtime.teardown import stop_service_handle


@pytest.fixture
def sleeper():
    """A real child process we own, safe to signal."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc
    if proc.poll() is None:
        proc.kill()
        proc.wait()


def test_stop_matching_create_time_terminates(sleeper) -> None:
    create_time = psutil.Process(sleeper.pid).create_time()
    handle = {"kind": "native", "pid": sleeper.pid, "create_time": create_time}
    assert stop_service_handle(handle) == "stopped"
    assert not psutil.pid_exists(sleeper.pid) or sleeper.poll() is not None  # actually gone


def test_stop_mismatched_create_time_refuses_to_kill(sleeper) -> None:
    """A recycled PID (create_time far from the recorded one) must not be signalled."""
    handle = {
        "kind": "native",
        "pid": sleeper.pid,
        "create_time": psutil.Process(sleeper.pid).create_time() - 3600.0,
    }
    assert stop_service_handle(handle) == "already stopped"
    assert sleeper.poll() is None  # untouched — never kill an unverified PID


def test_stop_create_time_within_tolerance_still_stops(sleeper) -> None:
    """±1s tolerance absorbs clock granularity between psutil readings."""
    create_time = psutil.Process(sleeper.pid).create_time()
    handle = {"kind": "native", "pid": sleeper.pid, "create_time": create_time + 0.5}
    assert stop_service_handle(handle) == "stopped"


def test_stop_legacy_handle_without_create_time_keeps_old_behavior(sleeper) -> None:
    """Handles written by older versions lack create_time — still stop the PID."""
    handle = {"kind": "native", "pid": sleeper.pid}
    assert stop_service_handle(handle) == "stopped"


def test_stop_dead_pid_reports_already_stopped(sleeper) -> None:
    sleeper.kill()
    sleeper.wait()
    handle = {"kind": "native", "pid": sleeper.pid, "create_time": 123.0}
    assert stop_service_handle(handle) == "already stopped"


def test_unknown_handle_kind() -> None:
    assert stop_service_handle({"kind": "martian"}) == "unknown handle"
