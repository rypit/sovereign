"""Clean-room lockfile (§6b) — so bench can't fight the daemon.

Clean-room mode boots and tears down stacks itself, which would race a
`sovereign up`/`serve` daemon managing the same state dir. Two guards:
refuse to start a clean-room run while a daemon-managed stack is up, and
hold a lockfile for the run's duration so a second concurrent bench run (or
`sovereign up`) sees clear contention instead of silently colliding.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import psutil

from sovereign.core.state import read_json, read_json_or_none

_LOCK_FILENAME = "bench.lock"


class BenchLockError(Exception):
    """Raised when a clean-room run can't acquire exclusive use of the state dir."""


def lock_path(state_dir: str | Path) -> Path:
    return Path(state_dir) / _LOCK_FILENAME


def check_no_live_daemon_stack(state_dir: str | Path) -> None:
    """Refuse if `state.json` shows a daemon (`sovereign up`/`serve`) is running."""
    state_path = Path(state_dir) / "state.json"
    if not state_path.exists():
        return
    state = read_json(state_path)
    if state.get("runtime"):
        raise BenchLockError(
            f"a daemon-managed stack is currently up in {state_dir} — "
            "`sovereign down` it first, or use a different --state-dir for clean-room runs."
        )


def _lock_holder_alive(existing: dict) -> bool:
    """Whether the process recorded in a lockfile still exists.

    ``create_time`` (when recorded) guards against PID recycling: a different
    process that inherited the PID doesn't hold this lock. An unreadable or
    holder-less lock reads as dead (stale) — safe, since acquisition is atomic.
    """
    pid = existing.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        proc = psutil.Process(pid)
        recorded = existing.get("create_time")
        if recorded is not None and abs(proc.create_time() - float(recorded)) > 1.0:
            return False  # PID recycled — the original holder is gone
        return True
    except psutil.NoSuchProcess:
        return False


def _try_create_lock(path: Path, run_id: str) -> bool:
    """Atomically create the lockfile (O_CREAT|O_EXCL); False when it already exists.

    The create-exclusive open *is* the acquisition — no exists()-then-write
    window for two processes to slip through together.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    payload = {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "run_id": run_id,
        "acquired_at": datetime.now(UTC).isoformat(),
    }
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return True


def _contention_error(path: Path) -> BenchLockError:
    existing = read_json_or_none(path) or {}
    return BenchLockError(
        f"another bench run (run_id={existing.get('run_id')}, pid={existing.get('pid')}) "
        f"holds the lock at {path}"
    )


@contextlib.contextmanager
def acquire_bench_lock(state_dir: str | Path, run_id: str) -> Iterator[None]:
    """Hold `bench.lock` for the duration of a clean-room run."""
    check_no_live_daemon_stack(state_dir)
    path = lock_path(state_dir)

    if not _try_create_lock(path, run_id):
        # Contention: is the recorded holder still alive, or did it die without
        # cleaning up (SIGKILL, power loss)? Clear a stale lock and retry once.
        existing = read_json_or_none(path) or {}
        if _lock_holder_alive(existing):
            raise _contention_error(path)
        path.unlink(missing_ok=True)
        if not _try_create_lock(path, run_id):
            raise _contention_error(path)  # lost the re-acquisition race
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
