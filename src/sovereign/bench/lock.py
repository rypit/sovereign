"""Clean-room lockfile (§6b) — so bench can't fight the daemon.

Clean-room mode boots and tears down stacks itself, which would race a
`sovereign up`/`serve` daemon managing the same state dir. Two guards:
refuse to start a clean-room run while a daemon-managed stack is up, and
hold a lockfile for the run's duration so a second concurrent bench run (or
`sovereign up`) sees clear contention instead of silently colliding.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from sovereign.utils.state import read_json, write_json

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


@contextlib.contextmanager
def acquire_bench_lock(state_dir: str | Path, run_id: str) -> Iterator[None]:
    """Hold `bench.lock` for the duration of a clean-room run."""
    check_no_live_daemon_stack(state_dir)
    path = lock_path(state_dir)
    if path.exists():
        existing = read_json(path)
        raise BenchLockError(
            f"another bench run (run_id={existing.get('run_id')}, pid={existing.get('pid')}) "
            f"holds the lock at {path}"
        )
    write_json(
        path,
        {"pid": os.getpid(), "run_id": run_id, "acquired_at": datetime.now(UTC).isoformat()},
    )
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
