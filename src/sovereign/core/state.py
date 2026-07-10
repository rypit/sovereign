"""`.sovereign/state.json` read/write + variant file hashing (§7b, §9).

State is a file, consistent with the "results are files" philosophy: read-only CLI
commands (``status``) read it without needing to reach the daemon.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes — used to detect drift from the source variant."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Atomically write ``data`` as pretty JSON, creating parent dirs as needed.

    Writes to a temp file in the same directory, then ``os.replace()`` — a
    same-filesystem rename is atomic, so a reader (a separate CLI process, no
    locking) never observes a partially-written file, and a crash between the
    write and the rename leaves the previous ``path`` untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())


def read_json_or_none(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON file, tolerating an absent or (rarely, mid-write) unparsable file.

    For pollers (``monitor``, the dashboard) that re-read a live file on an
    interval: a missing file or a decode error just means "no update this
    tick" — the caller keeps its last-known-good snapshot rather than crashing
    or showing a torn read.
    """
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def mark_stack_stopped(state_path: str | Path) -> None:
    """Rewrite ``state.json`` after an out-of-process teardown (`sovereign down`).

    The single place besides ``Orchestrator.persist()`` that writes the state-file
    schema: every service becomes "stopped" and the runtime handles are cleared,
    so a later ``status`` agrees with what ``down`` just did.
    """
    state = read_json(state_path)
    state["services"] = dict.fromkeys(state.get("services", {}), "stopped")
    state["runtime"] = {}
    state["updated_at"] = datetime.now(UTC).isoformat()
    write_json(state_path, state)
