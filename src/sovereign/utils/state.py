"""`.sovereign/state.json` read/write + variant file hashing (§7b, §9).

State is a file, consistent with the "results are files" philosophy: read-only CLI
commands (``status``) read it without needing to reach the daemon.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes — used to detect drift from the source variant."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write ``data`` as pretty JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())


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
