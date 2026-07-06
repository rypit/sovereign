"""`.sovereign/state.json` read/write + variant file hashing (§7b, §9).

State is a file, consistent with the "results are files" philosophy: read-only CLI
commands (``status``) read it without needing to reach the daemon.
"""

from __future__ import annotations

import hashlib
import json
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
