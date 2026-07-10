"""Content-addressed bench cells (§6b) — the feature that makes iteration fast.

A cell key is a hash of everything that determines its outcome: the stack under
test, the harness, the suite, the seed, and the trial. Completed cells skip on
re-run — change one sweep axis and only that slice re-executes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sovereign.core.state import read_json, write_json

_CELLS_SUBDIR = "cells"
_RUNS_SUBDIR = "runs"


def cell_key(**parts: Any) -> str:
    """A stable hash over arbitrary, JSON-serializable cell-identity parts.

    Callers pass whatever they have resolved: e.g. stack file + hash in B1,
    a full stack-manifest slice + harness fingerprint once B2 wires in a live
    stack. Same inputs -> same key; changing one part changes the key.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def cells_dir(bench_dir: str | Path) -> Path:
    return Path(bench_dir) / _CELLS_SUBDIR


def runs_dir(bench_dir: str | Path) -> Path:
    return Path(bench_dir) / _RUNS_SUBDIR


def cell_dir(bench_dir: str | Path, key: str) -> Path:
    return cells_dir(bench_dir) / key


def is_complete(bench_dir: str | Path, key: str) -> bool:
    """Whether this cell already has a recorded result (skip-completed re-runs)."""
    return (cell_dir(bench_dir, key) / "result.json").exists()


def write_cell_result(bench_dir: str | Path, key: str, result: dict[str, Any]) -> None:
    write_json(cell_dir(bench_dir, key) / "result.json", result)


def read_cell_result(bench_dir: str | Path, key: str) -> dict[str, Any]:
    return read_json(cell_dir(bench_dir, key) / "result.json")
