"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sparse_file():
    """Create a file reporting a given size via stat().st_size without writing real
    bytes to disk (a sparse file) — for tests that only care about file *size*
    (e.g. model-weight byte-count estimation), not content.
    """

    def _make(path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            if size > 0:
                f.truncate(size)

    return _make
