"""core/state.py — atomic JSON writes + tolerant reads (the coordination layer).

Separate CLI processes coordinate through `.sovereign/*.json` with no locking,
so `write_json` must be atomic (temp file + os.replace) and pollers must
tolerate a missing/garbage file via `read_json_or_none`.
"""

from __future__ import annotations

import json
import os

import pytest

from sovereign.core import state as state_mod
from sovereign.core.state import read_json, read_json_or_none, write_json


# --- write_json atomicity ---
def test_write_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}


def test_write_json_creates_parents(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "state.json"
    write_json(path, {"a": 1})
    assert read_json(path) == {"a": 1}


def test_write_json_crash_before_replace_preserves_old_content(tmp_path, monkeypatch) -> None:
    """A crash between writing the temp file and os.replace() must leave the
    previous file intact — a reader never sees a torn/partial write."""
    path = tmp_path / "state.json"
    write_json(path, {"generation": 1})

    def boom(src, dst):
        raise OSError("simulated crash between open and replace")

    monkeypatch.setattr(state_mod.os, "replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        write_json(path, {"generation": 2})

    assert read_json(path) == {"generation": 1}  # old content untouched
    assert list(tmp_path.glob("*.tmp")) == []  # temp file cleaned up


def test_write_json_never_exposes_partial_file(tmp_path, monkeypatch) -> None:
    """At the moment of os.replace() the destination flips from complete old
    content to complete new content — verify the temp path is distinct from
    the destination (an in-place write would be observable mid-write)."""
    path = tmp_path / "state.json"
    write_json(path, {"generation": 1})

    observed: dict = {}
    real_replace = os.replace

    def spy(src, dst):
        # Just before the swap, the destination still parses as the OLD data.
        observed["during"] = json.loads(path.read_text())
        real_replace(src, dst)

    monkeypatch.setattr(state_mod.os, "replace", spy)
    write_json(path, {"generation": 2})
    assert observed["during"] == {"generation": 1}
    assert read_json(path) == {"generation": 2}


# --- read_json_or_none ---
def test_read_json_or_none_missing_file(tmp_path) -> None:
    assert read_json_or_none(tmp_path / "absent.json") is None


def test_read_json_or_none_garbage(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"truncated": ')
    assert read_json_or_none(path) is None


def test_read_json_or_none_valid(tmp_path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"ok": True})
    assert read_json_or_none(path) == {"ok": True}


def test_read_json_still_strict(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        read_json(path)
