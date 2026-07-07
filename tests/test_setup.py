"""Lightweight tests for scripts/setup.py (loaded by path — it's not a package)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"


def _load():
    spec = importlib.util.spec_from_file_location("sovereign_setup", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = _load()


class _Result:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def test_brewfile_declares_bootstrap_only() -> None:
    """Root Brewfile is bootstrap-only; integration deps live in their own folders."""
    text = setup.BREWFILE.read_text()
    assert 'brew "uv"' in text
    assert 'brew "llama.cpp"' not in text  # moved to services/llama_cpp/Brewfile
    assert 'cask "docker-desktop"' not in text  # moved to services/docker_engine/Brewfile


def test_main_runs_bootstrap_then_provision(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(setup.shutil, "which", lambda _name: "/opt/homebrew/bin/brew")
    monkeypatch.setattr(
        setup.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Result(0)
    )
    assert setup.main() == 0
    assert calls[0][:2] == ["brew", "bundle"]
    assert calls[1] == ["uv", "sync"]
    assert calls[3] == ["uv", "run", "sovereign", "provision"]


def test_main_fails_without_homebrew(monkeypatch) -> None:
    ran: list[list[str]] = []
    monkeypatch.setattr(setup.shutil, "which", lambda _name: None)
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    assert setup.main() == 1
    assert ran == []  # bails before running anything


def test_main_propagates_command_failure(monkeypatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda _name: "/opt/homebrew/bin/brew")
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **kw: _Result(1))
    assert setup.main() == 1


def test_script_is_loadable() -> None:
    assert callable(setup.main)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
