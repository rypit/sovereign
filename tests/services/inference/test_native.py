"""Shared native-engine helpers: looks_local / local_model_bytes / check_local_artifact."""

from __future__ import annotations

import struct

import pytest

from sovereign.core import procmem
from sovereign.services.inference.base import check_local_artifact
from sovereign.services.inference.hf import local_model_bytes, looks_local


# --- looks_local ---
def test_looks_local_absolute_path() -> None:
    assert looks_local("/models/x.gguf") is True


def test_looks_local_home_path() -> None:
    assert looks_local("~/models/x.gguf") is True


def test_looks_local_dot_path() -> None:
    assert looks_local("./models/x.gguf") is True


def test_looks_local_repo_id_is_not_local() -> None:
    assert looks_local("mlx-community/foo-4bit") is False


def test_looks_local_existing_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.gguf").write_bytes(b"x")
    assert looks_local("model.gguf") is True


def test_looks_local_repo_id_not_hijacked_by_same_named_dir(tmp_path, monkeypatch) -> None:
    """A bare org/name is a repo id even when a same-named directory exists in
    the CWD — otherwise `mlx-community/foo` silently resolves to a local dir."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mlx-community" / "foo-4bit").mkdir(parents=True)
    assert looks_local("mlx-community/foo-4bit") is False


def test_looks_local_repo_with_quant_not_hijacked(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org").mkdir()
    assert looks_local("org/model:Q4_K_M") is False


def test_looks_local_multi_segment_existing_path_still_local(tmp_path, monkeypatch) -> None:
    """The existence fallback still applies to strings that can't be repo ids."""
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "models" / "sub" / "x.gguf"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"x")
    assert looks_local("models/sub/x.gguf") is True


def test_looks_local_dot_prefixed_missing_path_is_local() -> None:
    """An explicit ./ prefix is local even when the path doesn't exist (so the
    pre-flight check reports a missing file instead of querying the Hub)."""
    assert looks_local("./definitely/not/there.gguf") is True


# --- local_model_bytes ---
def test_local_model_bytes_file(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1024)
    assert local_model_bytes(str(model)) == 1024


def test_local_model_bytes_directory_recurses(tmp_path) -> None:
    d = tmp_path / "mlx-model"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"x" * 100)
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.safetensors").write_bytes(b"x" * 50)
    assert local_model_bytes(str(d)) == 150


def test_local_model_bytes_missing_or_repo_id_is_zero() -> None:
    assert local_model_bytes("/nope/missing") == 0
    assert local_model_bytes("mlx-community/foo-4bit") == 0


# --- check_local_artifact ---
def test_check_local_artifact_missing_local_raises() -> None:
    with pytest.raises(FileNotFoundError, match="thing for 'svc' not found"):
        check_local_artifact("/nope/missing", kind="thing", service="svc")


def test_check_local_artifact_repo_id_passes() -> None:
    check_local_artifact("org/repo", kind="thing", service="svc")  # must not raise


def test_check_local_artifact_existing_local_passes(tmp_path) -> None:
    p = tmp_path / "x.gguf"
    p.write_bytes(b"x")
    check_local_artifact(str(p), kind="thing", service="svc")  # must not raise


# --- macos_phys_footprint ---
# The implementation lives in `sovereign.core.procmem` — a leaf module shared
# by the parent manager and the worker heartbeat thread. Exercised directly
# here (no re-export from services.inference.base).
def test_parse_phys_footprint_reads_correct_offset() -> None:
    raw = bytearray(512)
    struct.pack_into("<Q", raw, 72, 12345678900)
    assert procmem._parse_phys_footprint(bytes(raw)) == 12345678900


def test_macos_phys_footprint_none_on_non_darwin(monkeypatch) -> None:
    monkeypatch.setattr(procmem.sys, "platform", "linux")
    assert procmem.macos_phys_footprint(123) is None


def test_macos_phys_footprint_none_when_cdll_raises(monkeypatch) -> None:
    monkeypatch.setattr(procmem.sys, "platform", "darwin")

    def boom(*args, **kwargs):
        raise OSError("no lib")

    monkeypatch.setattr(procmem.ctypes, "CDLL", boom)
    assert procmem.macos_phys_footprint(123) is None


class _FakeLibc:
    """A fake libc whose ``proc_pid_rusage`` is a plain instance-attribute
    closure (not a bound method) so real ctypes code can assign ``.argtypes``/
    ``.restype`` on it, exactly as it would on a real ctypes function pointer.
    """

    def __init__(self, returncode: int = 0, write: int | None = None) -> None:
        def proc_pid_rusage(pid, flavor, buf):
            if write is not None:
                struct.pack_into("<Q", buf, 72, write)
            return returncode

        self.proc_pid_rusage = proc_pid_rusage


def test_macos_phys_footprint_none_on_nonzero_returncode(monkeypatch) -> None:
    monkeypatch.setattr(procmem.sys, "platform", "darwin")
    monkeypatch.setattr(procmem.ctypes, "CDLL", lambda *a, **k: _FakeLibc(returncode=1))
    assert procmem.macos_phys_footprint(123) is None


def test_macos_phys_footprint_success(monkeypatch) -> None:
    monkeypatch.setattr(procmem.sys, "platform", "darwin")
    monkeypatch.setattr(
        procmem.ctypes, "CDLL", lambda *a, **k: _FakeLibc(returncode=0, write=999999)
    )
    assert procmem.macos_phys_footprint(123) == 999999
