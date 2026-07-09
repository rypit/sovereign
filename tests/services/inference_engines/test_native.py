"""Shared native-engine helpers: looks_local / local_model_bytes / check_local_artifact
/ macos_phys_footprint."""

from __future__ import annotations

import struct

import pytest

from sovereign.hf import local_model_bytes, looks_local
from sovereign.services.inference_engines import base as base_native
from sovereign.services.inference_engines.base import check_local_artifact, macos_phys_footprint


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
def test_parse_phys_footprint_reads_correct_offset() -> None:
    raw = bytearray(512)
    struct.pack_into("<Q", raw, 72, 12345678900)
    assert base_native._parse_phys_footprint(bytes(raw)) == 12345678900


def test_macos_phys_footprint_none_on_non_darwin(monkeypatch) -> None:
    monkeypatch.setattr(base_native.sys, "platform", "linux")
    assert macos_phys_footprint(123) is None


def test_macos_phys_footprint_none_when_cdll_raises(monkeypatch) -> None:
    monkeypatch.setattr(base_native.sys, "platform", "darwin")

    def boom(*args, **kwargs):
        raise OSError("no lib")

    monkeypatch.setattr(base_native.ctypes, "CDLL", boom)
    assert macos_phys_footprint(123) is None


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
    monkeypatch.setattr(base_native.sys, "platform", "darwin")
    monkeypatch.setattr(base_native.ctypes, "CDLL", lambda *a, **k: _FakeLibc(returncode=1))
    assert macos_phys_footprint(123) is None


def test_macos_phys_footprint_success(monkeypatch) -> None:
    monkeypatch.setattr(base_native.sys, "platform", "darwin")
    monkeypatch.setattr(
        base_native.ctypes, "CDLL", lambda *a, **k: _FakeLibc(returncode=0, write=999999)
    )
    assert macos_phys_footprint(123) == 999999
