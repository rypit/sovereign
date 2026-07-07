"""Shared native-engine helpers: looks_local / local_model_bytes / check_local_artifact."""

from __future__ import annotations

import pytest

from sovereign.core.base_native import check_local_artifact, local_model_bytes, looks_local


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
