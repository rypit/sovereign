"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sovereign.core import provisioning


@pytest.fixture(autouse=True)
def _no_real_provisioning(monkeypatch, request):
    """Neutralize Provisioner.provision() suite-wide so no test ever runs real
    installers (brew/npm/uv pip) as a side effect of booting integrations.

    Provisioning-behavior tests opt back in with @pytest.mark.allow_provisioning
    and mock the subprocess layer themselves.
    """
    provisioning.reset_attempts()
    if "allow_provisioning" in request.keywords:
        return
    monkeypatch.setattr(
        provisioning.Provisioner, "provision", classmethod(lambda cls: None)
    )


@pytest.fixture(autouse=True)
def _no_real_hf_api(monkeypatch):
    """Make any unmocked HuggingFace metadata call behave like CI-with-network.

    The suite's fake repo ids don't exist on the Hub, so on a networked runner the
    real HfApi raises RepositoryNotFoundError — while on an offline machine the
    same call quietly returns None (the fallback path) and the mistake hides.
    Stubbing HfApi to raise RepositoryNotFoundError reproduces the loud CI
    behavior everywhere; tests that want a real-looking response monkeypatch
    ``sovereign.services.inference.hf.fetch_repo_info`` (or HfApi itself)
    on top of this.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    from sovereign.services.inference import hf as models

    # Successful fetches are memoised for the process lifetime — drop them so a
    # result cached by one test can never leak into another.
    models._repo_info_cache.clear()

    def _raise(*args, **kwargs):
        response = MagicMock()
        response.headers = {}
        raise RepositoryNotFoundError("stubbed by conftest: no network in tests", response=response)

    api = MagicMock()
    api.model_info.side_effect = _raise
    monkeypatch.setattr(models, "HfApi", lambda *a, **k: api)


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


@pytest.fixture
def socket_path() -> Iterator[Path]:
    """A unix-domain-socket path short enough to bind on every platform.

    macOS caps AF_UNIX sun_path at ~104 bytes and pytest's tmp_path nests
    deeply enough on CI runners to exceed it, so socket tests use a dedicated
    short-lived directory under the system temp root instead of tmp_path.
    """
    tmp_dir = tempfile.mkdtemp(prefix="sov-")
    try:
        yield Path(tmp_dir) / "t.sock"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
