"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

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
