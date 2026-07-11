"""Provisioning track: the shared `Provisioner` contract (`core/provisioning.py`).

These tests opt back into the real `provision()` (the suite-wide autouse fixture
in conftest.py neutralizes it) and mock the subprocess layer (`_run`) plus
`shutil.which`, so nothing real is ever installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sovereign.core import provisioning
from sovereign.core.provisioning import Provisioner, ProvisioningError

pytestmark = pytest.mark.allow_provisioning


class BinaryProvisioner(Provisioner):
    provisioning_binary = "xyz-fake-binary"
    provisioning_commands = [["fake-installer", "install", "xyz"]]


class NothingToProvision(Provisioner):
    pass  # provisioning_binary None -> always satisfied


def _which_fake(available: set[str]):
    return lambda name: f"/fake/bin/{name}" if name in available else None


@pytest.fixture(autouse=True)
def _fresh_attempts():
    provisioning.reset_attempts()


# --- satisfied / discovery ---
def test_no_binary_means_always_satisfied() -> None:
    assert NothingToProvision.provisioning_satisfied() is True


def test_satisfied_when_binary_on_path(monkeypatch) -> None:
    monkeypatch.setattr(provisioning.shutil, "which", _which_fake({"xyz-fake-binary"}))
    assert BinaryProvisioner.provisioning_satisfied() is True


def test_brewfile_discovered_by_convention() -> None:
    """Real integrations ship a Brewfile next to their manager module."""
    import inspect

    from sovereign.harnesses.cline_cli.manager import ClineCliHarness
    from sovereign.services.docker.manager import DockerManager

    # llama_cpp is no longer a Brewfile-provisioned integration (§4 phase 4):
    # it's an embedded Python binding installed via provisioning_commands (a
    # prebuilt Metal wheel), not a Homebrew CLI toolchain — see
    # tests/services/inference/test_llama_cpp.py::test_provisioning_declaration.
    cases: list[tuple[type[provisioning.Provisioner], str]] = [
        (ClineCliHarness, 'brew "node"'),
        (DockerManager, 'cask "docker-desktop"'),
    ]
    for cls, needle in cases:
        brewfile = cls.provisioning_brewfile()
        assert brewfile is not None, cls.__name__
        assert brewfile.parent == Path(inspect.getfile(cls)).parent  # next to the module
        assert needle in brewfile.read_text()


def test_no_brewfile_returns_none() -> None:
    assert BinaryProvisioner.provisioning_brewfile() is None  # tests/core has no Brewfile


def _record_run(runs: list[list[str]]):
    def _run(cmd, **kw):
        runs.append(cmd)
        return (0, "")

    return _run


# --- provision() flow ---
def test_provision_noop_when_satisfied(monkeypatch) -> None:
    monkeypatch.setattr(provisioning.shutil, "which", _which_fake({"xyz-fake-binary"}))
    runs: list[list[str]] = []
    monkeypatch.setattr(provisioning, "_run", _record_run(runs))
    BinaryProvisioner.provision()
    assert runs == []


def test_provision_runs_commands_then_verifies(monkeypatch) -> None:
    available: set[str] = set()
    monkeypatch.setattr(
        provisioning.shutil, "which", lambda n: f"/b/{n}" if n in available else None
    )
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        available.add("xyz-fake-binary")  # the install "worked"
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    BinaryProvisioner.provision()
    assert runs == [["fake-installer", "install", "xyz"]]


def test_provision_runs_brewfile_before_commands(monkeypatch) -> None:
    class WithBrewfile(BinaryProvisioner):
        @classmethod
        def provisioning_brewfile(cls) -> Path | None:
            return Path("/fake/pkg/Brewfile")

    available = {"brew"}
    monkeypatch.setattr(
        provisioning.shutil, "which", lambda n: f"/b/{n}" if n in available else None
    )
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        if cmd[0] == "fake-installer":
            available.add("xyz-fake-binary")
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    WithBrewfile.provision()
    assert runs[0] == ["brew", "bundle", "--file", "/fake/pkg/Brewfile"]
    assert runs[1] == ["fake-installer", "install", "xyz"]


def test_provision_requires_brew_when_brewfile_present(monkeypatch) -> None:
    class WithBrewfile(BinaryProvisioner):
        @classmethod
        def provisioning_brewfile(cls) -> Path | None:
            return Path("/fake/pkg/Brewfile")

    monkeypatch.setattr(provisioning.shutil, "which", _which_fake(set()))
    runs: list[list[str]] = []
    monkeypatch.setattr(provisioning, "_run", _record_run(runs))
    with pytest.raises(ProvisioningError, match="brew.sh"):
        WithBrewfile.provision()
    assert runs == []  # never ran anything without brew


def test_provision_command_failure_raises_with_detail(monkeypatch) -> None:
    monkeypatch.setattr(provisioning.shutil, "which", _which_fake(set()))
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kw: (1, "registry unreachable"))
    with pytest.raises(ProvisioningError, match="registry unreachable"):
        BinaryProvisioner.provision()


def test_provision_still_unsatisfied_after_install_raises(monkeypatch) -> None:
    monkeypatch.setattr(provisioning.shutil, "which", _which_fake(set()))
    monkeypatch.setattr(provisioning, "_run", lambda cmd, **kw: (0, ""))  # "succeeds"
    with pytest.raises(ProvisioningError, match="still missing"):
        BinaryProvisioner.provision()


def test_provision_memoizes_failed_attempts(monkeypatch) -> None:
    monkeypatch.setattr(provisioning.shutil, "which", _which_fake(set()))
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        return 1, "boom"

    monkeypatch.setattr(provisioning, "_run", fake_run)
    with pytest.raises(ProvisioningError, match="boom"):
        BinaryProvisioner.provision()
    with pytest.raises(ProvisioningError, match="earlier provisioning attempt"):
        BinaryProvisioner.provision()
    assert len(runs) == 1  # installers not re-run after a failure


def test_provision_success_then_noop(monkeypatch) -> None:
    available: set[str] = set()
    monkeypatch.setattr(
        provisioning.shutil, "which", lambda n: f"/b/{n}" if n in available else None
    )
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        available.add("xyz-fake-binary")
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    BinaryProvisioner.provision()
    BinaryProvisioner.provision()  # now satisfied — must not run again
    assert len(runs) == 1
