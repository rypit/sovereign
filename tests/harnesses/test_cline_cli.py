"""Harness track (H3): `cline_cli` — subprocess-based headless harness.

The real `cline` npm binary is not required here: `subprocess.run` and
`shutil.which` are mocked, mirroring the `FakeProc` pattern used for the
native-engine managers (`tests/services/test_llama_cpp.py`).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from sovereign.config import HarnessEntry
from sovereign.core.base_harness import Harness, Task
from sovereign.core.registry import get_harness
from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.harnesses.cline_cli import manager as cline_mod
from sovereign.harnesses.cline_cli.manager import ClineCliHarness


def _entry(config: dict | None = None) -> HarnessEntry:
    return HarnessEntry(
        name="cline_local",
        base_type="cline_cli",
        dependencies=["engine"],
        config=config
        or {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
        },
    )


def _resolver() -> Resolver:
    reg = ServiceRegistry()
    reg.register("engine", ResolvedEndpoint("http", "127.0.0.1", 11435, model="llama3-70b"))
    return Resolver(reg, env={})


def _harness(config: dict | None = None) -> ClineCliHarness:
    h = ClineCliHarness(_entry(config))
    h.resolve(_resolver())
    return h


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --- registry / protocol ---
def test_registered_under_base_type() -> None:
    assert get_harness("cline_cli") is ClineCliHarness


def test_satisfies_harness_protocol() -> None:
    assert isinstance(_harness(), Harness)


# --- materialize ---
def test_materialize_writes_isolated_config(tmp_path) -> None:
    h = _harness(
        {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
            "config_dir": str(tmp_path / "cfg"),
        }
    )
    h.materialize()
    config_path = tmp_path / "cfg" / "cline_config.json"
    assert config_path.exists()
    settings = json.loads(config_path.read_text())
    assert settings["openAiCompatibleBaseUrl"] == "http://127.0.0.1:11435/v1"
    assert settings["openAiCompatibleModelId"] == "llama3-70b"
    assert settings["apiProvider"] == "openai-compatible"


def test_materialize_default_config_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    h = _harness()
    h.materialize()
    assert (tmp_path / ".sovereign" / "harnesses" / "cline_local" / "cline_config.json").exists()


# --- invoke ---
def test_invoke_missing_binary_raises(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: None)
    h = _harness()
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        h.invoke(Task(id="t1", prompt="x"))


def test_invoke_success_parses_ndjson(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")
    events = [{"type": "start"}, {"type": "complete", "success": True}]
    stdout = "\n".join(json.dumps(e) for e in events) + "\n"
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeCompleted(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(cline_mod.subprocess, "run", fake_run)
    h = _harness()
    result = h.invoke(Task(id="t1", prompt="add a hello() function", workdir="/tmp/work"))

    assert result.success is True
    assert result.exit_code == 0
    assert result.metadata["events"] == events
    assert captured["args"][0] == "cline"
    assert "--yolo" in captured["args"]
    assert "--json" in captured["args"]
    assert captured["args"][-1] == "add a hello() function"
    assert captured["kwargs"]["cwd"] == "/tmp/work"
    assert captured["kwargs"]["env"]["CLINE_DIR"]


def test_invoke_nonzero_exit_is_failure(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")
    monkeypatch.setattr(
        cline_mod.subprocess, "run", lambda *a, **kw: FakeCompleted(returncode=1, stdout="")
    )
    h = _harness()
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.exit_code == 1


def test_invoke_timeout_returns_failure(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="cline", timeout=1, output="partial output")

    monkeypatch.setattr(cline_mod.subprocess, "run", fake_run)
    h = _harness({"base_url": "x", "model": "y", "timeout_seconds": 1})
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.metadata["error"] == "timeout"
    assert result.output == "partial output"


def test_invoke_max_turns_flag(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        cline_mod.subprocess,
        "run",
        lambda args, **kw: captured.update(args=args) or FakeCompleted(),
    )
    h = _harness(
        {"base_url": "{{ engine.endpoint }}/v1", "model": "{{ engine.model }}", "max_turns": 10}
    )
    h.invoke(Task(id="t1", prompt="x"))
    assert captured["args"][captured["args"].index("--max-turns") + 1] == "10"


# --- provisioning ---
def test_provisioning_declaration() -> None:
    assert ClineCliHarness.provisioning_binary == "cline"
    assert ClineCliHarness.provisioning_commands == [["npm", "install", "-g", "cline"]]


def test_package_ships_brewfile_with_node() -> None:
    brewfile = ClineCliHarness.provisioning_brewfile()
    assert brewfile is not None
    assert 'brew "node"' in brewfile.read_text()


def test_prepare_environment_provisions(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    provisioned: list[type] = []
    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: provisioned.append(cls))
    )
    _harness().prepare_environment()
    assert provisioned == [ClineCliHarness]


@pytest.mark.allow_provisioning
def test_provision_full_chain_on_bare_machine(monkeypatch) -> None:
    """No cline, no npm: Brewfile installs Node, then npm installs cline."""
    from sovereign.core import provisioning

    available = {"brew"}
    monkeypatch.setattr(
        provisioning.shutil, "which", lambda n: f"/fake/bin/{n}" if n in available else None
    )
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        if cmd[0] == "brew":
            available.add("npm")
        elif cmd[0] == "npm":
            available.add("cline")
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    ClineCliHarness.provision()

    assert runs[0][:3] == ["brew", "bundle", "--file"]
    assert runs[0][3].endswith("cline_cli/Brewfile")
    assert runs[1] == ["npm", "install", "-g", "cline"]
    assert ClineCliHarness.provisioning_satisfied()


# --- fingerprint ---
def test_fingerprint_includes_cline_version(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")
    monkeypatch.setattr(
        cline_mod.subprocess,
        "run",
        lambda *a, **kw: FakeCompleted(returncode=0, stdout="4.2.0\n"),
    )
    h = _harness()
    fp = h.fingerprint()
    assert fp["cline_version"] == "4.2.0"
    assert fp["base_type"] == "cline_cli"


def test_fingerprint_skips_version_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: None)
    h = _harness()
    fp = h.fingerprint()
    assert "cline_version" not in fp
