"""Harness track: `pi` — subprocess-based headless harness.

The real `pi` npm binary is not required here: `subprocess.run` and
`shutil.which` are mocked, mirroring `tests/harnesses/test_cline_cli.py`.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from sovereign.config import HarnessEntry
from sovereign.core.base_harness import Harness, Task
from sovereign.core.registry import get_harness
from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.harnesses.pi import manager as pi_mod
from sovereign.harnesses.pi.manager import PiHarness


def _entry(config: dict | None = None) -> HarnessEntry:
    return HarnessEntry(
        name="pi_local",
        base_type="pi",
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


def _harness(config: dict | None = None) -> PiHarness:
    h = PiHarness(_entry(config))
    h.resolve(_resolver())
    return h


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --- registry / protocol ---
def test_registered_under_base_type() -> None:
    assert get_harness("pi") is PiHarness


def test_satisfies_harness_protocol() -> None:
    assert isinstance(_harness(), Harness)


# --- materialize ---
def test_materialize_writes_custom_provider(tmp_path) -> None:
    h = _harness(
        {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
            "config_dir": str(tmp_path / "cfg"),
        }
    )
    h.materialize()
    models_path = tmp_path / "cfg" / "models.json"
    assert models_path.exists()
    provider = json.loads(models_path.read_text())["providers"]["sovereign"]
    assert provider["baseUrl"] == "http://127.0.0.1:11435/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "sovereign"
    assert provider["models"] == [{"id": "llama3-70b"}]
    assert "compat" not in provider


def test_materialize_includes_limits_and_compat(tmp_path) -> None:
    h = _harness(
        {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
            "config_dir": str(tmp_path / "cfg"),
            "provider_id": "local",
            "context_window": 128000,
            "max_tokens": 8192,
            "compat": {"supportsDeveloperRole": False},
        }
    )
    h.materialize()
    provider = json.loads((tmp_path / "cfg" / "models.json").read_text())["providers"]["local"]
    assert provider["models"] == [
        {"id": "llama3-70b", "contextWindow": 128000, "maxTokens": 8192}
    ]
    assert provider["compat"] == {"supportsDeveloperRole": False}


def test_materialize_default_config_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    h = _harness()
    h.materialize()
    assert (tmp_path / ".sovereign" / "harnesses" / "pi_local" / "models.json").exists()


# --- invoke ---
def test_invoke_missing_binary_raises(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: None)
    h = _harness()
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        h.invoke(Task(id="t1", prompt="x"))


def test_invoke_success_parses_ndjson(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")
    events = [{"type": "agent_start"}, {"type": "agent_end", "messages": []}]
    stdout = "\n".join(json.dumps(e) for e in events) + "\n"
    captured: dict = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeCompleted(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pi_mod.subprocess, "run", fake_run)
    h = _harness(
        {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
            "config_dir": str(tmp_path / "cfg"),
        }
    )
    result = h.invoke(Task(id="t1", prompt="add a hello() function", workdir="/tmp/work"))

    assert result.success is True
    assert result.exit_code == 0
    assert result.metadata["events"] == events
    args = captured["args"]
    assert args[0] == "pi"
    assert args[args.index("--mode") + 1] == "json"
    assert args[args.index("--provider") + 1] == "sovereign"
    assert args[args.index("--model") + 1] == "llama3-70b"
    assert args[args.index("--api-key") + 1] == "sovereign"
    assert args[-1] == "add a hello() function"
    assert captured["kwargs"]["cwd"] == "/tmp/work"
    env = captured["kwargs"]["env"]
    assert env["PI_CODING_AGENT_DIR"] == str(tmp_path / "cfg")
    assert env["PI_OFFLINE"] == "1"


def test_invoke_offline_disabled_leaves_env_alone(monkeypatch) -> None:
    monkeypatch.delenv("PI_OFFLINE", raising=False)
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")
    captured: dict = {}
    monkeypatch.setattr(
        pi_mod.subprocess,
        "run",
        lambda args, **kw: captured.update(kw) or FakeCompleted(),
    )
    h = _harness({"base_url": "x", "model": "y", "offline": False})
    h.invoke(Task(id="t1", prompt="x"))
    assert "PI_OFFLINE" not in captured["env"]


def test_invoke_tools_and_thinking_flags(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")
    captured: dict = {}
    monkeypatch.setattr(
        pi_mod.subprocess,
        "run",
        lambda args, **kw: captured.update(args=args) or FakeCompleted(),
    )
    h = _harness(
        {
            "base_url": "x",
            "model": "y",
            "tools": ["read", "grep", "ls"],
            "thinking": "high",
        }
    )
    h.invoke(Task(id="t1", prompt="x"))
    args = captured["args"]
    assert args[args.index("--tools") + 1] == "read,grep,ls"
    assert args[args.index("--thinking") + 1] == "high"


def test_invoke_nonzero_exit_is_failure(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")
    monkeypatch.setattr(
        pi_mod.subprocess, "run", lambda *a, **kw: FakeCompleted(returncode=1, stdout="")
    )
    h = _harness()
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.exit_code == 1


def test_invoke_timeout_returns_failure(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pi", timeout=1, output="partial output")

    monkeypatch.setattr(pi_mod.subprocess, "run", fake_run)
    h = _harness({"base_url": "x", "model": "y", "timeout_seconds": 1})
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.metadata["error"] == "timeout"
    assert result.output == "partial output"


# --- provisioning ---
def test_provisioning_declaration() -> None:
    assert PiHarness.provisioning_binary == "pi"
    assert PiHarness.provisioning_commands == [
        ["npm", "install", "-g", "--ignore-scripts", "@earendil-works/pi-coding-agent"]
    ]


def test_package_ships_brewfile_with_node() -> None:
    brewfile = PiHarness.provisioning_brewfile()
    assert brewfile is not None
    assert 'brew "node"' in brewfile.read_text()


def test_prepare_environment_provisions(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    provisioned: list[type] = []
    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: provisioned.append(cls))
    )
    _harness().prepare_environment()
    assert provisioned == [PiHarness]


@pytest.mark.allow_provisioning
def test_provision_full_chain_on_bare_machine(monkeypatch) -> None:
    """No pi, no npm: Brewfile installs Node, then npm installs pi."""
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
            available.add("pi")
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    PiHarness.provision()

    assert runs[0][:3] == ["brew", "bundle", "--file"]
    assert runs[0][3].endswith("pi/Brewfile")
    assert runs[1][-1] == "@earendil-works/pi-coding-agent"
    assert PiHarness.provisioning_satisfied()


# --- fingerprint ---
def test_fingerprint_includes_pi_version(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: "/usr/local/bin/pi")
    monkeypatch.setattr(
        pi_mod.subprocess,
        "run",
        lambda *a, **kw: FakeCompleted(returncode=0, stdout="0.84.1\n"),
    )
    h = _harness()
    fp = h.fingerprint()
    assert fp["pi_version"] == "0.84.1"
    assert fp["base_type"] == "pi"


def test_fingerprint_skips_version_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod.shutil, "which", lambda _b: None)
    h = _harness()
    fp = h.fingerprint()
    assert "pi_version" not in fp
