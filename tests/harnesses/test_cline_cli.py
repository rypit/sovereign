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
    reg.register("searxng", ResolvedEndpoint("http", "127.0.0.1", 8888))
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
    captured: dict = {}
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


# --- sandboxed invoke (ADR 0009) ---
def _sandbox_config(tmp_path, **sandbox) -> dict:
    return {
        "base_url": "{{ engine.endpoint }}/v1",
        "model": "{{ engine.model }}",
        "config_dir": str(tmp_path / "cfg"),
        "sandbox": sandbox or {},
    }


class FakeDocker:
    """Records every docker argv; scripted per-subcommand results (see
    tests/services/test_sandbox.py for the fuller twin)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.results: dict[str, subprocess.CompletedProcess] = {}
        self.raises: dict[str, BaseException] = {}

    def __call__(self, args, *, binary="docker", timeout=None, check=False):
        self.calls.append(list(args))
        key = " ".join(args)
        for prefix, exc in self.raises.items():
            if key.startswith(prefix):
                raise exc
        for prefix, result in self.results.items():
            if key.startswith(prefix):
                return result
        return subprocess.CompletedProcess(args, 0, "", "")

    def argv(self, prefix: str) -> list[list[str]]:
        return [c for c in self.calls if " ".join(c).startswith(prefix)]


@pytest.fixture
def sbx_docker(monkeypatch):
    from sovereign.services.docker import sandbox as sbx_mod

    fake = FakeDocker()
    fake.results["inspect"] = subprocess.CompletedProcess([], 0, "true\n", "")
    monkeypatch.setattr(sbx_mod, "run_docker", fake)
    return fake


def test_sandboxed_invoke_full_flow(tmp_path, sbx_docker) -> None:
    h = _harness(_sandbox_config(tmp_path))
    result = h.invoke(Task(id="t1", prompt="fix the bug", workdir=str(tmp_path / "work")))

    assert result.success is True
    # Proxy forwards to the resolved engine port; agent joins the internal net.
    (net_argv,) = sbx_docker.argv("network create")
    assert "--internal" in net_argv
    proxy_argv = next(a for a in sbx_docker.argv("run") if f"{a[3]}".endswith("-sovereign-llm"))
    assert "TCP:host.docker.internal:11435" in proxy_argv
    agent_argv = next(a for a in sbx_docker.argv("run") if f"{a[3]}".endswith("-agent"))
    assert agent_argv[-5:] == [
        "sovereign/cline-sandbox:latest",
        "cline",
        "--yolo",
        "--json",
        "fix the bug",
    ]
    # The staged CLINE_DIR config points at the in-sandbox proxy alias.
    staged = list((tmp_path / "cfg" / "sandbox").rglob("cline_config.json"))
    assert len(staged) == 1
    settings = json.loads(staged[0].read_text())
    assert settings["openAiCompatibleBaseUrl"] == "http://sovereign-llm:8000/v1"
    assert settings["openAiCompatibleModelId"] == "llama3-70b"
    # Teardown removed the network afterwards.
    assert any(" ".join(a).startswith("network rm") for a in sbx_docker.calls)


def test_sandboxed_invoke_env_is_allowlist_only(tmp_path, sbx_docker, monkeypatch) -> None:
    monkeypatch.setenv("HOST_SECRET", "leak")
    h = _harness(_sandbox_config(tmp_path))
    h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    agent_argv = next(a for a in sbx_docker.argv("run") if f"{a[3]}".endswith("-agent"))
    env_pairs = [agent_argv[i + 1] for i, a in enumerate(agent_argv) if a == "-e"]
    assert sorted(env_pairs) == ["CLINE_DIR=/cline", "HOME=/home/node"]
    mounts = [agent_argv[i + 1] for i, a in enumerate(agent_argv) if a == "-v"]
    assert f"{tmp_path}:/workspace" in mounts


def test_sandboxed_invoke_requires_workdir(tmp_path, sbx_docker) -> None:
    h = _harness(_sandbox_config(tmp_path))
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert "workdir" in str(result.metadata["error"])


def test_sandboxed_invoke_docker_exit_maps_to_infra_error(tmp_path, sbx_docker) -> None:
    sbx_docker.results["run --rm"] = subprocess.CompletedProcess([], 125, "", "no such image")
    h = _harness(_sandbox_config(tmp_path))
    result = h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    assert result.success is False
    assert result.exit_code == 125
    assert result.metadata["error"] == "docker"


def test_sandboxed_invoke_timeout_tears_down(tmp_path, sbx_docker) -> None:
    sbx_docker.raises["run --rm"] = subprocess.TimeoutExpired(
        cmd="run", timeout=1, output=b"partial"
    )
    h = _harness(_sandbox_config(tmp_path))
    result = h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    assert result.success is False
    assert result.metadata["error"] == "timeout"
    assert result.output == "partial"
    assert any(" ".join(a).startswith("network rm sov-sbx-") for a in sbx_docker.calls)


def test_sandboxed_invoke_network_none_skips_proxy(tmp_path, sbx_docker) -> None:
    h = _harness(_sandbox_config(tmp_path, network="none"))
    h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    assert not sbx_docker.argv("network create")
    agent_argv = next(a for a in sbx_docker.argv("run") if f"{a[3]}".endswith("-agent"))
    assert agent_argv[agent_argv.index("--network") + 1] == "none"


def test_sandbox_disabled_falls_back_to_native(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cline_mod.shutil, "which", lambda _b: "/usr/local/bin/cline")
    monkeypatch.setattr(cline_mod.subprocess, "run", lambda *a, **kw: FakeCompleted())
    h = _harness(_sandbox_config(tmp_path, enabled=False))
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is True  # went through the host-subprocess path


def test_sandboxed_prepare_environment_builds_default_image(tmp_path, sbx_docker) -> None:
    sbx_docker.results["version"] = subprocess.CompletedProcess([], 0, "27.0\n", "")
    sbx_docker.results["image inspect"] = subprocess.CompletedProcess([], 1, "", "")
    h = _harness(_sandbox_config(tmp_path))
    h.prepare_environment()
    (build_argv,) = sbx_docker.argv("build")
    assert build_argv[2] == "sovereign/cline-sandbox:latest"
    assert build_argv[3].endswith("cline_cli/sandbox")


def test_sandboxed_prepare_environment_skips_host_provisioning(
    tmp_path, sbx_docker, monkeypatch
) -> None:
    from sovereign.core.provisioning import Provisioner

    sbx_docker.results["version"] = subprocess.CompletedProcess([], 0, "27.0\n", "")
    provisioned: list[type] = []
    monkeypatch.setattr(Provisioner, "provision", classmethod(lambda cls: provisioned.append(cls)))
    _harness(_sandbox_config(tmp_path)).prepare_environment()
    assert provisioned == []


def test_sandboxed_prepare_environment_daemon_down_raises(tmp_path, sbx_docker) -> None:
    sbx_docker.results["version"] = subprocess.CompletedProcess([], 1, "", "")
    h = _harness(_sandbox_config(tmp_path))
    with pytest.raises(RuntimeError, match="daemon is not reachable"):
        h.prepare_environment()


def test_sandboxed_fingerprint_uses_image_not_host_binary(tmp_path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(cline_mod.subprocess, "run", lambda *a, **kw: calls.append(a))
    h = _harness(_sandbox_config(tmp_path))
    fp = h.fingerprint()
    assert fp["sandbox_image"] == "sovereign/cline-sandbox:latest"
    assert "cline_version" not in fp and calls == []


def test_sandboxed_allow_routes_extra_service(tmp_path, sbx_docker) -> None:
    h = _harness(_sandbox_config(tmp_path, allow=["searxng"]))
    h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    # Two proxy sidecars: the engine + the allow-listed service.
    proxy_targets = {a[-1] for a in sbx_docker.argv("run") if a[-1].startswith("TCP:")}
    assert proxy_targets == {
        "TCP:host.docker.internal:11435",
        "TCP:host.docker.internal:8888",
    }
    connects = sbx_docker.argv("network connect")
    aliases = {a[a.index("--alias") + 1] for a in connects}
    assert aliases == {"sovereign-llm", "sovereign-searxng"}
    # The agent learns the reachable address via the published env var.
    agent_argv = next(a for a in sbx_docker.argv("run") if f"{a[3]}".endswith("-agent"))
    env_pairs = [agent_argv[i + 1] for i, a in enumerate(agent_argv) if a == "-e"]
    assert "SOVEREIGN_SEARXNG_URL=http://sovereign-searxng:8000" in env_pairs


def test_sandboxed_allow_unknown_service_raises(tmp_path, sbx_docker) -> None:
    from sovereign.services.docker.sandbox import SandboxError

    h = _harness(_sandbox_config(tmp_path, allow=["ghost"]))
    with pytest.raises(SandboxError, match="'ghost' is not in the running stack"):
        h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))


def test_sandboxed_allow_remote_service_raises(tmp_path, sbx_docker) -> None:
    from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
    from sovereign.services.docker.sandbox import SandboxError

    reg = ServiceRegistry()
    reg.register("engine", ResolvedEndpoint("http", "127.0.0.1", 11435, model="m"))
    reg.register("remote_svc", ResolvedEndpoint("http", "gpu-box.local", 9999))
    h = ClineCliHarness(_entry(_sandbox_config(tmp_path, allow=["remote_svc"])))
    h.resolve(Resolver(reg, env={}))
    with pytest.raises(SandboxError, match="non-local host"):
        h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))


def test_sandboxed_allow_ignored_under_network_none(tmp_path, sbx_docker) -> None:
    h = _harness(_sandbox_config(tmp_path, network="none", allow=["ghost"]))
    result = h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
    # No proxies, no registry lookup for 'ghost' — the run proceeds offline.
    assert result.success is True
    assert not sbx_docker.argv("network create")


def test_sandboxed_invoke_rejects_remote_engine(tmp_path, sbx_docker) -> None:
    from sovereign.services.docker.sandbox import SandboxError

    h = _harness(
        {
            "base_url": "http://gpu-box.local:8080/v1",
            "model": "m",
            "config_dir": str(tmp_path / "cfg"),
            "sandbox": {},
        }
    )
    with pytest.raises(SandboxError, match="loopback"):
        h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path)))
