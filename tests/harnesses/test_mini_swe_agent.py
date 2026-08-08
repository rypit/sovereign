"""Harness track (H2): `mini_swe_agent` — the first concrete harness.

The real `minisweagent` package is not required here: `_build_agent()` imports
it lazily, so tests inject a fake module tree via `sys.modules` and never need
the optional `harness` extra installed.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

from sovereign.config import HarnessEntry
from sovereign.core.base_harness import Harness, Task
from sovereign.core.registry import get_harness
from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.harnesses.mini_swe_agent.manager import MiniSweAgentHarness


def _entry(config: dict | None = None) -> HarnessEntry:
    return HarnessEntry(
        name="mini_swe_local",
        base_type="mini_swe_agent",
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


def _harness(config: dict | None = None) -> MiniSweAgentHarness:
    h = MiniSweAgentHarness(_entry(config))
    h.resolve(_resolver())
    return h


class FakeResult:
    def __init__(self, exit_status="Submitted", messages=None, cost=0.01):
        self.exit_status = exit_status
        self.messages = messages or []
        self.cost = cost

    def __str__(self) -> str:
        return f"<FakeResult {self.exit_status}>"


class FakeAgent:
    last_instance: FakeAgent | None = None

    def __init__(self, model, env, **kwargs):
        self.model = model
        self.env = env
        self.kwargs = kwargs
        self.run_calls: list[str] = []
        FakeAgent.last_instance = self

    def run(self, prompt: str):
        self.run_calls.append(prompt)
        return FakeResult()


class SleepyAgent(FakeAgent):
    def run(self, prompt: str):
        time.sleep(1.3)
        return FakeResult()


class RaisingAgent(FakeAgent):
    def run(self, prompt: str):
        raise RuntimeError("boom")


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeEnv:
    def __init__(self, cwd=None):
        self.cwd = cwd


class FakeDockerEnv:
    """Stands in for minisweagent.environments.docker.DockerEnvironment."""

    last_instance: FakeDockerEnv | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cleaned_up = False
        FakeDockerEnv.last_instance = self

    def cleanup(self):
        self.cleaned_up = True


_MSA_MODULES = (
    "minisweagent",
    "minisweagent.agents",
    "minisweagent.agents.default",
    "minisweagent.environments",
    "minisweagent.environments.local",
    "minisweagent.environments.docker",
    "minisweagent.models",
    "minisweagent.models.litellm_model",
)


def _block_minisweagent(monkeypatch) -> None:
    """Simulate the package being uninstalled, even when it's really present:
    a None entry in sys.modules makes `import <name>` raise ImportError."""
    for name in _MSA_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def _install_fake_minisweagent(monkeypatch, agent_cls=FakeAgent) -> None:
    modules = {name: types.ModuleType(name) for name in _MSA_MODULES}
    modules["minisweagent.agents.default"].DefaultAgent = agent_cls  # type: ignore[attr-defined]
    modules["minisweagent.environments.local"].LocalEnvironment = FakeEnv  # type: ignore[attr-defined]
    modules["minisweagent.environments.docker"].DockerEnvironment = FakeDockerEnv  # type: ignore[attr-defined]
    modules["minisweagent.models.litellm_model"].LitellmModel = FakeModel  # type: ignore[attr-defined]
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)


# --- registry / protocol ---
def test_registered_under_base_type() -> None:
    assert get_harness("mini_swe_agent") is MiniSweAgentHarness


def test_satisfies_harness_protocol() -> None:
    assert isinstance(_harness(), Harness)


# --- materialize ---
def test_materialize_writes_settings_file(tmp_path) -> None:
    h = _harness({"base_url": "{{ engine.endpoint }}/v1", "model": "{{ engine.model }}",
                  "config_dir": str(tmp_path / "cfg")})
    h.materialize()
    settings_path = tmp_path / "cfg" / "settings.yaml"
    assert settings_path.exists()
    import yaml

    settings = yaml.safe_load(settings_path.read_text())
    assert settings["base_url"] == "http://127.0.0.1:11435/v1"
    assert settings["model"] == "llama3-70b"
    assert settings["step_limit"] == 40


def test_materialize_default_config_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    h = _harness()
    h.materialize()
    assert (tmp_path / ".sovereign" / "harnesses" / "mini_swe_local" / "settings.yaml").exists()


# --- provisioning ---
def test_provisioning_satisfied_with_fake_module(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch)
    assert MiniSweAgentHarness.provisioning_satisfied() is True


def test_provisioning_not_satisfied_when_absent(monkeypatch) -> None:
    _block_minisweagent(monkeypatch)
    assert MiniSweAgentHarness.provisioning_satisfied() is False


def test_provisioning_command_uses_uv_pip() -> None:
    (cmd,) = MiniSweAgentHarness.provisioning_commands
    assert cmd[:3] == ["uv", "pip", "install"]
    assert cmd[-1] == "mini-swe-agent"


@pytest.mark.allow_provisioning
def test_provision_installs_package(monkeypatch) -> None:
    from sovereign.core import provisioning

    _block_minisweagent(monkeypatch)
    runs: list[list[str]] = []

    def fake_run(cmd, **kw):
        runs.append(cmd)
        _install_fake_minisweagent(monkeypatch)  # the install "worked"
        return 0, ""

    monkeypatch.setattr(provisioning, "_run", fake_run)
    MiniSweAgentHarness.provision()
    assert len(runs) == 1
    assert runs[0][-1] == "mini-swe-agent"


def test_prepare_environment_provisions(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    provisioned: list[type] = []
    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: provisioned.append(cls))
    )
    _harness().prepare_environment()
    assert provisioned == [MiniSweAgentHarness]


# --- invoke ---
def test_invoke_without_dependency_raises_import_error_message(monkeypatch) -> None:
    _block_minisweagent(monkeypatch)
    h = _harness()
    with pytest.raises(ImportError, match="mini-swe-agent is not installed"):
        h._build_agent()


def test_invoke_success_maps_run_result(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch, FakeAgent)
    h = _harness()
    result = h.invoke(Task(id="t1", prompt="do the thing"))
    assert result.task_id == "t1"
    assert result.success is True
    assert result.metadata["exit_status"] == "Submitted"
    assert result.metadata["cost"] == 0.01
    assert FakeAgent.last_instance is not None
    assert FakeAgent.last_instance.run_calls == ["do the thing"]


def test_invoke_passes_model_and_endpoint(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch, FakeAgent)
    h = _harness()
    h.invoke(Task(id="t1", prompt="x"))
    assert FakeAgent.last_instance is not None
    model_kwargs = FakeAgent.last_instance.model.kwargs
    assert model_kwargs["model_name"] == "openai/llama3-70b"
    assert model_kwargs["model_kwargs"]["api_base"] == "http://127.0.0.1:11435/v1"


def test_invoke_timeout_returns_failure(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch, SleepyAgent)
    h = _harness({"base_url": "{{ engine.endpoint }}/v1", "model": "{{ engine.model }}",
                  "timeout_seconds": 1})
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.metadata["error"] == "timeout"


def test_invoke_exception_returns_failure(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch, RaisingAgent)
    h = _harness()
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert result.metadata["error"] == "RuntimeError"
    assert "boom" in result.output


# --- sandboxed invoke (ADR 0009) ---
def _sandbox_harness(tmp_path, monkeypatch, agent_cls=FakeAgent, **sandbox) -> MiniSweAgentHarness:
    _install_fake_minisweagent(monkeypatch, agent_cls)
    return _harness(
        {
            "base_url": "{{ engine.endpoint }}/v1",
            "model": "{{ engine.model }}",
            "config_dir": str(tmp_path / "cfg"),
            "sandbox": sandbox or {},
        }
    )


@pytest.fixture
def sbx_docker(monkeypatch):
    """Patch the sandbox module's docker seam (for the sweep + daemon probe)."""
    import subprocess as sp

    from sovereign.services.docker import sandbox as sbx_mod

    calls: list[list[str]] = []

    def fake_run(args, *, binary="docker", timeout=None, check=False):
        calls.append(list(args))
        stdout = "27.0\n" if args[:1] == ["version"] else ""
        return sp.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(sbx_mod, "run_docker", fake_run)
    return calls


def test_sandboxed_invoke_uses_docker_env_network_none(tmp_path, sbx_docker, monkeypatch) -> None:
    h = _sandbox_harness(tmp_path, monkeypatch, network="engine_only", memory_gb=1)
    work = tmp_path / "work"
    work.mkdir()
    result = h.invoke(Task(id="t1", prompt="fix it", workdir=str(work)))

    assert result.success is True
    env = FakeDockerEnv.last_instance
    assert env is not None
    assert env.kwargs["image"] == "python:3.12-slim"
    assert env.kwargs["cwd"] == "/workspace"
    run_args = env.kwargs["run_args"]
    # engine_only is ignored for this harness: commands always run offline.
    assert run_args[run_args.index("--network") + 1] == "none"
    assert f"{work}:/workspace" in run_args
    assert "sovereign.sandbox=1" in run_args
    assert "sovereign.harness=mini_swe_local" in run_args
    assert run_args[run_args.index("--memory") + 1] == str(10**9)
    # The LLM still talks to the host endpoint — no proxy rewrite.
    assert FakeAgent.last_instance is not None
    model_kwargs = FakeAgent.last_instance.model.kwargs
    assert model_kwargs["model_kwargs"]["api_base"] == "http://127.0.0.1:11435/v1"


def test_sandboxed_invoke_cleans_up_and_sweeps(tmp_path, sbx_docker, monkeypatch) -> None:
    h = _sandbox_harness(tmp_path, monkeypatch)
    (tmp_path / "w").mkdir()
    h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path / "w")))
    env = FakeDockerEnv.last_instance
    assert env is not None and env.cleaned_up
    assert ["ps", "-aq", "--filter", "label=sovereign.sandbox", "--filter",
            "label=sovereign.harness=mini_swe_local"] in sbx_docker


def test_sandboxed_invoke_sweeps_on_timeout(tmp_path, sbx_docker, monkeypatch) -> None:
    h = _sandbox_harness(tmp_path, monkeypatch, agent_cls=SleepyAgent)
    h.config.timeout_seconds = 1
    (tmp_path / "w").mkdir()
    result = h.invoke(Task(id="t1", prompt="x", workdir=str(tmp_path / "w")))
    assert result.metadata["error"] == "timeout"
    env = FakeDockerEnv.last_instance
    assert env is not None and env.cleaned_up
    assert any(c[:2] == ["ps", "-aq"] for c in sbx_docker)


def test_sandboxed_invoke_requires_workdir(tmp_path, sbx_docker, monkeypatch) -> None:
    FakeDockerEnv.last_instance = None
    h = _sandbox_harness(tmp_path, monkeypatch)
    result = h.invoke(Task(id="t1", prompt="x"))
    assert result.success is False
    assert "workdir" in str(result.metadata["error"])
    assert FakeDockerEnv.last_instance is None  # never even built the container


def test_sandbox_disabled_uses_local_environment(tmp_path, monkeypatch) -> None:
    h = _sandbox_harness(tmp_path, monkeypatch, enabled=False)
    h.invoke(Task(id="t1", prompt="x"))
    assert FakeAgent.last_instance is not None
    assert isinstance(FakeAgent.last_instance.env, FakeEnv)


def test_sandboxed_prepare_environment_pulls_image_and_provisions_host(
    tmp_path, sbx_docker, monkeypatch
) -> None:
    from sovereign.core.provisioning import Provisioner

    provisioned: list[type] = []
    monkeypatch.setattr(Provisioner, "provision", classmethod(lambda cls: provisioned.append(cls)))
    h = _sandbox_harness(tmp_path, monkeypatch)
    h.prepare_environment()
    # Host package still required (agent runs on the host) …
    assert provisioned == [MiniSweAgentHarness]
    # … and the exec image is probed/ensured.
    assert any(c[:2] == ["image", "inspect"] for c in sbx_docker)


def test_sandboxed_prepare_environment_daemon_down_raises(tmp_path, monkeypatch) -> None:
    import subprocess as sp

    from sovereign.core.provisioning import Provisioner
    from sovereign.services.docker import sandbox as sbx_mod

    monkeypatch.setattr(Provisioner, "provision", classmethod(lambda cls: None))
    monkeypatch.setattr(
        sbx_mod, "run_docker", lambda args, **kw: sp.CompletedProcess(args, 1, "", "")
    )
    h = _sandbox_harness(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="daemon is not reachable"):
        h.prepare_environment()
