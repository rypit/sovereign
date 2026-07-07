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
    last_instance = None

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


def _install_fake_minisweagent(monkeypatch, agent_cls=FakeAgent) -> None:
    modules = {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.agents": types.ModuleType("minisweagent.agents"),
        "minisweagent.agents.default": types.ModuleType("minisweagent.agents.default"),
        "minisweagent.environments": types.ModuleType("minisweagent.environments"),
        "minisweagent.environments.local": types.ModuleType("minisweagent.environments.local"),
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": types.ModuleType(
            "minisweagent.models.litellm_model"
        ),
    }
    modules["minisweagent.agents.default"].DefaultAgent = agent_cls
    modules["minisweagent.environments.local"].LocalEnvironment = FakeEnv
    modules["minisweagent.models.litellm_model"].LitellmModel = FakeModel
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


# --- invoke ---
def test_invoke_without_dependency_raises_import_error_message(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "minisweagent", raising=False)
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
    assert FakeAgent.last_instance.run_calls == ["do the thing"]


def test_invoke_passes_model_and_endpoint(monkeypatch) -> None:
    _install_fake_minisweagent(monkeypatch, FakeAgent)
    h = _harness()
    h.invoke(Task(id="t1", prompt="x"))
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
