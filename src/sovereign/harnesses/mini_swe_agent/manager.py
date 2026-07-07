"""``mini_swe_agent`` — a pure-Python coding harness (§4b, harness track).

Wraps `mini-swe-agent <https://github.com/SWE-agent/mini-swe-agent>`_'s
``DefaultAgent`` Python API, pointed at a local OpenAI-compatible endpoint via
LiteLLM's ``openai/`` provider prefix. Chosen as the first concrete harness
because it has no subprocess/Node toolchain to manage — ``invoke()`` runs the
agent in-process, which proves the whole harness pipeline with the least
moving parts.

The ``minisweagent`` package is an optional dependency (the ``harness`` extra):
imported lazily so the base install stays lean and importing this module never
fails just because it isn't installed. Declaring the harness in
``sovereign.yaml`` provisions it automatically (via the shared ``Provisioner``
contract): ``prepare_environment()`` installs the package with
``uv pip install`` when the import is missing.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from sovereign.core.base_harness import BaseHarness, RunResult, Task
from sovereign.core.registry import register_harness
from sovereign.harnesses.mini_swe_agent.config import MiniSweAgentConfig

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry

_INSTALL_HINT = (
    "mini-swe-agent is not installed. Install the optional harness extra: "
    "`uv sync --extra harness` (or `pip install sovereign[harness]`)."
)


@register_harness("mini_swe_agent")
class MiniSweAgentHarness(BaseHarness):
    """Configures + invokes ``mini-swe-agent``'s ``DefaultAgent`` in-process."""

    base_type = "mini_swe_agent"
    #: uv-managed venvs don't bundle pip; `uv pip --python` targets the running env.
    provisioning_commands: ClassVar[list[list[str]]] = [
        ["uv", "pip", "install", "--python", sys.executable, "mini-swe-agent"]
    ]

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Satisfied when the ``minisweagent`` package is importable."""
        importlib.invalidate_caches()  # a just-installed package must be visible
        try:
            import minisweagent  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, entry: HarnessEntry) -> None:
        super().__init__(entry)
        self.config = MiniSweAgentConfig.model_validate(entry.config)

    # --- wiring ---
    def _config_dir(self) -> Path:
        configured = self.config.config_dir or f"~/.sovereign/harnesses/{self.name}"
        return Path(configured).expanduser()

    def materialize(self) -> None:
        """Write the resolved endpoint/model/key as a settings file a human can
        also use to run ``mini-swe-agent`` by hand against this stack."""
        resolved = self.resolved_config or self.config.model_dump(mode="json")
        config_dir = self._config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "base_url": resolved.get("base_url", self.config.base_url),
            "model": resolved.get("model", self.config.model),
            "api_key": resolved.get("api_key", self.config.api_key),
            "step_limit": self.config.step_limit,
            "timeout_seconds": self.config.timeout_seconds,
        }
        (config_dir / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))

    # --- invocation ---
    def _build_agent(self, workdir: str | None = None):
        try:
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.environments.local import LocalEnvironment
            from minisweagent.models.litellm_model import LitellmModel
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        resolved = self.resolved_config or {}
        base_url = resolved.get("base_url", self.config.base_url)
        model_name = resolved.get("model", self.config.model)
        api_key = resolved.get("api_key", self.config.api_key)

        model = LitellmModel(
            model_name=f"openai/{model_name}",
            model_kwargs={"api_base": base_url, "api_key": api_key},
        )
        # LocalEnvironment requires a string cwd (mini-swe-agent v2 validates it).
        env = LocalEnvironment(cwd=workdir or os.getcwd())
        return DefaultAgent(
            model,
            env,
            step_limit=self.config.step_limit,
            **self.config.extra,
        )

    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``mini-swe-agent`` session against ``task.prompt``.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        agent = self._build_agent(task.workdir)
        cwd = os.getcwd()
        try:
            if task.workdir:
                os.chdir(task.workdir)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(agent.run, task.prompt)
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    return RunResult(
                        task_id=task.id,
                        success=False,
                        output=f"timed out after {self.config.timeout_seconds}s",
                        metadata={"error": "timeout"},
                    )
        except Exception as exc:  # noqa: BLE001 - surface any agent failure as a RunResult
            return RunResult(
                task_id=task.id,
                success=False,
                output=str(exc),
                metadata={"error": type(exc).__name__},
            )
        finally:
            os.chdir(cwd)

        return self._to_run_result(task.id, result)

    @staticmethod
    def _to_run_result(task_id: str, result: Any) -> RunResult:
        exit_status = getattr(result, "exit_status", None)
        messages = getattr(result, "messages", None)
        metadata: dict[str, object] = {}
        if exit_status is not None:
            metadata["exit_status"] = exit_status
        if messages is not None:
            metadata["messages"] = messages
        cost = getattr(result, "cost", None) or getattr(result, "total_cost", None)
        if cost is not None:
            metadata["cost"] = cost
        success = exit_status in (None, "Submitted", "submitted", "completed")
        return RunResult(
            task_id=task_id,
            success=success,
            output=str(result),
            metadata=metadata,
        )
