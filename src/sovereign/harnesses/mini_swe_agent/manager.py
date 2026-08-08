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
import logging
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

logger = logging.getLogger("sovereign")

_INSTALL_HINT = (
    "mini-swe-agent is not installed. Install the optional harness extra: "
    "`uv sync --extra harness` (or `pip install sovereign[harness]`)."
)

# Where the agent's shell commands run when sandboxed (ADR 0009). The agent
# itself stays a host process, so this only needs a shell + common tools.
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"


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
        #: The DockerEnvironment of the current sandboxed invoke, for cleanup.
        self._sandbox_env: Any = None

    # --- sandbox (ADR 0009) ---
    def _sandboxed(self) -> bool:
        return self.config.sandbox is not None and self.config.sandbox.enabled

    def _sandbox_image(self) -> str:
        assert self.config.sandbox is not None
        return self.config.sandbox.image or DEFAULT_SANDBOX_IMAGE

    def prepare_environment(self) -> None:
        """Host package install always (the agent runs on the host), plus the
        daemon probe + exec-image pull when sandboxed."""
        super().prepare_environment()
        if not self._sandboxed():
            return
        from sovereign.services.docker import sandbox as sbx

        cfg = self.config.sandbox
        assert cfg is not None
        if not sbx.daemon_reachable(binary=cfg.binary):
            raise RuntimeError(
                f"Docker daemon is not reachable via '{cfg.binary}' (needed by the "
                f"'{self.name}' sandbox). Start Docker Desktop or OrbStack and retry."
            )
        sbx.ensure_image(self._sandbox_image(), binary=cfg.binary)

    def _sandbox_environment(self, workdir: str) -> Any:
        """Build the upstream DockerEnvironment the agent execs commands into."""
        from minisweagent.environments.docker import DockerEnvironment

        from sovereign.services.docker.sandbox import HARNESS_LABEL, SANDBOX_LABEL

        cfg = self.config.sandbox
        assert cfg is not None
        if cfg.network != "none":
            logger.debug(
                "harness '%s': mini_swe_agent sandboxes always run --network none "
                "(LLM traffic originates on the host); ignoring network=%r",
                self.name,
                cfg.network,
            )
        run_args = [
            "--rm",
            "--network",
            "none",
            "--label",
            f"{SANDBOX_LABEL}=1",
            "--label",
            f"{HARNESS_LABEL}={self.name}",
            "-v",
            f"{workdir}:/workspace",
        ]
        for mount in cfg.mounts:
            run_args += ["-v", mount]
        if cfg.memory_bytes is not None:
            run_args += ["--memory", str(cfg.memory_bytes)]
        return DockerEnvironment(
            image=self._sandbox_image(),
            cwd="/workspace",
            executable=cfg.binary,
            run_args=run_args,
        )

    def _cleanup_sandbox(self) -> None:
        """Stop the exec container; reap anything a timed-out run left behind."""
        env, self._sandbox_env = self._sandbox_env, None
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - best effort, the sweep is the backstop
                pass
        from sovereign.services.docker import sandbox as sbx

        cfg = self.config.sandbox
        assert cfg is not None
        sbx.sweep_leftovers(binary=cfg.binary, harness=self.name)

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
        if self._sandboxed():
            # LLM traffic stays on the host (base_url untouched); only the
            # agent's shell commands move into the throwaway container.
            env = self._sandbox_environment(workdir or os.getcwd())
            self._sandbox_env = env
        else:
            # LocalEnvironment requires a string cwd (mini-swe-agent v2 validates it).
            env = LocalEnvironment(cwd=workdir or os.getcwd())
        return DefaultAgent(
            model,
            env,
            step_limit=self.config.step_limit,
            **self.config.extra,
        )

    def _run_agent(self, agent: Any, task: Task) -> RunResult:
        try:
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
        return self._to_run_result(task.id, result)

    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``mini-swe-agent`` session against ``task.prompt``.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        if self._sandboxed():
            if not task.workdir:
                return RunResult(
                    task_id=task.id,
                    success=False,
                    metadata={"error": "sandbox requires a task workdir"},
                )
            agent = self._build_agent(task.workdir)
            try:
                # No os.chdir: the cwd lives inside the container.
                return self._run_agent(agent, task)
            finally:
                self._cleanup_sandbox()

        agent = self._build_agent(task.workdir)
        cwd = os.getcwd()
        try:
            if task.workdir:
                try:
                    os.chdir(task.workdir)
                except OSError as exc:
                    return RunResult(
                        task_id=task.id,
                        success=False,
                        output=str(exc),
                        metadata={"error": type(exc).__name__},
                    )
            return self._run_agent(agent, task)
        finally:
            os.chdir(cwd)

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
