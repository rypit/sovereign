"""``cline_cli`` — a subprocess-based coding harness (§4b, harness track).

Wraps the `Cline CLI <https://github.com/cline/cline/tree/main/apps/cli>`_
(an npm binary) in headless mode: ``--yolo`` auto-approves every tool call so
a run completes without a human in the loop, and ``--json`` streams NDJSON
events instead of the interactive TUI. Settings live under an isolated
``CLINE_DIR`` per instance (§11 locked decision) so this never merges into a
human's shared global Cline config.

Declaring this harness in ``sovereign.yaml`` provisions its full dependency
chain automatically (via the shared ``Provisioner`` contract): the package
``Brewfile`` installs Node/npm if missing, then ``npm install -g cline``
installs the CLI itself — so it's ready both for headless bench runs and for
daily-driver use (`CLINE_DIR=... cline`) with zero manual setup.

The on-disk settings schema below approximates Cline's OpenAI-compatible
provider config as documented at implementation time
(docs.cline.bot/provider-config/openai-compatible); reconcile field names
against ``cline config --help`` (or the installed CLI's own schema) if they
drift from a future Cline release.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.core.base_harness import BaseHarness, RunResult, Task
from sovereign.core.registry import register_harness
from sovereign.harnesses.cline_cli.config import ClineCliConfig

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry

logger = logging.getLogger("sovereign")

# Per-command timeout for `cline --version` when building the fingerprint.
_VERSION_PROBE_TIMEOUT = 5.0

# Sovereign-owned sandbox image, built from the package's sandbox/ Dockerfile
# on first use (ADR 0009). A user-supplied sandbox.image is pulled instead.
DEFAULT_SANDBOX_IMAGE = "sovereign/cline-sandbox:latest"


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of ``--json`` NDJSON output; skips malformed lines."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


@register_harness("cline_cli")
class ClineCliHarness(BaseHarness):
    """Configures + invokes an isolated Cline CLI instance."""

    base_type = "cline_cli"
    #: Provisioning chain: package Brewfile installs Node/npm, then this
    #: installs the CLI; satisfied once `cline` resolves on PATH.
    provisioning_binary = "cline"
    provisioning_commands = [["npm", "install", "-g", "cline"]]

    def __init__(self, entry: HarnessEntry) -> None:
        super().__init__(entry)
        self.config = ClineCliConfig.model_validate(entry.config)

    # --- sandbox (ADR 0009) ---
    def _sandboxed(self) -> bool:
        return self.config.sandbox is not None and self.config.sandbox.enabled

    def _sandbox_image(self) -> str:
        assert self.config.sandbox is not None
        return self.config.sandbox.image or DEFAULT_SANDBOX_IMAGE

    def prepare_environment(self) -> None:
        """Host provisioning natively; daemon probe + image build/pull sandboxed."""
        if not self._sandboxed():
            super().prepare_environment()
            return
        from sovereign.services.docker import sandbox as sbx

        cfg = self.config.sandbox
        assert cfg is not None
        if not sbx.daemon_reachable(binary=cfg.binary):
            raise RuntimeError(
                f"Docker daemon is not reachable via '{cfg.binary}' (needed by the "
                f"'{self.name}' sandbox). Start Docker Desktop or OrbStack and retry."
            )
        context = Path(__file__).parent / "sandbox" if cfg.image is None else None
        sbx.ensure_image(self._sandbox_image(), context_dir=context, binary=cfg.binary)

    # --- wiring ---
    def _config_dir(self) -> Path:
        configured = self.config.config_dir or f"~/.sovereign/harnesses/{self.name}"
        return Path(configured).expanduser()

    def _resolved(self, key: str, default: str) -> str:
        return str((self.resolved_config or {}).get(key, default))

    def materialize(self) -> None:
        """Write the resolved provider settings into an isolated ``CLINE_DIR``."""
        config_dir = self._config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "apiProvider": "openai-compatible",
            "openAiCompatibleBaseUrl": self._resolved("base_url", self.config.base_url),
            "openAiCompatibleModelId": self._resolved("model", self.config.model),
            "openAiCompatibleApiKey": self._resolved("api_key", self.config.api_key),
        }
        (config_dir / "cline_config.json").write_text(json.dumps(settings, indent=2) + "\n")

    def _resolve_binary(self) -> str | None:
        binary = self.config.binary
        if shutil.which(binary) is not None:
            return binary
        if Path(binary).expanduser().is_file():
            return str(Path(binary).expanduser())
        return None

    def fingerprint(self) -> dict[str, object]:
        fp = super().fingerprint()
        if self._sandboxed():
            # The image tag is the version identity inside the sandbox; the
            # host binary (possibly absent) is irrelevant to the run.
            fp["sandbox_image"] = self._sandbox_image()
            return fp
        binary = self._resolve_binary()
        if binary is not None:
            try:
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=_VERSION_PROBE_TIMEOUT,
                )
                fp["cline_version"] = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return fp

    def _cline_argv(self, binary: str, task: Task) -> list[str]:
        args = [binary, "--yolo", "--json", task.prompt]
        if self.config.max_turns is not None:
            args += ["--max-turns", str(self.config.max_turns)]
        return args

    def _run_result(self, task: Task, returncode: int, stdout: str, stderr: str) -> RunResult:
        events = _parse_ndjson(stdout)
        return RunResult(
            task_id=task.id,
            success=returncode == 0,
            exit_code=returncode,
            output=stdout,
            metadata={"events": events, "stderr": stderr},
        )

    def _allowed_routes(self) -> tuple[dict[str, int], dict[str, str]]:
        """Proxy routes + published env vars for ``sandbox.allow`` services.

        Each allow-listed service name is resolved against the live registry
        (the same one ``resolve()`` ran with) and becomes an in-sandbox alias
        ``sovereign-<name>`` plus a ``SOVEREIGN_<NAME>_URL`` env var, so the
        agent can discover the reachable address without guessing ports.
        """
        from sovereign.services.docker import sandbox as sbx

        cfg = self.config.sandbox
        assert cfg is not None
        routes: dict[str, int] = {}
        env: dict[str, str] = {}
        if not cfg.allow:
            return routes, env
        if self.resolver is None:
            raise sbx.SandboxError(
                f"harness '{self.name}' has sandbox.allow set but was never resolved "
                "against a running stack"
            )
        for name in cfg.allow:
            if name not in self.resolver.registry:
                raise sbx.SandboxError(
                    f"sandbox.allow service '{name}' is not in the running stack's "
                    f"registry for harness '{self.name}' — is it booted (and listed "
                    "in this harness's dependencies)?"
                )
            endpoint = self.resolver.registry.get(name)
            if not sbx.is_loopback_host(endpoint.host):
                raise sbx.SandboxError(
                    f"sandbox.allow service '{name}' resolves to non-local host "
                    f"'{endpoint.host}' — only local services can be proxy-routed"
                )
            alias = sbx.service_alias(name)
            routes[alias] = endpoint.port
            env[sbx.service_env_var(name)] = f"http://{alias}:{sbx.PROXY_PORT}"
        return routes, env

    def _invoke_sandboxed(self, task: Task) -> RunResult:
        """One throwaway container per task; egress only to the engine plus
        explicitly allow-listed services (ADR 0009)."""
        from sovereign.services.docker import sandbox as sbx

        cfg = self.config.sandbox
        assert cfg is not None
        if not task.workdir:
            return RunResult(
                task_id=task.id,
                success=False,
                metadata={"error": "sandbox requires a task workdir"},
            )

        sbx.sweep_leftovers(binary=cfg.binary, harness=self.name)
        sid = sbx.sandbox_id(self.name)
        base_url = self._resolved("base_url", self.config.base_url)
        engine_port, container_url = sbx.rewrite_base_url(base_url)
        allow_routes: dict[str, int] = {}
        allow_env: dict[str, str] = {}
        if cfg.network == "engine_only":
            allow_routes, allow_env = self._allowed_routes()
        elif cfg.allow:
            logger.debug(
                "harness '%s': sandbox.allow is ignored under network 'none'", self.name
            )

        # Per-invoke staging CLINE_DIR: same settings materialize() writes, but
        # the base URL points at the in-sandbox proxy alias instead of the host.
        staging = self._config_dir() / "sandbox" / sid
        staging.mkdir(parents=True, exist_ok=True)
        settings = {
            "apiProvider": "openai-compatible",
            "openAiCompatibleBaseUrl": container_url,
            "openAiCompatibleModelId": self._resolved("model", self.config.model),
            "openAiCompatibleApiKey": self._resolved("api_key", self.config.api_key),
        }
        (staging / "cline_config.json").write_text(json.dumps(settings, indent=2) + "\n")

        try:
            if cfg.network == "engine_only":
                sbx.create_internal_network(sid, harness_name=self.name, binary=cfg.binary)
                sbx.start_egress_proxies(
                    sid,
                    {sbx.PROXY_ALIAS: engine_port, **allow_routes},
                    harness_name=self.name,
                    image=cfg.proxy_image,
                    binary=cfg.binary,
                )
                network = sid
            else:
                network = "none"
            proc = sbx.run_sandboxed(
                sid,
                harness_name=self.name,
                image=self._sandbox_image(),
                argv=self._cline_argv("cline", task),
                mounts=[f"{task.workdir}:/workspace", f"{staging}:/cline", *cfg.mounts],
                # The container env is exactly this allowlist — never os.environ.
                env={"CLINE_DIR": "/cline", "HOME": "/home/node", **allow_env},
                network=network,
                memory_bytes=cfg.memory_bytes,
                workdir="/workspace",
                timeout=self.config.timeout_seconds,
                binary=cfg.binary,
            )
        except sbx.SandboxTimeout as exc:
            return RunResult(
                task_id=task.id,
                success=False,
                output=exc.partial_stdout,
                metadata={"error": "timeout"},
            )
        finally:
            sbx.teardown_sandbox(sid, binary=cfg.binary)

        if proc.returncode in (125, 126, 127):
            # Docker infrastructure failure (image/daemon/entrypoint), not cline.
            return RunResult(
                task_id=task.id,
                success=False,
                exit_code=proc.returncode,
                output=proc.stdout,
                metadata={"error": "docker", "stderr": proc.stderr},
            )
        return self._run_result(task, proc.returncode, proc.stdout, proc.stderr)

    # --- invocation ---
    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``cline --yolo --json`` session and parse its NDJSON.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        if self._sandboxed():
            return self._invoke_sandboxed(task)
        binary = self._resolve_binary()
        if binary is None:
            raise FileNotFoundError(
                f"'{self.config.binary}' not found on PATH for harness '{self.name}'. "
                "Install Cline CLI (e.g. `npm install -g cline`)."
            )

        env = {**os.environ, "CLINE_DIR": str(self._config_dir())}
        args = self._cline_argv(binary, task)

        try:
            proc = subprocess.run(  # noqa: S603 - argv is constructed, not shell
                args,
                cwd=task.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout
            output = raw.decode(errors="replace") if isinstance(raw, bytes) else (raw or "")
            return RunResult(
                task_id=task.id,
                success=False,
                output=output,
                metadata={"error": "timeout"},
            )

        return self._run_result(task, proc.returncode, proc.stdout, proc.stderr)
