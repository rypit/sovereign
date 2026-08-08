"""``pi`` — a subprocess-based coding harness (§4b, harness track).

Wraps the `pi <https://pi.dev>`_ coding agent (an npm binary) in its headless
JSON mode: ``pi --mode json`` runs one non-interactive session and streams every
agent/turn/tool event as NDJSON instead of the TUI. pi's built-in tools run
without a permission prompt, so no auto-approve flag is needed — the run
completes with no human in the loop.

Settings live under an isolated ``PI_CODING_AGENT_DIR`` per instance (§11 locked
decision) so this never merges into a human's shared ``~/.pi/agent``:
``materialize()`` writes a ``models.json`` there declaring a custom provider
pointed at the local engine, and ``invoke()`` selects it with
``--provider``/``--model`` (two flags rather than ``provider/model``, so a model
id containing ``/`` — an HF-style ref — stays unambiguous). ``--api-key`` is
passed as well: pi hides models it considers unauthenticated, and local engines
don't check the value.

Declaring this harness in ``sovereign.yaml`` provisions its full dependency
chain automatically (via the shared ``Provisioner`` contract): the package
``Brewfile`` installs Node/npm if missing, then
``npm install -g --ignore-scripts @earendil-works/pi-coding-agent`` installs the
CLI itself — so it's ready both for headless bench runs and for daily-driver use
(``PI_CODING_AGENT_DIR=... pi``) with zero manual setup.

The on-disk ``models.json`` schema below tracks pi's custom-provider format
(``providers.<id>.{baseUrl,api,apiKey,compat,models[]}``) as documented at
implementation time (``docs/models.md`` in the pi repo); reconcile it against
the installed CLI if a future release drifts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from sovereign.core.base_harness import BaseHarness, RunResult, Task
from sovereign.core.registry import register_harness
from sovereign.harnesses.pi.config import PiConfig

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry

# Per-command timeout for `pi --version` when building the fingerprint.
_VERSION_PROBE_TIMEOUT = 5.0

# Startup network calls pi makes unless told otherwise (update check, install
# telemetry) — disabled by default so a bench run stays local and reproducible.
_OFFLINE_ENV = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"}


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of ``--mode json`` output; skips malformed lines."""
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


@register_harness("pi")
class PiHarness(BaseHarness):
    """Configures + invokes an isolated ``pi`` CLI instance."""

    base_type = "pi"
    #: Provisioning chain: package Brewfile installs Node/npm, then this
    #: installs the CLI; satisfied once `pi` resolves on PATH.
    provisioning_binary = "pi"
    #: `--ignore-scripts`: pi needs no dependency lifecycle scripts to install.
    provisioning_commands: ClassVar[list[list[str]]] = [
        ["npm", "install", "-g", "--ignore-scripts", "@earendil-works/pi-coding-agent"]
    ]

    def __init__(self, entry: HarnessEntry) -> None:
        super().__init__(entry)
        self.config = PiConfig.model_validate(entry.config)

    # --- wiring ---
    def _config_dir(self) -> Path:
        configured = self.config.config_dir or f"~/.sovereign/harnesses/{self.name}"
        return Path(configured).expanduser()

    def _resolved(self, key: str, default: str) -> str:
        return str((self.resolved_config or {}).get(key, default))

    def materialize(self) -> None:
        """Write the resolved endpoint as a custom provider in ``models.json``."""
        config_dir = self._config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        model: dict[str, Any] = {"id": self._resolved("model", self.config.model)}
        if self.config.context_window is not None:
            model["contextWindow"] = self.config.context_window
        if self.config.max_tokens is not None:
            model["maxTokens"] = self.config.max_tokens
        provider: dict[str, Any] = {
            "baseUrl": self._resolved("base_url", self.config.base_url),
            "api": self.config.api,
            "apiKey": self._resolved("api_key", self.config.api_key),
            "models": [model],
        }
        if self.config.compat:
            provider["compat"] = self.config.compat
        settings = {"providers": {self.config.provider_id: provider}}
        (config_dir / "models.json").write_text(json.dumps(settings, indent=2) + "\n")

    def _resolve_binary(self) -> str | None:
        binary = self.config.binary
        if shutil.which(binary) is not None:
            return binary
        if Path(binary).expanduser().is_file():
            return str(Path(binary).expanduser())
        return None

    def fingerprint(self) -> dict[str, object]:
        fp = super().fingerprint()
        binary = self._resolve_binary()
        if binary is not None:
            try:
                result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=_VERSION_PROBE_TIMEOUT,
                )
                fp["pi_version"] = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return fp

    # --- invocation ---
    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``pi --mode json`` session and parse its NDJSON.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        binary = self._resolve_binary()
        if binary is None:
            raise FileNotFoundError(
                f"'{self.config.binary}' not found on PATH for harness '{self.name}'. "
                "Install pi (e.g. `npm install -g --ignore-scripts "
                "@earendil-works/pi-coding-agent`)."
            )

        env = {**os.environ, "PI_CODING_AGENT_DIR": str(self._config_dir())}
        if self.config.offline:
            env.update(_OFFLINE_ENV)

        args = [
            binary,
            "--mode",
            "json",
            "--provider",
            self.config.provider_id,
            "--model",
            self._resolved("model", self.config.model),
            "--api-key",
            self._resolved("api_key", self.config.api_key),
        ]
        if self.config.thinking is not None:
            args += ["--thinking", self.config.thinking]
        if self.config.tools:
            args += ["--tools", ",".join(self.config.tools)]
        args.append(task.prompt)

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

        events = _parse_ndjson(proc.stdout)
        return RunResult(
            task_id=task.id,
            success=proc.returncode == 0,
            exit_code=proc.returncode,
            output=proc.stdout,
            metadata={"events": events, "stderr": proc.stderr},
        )
