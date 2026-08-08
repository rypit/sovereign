"""``opencode`` — a subprocess-based coding harness (§4b, harness track).

Wraps the `opencode <https://github.com/sst/opencode>`_ CLI (an npm binary) in
its non-interactive mode: ``opencode run --format json`` streams NDJSON events
instead of the TUI, and ``--auto`` approves tool permissions so a run completes
without a human in the loop (without it ``run`` auto-*rejects* every permission
request, and the agent can't touch the workspace).

opencode has no ``CLINE_DIR``-style config directory, but it does honour
``OPENCODE_CONFIG`` — a path to one extra config file merged over the user's
global settings. ``materialize()`` writes an ``opencode.json`` into this
harness's isolated directory and ``invoke()`` points ``OPENCODE_CONFIG`` at it,
so the local stack never merges into a human's daily-driver opencode setup
(§11 locked decision). The generated file declares a custom OpenAI-compatible
provider (``@ai-sdk/openai-compatible``) whose ``baseURL`` is the engine
endpoint; ``--model <provider_id>/<model>`` on the command line then wins over
any project-level ``opencode.json`` in the task's workdir.

Note that opencode installs the provider's npm package on first use, so the
very first ``invoke()`` on a machine needs registry access even though every
model call stays local.

The on-disk provider schema below tracks opencode's ``ProviderConfig``
(``provider.<id>.{npm,name,options.{baseURL,apiKey},models}``) as documented at
implementation time; reconcile it against the installed CLI's own schema
(``https://opencode.ai/config.json``) if a future release drifts.
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
from sovereign.harnesses.opencode.config import OpencodeConfig

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry

# Per-command timeout for `opencode --version` when building the fingerprint.
_VERSION_PROBE_TIMEOUT = 5.0


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of ``--format json`` output; skips malformed lines."""
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


@register_harness("opencode")
class OpencodeHarness(BaseHarness):
    """Configures + invokes an isolated ``opencode`` CLI instance."""

    base_type = "opencode"
    #: Provisioning chain: package Brewfile installs Node/npm, then this
    #: installs the CLI; satisfied once `opencode` resolves on PATH.
    provisioning_binary = "opencode"
    provisioning_commands: ClassVar[list[list[str]]] = [["npm", "install", "-g", "opencode-ai"]]

    def __init__(self, entry: HarnessEntry) -> None:
        super().__init__(entry)
        self.config = OpencodeConfig.model_validate(entry.config)

    # --- wiring ---
    def _config_dir(self) -> Path:
        configured = self.config.config_dir or f"~/.sovereign/harnesses/{self.name}"
        return Path(configured).expanduser()

    def _config_path(self) -> Path:
        return self._config_dir() / "opencode.json"

    def _resolved(self, key: str, default: str) -> str:
        return str((self.resolved_config or {}).get(key, default))

    def _model_ref(self) -> str:
        """The ``provider/model`` string opencode selects a model by."""
        return f"{self.config.provider_id}/{self._resolved('model', self.config.model)}"

    def materialize(self) -> None:
        """Write the custom-provider block into an isolated ``opencode.json``."""
        config_dir = self._config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        model = self._resolved("model", self.config.model)
        settings = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                self.config.provider_id: {
                    "npm": self.config.npm_package,
                    "name": f"Sovereign ({self.name})",
                    "options": {
                        "baseURL": self._resolved("base_url", self.config.base_url),
                        "apiKey": self._resolved("api_key", self.config.api_key),
                    },
                    "models": {model: {"name": model}},
                }
            },
            "model": self._model_ref(),
        }
        self._config_path().write_text(json.dumps(settings, indent=2) + "\n")

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
                fp["opencode_version"] = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
        return fp

    # --- invocation ---
    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``opencode run --format json`` session and parse its NDJSON.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        binary = self._resolve_binary()
        if binary is None:
            raise FileNotFoundError(
                f"'{self.config.binary}' not found on PATH for harness '{self.name}'. "
                "Install the opencode CLI (e.g. `npm install -g opencode-ai`)."
            )

        env = {**os.environ, "OPENCODE_CONFIG": str(self._config_path())}
        args = [binary, "run", "--format", "json", "--model", self._model_ref()]
        if self.config.auto_approve:
            args.append("--auto")
        if self.config.agent is not None:
            args += ["--agent", self.config.agent]
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
