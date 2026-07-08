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

# Per-command timeout for `cline --version` when building the fingerprint.
_VERSION_PROBE_TIMEOUT = 5.0


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

    # --- invocation ---
    def invoke(self, task: Task) -> RunResult:
        """Run one headless ``cline --yolo --json`` session and parse its NDJSON.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        binary = self._resolve_binary()
        if binary is None:
            raise FileNotFoundError(
                f"'{self.config.binary}' not found on PATH for harness '{self.name}'. "
                "Install Cline CLI (e.g. `npm install -g cline`)."
            )

        env = {**os.environ, "CLINE_DIR": str(self._config_dir())}
        args = [binary, "--yolo", "--json", task.prompt]
        if self.config.max_turns is not None:
            args += ["--max-turns", str(self.config.max_turns)]

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
