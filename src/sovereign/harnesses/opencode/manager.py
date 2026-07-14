"""``opencode`` — a subprocess-based coding harness (§4b, harness track).

Wraps `opencode <https://opencode.ai>`_ (sst's terminal coding agent, an npm
binary) in headless mode: ``opencode run --format json`` streams JSON events
instead of the interactive TUI, and ``--auto`` auto-approves permissions so a
run completes without a human in the loop. Settings live in an isolated
``opencode.json`` pointed at via the ``OPENCODE_CONFIG`` environment variable
(§11 locked decision) so this never merges into a human's shared global
opencode config.

Declaring this harness in ``sovereign.yaml`` provisions its full dependency
chain automatically (via the shared ``Provisioner`` contract): the package
``Brewfile`` installs Node/npm if missing, then ``npm install -g opencode-ai``
installs the CLI itself — so it's ready both for headless bench runs and for
daily-driver use (``OPENCODE_CONFIG=... opencode``) with zero manual setup.

The materialized config declares the stack's engine as a custom
OpenAI-compatible provider (id ``sovereign``, backed by
``@ai-sdk/openai-compatible``) and selects ``sovereign/<model>`` as the default
model; the flags and schema were validated against opencode 1.17.20 —
reconcile against ``opencode run --help`` if they drift in a future release.
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
from sovereign.harnesses.opencode.config import OpencodeConfig

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry

# Per-command timeout for `opencode --version` when building the fingerprint.
_VERSION_PROBE_TIMEOUT = 5.0

#: Provider id the materialized config registers the stack's engine under —
#: models are addressed as ``sovereign/<model>``.
_PROVIDER_ID = "sovereign"


def _parse_ndjson(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of ``--format json`` event output; skips malformed lines."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


@register_harness("opencode")
class OpencodeHarness(BaseHarness):
    """Configures + invokes an isolated opencode CLI instance."""

    base_type = "opencode"
    #: Provisioning chain: package Brewfile installs Node/npm, then this
    #: installs the CLI; satisfied once `opencode` resolves on PATH.
    provisioning_binary = "opencode"
    provisioning_commands = [["npm", "install", "-g", "opencode-ai"]]

    def __init__(self, entry: HarnessEntry) -> None:
        super().__init__(entry)
        self.config = OpencodeConfig.model_validate(entry.config)

    # --- wiring ---
    def _config_dir(self) -> Path:
        configured = self.config.config_dir or f"~/.sovereign/harnesses/{self.name}"
        return Path(configured).expanduser()

    def _config_file(self) -> Path:
        return self._config_dir() / "opencode.json"

    def _resolved(self, key: str, default: str) -> str:
        return str((self.resolved_config or {}).get(key, default))

    def materialize(self) -> None:
        """Write the resolved endpoint as an isolated ``opencode.json``.

        The stack's engine becomes a custom OpenAI-compatible provider and the
        default model, with permissions pre-approved for headless runs.
        """
        model = self._resolved("model", self.config.model)
        settings = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                _PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Sovereign",
                    "options": {
                        "baseURL": self._resolved("base_url", self.config.base_url),
                        "apiKey": self._resolved("api_key", self.config.api_key),
                    },
                    "models": {model: {"name": model}},
                }
            },
            "model": f"{_PROVIDER_ID}/{model}",
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "allow"},
        }
        config_file = self._config_file()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps(settings, indent=2) + "\n")

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
        """Run one headless ``opencode run --format json --auto`` session.

        Self-reported success is metadata, not proof — grading (diff + tests)
        happens separately at the bench layer.
        """
        binary = self._resolve_binary()
        if binary is None:
            raise FileNotFoundError(
                f"'{self.config.binary}' not found on PATH for harness '{self.name}'. "
                "Install opencode (e.g. `npm install -g opencode-ai`)."
            )

        env = {**os.environ, "OPENCODE_CONFIG": str(self._config_file())}
        model = self._resolved("model", self.config.model)
        args = [
            binary,
            "run",
            "--format",
            "json",
            "--auto",
            "--model",
            f"{_PROVIDER_ID}/{model}",
        ]
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
