"""Native task-suite format (§6b) — the highest-value suite is your own tasks.

A suite is a directory with a ``suite.yaml`` describing tasks: a fixture repo,
a prompt, and a programmatic grader. Deliberately simple — adapt existing
suites (SWE-bench, Aider polyglot) later behind the same ``SuiteTask`` shape
rather than growing this into a format of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from sovereign.core.base_config import SovereignBaseModel


class SuiteError(Exception):
    """Raised when a task suite can't be read or validated."""


class RepoSpec(SovereignBaseModel):
    """Where the task's fixture repo comes from — exactly one of the two."""

    path: str | None = None
    git_url: str | None = None
    rev: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> RepoSpec:
        if bool(self.path) == bool(self.git_url):
            raise ValueError("repo must set exactly one of `path` or `git_url`")
        return self


class GraderSpec(SovereignBaseModel):
    """A programmatic grader run against the workspace after `invoke()`.

    Grade the repo, not the transcript (§6b): the agent's self-reported
    success is metadata, this command's exit code (plus whether a diff
    exists) is ground truth.
    """

    type: Literal["command", "pytest"] = "command"
    #: Shell command to run in the workspace. Defaults to ``pytest`` for type
    #: ``pytest`` when unset.
    cmd: str | None = None
    #: Require a non-empty `git diff` for the task to pass (catches
    #: false-completions where the harness claims success but touched nothing).
    expect_diff: bool = True

    def command(self) -> str:
        if self.cmd:
            return self.cmd
        if self.type == "pytest":
            return "pytest"
        raise SuiteError("grader.cmd is required for type 'command'")


class SuiteTask(SovereignBaseModel):
    """One task: a fixture repo, a prompt, and how to grade the result."""

    id: str
    repo: RepoSpec
    prompt: str
    grader: GraderSpec
    timeout_s: int = Field(default=900, gt=0)


class Suite(SovereignBaseModel):
    """A named collection of tasks, loaded from one `suite.yaml`."""

    version: str = "1"
    #: Directory the suite was loaded from — resolves each task's `repo.path`
    #: relative to the suite, not the caller's cwd.
    root: str = "."
    tasks: list[SuiteTask] = Field(default_factory=list)


def load_suite(path: str | Path) -> Suite:
    """Load a suite from a directory (containing `suite.yaml`) or a YAML file directly."""
    path = Path(path)
    suite_file = path / "suite.yaml" if path.is_dir() else path
    try:
        raw = suite_file.read_text()
    except OSError as exc:
        raise SuiteError(f"cannot read suite {suite_file}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SuiteError(f"invalid YAML in {suite_file}: {exc}") from exc

    if not isinstance(data, dict):
        raise SuiteError(f"{suite_file} must contain a mapping at the top level")
    data.setdefault("root", str(suite_file.parent))

    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        raise SuiteError(f"invalid suite in {suite_file}:\n{exc}") from exc
