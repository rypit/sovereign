"""Workspace lifecycle + grading (§6b) — grade the repo, not the transcript.

Git diff + grader exit code are ground truth; the harness's self-reported
`RunResult.success` is metadata only. **False-completion rate** — the harness
claims success but the diff is empty or the grader fails — is a first-class
metric computed one level up, in `bench/quality.py`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sovereign.bench.suites import Suite, SuiteTask

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "sovereign-bench",
    "GIT_AUTHOR_EMAIL": "bench@sovereign.local",
    "GIT_COMMITTER_NAME": "sovereign-bench",
    "GIT_COMMITTER_EMAIL": "bench@sovereign.local",
}


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **_GIT_ENV},
        capture_output=True,
        text=True,
    )


def prepare_workspace(task: SuiteTask, suite: Suite, workspace_dir: Path) -> Path:
    """Materialize the task's fixture repo into a throwaway git workspace.

    A ``repo.path`` fixture is copied and committed as a clean baseline (so any
    agent edit shows up as a diff, whether or not the fixture already had its
    own ``.git``); a ``repo.git_url`` fixture is cloned directly.
    """
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.parent.mkdir(parents=True, exist_ok=True)

    if task.repo.path:
        src = Path(task.repo.path)
        if not src.is_absolute():
            src = Path(suite.root) / src
        shutil.copytree(src, workspace_dir, ignore=shutil.ignore_patterns(".git"))
        _run_git(["init"], workspace_dir)
        _run_git(["add", "-A"], workspace_dir)
        _run_git(
            ["commit", "-m", "sovereign-bench: fixture baseline", "--allow-empty"], workspace_dir
        )
    else:
        if task.repo.git_url is None:
            raise ValueError(f"task {task.id}: repo needs either a fixture path or a git_url")
        _run_git(["clone", task.repo.git_url, str(workspace_dir)], workspace_dir.parent)
        if task.repo.rev:
            _run_git(["checkout", task.repo.rev], workspace_dir)
    return workspace_dir


def _has_diff(workspace_dir: Path) -> tuple[bool, str]:
    diff = _run_git(["diff", "--stat", "HEAD"], workspace_dir)
    status = _run_git(["status", "--porcelain"], workspace_dir)
    diff_stat = diff.stdout.strip()
    has_changes = bool(diff_stat) or bool(status.stdout.strip())
    return has_changes, diff_stat


def _decoded(stream: str | bytes | None) -> str:
    """TimeoutExpired carries bytes even in text mode; normalise to str."""
    if stream is None:
        return ""
    return stream.decode(errors="replace") if isinstance(stream, bytes) else stream


def grade_task(task: SuiteTask, workspace_dir: Path) -> dict[str, Any]:
    """Run the task's grader against the workspace and return ground truth."""
    has_diff, diff_stat = _has_diff(workspace_dir)
    grader = task.grader

    if grader.expect_diff and not has_diff:
        return {
            "has_diff": has_diff,
            "diff_stat": diff_stat,
            "grader_exit_code": None,
            "grader_output": "",
            "passed": False,
            "reason": "no diff produced",
        }

    try:
        proc = subprocess.run(  # noqa: S602 - grader command is authored in a local, trusted suite file
            grader.command(),
            shell=True,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=task.timeout_s,
        )
        exit_code: int | None = proc.returncode
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        output = _decoded(exc.stdout) + _decoded(exc.stderr)

    return {
        "has_diff": has_diff,
        "diff_stat": diff_stat,
        "grader_exit_code": exit_code,
        "grader_output": output,
        "passed": exit_code == 0,
    }
