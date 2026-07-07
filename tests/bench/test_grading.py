"""Bench track (B4): workspace lifecycle + grading — grade the repo, not the transcript."""

from __future__ import annotations

from sovereign.bench.grading import grade_task, prepare_workspace
from sovereign.bench.suites import GraderSpec, RepoSpec, Suite, SuiteTask


def _fixture_repo(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "util.py").write_text("def existing():\n    return 1\n")
    return fixture


def _suite(tmp_path, fixture) -> Suite:
    return Suite(root=str(tmp_path), tasks=[])


def test_prepare_workspace_creates_git_baseline(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="true"),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    assert (workspace / "util.py").exists()
    assert (workspace / ".git").is_dir()


def test_prepare_workspace_relative_path_resolves_against_suite_root(tmp_path) -> None:
    _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path="fixture"),  # relative
        prompt="x",
        grader=GraderSpec(type="command", cmd="true"),
    )
    suite = Suite(root=str(tmp_path), tasks=[])
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    assert (workspace / "util.py").exists()


def test_no_diff_fails_when_expected(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="true", expect_diff=True),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    grade = grade_task(task, workspace)
    assert grade["has_diff"] is False
    assert grade["passed"] is False
    assert grade["reason"] == "no diff produced"


def test_diff_present_and_grader_passes(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="true", expect_diff=True),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    (workspace / "util.py").write_text(
        "def existing():\n    return 1\n\ndef added():\n    return 2\n"
    )
    grade = grade_task(task, workspace)
    assert grade["has_diff"] is True
    assert grade["passed"] is True
    assert grade["grader_exit_code"] == 0


def test_diff_present_but_grader_fails(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="false", expect_diff=True),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    (workspace / "util.py").write_text("changed\n")
    grade = grade_task(task, workspace)
    assert grade["has_diff"] is True
    assert grade["passed"] is False
    assert grade["grader_exit_code"] != 0


def test_expect_diff_false_ignores_missing_diff(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="true", expect_diff=False),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    grade = grade_task(task, workspace)
    assert grade["has_diff"] is False
    assert grade["passed"] is True


def test_new_untracked_file_counts_as_diff(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="true", expect_diff=True),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    (workspace / "new_file.py").write_text("x = 1\n")
    grade = grade_task(task, workspace)
    assert grade["has_diff"] is True
    assert grade["passed"] is True


def test_grader_timeout_returns_none_exit_code(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="command", cmd="sleep 5", expect_diff=False),
        timeout_s=1,
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    grade = grade_task(task, workspace)
    assert grade["grader_exit_code"] is None
    assert grade["passed"] is False


def test_pytest_type_defaults_command(tmp_path) -> None:
    fixture = _fixture_repo(tmp_path)
    (fixture / "test_util.py").write_text("def test_ok():\n    assert True\n")
    task = SuiteTask(
        id="t1",
        repo=RepoSpec(path=str(fixture)),
        prompt="x",
        grader=GraderSpec(type="pytest", expect_diff=False),
    )
    suite = _suite(tmp_path, fixture)
    workspace = prepare_workspace(task, suite, tmp_path / "workspaces" / "t1")
    grade = grade_task(task, workspace)
    assert grade["grader_exit_code"] == 0
    assert grade["passed"] is True
