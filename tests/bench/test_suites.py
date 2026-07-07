"""Bench track (B4): native task-suite format."""

from __future__ import annotations

import pytest

from sovereign.bench.suites import SuiteError, load_suite


def _write_suite(tmp_path, text: str):
    d = tmp_path / "my_suite"
    d.mkdir()
    (d / "suite.yaml").write_text(text)
    return d


def test_load_suite_from_directory(tmp_path) -> None:
    d = _write_suite(
        tmp_path,
        """
tasks:
  - id: add_hello
    repo: {path: "./fixture"}
    prompt: "Add a hello() function."
    grader: {type: command, cmd: "pytest"}
""",
    )
    suite = load_suite(d)
    assert len(suite.tasks) == 1
    assert suite.tasks[0].id == "add_hello"
    assert suite.tasks[0].repo.path == "./fixture"
    assert suite.root == str(d)


def test_load_suite_from_yaml_file_directly(tmp_path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text("tasks: []\n")
    suite = load_suite(path)
    assert suite.tasks == []


def test_repo_requires_exactly_one_source(tmp_path) -> None:
    d = _write_suite(
        tmp_path,
        """
tasks:
  - id: t1
    repo: {path: "./a", git_url: "https://example.com/x.git"}
    prompt: "x"
    grader: {type: command, cmd: "true"}
""",
    )
    with pytest.raises(SuiteError, match="invalid suite"):
        load_suite(d)


def test_repo_requires_at_least_one_source(tmp_path) -> None:
    d = _write_suite(
        tmp_path,
        """
tasks:
  - id: t1
    repo: {}
    prompt: "x"
    grader: {type: command, cmd: "true"}
""",
    )
    with pytest.raises(SuiteError):
        load_suite(d)


def test_grader_command_defaults_for_pytest_type() -> None:
    from sovereign.bench.suites import GraderSpec

    g = GraderSpec(type="pytest")
    assert g.command() == "pytest"


def test_grader_command_requires_cmd_for_command_type() -> None:
    from sovereign.bench.suites import GraderSpec

    g = GraderSpec(type="command")
    with pytest.raises(SuiteError, match="grader.cmd is required"):
        g.command()


def test_missing_suite_file_raises(tmp_path) -> None:
    with pytest.raises(SuiteError, match="cannot read"):
        load_suite(tmp_path / "nonexistent")


def test_invalid_yaml_raises(tmp_path) -> None:
    d = tmp_path / "bad_suite"
    d.mkdir()
    (d / "suite.yaml").write_text("tasks: [unterminated\n")
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(d)
