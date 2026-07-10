"""Bench track (B4): agentic quality runner — grade the repo, funnel enforcement."""

from __future__ import annotations

import pytest

from sovereign.bench.quality import QualityError, make_quality_executor, run_quality_cell
from sovereign.bench.runner import Job
from sovereign.bench.spec import BenchSpec
from sovereign.core.base_harness import RunResult
from sovereign.core.registry import _HARNESSES
from sovereign.core.resolver import ConsumerKind
from sovereign.core.state import write_json

_BASE_TYPE = "bench_quality_fake_harness"


class ScriptedHarness:
    """A harness whose invoke() outcomes are scripted per-call for tests."""

    consumer_kind = ConsumerKind.NATIVE
    outcomes: list[bool] = []  # class-level: set per test before invoking
    call_count = 0

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.entry = entry
        self.materialized = False

    def resolve(self, resolver) -> None:
        pass

    def prepare_environment(self) -> None:
        pass

    def materialize(self) -> None:
        self.materialized = True

    def invoke(self, task):
        outcome = ScriptedHarness.outcomes[ScriptedHarness.call_count]
        ScriptedHarness.call_count += 1
        return RunResult(task_id=task.id, success=outcome, exit_code=0 if outcome else 1)


@pytest.fixture(autouse=True)
def _register_fake_harness(monkeypatch):
    monkeypatch.setitem(_HARNESSES, _BASE_TYPE, ScriptedHarness)
    ScriptedHarness.call_count = 0
    ScriptedHarness.outcomes = []


def _write_stack(tmp_path) -> str:
    path = tmp_path / "stack.yaml"
    path.write_text(
        f"""
version: "1.1"
resources:
  max_unified_memory_gb: 64
  safety_margin_gb: 4
services:
  - name: engine
    base_type: llama_cpp
    health_check: {{type: http, endpoint: /health, port: 11435}}
    config: {{model: /models/x.gguf}}
harnesses:
  - name: h1
    base_type: {_BASE_TYPE}
    dependencies: [engine]
    config: {{base_url: "{{{{ engine.endpoint }}}}/v1", model: "{{{{ engine.model }}}}"}}
"""
    )
    return str(path)


def _write_manifest(state_dir) -> None:
    write_json(
        state_dir / "manifest.json",
        {
            "services": [
                {
                    "name": "engine",
                    "endpoint": {
                        "scheme": "http",
                        "host": "127.0.0.1",
                        "port": 11435,
                        "model": "llama3-70b",
                    },
                }
            ]
        },
    )


def _write_suite(tmp_path, task_ids: list[str], *, expect_diff=True):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "util.py").write_text("def existing():\n    return 1\n")
    tasks_yaml = "\n".join(
        f"""  - id: {tid}
    repo: {{path: "{fixture}"}}
    prompt: "do {tid}"
    grader: {{type: command, cmd: "true", expect_diff: {str(expect_diff).lower()}}}
"""
        for tid in task_ids
    )
    (suite_dir / "suite.yaml").write_text(f"tasks:\n{tasks_yaml}")
    return suite_dir


def _job(stack: str, suite: str, harness="h1") -> Job:
    return Job(id="cell", cell_key="qkey", stack=stack, harness=harness, suite=suite)


def test_no_live_manifest_raises(tmp_path) -> None:
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1"])
    spec = BenchSpec(stacks=[stack], harnesses=["h1"], suites=[str(suite)], trials=1)
    with pytest.raises(QualityError, match="no live stack found"):
        run_quality_cell(
            _job(stack, str(suite)),
            spec,
            manifest_state_dir=tmp_path,
            bench_dir=tmp_path / "benchmarks",
        )


def test_unknown_harness_raises(tmp_path) -> None:
    _write_manifest(tmp_path)
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1"])
    spec = BenchSpec(stacks=[stack], harnesses=["ghost"], suites=[str(suite)], trials=1)
    with pytest.raises(QualityError, match="unknown harness"):
        run_quality_cell(
            _job(stack, str(suite), harness="ghost"),
            spec,
            manifest_state_dir=tmp_path,
            bench_dir=tmp_path / "benchmarks",
        )


def test_gated_when_perf_cell_failed_threshold(tmp_path) -> None:
    from sovereign.bench.cells import cell_key, write_cell_result
    from sovereign.bench.runner import stack_identity

    _write_manifest(tmp_path)
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1"])
    spec = BenchSpec(stacks=[stack], harnesses=["h1"], suites=[str(suite)], trials=1)

    perf_key = cell_key(
        stack=stack_identity(stack), harness="_none", suite="_none", seed=0, trials=1
    )
    write_cell_result(tmp_path / "benchmarks", perf_key, {"gate_passed": False})

    with pytest.raises(QualityError, match="gated"):
        run_quality_cell(
            _job(stack, str(suite)),
            spec,
            manifest_state_dir=tmp_path,
            bench_dir=tmp_path / "benchmarks",
        )


def test_runs_when_no_perf_cell_recorded(tmp_path) -> None:
    _write_manifest(tmp_path)
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1"])
    spec = BenchSpec(stacks=[stack], harnesses=["h1"], suites=[str(suite)], trials=1)
    ScriptedHarness.outcomes = [True]

    result = run_quality_cell(
        _job(stack, str(suite)),
        spec,
        manifest_state_dir=tmp_path,
        bench_dir=tmp_path / "benchmarks",
    )
    assert result["total"] == 1


def test_pass_and_false_completion_accounting(tmp_path) -> None:
    _write_manifest(tmp_path)
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1", "t2"], expect_diff=True)
    spec = BenchSpec(stacks=[stack], harnesses=["h1"], suites=[str(suite)], trials=1)
    # t1: harness claims success, but grader will fail (no diff produced by the
    # scripted harness — this is exactly a false completion). t2: harness
    # claims failure too (no false completion, just a real failure).
    ScriptedHarness.outcomes = [True, False]

    result = run_quality_cell(
        _job(stack, str(suite)),
        spec,
        manifest_state_dir=tmp_path,
        bench_dir=tmp_path / "benchmarks",
    )
    assert result["total"] == 2
    assert result["passed"] == 0  # neither task produced a diff
    assert result["false_completions"] == 1  # only t1 claimed success falsely
    task_by_id = {t["task_id"]: t for t in result["tasks"]}
    assert task_by_id["t1"]["false_completion"] is True
    assert task_by_id["t2"]["false_completion"] is False


def test_make_quality_executor_wires_state_dir(tmp_path) -> None:
    _write_manifest(tmp_path)
    stack = _write_stack(tmp_path)
    suite = _write_suite(tmp_path, ["t1"])
    spec = BenchSpec(stacks=[stack], harnesses=["h1"], suites=[str(suite)], trials=1)
    ScriptedHarness.outcomes = [True]

    executor = make_quality_executor(spec, tmp_path)
    result = executor(_job(stack, str(suite)))
    assert result["total"] == 1
