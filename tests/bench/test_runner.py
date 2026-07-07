"""Bench track (B1): `Job`/`enumerate_cells`/`run_bench` — the sweep skeleton."""

from __future__ import annotations

import json

import pytest

from sovereign.bench.runner import JobState, enumerate_cells, run_bench
from sovereign.bench.spec import BenchSpec


def _spec(**overrides) -> BenchSpec:
    data = {"stacks": ["a.yaml"], "harnesses": ["h1"], "suites": ["s1"], "trials": 2}
    data.update(overrides)
    return BenchSpec.model_validate(data)


def test_enumerate_cells_expands_full_matrix() -> None:
    spec = _spec(stacks=["a.yaml", "b.yaml"], harnesses=["h1", "h2"], suites=["s1"])
    jobs = enumerate_cells(spec)
    # 2 stacks x 2 harnesses x 1 suite = 4 cells (trials run *within* a cell,
    # not as a separate axis — see Job's docstring).
    assert len(jobs) == 4
    assert all(job.state == JobState.PENDING for job in jobs)


def test_enumerate_cells_defaults_when_no_harnesses_or_suites() -> None:
    spec = BenchSpec.model_validate({"stacks": ["a.yaml"], "trials": 1})
    jobs = enumerate_cells(spec)
    assert len(jobs) == 1
    assert jobs[0].harness == "_none"
    assert jobs[0].suite == "_none"


def test_enumerate_cells_keys_are_stable_and_distinct() -> None:
    jobs = enumerate_cells(_spec(stacks=["a.yaml", "b.yaml"]))
    keys = [j.cell_key for j in jobs]
    assert len(set(keys)) == len(keys)  # every stack gets a distinct key

    jobs_again = enumerate_cells(_spec(stacks=["a.yaml", "b.yaml"]))
    assert [j.cell_key for j in jobs_again] == keys  # deterministic re-enumeration


def test_enumerate_cells_trial_count_change_changes_key() -> None:
    key_a = enumerate_cells(_spec(trials=2))[0].cell_key
    key_b = enumerate_cells(_spec(trials=5))[0].cell_key
    assert key_a != key_b  # changing the trial count changes what the cell measures


def test_run_bench_without_executor_marks_cells_failed(tmp_path) -> None:
    spec = _spec(trials=1)
    manifest = run_bench(spec, state_dir=tmp_path)
    assert len(manifest["cells"]) == 1
    cell = manifest["cells"][0]
    assert cell["state"] == "failed"
    assert "no executor configured" in cell["error"]


def test_run_bench_writes_run_manifest(tmp_path) -> None:
    spec = _spec(trials=1)
    manifest = run_bench(spec, state_dir=tmp_path)
    run_file = tmp_path / "benchmarks" / "runs" / f"{manifest['run_id']}.json"
    assert run_file.exists()
    on_disk = json.loads(run_file.read_text())
    assert on_disk == manifest


def test_run_bench_with_executor_completes_and_persists_result(tmp_path) -> None:
    spec = _spec(trials=1)

    def executor(job):
        return {"tok_s": 99.5, "cell": job.id}

    manifest = run_bench(spec, state_dir=tmp_path, executor=executor)
    cell = manifest["cells"][0]
    assert cell["state"] == "completed"
    assert cell["result"] == {"tok_s": 99.5, "cell": cell["id"]}

    from sovereign.bench.cells import is_complete

    assert is_complete(tmp_path / "benchmarks", cell["cell_key"])


def test_run_bench_executor_exception_marks_failed(tmp_path) -> None:
    spec = _spec(trials=1)

    def executor(job):
        raise RuntimeError("engine unreachable")

    manifest = run_bench(spec, state_dir=tmp_path, executor=executor)
    cell = manifest["cells"][0]
    assert cell["state"] == "failed"
    assert cell["error"] == "engine unreachable"


def test_run_bench_skips_already_completed_cells(tmp_path) -> None:
    spec = _spec(trials=1)
    calls = []

    def executor(job):
        calls.append(job.id)
        return {"tok_s": 1.0}

    run_bench(spec, state_dir=tmp_path, executor=executor)
    assert len(calls) == 1

    # Re-run with the same spec: the cell is already complete, so the executor
    # must not be invoked again — this is the feature that makes iteration fast.
    manifest2 = run_bench(spec, state_dir=tmp_path, executor=executor)
    assert len(calls) == 1
    assert manifest2["cells"][0]["state"] == "completed"


def test_run_bench_changing_an_axis_forces_recompute(tmp_path) -> None:
    calls = []

    def executor(job):
        calls.append(job.id)
        return {"tok_s": 1.0}

    run_bench(_spec(trials=1, seed=1), state_dir=tmp_path, executor=executor)
    run_bench(_spec(trials=1, seed=2), state_dir=tmp_path, executor=executor)
    assert len(calls) == 2  # different seed -> different cell key -> recomputed


@pytest.mark.parametrize("bad_field", ["stacks"])
def test_bench_spec_requires_stacks(bad_field) -> None:
    with pytest.raises(Exception):  # noqa: B017,PT011 - pydantic ValidationError
        BenchSpec.model_validate({})
