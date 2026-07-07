"""Bench track (B1): `BenchSpec` — imperative, never part of `sovereign.yaml`."""

from __future__ import annotations

import pytest

from sovereign.bench.spec import BenchMode, BenchSpecError, load_bench_spec


def _write(tmp_path, text: str):
    path = tmp_path / "bench.yaml"
    path.write_text(text)
    return path


def test_minimal_spec_defaults() -> None:
    from sovereign.bench.spec import BenchSpec

    spec = BenchSpec.model_validate({"stacks": ["stack.yaml"]})
    assert spec.trials == 3
    assert spec.seed == 0
    assert spec.mode == BenchMode.ATTACH
    assert spec.harnesses == []
    assert spec.suites == []


def test_load_bench_spec_from_file(tmp_path) -> None:
    path = _write(
        tmp_path,
        """
stacks: [stack.yaml]
harnesses: [cline_local]
suites: [suite_a]
trials: 5
seed: 42
mode: cleanroom
thresholds:
  min_tok_s: 10
  max_ttft_ms: 500
budgets:
  task_timeout_s: 60
""",
    )
    spec = load_bench_spec(path)
    assert spec.stacks == ["stack.yaml"]
    assert spec.harnesses == ["cline_local"]
    assert spec.trials == 5
    assert spec.seed == 42
    assert spec.mode == BenchMode.CLEANROOM
    assert spec.thresholds.min_tok_s == 10
    assert spec.budgets.task_timeout_s == 60


def test_missing_file_raises() -> None:
    with pytest.raises(BenchSpecError, match="cannot read"):
        load_bench_spec("/nonexistent/bench.yaml")


def test_invalid_yaml_raises(tmp_path) -> None:
    path = _write(tmp_path, "stacks: [unterminated\n")
    with pytest.raises(BenchSpecError, match="invalid YAML"):
        load_bench_spec(path)


def test_missing_required_field_raises(tmp_path) -> None:
    path = _write(tmp_path, "trials: 3\n")  # no `stacks`
    with pytest.raises(BenchSpecError, match="invalid bench spec"):
        load_bench_spec(path)


def test_extra_field_rejected(tmp_path) -> None:
    path = _write(tmp_path, "stacks: [x.yaml]\nbogus: true\n")
    with pytest.raises(BenchSpecError):
        load_bench_spec(path)


def test_not_a_mapping_raises(tmp_path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(BenchSpecError, match="must contain a mapping"):
        load_bench_spec(path)
