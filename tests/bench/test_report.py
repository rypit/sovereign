"""Bench track (B5): `bench compare` — join cells into a Pareto comparison."""

from __future__ import annotations

from sovereign.bench.report import build_comparison, flag_pareto
from sovereign.core.state import write_json


def _write_run(state_dir, run_id: str, cells: list[dict]) -> None:
    write_json(
        state_dir / "benchmarks" / "runs" / f"{run_id}.json",
        {"run_id": run_id, "cells": cells},
    )


def _perf_cell(stack: str, tok_s: float, ttft_ms: float, engine="engine") -> dict:
    return {
        "id": f"{stack}-_none-_none",
        "cell_key": f"perf-{stack}",
        "stack": stack,
        "harness": "_none",
        "suite": "_none",
        "state": "completed",
        "result": {
            "engine": engine,
            "tok_s": {"mean": tok_s},
            "ttft_ms": {"mean": ttft_ms},
        },
    }


def _quality_cell(stack: str, harness: str, pass_rate: float, false_rate: float) -> dict:
    return {
        "id": f"{stack}-{harness}-suite",
        "cell_key": f"quality-{stack}-{harness}",
        "stack": stack,
        "harness": harness,
        "suite": "suite",
        "state": "completed",
        "result": {"pass_rate": pass_rate, "false_completion_rate": false_rate},
    }


def test_build_comparison_joins_perf_and_quality_for_same_stack(tmp_path) -> None:
    _write_run(
        tmp_path,
        "run1",
        [
            _perf_cell("stack_a.yaml", tok_s=20.0, ttft_ms=100.0),
            _quality_cell("stack_a.yaml", "h1", pass_rate=0.8, false_rate=0.1),
        ],
    )
    rows = build_comparison(tmp_path)
    quality_row = next(r for r in rows if r["harness"] == "h1")
    assert quality_row["tok_s"] == 20.0
    assert quality_row["ttft_ms"] == 100.0
    assert quality_row["pass_rate"] == 0.8


def test_build_comparison_includes_perf_only_row(tmp_path) -> None:
    _write_run(tmp_path, "run1", [_perf_cell("stack_a.yaml", tok_s=20.0, ttft_ms=100.0)])
    rows = build_comparison(tmp_path)
    assert len(rows) == 1
    assert rows[0]["harness"] is None
    assert rows[0]["pass_rate"] is None


def test_build_comparison_skips_incomplete_cells(tmp_path) -> None:
    cell = _perf_cell("stack_a.yaml", tok_s=20.0, ttft_ms=100.0)
    cell["state"] = "failed"
    _write_run(tmp_path, "run1", [cell])
    assert build_comparison(tmp_path) == []


def test_build_comparison_filters_by_run_ids(tmp_path) -> None:
    _write_run(tmp_path, "run1", [_perf_cell("a.yaml", tok_s=10.0, ttft_ms=50.0)])
    _write_run(tmp_path, "run2", [_perf_cell("b.yaml", tok_s=20.0, ttft_ms=60.0)])
    rows = build_comparison(tmp_path, run_ids=["run1"])
    assert len(rows) == 1
    assert rows[0]["stack"] == "a.yaml"


def test_no_runs_returns_empty(tmp_path) -> None:
    assert build_comparison(tmp_path) == []


# --- Pareto flagging ---
def test_flag_pareto_marks_dominated_row_false() -> None:
    rows = [
        {"stack": "a", "harness": "h", "tok_s": 30.0, "pass_rate": 0.9},
        {"stack": "b", "harness": "h", "tok_s": 10.0, "pass_rate": 0.5},  # dominated by a
    ]
    flag_pareto(rows)
    assert rows[0]["pareto"] is True
    assert rows[1]["pareto"] is False


def test_flag_pareto_tradeoff_both_nondominated() -> None:
    rows = [
        {"stack": "a", "harness": "h", "tok_s": 30.0, "pass_rate": 0.5},  # fast, lower quality
        {"stack": "b", "harness": "h", "tok_s": 10.0, "pass_rate": 0.9},  # slow, higher quality
    ]
    flag_pareto(rows)
    assert rows[0]["pareto"] is True
    assert rows[1]["pareto"] is True


def test_flag_pareto_incomparable_rows_marked_none() -> None:
    rows = [{"stack": "a", "harness": None, "tok_s": 30.0, "pass_rate": None}]
    flag_pareto(rows)
    assert rows[0]["pareto"] is None
