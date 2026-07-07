"""Bench track (B1): content-addressed cell keys — skip-completed re-runs."""

from __future__ import annotations

from sovereign.bench.cells import (
    cell_key,
    is_complete,
    read_cell_result,
    write_cell_result,
)


def test_same_inputs_produce_same_key() -> None:
    k1 = cell_key(stack="a.yaml", harness="h", suite="s", seed=1, trial=0)
    k2 = cell_key(stack="a.yaml", harness="h", suite="s", seed=1, trial=0)
    assert k1 == k2


def test_changing_one_axis_changes_the_key() -> None:
    base = cell_key(stack="a.yaml", harness="h", suite="s", seed=1, trial=0)
    changed = cell_key(stack="b.yaml", harness="h", suite="s", seed=1, trial=0)
    assert base != changed


def test_key_independent_of_kwarg_order() -> None:
    k1 = cell_key(a=1, b=2)
    k2 = cell_key(b=2, a=1)
    assert k1 == k2


def test_is_complete_false_until_written(tmp_path) -> None:
    key = cell_key(stack="a.yaml")
    assert not is_complete(tmp_path, key)
    write_cell_result(tmp_path, key, {"tok_s": 42})
    assert is_complete(tmp_path, key)


def test_read_cell_result_roundtrip(tmp_path) -> None:
    key = cell_key(stack="a.yaml")
    write_cell_result(tmp_path, key, {"tok_s": 42, "ttft_ms": 100})
    assert read_cell_result(tmp_path, key) == {"tok_s": 42, "ttft_ms": 100}
