"""`bench compare` (§6b) — join cell results into a speed/quality Pareto frontier.

No auto-optimizer: this only joins and flags the frontier. You're the optimizer
(§6b measurement discipline) — the output is a tradeoff surface, not a winner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sovereign.state import read_json


def _all_cells(state_dir: str | Path, run_ids: list[str] | None) -> list[dict[str, Any]]:
    """Every cell from the given (or all) recorded runs, tagged with its run_id."""
    runs_dir = Path(state_dir) / "benchmarks" / "runs"
    if not runs_dir.exists():
        return []
    cells: list[dict[str, Any]] = []
    for run_file in sorted(runs_dir.glob("*.json")):
        manifest = read_json(run_file)
        if run_ids and manifest.get("run_id") not in run_ids:
            continue
        for cell in manifest.get("cells", []):
            cells.append({**cell, "run_id": manifest.get("run_id")})
    return cells


def _perf_metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
    result = result or {}
    return {
        "engine": result.get("engine"),
        "tok_s": (result.get("tok_s") or {}).get("mean"),
        "ttft_ms": (result.get("ttft_ms") or {}).get("mean"),
    }


def _quality_metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
    result = result or {}
    return {
        "pass_rate": result.get("pass_rate"),
        "false_completion_rate": result.get("false_completion_rate"),
    }


def build_comparison(
    state_dir: str | Path, run_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """One row per (stack) perf-only cell, or per (stack, harness) quality cell —
    quality rows are joined against their stack's perf cell for tok/s + TTFT."""
    cells = [c for c in _all_cells(state_dir, run_ids) if c.get("state") == "completed"]

    perf_by_stack: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if cell["harness"] == "_none" and cell["suite"] == "_none":
            perf_by_stack[cell["stack"]] = cell.get("result") or {}

    rows: list[dict[str, Any]] = []
    seen_perf_stacks: set[str] = set()
    for cell in cells:
        is_perf = cell["harness"] == "_none" and cell["suite"] == "_none"
        if is_perf:
            if cell["stack"] in seen_perf_stacks:
                continue
            seen_perf_stacks.add(cell["stack"])
            row = {
                "stack": cell["stack"],
                "harness": None,
                **_perf_metrics(cell.get("result")),
                "pass_rate": None,
                "false_completion_rate": None,
            }
        else:
            perf = perf_by_stack.get(cell["stack"], {})
            row = {
                "stack": cell["stack"],
                "harness": cell["harness"],
                **_perf_metrics(perf),
                **_quality_metrics(cell.get("result")),
            }
        rows.append(row)
    return rows


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether ``a`` Pareto-dominates ``b`` on (tok_s, pass_rate): >= on both, > on one."""
    ge_both = a["tok_s"] >= b["tok_s"] and a["pass_rate"] >= b["pass_rate"]
    gt_one = a["tok_s"] > b["tok_s"] or a["pass_rate"] > b["pass_rate"]
    return ge_both and gt_one


def flag_pareto(rows: list[dict[str, Any]]) -> None:
    """Set ``row["pareto"]`` in place: True/False when comparable, else `None`."""
    comparable = [r for r in rows if r["tok_s"] is not None and r["pass_rate"] is not None]
    for row in rows:
        if row["tok_s"] is None or row["pass_rate"] is None:
            row["pareto"] = None
            continue
        row["pareto"] = not any(_dominates(other, row) for other in comparable if other is not row)
