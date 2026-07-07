"""Agentic quality runner (§6b) — harness x model on task suites, graded programmatically.

Quality cells only run for stacks whose perf cell (§B2) already cleared the
funnel's thresholds — the perf/quality gate keeps expensive agentic runs from
burning time on a config that's already known to be too slow or too tight on
memory. If no perf cell has been recorded yet, the funnel can't be enforced
(so the quality cell runs anyway) — run the perf cell first for real gating.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.bench.cells import cell_key, is_complete, read_cell_result
from sovereign.bench.grading import grade_task, prepare_workspace
from sovereign.bench.runner import _stack_identity
from sovereign.bench.suites import load_suite
from sovereign.config import load_config
from sovereign.core.base_harness import Task
from sovereign.core.registry import get_harness
from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.utils.state import read_json

if TYPE_CHECKING:
    from sovereign.bench.runner import CellExecutor, Job
    from sovereign.bench.spec import BenchSpec


class QualityError(Exception):
    """Raised when a quality cell can't be set up: no live manifest, unknown
    harness, or gated by an already-failing perf cell."""


def _registry_from_manifest(manifest: dict[str, Any]) -> ServiceRegistry:
    """Rebuild a `ServiceRegistry` from a persisted manifest.json."""
    registry = ServiceRegistry()
    for svc in manifest.get("services", []):
        endpoint = svc.get("endpoint")
        if not endpoint:
            continue
        registry.register(
            svc["name"],
            ResolvedEndpoint(
                scheme=endpoint["scheme"],
                host=endpoint["host"],
                port=endpoint["port"],
                model=endpoint.get("model"),
            ),
        )
    return registry


def _perf_gate_status(spec: BenchSpec, stack: str, bench_dir: Path) -> bool | None:
    """`None` when no perf cell is recorded yet (funnel can't be enforced)."""
    key = cell_key(
        stack=_stack_identity(stack),
        harness="_none",
        suite="_none",
        seed=spec.seed,
        trials=spec.trials,
    )
    if not is_complete(bench_dir, key):
        return None
    return read_cell_result(bench_dir, key).get("gate_passed")


def run_quality_cell(
    job: Job,
    spec: BenchSpec,
    *,
    manifest_state_dir: str | Path,
    bench_dir: str | Path,
) -> dict[str, Any]:
    """Invoke ``job.harness`` against every task in ``job.suite``, grading each.

    ``manifest_state_dir`` is where the live stack's ``manifest.json`` lives
    (the shared state dir in attach mode; a cell-local Orchestrator state dir
    in clean-room mode). ``bench_dir`` is always the overall
    ``<state_dir>/benchmarks`` — where the perf funnel gate and per-task
    workspaces live, regardless of mode.
    """
    bench_dir = Path(bench_dir)

    gate = _perf_gate_status(spec, job.stack, bench_dir)
    if gate is False:
        raise QualityError(
            f"gated: perf thresholds not met for stack '{job.stack}' — skipping quality run"
        )

    manifest_path = Path(manifest_state_dir) / "manifest.json"
    if not manifest_path.exists():
        raise QualityError(
            f"no live stack found at {manifest_state_dir} (need a running `sovereign up`)"
        )
    manifest = read_json(manifest_path)

    config = load_config(job.stack)
    entry = next((h for h in config.harnesses if h.name == job.harness), None)
    if entry is None:
        raise QualityError(f"unknown harness '{job.harness}' in stack '{job.stack}'")

    harness = get_harness(entry.base_type)(entry)
    resolve = getattr(harness, "resolve", None)
    if callable(resolve):
        resolve(Resolver(_registry_from_manifest(manifest)))
    harness.materialize()

    suite = load_suite(job.suite)
    workspace_root = bench_dir / "workspaces" / job.cell_key

    task_results: list[dict[str, Any]] = []
    false_completions = 0
    passed_count = 0
    for suite_task in suite.tasks:
        workspace = prepare_workspace(suite_task, suite, workspace_root / suite_task.id)
        task = Task(
            id=f"{job.id}-{suite_task.id}", prompt=suite_task.prompt, workdir=str(workspace)
        )
        run_result = harness.invoke(task)
        grade = grade_task(suite_task, workspace)
        false_completion = bool(run_result.success) and not grade["passed"]
        false_completions += false_completion
        passed_count += grade["passed"]
        task_results.append(
            {
                "task_id": suite_task.id,
                "harness_success": run_result.success,
                "harness_exit_code": run_result.exit_code,
                **grade,
                "false_completion": false_completion,
            }
        )

    total = len(suite.tasks)
    return {
        "harness": job.harness,
        "suite": job.suite,
        "tasks": task_results,
        "total": total,
        "passed": passed_count,
        "pass_rate": passed_count / total if total else None,
        "false_completions": false_completions,
        "false_completion_rate": false_completions / total if total else None,
    }


def make_quality_executor(spec: BenchSpec, state_dir: str | Path) -> CellExecutor:
    """Attach-mode quality executor — manifest and bench dir share one root."""
    state_dir = Path(state_dir)

    def executor(job: Job) -> dict[str, Any]:
        return run_quality_cell(
            job, spec, manifest_state_dir=state_dir, bench_dir=state_dir / "benchmarks"
        )

    return executor
