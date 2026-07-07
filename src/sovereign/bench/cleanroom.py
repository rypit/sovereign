"""Clean-room execution (§6b) — bench owns the stack for each cell it measures.

For each cell, boot the cell's variant file via the Orchestrator-as-library
(the same `boot()`/`shutdown()` used by `sovereign serve`, just driven directly
instead of through `serve_forever`'s signal-handling loop), measure it — a perf
probe (B2) for a bare cell, or a full quality suite (B4) when the cell has a
harness+suite — against the manifest that boot just wrote, then tear down
before the next cell. Model-load time is why cells are grouped by stack
rather than interleaved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.bench.perf import run_perf_attach_cell
from sovereign.bench.quality import run_quality_cell
from sovereign.bench.runner import is_perf_only_cell
from sovereign.config import SovereignConfig, load_config
from sovereign.core.resources import ResourceBudgeter, estimate_service_memory
from sovereign.orchestrator import Orchestrator

if TYPE_CHECKING:
    from sovereign.bench.runner import CellExecutor, Job
    from sovereign.bench.spec import BenchSpec


class CleanroomError(Exception):
    """Raised when a clean-room cell can't boot (or is gated before trying)."""


def _would_fit(config: SovereignConfig) -> tuple[bool, float, float]:
    """Pre-prune: would this stack's own services fit its own declared budget?

    Cheap and boot-free — builds managers (no subprocesses spawned) and reuses
    the same `estimate_service_memory`/`ResourceBudgeter` admission-control
    machinery `boot()` already uses, just summed across every service up front
    so a doomed stack never gets a health-check timeout's worth of a chance.
    """
    orch = Orchestrator(config)
    orch.build()
    budgeter = ResourceBudgeter(
        config.resources.max_unified_memory_gb, config.resources.safety_margin_gb
    )
    total = sum(
        estimate_service_memory(manager, orch.entry(name))
        for name, manager in orch.managers.items()
    )
    return budgeter.can_fit(total), total, budgeter.available_gb


async def run_cell_cleanroom(
    job: Job, spec: BenchSpec, cell_state_dir: str | Path, *, bench_dir: str | Path
) -> dict[str, Any]:
    """Boot ``job.stack``, measure it, tear down — regardless of how it ends.

    ``bench_dir`` is the overall ``<state_dir>/benchmarks`` — where the perf
    funnel gate and quality-suite workspaces live, distinct from
    ``cell_state_dir`` (this cell's own throwaway Orchestrator state).
    """
    config = load_config(job.stack)

    fits, needed_gb, available_gb = _would_fit(config)
    if not fits:
        raise CleanroomError(
            f"gated: stack '{job.stack}' needs ~{needed_gb:.1f}GB, only "
            f"{available_gb:.1f}GB available under its own declared budget"
        )

    orch = Orchestrator(config, variant_file=job.stack, state_dir=cell_state_dir)
    try:
        await orch.boot()
        if is_perf_only_cell(job):
            return await run_perf_attach_cell(job, spec, cell_state_dir)
        return run_quality_cell(
            job, spec, manifest_state_dir=cell_state_dir, bench_dir=bench_dir
        )
    finally:
        await orch.shutdown()


def make_cleanroom_executor(spec: BenchSpec, state_dir: str | Path) -> CellExecutor:
    """Build a `CellExecutor` (§B1 seam) that owns boot/measure/teardown per cell."""
    bench_dir = Path(state_dir) / "benchmarks"
    base_dir = bench_dir / "_cleanroom"

    def executor(job: Job) -> dict[str, Any]:
        cell_state_dir = base_dir / job.cell_key
        return asyncio.run(run_cell_cleanroom(job, spec, cell_state_dir, bench_dir=bench_dir))

    return executor
