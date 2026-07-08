"""The bench `Job` type (§6b) — a distinct run-to-completion lifecycle.

Services never finish; benchmark cells do. ``JobState`` has terminal states
(``COMPLETED``/``FAILED``), unlike ``ServiceState`` — forcing a benchmark into the
service state machine would turn most of that Protocol into no-ops, which is the
signal it needs its own contract (§0 core taxonomy).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from sovereign.bench.cells import cell_key, is_complete, read_cell_result, write_cell_result
from sovereign.bench.lock import acquire_bench_lock
from sovereign.bench.spec import BenchMode, BenchSpec
from sovereign.utils.state import file_hash, write_json


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    """One benchmark cell: a single (stack, harness, suite) combination.

    ``spec.trials`` is *not* a cell axis — it's how many trials the executor
    runs *within* this one cell to report a mean and spread (§6b measurement
    discipline: "3+ trials/cell"). Splitting trials into separate cells would
    make each one a single noisy sample instead.
    """

    id: str
    cell_key: str
    stack: str
    harness: str
    suite: str
    state: JobState = JobState.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    #: True when this cell was already complete from a prior run (content-
    #: addressed skip) rather than freshly executed this time.
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cell_key": self.cell_key,
            "stack": self.stack,
            "harness": self.harness,
            "suite": self.suite,
            "state": str(self.state),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "skipped": self.skipped,
        }


#: Executes one cell and returns its result payload, or raises on failure.
#: ``None`` (the default wired up through B1) means "no runner yet" — every
#: not-already-complete cell is recorded FAILED with an explanatory error.
CellExecutor = Callable[[Job], dict[str, Any]]


def is_perf_only_cell(job: Job) -> bool:
    """Whether a cell has no harness/suite — i.e. it's a pure performance probe."""
    return job.harness == "_none" and job.suite == "_none"


def combine_executors(perf_executor: CellExecutor, quality_executor: CellExecutor) -> CellExecutor:
    """Dispatch each cell to ``perf_executor`` (no harness/suite) or
    ``quality_executor`` (both set), so one spec can sweep pure-perf and
    agentic-quality cells side by side."""

    def executor(job: Job) -> dict[str, Any]:
        return perf_executor(job) if is_perf_only_cell(job) else quality_executor(job)

    return executor


def stack_identity(stack: str) -> dict[str, Any]:
    path = Path(stack)
    if path.is_file():
        return {"path": str(path), "hash": file_hash(path)}
    return {"path": str(path), "hash": None}


def enumerate_cells(spec: BenchSpec) -> list[Job]:
    """Expand the spec's sweep matrix into one `Job` per (stack, harness, suite).

    The cell key folds in ``spec.seed`` and ``spec.trials`` (changing the trial
    count changes what the cell measures) but not a per-trial index.
    """
    suites = spec.suites or ["_none"]
    harnesses = spec.harnesses or ["_none"]
    jobs: list[Job] = []
    for stack in spec.stacks:
        identity = stack_identity(stack)
        stack_label = Path(stack).stem
        for harness in harnesses:
            for suite in suites:
                suite_label = Path(suite).stem if suite != "_none" else suite
                key = cell_key(
                    stack=identity,
                    harness=harness,
                    suite=suite,
                    seed=spec.seed,
                    trials=spec.trials,
                )
                jobs.append(
                    Job(
                        id=f"{stack_label}-{harness}-{suite_label}",
                        cell_key=key,
                        stack=stack,
                        harness=harness,
                        suite=suite,
                    )
                )
    return jobs


def _execute_cells(jobs: list[Job], bench_dir: Path, executor: CellExecutor | None) -> None:
    for job in jobs:
        if is_complete(bench_dir, job.cell_key):
            job.state = JobState.COMPLETED
            job.result = read_cell_result(bench_dir, job.cell_key)
            job.skipped = True
            continue

        if executor is None:
            job.state = JobState.FAILED
            job.error = (
                "no executor configured for this cell "
                "(perf/quality runners land in later phases)"
            )
            continue

        job.state = JobState.RUNNING
        job.started_at = _now()
        try:
            job.result = executor(job)
            write_cell_result(bench_dir, job.cell_key, job.result)
            job.state = JobState.COMPLETED
        except Exception as exc:  # noqa: BLE001 - a failed cell shouldn't abort the sweep
            job.state = JobState.FAILED
            job.error = str(exc)
        job.finished_at = _now()


def run_bench(
    spec: BenchSpec,
    *,
    state_dir: str | Path = ".sovereign",
    executor: CellExecutor | None = None,
) -> dict[str, Any]:
    """Enumerate cells, skip already-completed ones, execute the rest via ``executor``.

    With no ``executor`` (B1's wiring — perf/quality runners land in B2+), every
    cell that isn't already complete is recorded ``FAILED`` with an explanatory
    error rather than silently doing nothing.

    Clean-room mode holds ``bench.lock`` for the run's duration (§6b) so it can't
    race a daemon-managed ``sovereign up``/``serve`` in the same state dir; attach
    mode is read-only and needs no lock.
    """
    bench_dir = Path(state_dir) / "benchmarks"
    jobs = enumerate_cells(spec)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

    if spec.mode is BenchMode.CLEANROOM:
        with acquire_bench_lock(state_dir, run_id):
            _execute_cells(jobs, bench_dir, executor)
    else:
        _execute_cells(jobs, bench_dir, executor)

    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "mode": str(spec.mode),
        "cells": [job.to_dict() for job in jobs],
    }
    write_json(bench_dir / "runs" / f"{run_id}.json", manifest)
    return manifest
