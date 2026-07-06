"""The core contract for coding harnesses: ``Harness`` (§4b).

Harnesses are **leaf consumers** of the service registry: they reuse the resolver
and dependency edges, but nothing depends on them. Two separable capabilities:

* ``materialize()`` — project resolved endpoints/secrets into the tool's own
  config format. Runs only after dependencies are ``READY``; re-runs when an
  endpoint changes.
* ``invoke(task)`` — run one headless, non-interactive session to completion.
  Not all harnesses support this.

``Task`` and ``RunResult`` are intentionally minimal here; fields grow when the
harness/bench tracks land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Task:
    """A single unit of work handed to a harness for a headless run."""

    #: Human-readable identifier for the task (used in bench cell keys / reports).
    id: str
    #: The instruction / prompt given to the harness.
    prompt: str
    #: Working directory the harness operates in (e.g. a throwaway sandbox).
    workdir: str | None = None
    #: Arbitrary per-task metadata (suite name, seed, budgets, ...).
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RunResult:
    """The outcome of a single ``invoke()`` session."""

    task_id: str
    #: Whether the harness reports the run as complete. Ground truth is graded
    #: separately (diff + tests) — a self-report of success is metadata, not proof.
    success: bool
    #: Process exit code, when the harness exposes one.
    exit_code: int | None = None
    #: Captured output / transcript path / free-form notes.
    output: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Harness(Protocol):
    """Contract for a configure-then-run-on-demand coding harness."""

    #: Unique instance ID (e.g. ``"cline_local"``).
    name: str
    #: Names of services that must be ``READY`` before this harness is usable.
    dependencies: list[str]

    def materialize(self) -> None:
        """Write resolved endpoints/secrets into the tool's own config format."""
        ...

    def invoke(self, task: Task) -> RunResult:
        """Run one headless, non-interactive session and return its result."""
        ...
