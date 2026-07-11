"""Bench-spec schema (§6b) — the sweep matrix + thresholds for `sovereign bench run`.

Deliberately separate from ``sovereign.yaml``/``SovereignConfig``: benchmarks are
run imperatively, never at boot, so this is its own small Pydantic model loaded
straight from a YAML file, not folded into the stack config.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from sovereign.core.base_config import GbBytes, SovereignBaseModel


class BenchSpecError(Exception):
    """Raised when a bench spec file cannot be read or validated."""


class BenchMode(StrEnum):
    """How the bench runner obtains a live stack for a cell (§6b)."""

    #: Measure the live stack read-only; never boots/stops anything.
    ATTACH = "attach"
    #: Bench owns the stack: boots each variant, runs its cells, tears down.
    CLEANROOM = "cleanroom"


class Thresholds(SovereignBaseModel):
    """Perf gates that must pass before a cell proceeds to quality runs (§6b funnel)."""

    min_tok_s: float | None = Field(default=None, gt=0)
    max_ttft_ms: float | None = Field(default=None, gt=0)
    min_headroom_bytes: GbBytes | None = Field(default=None, ge=0, validation_alias="min_headroom_gb")


class Budgets(SovereignBaseModel):
    """Per-task limits so a looping agent can't eat the sweep (§6b measurement discipline)."""

    task_timeout_s: int = Field(default=900, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)


class BenchSpec(SovereignBaseModel):
    """A sweep: ``suite x (stack) x harness``, run via ``sovereign bench run``."""

    version: str = "1"
    #: Paths to native task-suite directories (each with a ``suite.yaml``).
    suites: list[str] = Field(default_factory=list)
    #: Stack variant files to sweep (reuses §7b variant capture).
    stacks: list[str]
    #: Harness instance names (within the stacks above) to invoke per suite task.
    harnesses: list[str] = Field(default_factory=list)
    #: Trials per cell — 3+ recommended so mean/spread are meaningful (§6b).
    trials: int = Field(default=3, gt=0)
    seed: int = 0
    thresholds: Thresholds = Field(default_factory=Thresholds)
    budgets: Budgets = Field(default_factory=Budgets)
    mode: BenchMode = BenchMode.ATTACH


def load_bench_spec(path: str | Path) -> BenchSpec:
    """Read and validate a bench-spec YAML file."""
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise BenchSpecError(f"cannot read bench spec {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise BenchSpecError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise BenchSpecError(f"{path} must contain a mapping at the top level")

    try:
        return BenchSpec.model_validate(data)
    except ValidationError as exc:
        raise BenchSpecError(f"invalid bench spec in {path}:\n{exc}") from exc
