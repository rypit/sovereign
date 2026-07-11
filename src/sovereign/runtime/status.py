"""The status-snapshot schema shared by producer and consumers (§8).

``Orchestrator.status_snapshot()`` produces this shape; it is persisted as
``status.json`` and rendered by :mod:`sovereign.runtime.dashboard`. TypedDicts (rather
than Pydantic models) because the snapshot is written/read as plain JSON on a
2-second cadence — this is a type-checker contract, not runtime validation.
"""

from __future__ import annotations

from typing import TypedDict


class BudgetStatus(TypedDict):
    """Unified-memory budget summary for the dashboard footer."""

    usable_bytes: int
    reserved_bytes: int
    available_bytes: int


class ActivityStatus(TypedDict):
    """A service's current activity as discrete lines (empty when idle).

    Structured rather than a newline-joined string so consumers render the lines
    themselves — e.g. huggingface_hub's several concurrent download bars.
    """

    lines: list[str]


class PrefillStatus(TypedDict):
    """One in-flight prefill request tracked by the telemetry cache."""

    request_id: str
    processed: int
    total: int | None


class TelemetryStatus(TypedDict):
    """A service's telemetry block — shaped exactly like
    :meth:`~sovereign.runtime.telemetry.TelemetryStateCache.snapshot`'s output
    (the producer of record; this TypedDict follows its field names).
    """

    worker_state: str | None
    last_heartbeat: float | None
    prefill: list[PrefillStatus]
    generation_tps: float | None
    prompt_tps: float | None
    tps_history: list[tuple[float, float]]


class ServiceStatus(TypedDict):
    """One service's row in the dashboard."""

    state: str
    since: str | None
    endpoint: str | None
    descriptor: str | None
    engine: str
    estimated_bytes: int | None
    metrics: dict[str, float | str]
    activity: ActivityStatus
    telemetry: TelemetryStatus


class StatusSnapshot(TypedDict):
    """The full ``status.json`` payload."""

    updated_at: str
    budget: BudgetStatus
    services: dict[str, ServiceStatus]
