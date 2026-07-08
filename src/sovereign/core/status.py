"""The status-snapshot schema shared by producer and consumers (§8).

``Orchestrator.status_snapshot()`` produces this shape; it is persisted as
``status.json`` and rendered by :mod:`sovereign.dashboard`. TypedDicts (rather
than Pydantic models) because the snapshot is written/read as plain JSON on a
2-second cadence — this is a type-checker contract, not runtime validation.
"""

from __future__ import annotations

from typing import TypedDict


class BudgetStatus(TypedDict):
    """Unified-memory budget summary for the dashboard footer."""

    usable_gb: float
    reserved_gb: float
    available_gb: float


class ServiceStatus(TypedDict):
    """One service's row in the dashboard."""

    state: str
    since: str | None
    endpoint: str | None
    descriptor: str | None
    estimated_gb: float | None
    metrics: dict[str, float | str]
    activity: str


class StatusSnapshot(TypedDict):
    """The full ``status.json`` payload."""

    updated_at: str
    budget: BudgetStatus
    services: dict[str, ServiceStatus]
