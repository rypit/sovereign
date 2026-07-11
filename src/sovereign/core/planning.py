"""Dry-run stack planning — the boot path's routing + admission, without starting anything.

``sovereign plan`` and ``Orchestrator.boot()`` must never disagree, so this module
reuses the exact seams boot does: :func:`sovereign.core.registry.route_entry` for
engine routing and :func:`sovereign.core.resources.estimate_service_memory` (through
the real manager class) for the memory number. Constructing the manager also
validates the service's ``config:`` block, so typos surface at dry-run time instead
of at boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from sovereign.config import ServiceEntry, SovereignConfig
from sovereign.core.errors import ModelResolutionError
from sovereign.core.registry import get_service_manager, populate_registries, route_entry
from sovereign.core.resources import (
    ResourceBudgeter,
    ResourceExhaustedError,
    estimate_service_memory,
    estimate_source,
)

# Plan verdicts, one per failure mode (rendered by the CLI).
VERDICT_OK = "OK"
VERDICT_REFUSED = "REFUSED"
VERDICT_ROUTING_ERROR = "ROUTING ERROR"
VERDICT_CONFIG_ERROR = "CONFIG ERROR"


@dataclass
class ServicePlan:
    """One service's dry-run outcome."""

    name: str
    base_type: str  #: resolved concrete type (or the requested one when routing failed)
    requested_auto: bool  #: whether the YAML declared ``auto`` (or omitted base_type)
    model: str
    source: str  #: where the estimate came from: declared|local|cached|hub|unknown|-
    estimated_bytes: int | None  #: the admitted number (bytes); None when nothing is known
    verdict: str
    error: str | None = None  #: the underlying message for non-OK verdicts


@dataclass
class StackPlan:
    """Every service's plan plus the budget they were admitted against."""

    services: list[ServicePlan]
    budget: ResourceBudgeter

    @property
    def ok(self) -> bool:
        return all(s.verdict == VERDICT_OK for s in self.services)


def plan_stack(config: SovereignConfig, state_dir: Path) -> StackPlan:
    """Route, estimate, and admit every service exactly as boot would. No downloads."""
    populate_registries()
    budgeter = ResourceBudgeter(
        config.resources.max_unified_memory_bytes, config.resources.safety_margin_bytes
    )
    services = [_plan_service(entry, state_dir, budgeter) for entry in config.services]
    return StackPlan(services=services, budget=budgeter)


def _plan_service(
    entry: ServiceEntry, state_dir: Path, budgeter: ResourceBudgeter
) -> ServicePlan:
    model = str(entry.config.get("model") or "-")
    requested_auto = entry.base_type == "auto"

    # Route (auto entries need HF metadata / the routing cache) — same call as
    # Orchestrator._build().
    try:
        base_type = route_entry(entry, state_dir)
    except ModelResolutionError as exc:
        return ServicePlan(
            name=entry.name,
            base_type=entry.base_type,
            requested_auto=requested_auto,
            model=model,
            source="-",
            estimated_bytes=None,
            verdict=VERDICT_ROUTING_ERROR,
            error=str(exc),
        )

    # Construct the real manager: validates the config block and gives admission
    # the same estimator boot uses.
    try:
        manager_cls = get_service_manager(base_type)
        manager = manager_cls(entry)
    except (KeyError, ValidationError, ValueError) as exc:
        return ServicePlan(
            name=entry.name,
            base_type=base_type,
            requested_auto=requested_auto,
            model=model,
            source="-",
            estimated_bytes=None,
            verdict=VERDICT_CONFIG_ERROR,
            error=str(exc),
        )

    estimated = estimate_service_memory(manager, entry)
    source = estimate_source(manager, entry)
    display_bytes: int | None = estimated if (estimated or source != "unknown") else None

    # Admit against the running budget — same refuse-to-boot rule as boot.
    try:
        budgeter.admit(entry.name, estimated)
    except ResourceExhaustedError as exc:
        return ServicePlan(
            name=entry.name,
            base_type=base_type,
            requested_auto=requested_auto,
            model=model,
            source=source,
            estimated_bytes=display_bytes,
            verdict=VERDICT_REFUSED,
            error=str(exc),
        )

    return ServicePlan(
        name=entry.name,
        base_type=base_type,
        requested_auto=requested_auto,
        model=model,
        source=source,
        estimated_bytes=display_bytes,
        verdict=VERDICT_OK,
    )

