"""Dry-run stack planning — the boot path's routing + admission, without starting anything.

``sovereign plan`` and ``Orchestrator.boot()`` must never disagree, so this module
reuses the exact seams boot does: :func:`sovereign.hf.resolve_entry_base_type` for
engine routing and :func:`sovereign.core.resources.estimate_service_memory` (through
the real manager class) for the memory number. Constructing the manager also
validates the service's ``config:`` block, so typos surface at dry-run time instead
of at boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from sovereign import hf
from sovereign.config import ServiceEntry, SovereignConfig
from sovereign.core.registry import get_service_manager, populate_registries
from sovereign.core.resources import (
    ResourceBudgeter,
    ResourceExhaustedError,
    estimate_service_memory,
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
    estimated_gb: float | None  #: the admitted number; None when nothing is known
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
        config.resources.max_unified_memory_gb, config.resources.safety_margin_gb
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
        base_type = hf.resolve_entry_base_type(entry, state_dir)
    except hf.ModelResolutionError as exc:
        return ServicePlan(
            name=entry.name,
            base_type=entry.base_type,
            requested_auto=requested_auto,
            model=model,
            source="-",
            estimated_gb=None,
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
            estimated_gb=None,
            verdict=VERDICT_CONFIG_ERROR,
            error=str(exc),
        )

    estimated = estimate_service_memory(manager, entry)
    source = _estimate_source(entry, manager_cls, model)
    display_gb: float | None = estimated if (estimated or source != "unknown") else None

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
            estimated_gb=display_gb,
            verdict=VERDICT_REFUSED,
            error=str(exc),
        )

    return ServicePlan(
        name=entry.name,
        base_type=base_type,
        requested_auto=requested_auto,
        model=model,
        source=source,
        estimated_gb=display_gb,
        verdict=VERDICT_OK,
    )


def _estimate_source(entry: ServiceEntry, manager_cls: type, model: str) -> str:
    """Label where the memory number came from, for the plan table's SOURCE column."""
    if entry.memory_gb is not None:
        return "declared"
    kind = getattr(manager_cls, "model_artifact_kind", None)
    if kind is not None and model != "-":
        _, source = hf.estimate_model_bytes_with_source(hf.parse_model_ref(model), kind)
        return source
    return "unknown"
