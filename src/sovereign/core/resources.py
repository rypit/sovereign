"""Resource & memory management by proxy (§7).

Apple Silicon's unified memory can't be hard-partitioned like VRAM, so Sovereign
manages it *by proxy*: translate a declared budget into admission control that
refuses to boot a service that would overcommit — rather than letting macOS swap.

Eviction policy (§7, §11.5): **refuse-to-boot with a clear message; never auto-kill
a running service.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sovereign.config import Priority
from sovereign.core.base_manager import SupportsEstimateSource, SupportsMemoryEstimate
from sovereign.core.units import fmt_size

if TYPE_CHECKING:
    from sovereign.config import ServiceEntry
    from sovereign.core.base_manager import ServiceManager

# priority -> os.nice() value. Only non-negative (deprioritise) values are used,
# since raising priority (negative nice) needs privileges a LaunchAgent lacks (§9).
_NICE_BY_PRIORITY = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 0,
    Priority.MEDIUM: 5,
    Priority.LOW: 10,
}


class ResourceExhaustedError(Exception):
    """Raised when admitting a service would exceed the memory budget."""


class ResourceBudgeter:
    """Admission control over a declared unified-memory budget (int bytes)."""

    def __init__(self, total_bytes: int, safety_margin_bytes: int = 0) -> None:
        self.total_bytes = total_bytes
        self.safety_margin_bytes = safety_margin_bytes
        self._reserved: dict[str, int] = {}

    @property
    def reserved_bytes(self) -> int:
        return sum(self._reserved.values())

    @property
    def usable_bytes(self) -> int:
        return self.total_bytes - self.safety_margin_bytes

    @property
    def available_bytes(self) -> int:
        return self.usable_bytes - self.reserved_bytes

    def can_fit(self, estimated_bytes: int) -> bool:
        return estimated_bytes <= self.available_bytes

    def admit(self, name: str, estimated_bytes: int) -> None:
        """Reserve budget for ``name`` or raise a specific, actionable error."""
        if not self.can_fit(estimated_bytes):
            raise ResourceExhaustedError(self._denial_message(name, estimated_bytes))
        self._reserved[name] = estimated_bytes

    def release(self, name: str) -> None:
        self._reserved.pop(name, None)

    def reservations(self) -> dict[str, int]:
        return dict(self._reserved)

    def _denial_message(self, name: str, needed: int) -> str:
        if self._reserved:
            biggest = sorted(self._reserved.items(), key=lambda kv: kv[1], reverse=True)
            suggestions = ", ".join(f"{n} (~{fmt_size(b)})" for n, b in biggest)
            free_hint = f"Free memory by stopping: {suggestions}"
        else:
            free_hint = (
                "Nothing else is reserved — lower this service's needs or raise "
                "max_unified_memory_gb"
            )
        return (
            f"Cannot start '{name}': needs ~{fmt_size(needed)}, "
            f"only {fmt_size(self.available_bytes)} "
            f"available (budget {fmt_size(self.total_bytes)} "
            f"- {fmt_size(self.safety_margin_bytes)} safety "
            f"- {fmt_size(self.reserved_bytes)} in use). {free_hint}."
        )


def estimate_service_memory(manager: ServiceManager, entry: ServiceEntry) -> int:
    """Best-effort memory estimate (int bytes) for admission control.

    Prefers the manager's own ``estimated_memory_bytes()`` (e.g. llama_cpp sizes
    from the model file + KV cache); falls back to a declared ``config.memory_gb``
    hint (parsed as ``entry.memory_bytes``); otherwise 0 (unknown → admitted, so
    we only ever refuse on real estimates).
    """
    if isinstance(manager, SupportsMemoryEstimate):
        return manager.estimated_memory_bytes()
    if entry.memory_bytes is not None:
        return entry.memory_bytes
    return 0


def estimate_source(manager: ServiceManager, entry: ServiceEntry) -> str:
    """Label where the admission estimate came from: declared|local|cached|hub|unknown.

    The single labelling rule `sovereign plan` (SOURCE column) and boot (the
    fail-open warning) share — "unknown" means the deliberate unknown->admit
    policy applied: the service was admitted without counting against the budget.
    """
    if entry.memory_bytes is not None:
        return "declared"
    if isinstance(manager, SupportsEstimateSource):
        return manager.estimated_memory_source()
    return "unknown"


def priority_to_nice(priority: Priority | None) -> int:
    """Map a service priority to an ``os.nice()`` value (defaults to MEDIUM)."""
    return _NICE_BY_PRIORITY.get(priority or Priority.MEDIUM, 5)
