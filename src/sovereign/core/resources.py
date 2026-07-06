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
    """Admission control over a declared unified-memory budget."""

    def __init__(self, total_gb: float, safety_margin_gb: float = 0.0) -> None:
        self.total_gb = float(total_gb)
        self.safety_margin_gb = float(safety_margin_gb)
        self._reserved: dict[str, float] = {}

    @property
    def reserved_gb(self) -> float:
        return sum(self._reserved.values())

    @property
    def usable_gb(self) -> float:
        return self.total_gb - self.safety_margin_gb

    @property
    def available_gb(self) -> float:
        return self.usable_gb - self.reserved_gb

    def can_fit(self, estimated_gb: float) -> bool:
        return estimated_gb <= self.available_gb + 1e-9

    def admit(self, name: str, estimated_gb: float) -> None:
        """Reserve budget for ``name`` or raise a specific, actionable error."""
        if not self.can_fit(estimated_gb):
            raise ResourceExhaustedError(self._denial_message(name, estimated_gb))
        self._reserved[name] = estimated_gb

    def release(self, name: str) -> None:
        self._reserved.pop(name, None)

    def reservations(self) -> dict[str, float]:
        return dict(self._reserved)

    def _denial_message(self, name: str, needed: float) -> str:
        if self._reserved:
            biggest = sorted(self._reserved.items(), key=lambda kv: kv[1], reverse=True)
            suggestions = ", ".join(f"{n} (~{gb:.1f}GB)" for n, gb in biggest)
            free_hint = f"Free memory by stopping: {suggestions}"
        else:
            free_hint = (
                "Nothing else is reserved — lower this service's needs or raise "
                "max_unified_memory_gb"
            )
        return (
            f"Cannot start '{name}': needs ~{needed:.1f}GB, only {self.available_gb:.1f}GB "
            f"available (budget {self.total_gb:.0f}GB - {self.safety_margin_gb:.0f}GB safety "
            f"- {self.reserved_gb:.1f}GB in use). {free_hint}."
        )


def estimate_service_memory(manager: ServiceManager, entry: ServiceEntry) -> float:
    """Best-effort memory estimate (GB) for admission control.

    Prefers the manager's own ``estimated_memory_gb()`` (e.g. llama_cpp sizes from
    the model file + KV cache); falls back to a declared ``config.memory_gb`` hint;
    otherwise 0.0 (unknown → admitted, so we only ever refuse on real estimates).
    """
    estimate_fn = getattr(manager, "estimated_memory_gb", None)
    if callable(estimate_fn):
        return float(estimate_fn())
    if entry.memory_gb is not None:
        return float(entry.memory_gb)
    return 0.0


def priority_to_nice(priority: Priority | None) -> int:
    """Map a service priority to an ``os.nice()`` value (defaults to MEDIUM)."""
    return _NICE_BY_PRIORITY.get(priority or Priority.MEDIUM, 5)
