"""Phase 7: ResourceBudgeter admission control + estimation helpers."""

from __future__ import annotations

import pytest

from sovereign.config import Priority, ServiceEntry
from sovereign.core.resources import (
    ResourceBudgeter,
    ResourceExhaustedError,
    estimate_service_memory,
    priority_to_nice,
)


def test_available_accounts_for_safety_margin() -> None:
    b = ResourceBudgeter(total_gb=128, safety_margin_gb=8)
    assert b.usable_gb == 120
    assert b.available_gb == 120
    assert b.reserved_gb == 0


def test_admit_reserves_and_reduces_available() -> None:
    b = ResourceBudgeter(64, 4)
    b.admit("llama", 40)
    assert b.reserved_gb == 40
    assert b.available_gb == 20
    assert b.can_fit(20) is True
    assert b.can_fit(21) is False


def test_admit_over_budget_raises_actionable_error() -> None:
    b = ResourceBudgeter(64, 8)
    b.admit("comfyui", 25)
    with pytest.raises(ResourceExhaustedError) as exc:
        b.admit("llama_heavy", 40)
    msg = str(exc.value)
    assert "Cannot start 'llama_heavy'" in msg
    assert "needs ~40.0GB" in msg
    assert "comfyui (~25.0GB)" in msg  # suggests what to stop
    # nothing was reserved for the refused service
    assert "llama_heavy" not in b.reservations()


def test_release_frees_budget() -> None:
    b = ResourceBudgeter(64, 4)
    b.admit("a", 50)
    assert b.can_fit(10) is True
    assert b.can_fit(11) is False
    b.release("a")
    assert b.available_gb == 60
    assert b.can_fit(60) is True


def test_denial_message_when_nothing_reserved() -> None:
    b = ResourceBudgeter(16, 4)
    with pytest.raises(ResourceExhaustedError, match="raise max_unified_memory_gb"):
        b.admit("big", 40)


# --- estimation ---
class _ManagerStub:
    """Minimal but *complete* ServiceManager, so the fakes here actually
    conform to the Protocol estimate_service_memory() is typed against."""

    def __init__(self) -> None:
        self.name = "s"
        self.dependencies: list[str] = []
        self.activity: tuple[str, ...] = ()

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_healthy(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {}

    def prepare_environment(self) -> None: ...

    def adjust_resources(self, memory_limit_mb: int) -> None: ...


class _EstimatingManager(_ManagerStub):
    def __init__(self, gb: float):
        super().__init__()
        self._gb = gb

    def estimated_memory_gb(self) -> float:
        return self._gb


def _entry(memory_gb: float | None = None) -> ServiceEntry:
    return ServiceEntry(name="s", base_type="x", memory_gb=memory_gb)


def test_estimate_prefers_manager_method() -> None:
    assert estimate_service_memory(_EstimatingManager(12.5), _entry()) == 12.5


def test_estimate_falls_back_to_entry_hint() -> None:
    assert estimate_service_memory(_ManagerStub(), _entry(memory_gb=7)) == 7.0


def test_estimate_defaults_to_zero_when_unknown() -> None:
    assert estimate_service_memory(_ManagerStub(), _entry()) == 0.0


@pytest.mark.parametrize(
    ("priority", "expected"),
    [
        (Priority.CRITICAL, 0),
        (Priority.HIGH, 0),
        (Priority.MEDIUM, 5),
        (Priority.LOW, 10),
        (None, 5),
    ],
)
def test_priority_to_nice(priority, expected) -> None:
    assert priority_to_nice(priority) == expected
