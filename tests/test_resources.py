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
    b = ResourceBudgeter(total_bytes=128 * 10**9, safety_margin_bytes=8 * 10**9)
    assert b.usable_bytes == 120 * 10**9
    assert b.available_bytes == 120 * 10**9
    assert b.reserved_bytes == 0


def test_admit_reserves_and_reduces_available() -> None:
    b = ResourceBudgeter(64 * 10**9, 4 * 10**9)
    b.admit("llama", 40 * 10**9)
    assert b.reserved_bytes == 40 * 10**9
    assert b.available_bytes == 20 * 10**9
    assert b.can_fit(20 * 10**9) is True
    assert b.can_fit(21 * 10**9) is False


def test_admit_over_budget_raises_actionable_error() -> None:
    b = ResourceBudgeter(64 * 10**9, 8 * 10**9)
    b.admit("comfyui", 25 * 10**9)
    with pytest.raises(ResourceExhaustedError) as exc:
        b.admit("llama_heavy", 40 * 10**9)
    msg = str(exc.value)
    assert "Cannot start 'llama_heavy'" in msg
    assert "needs ~40.0 GB" in msg
    assert "comfyui (~25.0 GB)" in msg  # suggests what to stop
    # nothing was reserved for the refused service
    assert "llama_heavy" not in b.reservations()


def test_release_frees_budget() -> None:
    b = ResourceBudgeter(64 * 10**9, 4 * 10**9)
    b.admit("a", 50 * 10**9)
    assert b.can_fit(10 * 10**9) is True
    assert b.can_fit(11 * 10**9) is False
    b.release("a")
    assert b.available_bytes == 60 * 10**9
    assert b.can_fit(60 * 10**9) is True


def test_denial_message_when_nothing_reserved() -> None:
    b = ResourceBudgeter(16 * 10**9, 4 * 10**9)
    with pytest.raises(ResourceExhaustedError, match="raise max_unified_memory_gb"):
        b.admit("big", 40 * 10**9)


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

    def adjust_resources(self, memory_limit_bytes: int) -> None: ...


class _EstimatingManager(_ManagerStub):
    def __init__(self, memory_bytes: int):
        super().__init__()
        self._memory_bytes = memory_bytes

    def estimated_memory_bytes(self) -> int:
        return self._memory_bytes


def _entry(memory_gb: float | None = None) -> ServiceEntry:
    return ServiceEntry(name="s", base_type="x", memory_gb=memory_gb)


def test_estimate_prefers_manager_method() -> None:
    assert estimate_service_memory(_EstimatingManager(12_500_000_000), _entry()) == 12_500_000_000


def test_estimate_falls_back_to_entry_hint() -> None:
    assert estimate_service_memory(_ManagerStub(), _entry(memory_gb=7)) == 7 * 10**9


def test_estimate_defaults_to_zero_when_unknown() -> None:
    assert estimate_service_memory(_ManagerStub(), _entry()) == 0


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
