"""Phase 1 exit: trivial fakes satisfy the runtime-checkable Protocols."""

from __future__ import annotations

from typing import Any

from sovereign.core.base_harness import Harness, RunResult, SupportsInvoke, Task
from sovereign.core.base_manager import ServiceManager
from sovereign.core.resolver import Resolver


class FakeManager:
    def __init__(self) -> None:
        self.name = "fake"
        self.dependencies: list[str] = []
        self.activity: tuple[str, ...] = ()

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_healthy(self) -> bool:
        return True

    def get_metrics(self) -> dict[str, Any]:
        return {"status": "running"}

    def prepare_environment(self) -> None: ...
    def adjust_resources(self, memory_limit_mb: int) -> None: ...


class FakeHarness:
    def __init__(self) -> None:
        self.name = "fake_harness"
        self.dependencies: list[str] = []

    def resolve(self, resolver: Resolver) -> None: ...

    def prepare_environment(self) -> None: ...

    def materialize(self) -> None: ...

    def invoke(self, task: Task) -> RunResult:
        return RunResult(task_id=task.id, success=True, exit_code=0)


def test_fake_manager_satisfies_service_manager() -> None:
    assert isinstance(FakeManager(), ServiceManager)


def test_fake_harness_satisfies_harness() -> None:
    assert isinstance(FakeHarness(), Harness)


def test_incomplete_manager_is_rejected() -> None:
    class Incomplete:
        name = "x"
        dependencies: list[str] = []

        def start(self) -> None: ...

    assert not isinstance(Incomplete(), ServiceManager)


def test_harness_invoke_roundtrip() -> None:
    result = FakeHarness().invoke(Task(id="t1", prompt="do the thing"))
    assert result.task_id == "t1"
    assert result.success is True


def test_incomplete_harness_is_rejected() -> None:
    class Incomplete:
        name = "x"
        dependencies: list[str] = []

        def materialize(self) -> None: ...  # no resolve / prepare_environment

    assert not isinstance(Incomplete(), Harness)


def test_fake_harness_satisfies_supports_invoke() -> None:
    assert isinstance(FakeHarness(), SupportsInvoke)
