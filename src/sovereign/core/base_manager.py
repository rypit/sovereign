"""The core contract for supervised, long-running things: ``ServiceManager``.

Every engine and every container implements this Protocol. It is the single
interface the Orchestrator programs against, so native subprocesses and Docker
containers look identical from its point of view (§4 / §2.5).

Harnesses and Jobs do **not** implement this — they have their own contracts
(:mod:`sovereign.core.base_harness`, and the bench ``Job`` type, respectively).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ActivityMixin:
    """Standard progress reporting shared by every manager.

    ``activity`` is a short, human-readable line describing what a service is
    currently doing — a docker pull's layer count, a model load, a cache
    warm-up. The Orchestrator surfaces it in the live dashboard and status
    snapshot, so all services report progress the same way. Managers call
    ``set_activity()`` during long operations and ``clear_activity()`` when idle.
    """

    activity: str = ""

    def set_activity(self, message: str) -> None:
        self.activity = message

    def clear_activity(self) -> None:
        self.activity = ""


@runtime_checkable
class ServiceManager(Protocol):
    """Contract for a supervised, run-forever service.

    ``@runtime_checkable`` lets the Orchestrator ``isinstance(manager,
    ServiceManager)`` at registration time, so a malformed integration fails
    loudly before ``start()`` is ever called rather than mid-boot.

    Managers get ``activity`` for free by inheriting :class:`ActivityMixin`.
    """

    #: Unique instance ID (e.g. ``"llama_heavy_v1"``).
    name: str
    #: Names of services that must be ``READY`` before this one starts.
    dependencies: list[str]
    #: Human-readable current-activity line (see :class:`ActivityMixin`).
    activity: str

    # --- Lifecycle ---
    def start(self) -> None:
        """Spawn the process / container. Returns once launched, not once ready."""
        ...

    def stop(self) -> None:
        """Terminate gracefully (SIGTERM for native processes, so caches flush)."""
        ...

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        """Whether the service currently passes its configured health check."""
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Point-in-time resource metrics (memory, cpu, status) for the dashboard."""
        ...

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        """Pre-flight hook run before ``start()``.

        Validates preconditions (model file exists, cache dir writable, disk
        space) so failures surface as a clean error instead of a half-booted
        process.
        """
        ...

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """Shrink resource use in response to pressure (e.g. reduce cache size)."""
        ...
