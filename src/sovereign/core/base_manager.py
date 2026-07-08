"""The core contract for supervised, long-running things: ``ServiceManager``.

Every engine and every container implements this Protocol. It is the single
interface the Orchestrator programs against, so native subprocesses and Docker
containers look identical from its point of view (§4 / §2.5).

Harnesses and Jobs do **not** implement this — they have their own contracts
(:mod:`sovereign.core.base_harness`, and the bench ``Job`` type, respectively).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sovereign.config import ServiceEntry
    from sovereign.core.resolver import ResolvedEndpoint, Resolver


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

    def __init__(self, entry: ServiceEntry) -> None:
        """Managers are constructed from their service entry (the registry's contract)."""

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


# ---------------------------------------------------------------------------
# Optional capabilities
#
# Not every manager implements every hook: docker containers have no model to
# download, native engines have no `docker run` argv. The Orchestrator, the
# resource budgeter, and the manifest builder discover these capabilities via
# `isinstance()` against the runtime-checkable Protocols below — never via ad-hoc
# `getattr` probing — so the full manager contract is visible in one place.
#
# A manager opts into a capability simply by defining the method. Additionally,
# managers may expose these *data* attributes (which Protocols can't
# runtime-check on Python 3.11), read via `getattr` where needed:
#
# - ``model_path: Path | None`` — the resolved local model artifact, populated
#   by ``prepare_model()``; the manifest fingerprints it.
# - ``resolved_env: dict[str, Any]`` — endpoint-resolved environment for
#   container managers, recorded alongside ``run_args()`` in the manifest.
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsModelPreparation(Protocol):
    """Downloads/resolves model artifacts before ``start()`` (DOWNLOADING state)."""

    def prepare_model(self) -> None:
        """Resolve the model to a local path, downloading into the HF cache if needed."""
        ...


@runtime_checkable
class SupportsMemoryEstimate(Protocol):
    """Estimates resident memory for admission control (§7 refuse-to-boot)."""

    def estimated_memory_gb(self) -> float:
        """Expected unified-memory footprint in GB (0.0 when unknown)."""
        ...


@runtime_checkable
class SupportsResolve(Protocol):
    """Consumes dependency endpoints (``{{ service.url }}`` templates) before start."""

    def resolve(self, resolver: Resolver) -> None: ...


@runtime_checkable
class SupportsEndpoint(Protocol):
    """Exposes a network endpoint registered for dependents when READY."""

    def endpoint(self) -> ResolvedEndpoint | None: ...


@runtime_checkable
class SupportsRuntimeHandle(Protocol):
    """Provides a cross-process teardown handle (PID / container name) for `down`."""

    def runtime_handle(self) -> dict[str, Any] | None: ...


@runtime_checkable
class SupportsStartArgs(Protocol):
    """Native engines: the final resolved argv, recorded in the manifest."""

    def get_start_args(self) -> list[str]: ...


@runtime_checkable
class SupportsRunArgs(Protocol):
    """Container managers: the final ``docker run`` argv, recorded in the manifest."""

    def run_args(self) -> list[str]: ...


@runtime_checkable
class SupportsPerSlotContext(Protocol):
    """Engines with parallel slots: context window available per agent."""

    def per_slot_context(self) -> int | None: ...
