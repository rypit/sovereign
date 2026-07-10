"""The Orchestrator — DAG boot/shutdown + reconciliation loop (§6).

Programs against the ``ServiceManager`` Protocol only, so native processes and
containers are driven identically. Usable as a library: ``sovereign serve`` and
``sovereign bench`` are both consumers of it.

Boot (§6.2): topologically sort the dependency graph, then boot each service —
``prepare_environment`` → (admission, Phase 7) → resolve templates against the
runtime ``ServiceRegistry`` → ``start`` → poll ``is_healthy`` until READY → register
its endpoint. Independent branches run concurrently; dependents wait on their
dependencies' readiness. Once up, the resolved stack manifest is written (§7b).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from collections.abc import Callable, Coroutine, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sovereign.config import HarnessEntry, ServiceEntry, SovereignConfig
from sovereign.core.base_manager import (
    ServiceManager,
    SupportsEndpoint,
    SupportsModelPreparation,
    SupportsResolve,
    SupportsRuntimeHandle,
)
from sovereign.core.registry import route_entry
from sovereign.core.resolver import ConsumerKind, Resolver, ServiceRegistry
from sovereign.core.resources import (
    ResourceBudgeter,
    ResourceExhaustedError,
    estimate_service_memory,
)
from sovereign.core.status import StatusSnapshot
from sovereign.utils.manifest import write_manifest
from sovereign.utils.state import file_hash, write_json

log = logging.getLogger(__name__)

# Default cadences (§6.2/§6.3): 2s health polling, 2s metrics (fresh enough for sparklines).
_HEALTH_INTERVAL = 2.0
_METRICS_INTERVAL = 2.0


class ServiceState(StrEnum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    DOWNLOADING = "downloading"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


class CircularDependencyError(Exception):
    """Raised when the dependency graph contains a cycle."""


class BootError(Exception):
    """Raised when a service fails to boot (bad pre-flight, timeout, crash)."""


ManagerFactory = Callable[[ServiceEntry], ServiceManager]
TransitionHook = Callable[[str, ServiceState, ServiceState], None]


def _service_descriptor(entry: ServiceEntry) -> str | None:
    """What a service is running — the image for docker, the model for a
    native engine — surfaced as-is (untruncated) for the dashboard's MODEL column.
    """
    if entry.base_type == "docker":
        return entry.config.get("image")
    if entry.base_type in ("llama_cpp", "mlx_lm"):
        return entry.config.get("model")
    return None


class Orchestrator:
    """Boots, supervises, and tears down a declared stack."""

    def __init__(
        self,
        config: SovereignConfig,
        *,
        manager_factory: ManagerFactory | None = None,
        harness_factory: Callable[[HarnessEntry], object] | None = None,
        env: Mapping[str, str] | None = None,
        variant_file: str | Path | None = None,
        state_dir: str | Path = ".sovereign",
        health_interval: float = _HEALTH_INTERVAL,
        metrics_interval: float = _METRICS_INTERVAL,
        auto_restart: bool = False,
        on_transition: TransitionHook | None = None,
    ) -> None:
        self.config = config
        self.registry = ServiceRegistry()
        self.resolver = Resolver(self.registry, env)
        self.budgeter = ResourceBudgeter(
            config.resources.max_unified_memory_gb,
            config.resources.safety_margin_gb,
        )

        self.managers: dict[str, ServiceManager] = {}
        self.harnesses: dict[str, object] = {}
        #: name -> "auto" for services whose base_type was routed at build time.
        self.requested_base_types: dict[str, str] = {}
        self.states: dict[str, ServiceState] = {}
        self.state_since: dict[str, str] = {}
        self.metrics: dict[str, dict] = {}

        self._manager_factory = manager_factory or self._default_manager_factory
        self._harness_factory = harness_factory or self._default_harness_factory
        self._entries = {s.name: s for s in config.services}
        self._service_names = [s.name for s in config.services]
        self._boot_order: list[str] = []
        self._built = False
        self._boot_complete = False

        self.variant_file = Path(variant_file) if variant_file else None
        self.variant_hash = file_hash(self.variant_file) if self.variant_file else None
        self.state_dir = Path(state_dir)
        self.health_interval = health_interval
        self.metrics_interval = metrics_interval
        self.auto_restart = auto_restart
        self._on_transition = on_transition

    # --- public accessors (used by the manifest builder) ---
    @property
    def service_names(self) -> list[str]:
        return list(self._service_names)

    @property
    def boot_order(self) -> list[str]:
        return list(self._boot_order) if self._boot_order else list(self._service_names)

    def entry(self, name: str) -> ServiceEntry:
        return self._entries[name]

    # --- factories ---
    def _default_manager_factory(self, entry: ServiceEntry) -> ServiceManager:
        from sovereign.core.registry import get_service_manager, populate_registries

        populate_registries()
        return get_service_manager(entry.base_type)(entry)

    def _default_harness_factory(self, entry: HarnessEntry) -> object:
        from sovereign.core.registry import get_harness, populate_registries

        populate_registries()
        return get_harness(entry.base_type)(entry)

    # --- build + graph ---
    def build(self) -> None:
        """Instantiate managers and set PENDING states (idempotent, safe pre-boot)."""
        self._build()

    def _build(self) -> None:
        if self._built:
            return
        # Resolve `base_type: auto` (or omitted) to a concrete engine via HF metadata
        # before instantiating managers. Routing errors propagate as build failures
        # with actionable messages; the resolved type is written to the routing cache.
        # The caller's config object is never mutated — the orchestrator keeps its
        # own resolved copies in self._entries, so a config can be re-planned later.
        for name in self._service_names:
            entry = self._entries[name]
            if entry.base_type == "auto":
                self.requested_base_types[name] = "auto"
                resolved = route_entry(entry, self.state_dir)
                self._entries[name] = entry.model_copy(update={"base_type": resolved})
        for name in self._service_names:
            self.managers[name] = self._manager_factory(self._entries[name])
            self.states[name] = ServiceState.PENDING
            self.state_since[name] = datetime.now(UTC).isoformat()
        for harness_entry in self.config.harnesses:
            try:
                self.harnesses[harness_entry.name] = self._harness_factory(harness_entry)
            except (KeyError, ImportError):
                pass  # harness base_type not registered yet (harness track)
        self._built = True

    def _service_deps(self, name: str) -> list[str]:
        deps = self._entries[name].dependencies
        for dep in deps:
            if dep not in self._entries:
                raise BootError(
                    f"service '{name}' depends on '{dep}', which is not a service"
                )
        return deps

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; raises :class:`CircularDependencyError` on a cycle."""
        incoming = {n: set(self._service_deps(n)) for n in self._service_names}
        order: list[str] = []
        ready = sorted(n for n, deps in incoming.items() if not deps)
        while ready:
            node = ready.pop(0)
            order.append(node)
            for other in self._service_names:
                if node in incoming[other]:
                    incoming[other].discard(node)
                    if not incoming[other] and other not in order and other not in ready:
                        ready.append(other)
            ready.sort()
        if len(order) != len(self._service_names):
            remaining = [n for n in self._service_names if n not in order]
            raise CircularDependencyError(
                f"dependency cycle among: {', '.join(sorted(remaining))}"
            )
        return order

    # --- state ---
    def _set_state(self, name: str, state: ServiceState) -> None:
        old = self.states.get(name, ServiceState.PENDING)
        self.states[name] = state
        if old is not state:
            log.debug("%s: %s -> %s", name, old, state)
            self.state_since[name] = datetime.now(UTC).isoformat()
            if self._on_transition is not None:
                self._on_transition(name, old, state)

    # --- boot ---
    async def boot(self) -> None:
        self._build()
        self._boot_order = self.topological_order()

        ready_events = {n: asyncio.Event() for n in self._service_names}

        async def run(name: str) -> None:
            for dep in self._service_deps(name):
                await ready_events[dep].wait()
            await self._boot_service(name)
            ready_events[name].set()

        try:
            async with asyncio.TaskGroup() as tg:
                for name in self._service_names:
                    tg.create_task(run(name))
        except BaseExceptionGroup as group:
            # Report every concurrent failure, not just the first branch's.
            leaves = _leaf_exceptions(group)
            raise BootError("; ".join(str(exc) for exc in leaves)) from leaves[0]

        self._materialize_harnesses()
        self._boot_complete = True
        self.persist()

    async def _boot_service(self, name: str) -> None:
        manager = self.managers[name]

        self._set_state(name, ServiceState.PROVISIONING)
        await asyncio.to_thread(manager.prepare_environment)

        # Admission control (§7): refuse-to-boot rather than let macOS swap.
        # The estimate may still hit the network (metadata fetch) if the
        # PROVISIONING prefetch missed, so keep it off the event loop.
        estimated = await asyncio.to_thread(
            estimate_service_memory, manager, self._entries[name]
        )
        try:
            self.budgeter.admit(name, estimated)
            log.debug(
                "admitted %s at %.1f GB (%.1f GB still available)",
                name,
                estimated,
                self.budgeter.available_gb,
            )
        except ResourceExhaustedError:
            self._set_state(name, ServiceState.FAILED)
            raise

        # Pre-download the model (DOWNLOADING) so the server launches from a
        # resolved local path. Cached models return in ms — the state is entered
        # unconditionally when the hook exists for a deterministic machine; managers
        # without the hook (FakeManager, docker) skip it.
        if isinstance(manager, SupportsModelPreparation):
            self._set_state(name, ServiceState.DOWNLOADING)
            try:
                await asyncio.to_thread(manager.prepare_model)
            except Exception:
                self._set_state(name, ServiceState.FAILED)
                raise

        self._resolve_manager(manager)

        self._set_state(name, ServiceState.STARTING)
        await asyncio.to_thread(manager.start)

        await self._wait_healthy(name)
        self._set_state(name, ServiceState.READY)
        self._register_endpoint(name, manager)

    async def _wait_healthy(self, name: str) -> None:
        manager = self.managers[name]
        timeout = self._health_timeout(name)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if await asyncio.to_thread(manager.is_healthy):
                return
            if loop.time() >= deadline:
                self._set_state(name, ServiceState.FAILED)
                raise BootError(f"'{name}' did not become healthy within {timeout:.0f}s")
            await asyncio.sleep(self.health_interval)

    def _health_timeout(self, name: str) -> float:
        health_check = self._entries[name].health_check
        return float(health_check.timeout_seconds) if health_check else 60.0

    def _resolve_manager(self, manager: ServiceManager) -> None:
        if isinstance(manager, SupportsResolve):
            manager.resolve(self.resolver)

    def _register_endpoint(self, name: str, manager: ServiceManager) -> None:
        if isinstance(manager, SupportsEndpoint):
            endpoint = manager.endpoint()
            if endpoint is not None:
                previous = self.registry.get(name) if name in self.registry else None
                self.registry.register(name, endpoint)
                # A restart landing on a new endpoint (e.g. a different port) would
                # otherwise silently strand any harness pointing at the old one.
                if self._boot_complete and previous is not None and previous != endpoint:
                    self._materialize_harnesses()

    def _materialize_harnesses(self) -> None:
        for entry in self.config.harnesses:
            harness = self.harnesses.get(entry.name)
            if harness is None:
                continue
            if all(self.states.get(dep) is ServiceState.READY for dep in entry.dependencies):
                # Provision first, mirroring the service PROVISIONING phase —
                # a declared harness installs what it needs before it's wired.
                # Must be idempotent: re-materialization re-runs this.
                prepare = getattr(harness, "prepare_environment", None)
                if callable(prepare):
                    prepare()
                resolve = getattr(harness, "resolve", None)
                if callable(resolve):
                    resolve(self.resolver)
                materialize = getattr(harness, "materialize", None)
                if callable(materialize):
                    materialize()

    # --- reconciliation (§6.3) ---
    async def reconcile(self, stop: asyncio.Event) -> None:
        await asyncio.gather(self._health_loop(stop), self._metrics_loop(stop))

    async def _health_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            for name in self._service_names:
                if self.states.get(name) is not ServiceState.READY:
                    continue
                healthy = await asyncio.to_thread(self.managers[name].is_healthy)
                if not healthy:
                    self._set_state(name, ServiceState.DEGRADED)
                    if self.auto_restart:
                        await self._restart(name)
            self.write_status()
            await self._sleep_or_stop(stop, self.health_interval)

    async def _metrics_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            for name in self._service_names:
                if self.states.get(name) in (ServiceState.READY, ServiceState.DEGRADED):
                    self.metrics[name] = await asyncio.to_thread(self.managers[name].get_metrics)
            self.write_status()
            await self._sleep_or_stop(stop, self.metrics_interval)

    async def _restart(self, name: str) -> None:
        try:
            await asyncio.to_thread(self.managers[name].stop)
            self.budgeter.release(name)  # free before re-admitting on reboot
            await self._boot_service(name)
        except Exception:  # noqa: BLE001 - a failed restart leaves it DEGRADED/FAILED
            log.warning("restart of %s failed", name, exc_info=True)
            self._set_state(name, ServiceState.FAILED)

    @staticmethod
    async def _sleep_or_stop(stop: asyncio.Event, interval: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass

    # --- shutdown (§6.4) ---
    async def shutdown(self) -> None:
        for name in reversed(self.boot_order):
            manager = self.managers.get(name)
            if manager is None:
                continue
            try:
                await asyncio.to_thread(manager.stop)
            except Exception:  # noqa: BLE001 - keep tearing the rest down
                log.warning("stop of %s failed; continuing teardown", name, exc_info=True)
            self.budgeter.release(name)
            self._set_state(name, ServiceState.STOPPED)
        self.persist()

    # --- persistence (§7b) ---
    def _runtime_handles(self) -> dict[str, dict]:
        handles: dict[str, dict] = {}
        for name, manager in self.managers.items():
            if isinstance(manager, SupportsRuntimeHandle):
                handle = manager.runtime_handle()
                if handle:
                    handles[name] = handle
        return handles

    def persist(self) -> None:
        write_manifest(self, self.state_dir / "manifest.json")
        write_json(
            self.state_dir / "state.json",
            {
                "variant_file": str(self.variant_file) if self.variant_file else None,
                "variant_hash": self.variant_hash,
                "services": {n: str(self.states[n]) for n in self._service_names},
                "runtime": self._runtime_handles(),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        self.write_status()

    def status_snapshot(self) -> StatusSnapshot:
        """Live dashboard snapshot (§8) — the schema `sovereign.dashboard` renders."""
        reservations = self.budgeter.reservations()
        return {
            "updated_at": datetime.now(UTC).isoformat(),
            "budget": {
                "usable_gb": self.budgeter.usable_gb,
                "reserved_gb": round(self.budgeter.reserved_gb, 2),
                "available_gb": round(self.budgeter.available_gb, 2),
            },
            "services": {
                name: {
                    "state": str(self.states.get(name)),
                    "since": self.state_since.get(name),
                    "endpoint": (
                        self.registry.get(name).url_for(ConsumerKind.NATIVE)
                        if name in self.registry
                        and self.states.get(name) in (ServiceState.READY, ServiceState.DEGRADED)
                        else None
                    ),
                    "descriptor": _service_descriptor(self._entries[name]),
                    "estimated_gb": reservations.get(name),
                    "metrics": self.metrics.get(name, {}),
                    "activity": {
                        "lines": list(getattr(self.managers.get(name), "activity", ()) or ())
                    },
                }
                for name in self._service_names
            },
        }

    def write_status(self) -> None:
        """Persist the snapshot for a separate `sovereign monitor` process to read."""
        write_json(self.state_dir / "status.json", self.status_snapshot())


def _leaf_exceptions(group: BaseExceptionGroup) -> list[BaseException]:
    """Flatten a (possibly nested) exception group into its leaf exceptions."""
    leaves: list[BaseException] = []
    for exc in group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            leaves.extend(_leaf_exceptions(exc))
        else:
            leaves.append(exc)
    return leaves or [group]


ExtraTask = Callable[["Orchestrator", asyncio.Event], "Coroutine"]


async def serve_forever(
    config: SovereignConfig,
    *,
    variant_file: str | Path | None = None,
    state_dir: str | Path = ".sovereign",
    extra_tasks: Sequence[ExtraTask] = (),
    on_transition: TransitionHook | None = None,
) -> Orchestrator:
    """Boot the stack, run reconciliation (plus any extra tasks) until SIGINT/SIGTERM.

    ``extra_tasks`` are ``async (orch, stop) -> None`` coroutines run concurrently —
    including *during boot*, so a foreground dashboard can show services progressing
    through ``PENDING → PROVISIONING → STARTING → READY``. They share the ``stop``
    event, so a signal (even mid-boot) tears everything down together.
    """
    orch = Orchestrator(
        config, variant_file=variant_file, state_dir=state_dir, on_transition=on_transition
    )
    orch.build()  # populate PENDING states so watchers see the full list immediately

    stop = asyncio.Event()
    interrupts = 0

    def _request_stop() -> None:  # pragma: no cover - interactive
        # First signal begins graceful shutdown; a second forces an immediate
        # exit, skipping the join of an un-cancellable in-flight download thread
        # (huggingface_hub network I/O ignores asyncio cancellation).
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            stop.set()
        else:
            os._exit(130)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:  # pragma: no cover - non-Unix
            pass

    background = [asyncio.create_task(task(orch, stop)) for task in extra_tasks]
    try:
        boot_task = asyncio.create_task(orch.boot())
        stop_task = asyncio.create_task(stop.wait())
        await asyncio.wait({boot_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if boot_task.done():
            stop_task.cancel()
            await boot_task  # re-raise BootError if boot failed
            await orch.reconcile(stop)
        else:  # Ctrl+C arrived mid-boot
            boot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await boot_task
    finally:
        stop.set()
        for task in background:
            with contextlib.suppress(Exception):
                await task
        await orch.shutdown()
    return orch
