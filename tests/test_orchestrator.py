"""Phase 6: Orchestrator — topo boot, concurrency, reconciliation, shutdown."""

from __future__ import annotations

import asyncio
import json
import time
from typing import cast

import pytest

from sovereign.config import ServiceEntry, SovereignConfig
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.runtime.orchestrator import (
    BootError,
    CircularDependencyError,
    Orchestrator,
    ServiceState,
)


class FakeManager:
    """Records lifecycle calls into a shared log for ordering assertions."""

    consumer_kind = ConsumerKind.NATIVE

    def __init__(
        self,
        entry: ServiceEntry,
        log: list,
        *,
        healthy: bool = True,
        port: int = 0,
        mem_bytes: int = 0,
        prepare_delay: float = 0.0,
        has_prepare_model: bool = True,
        prepare_model_raises: bool = False,
        estimate_source: str | None = None,
    ):
        self.name = entry.name
        self.dependencies = entry.dependencies
        self._log = log
        self._healthy = healthy
        self._port = port
        self._mem_bytes = mem_bytes
        self._prepare_delay = prepare_delay
        self._prepare_model_raises = prepare_model_raises
        self.resolved_with: object | None = None
        self.activity: tuple[str, ...] = ()
        # A native engine exposes prepare_model (pre-download); managers without
        # the capability (docker, older fakes) simply lack the attribute,
        # which is what the SupportsModelPreparation isinstance check keys on.
        if has_prepare_model:
            self.prepare_model = self._prepare_model
        # Likewise, only managers that opt in report where their estimate came
        # from — the SupportsEstimateSource isinstance check keys on presence.
        if estimate_source is not None:
            self.estimated_memory_source = lambda: estimate_source

    def estimated_memory_bytes(self) -> int:
        return self._mem_bytes

    def prepare_environment(self) -> None:
        self._log.append((self.name, "prepare"))
        if self._prepare_delay:
            time.sleep(self._prepare_delay)  # runs in a worker thread during boot

    def _prepare_model(self) -> None:
        self._log.append((self.name, "prepare_model"))
        if self._prepare_model_raises:
            raise RuntimeError(f"download failed for {self.name}")

    def start(self) -> None:
        self._log.append((self.name, "start"))

    def stop(self) -> None:
        self._log.append((self.name, "stop"))

    def is_healthy(self) -> bool:
        return self._healthy

    def get_metrics(self) -> dict:
        return {"status": "running", "name": self.name}

    def prepare(self) -> None:  # unused
        ...

    def adjust_resources(self, memory_limit_bytes: int) -> None: ...

    def resolve(self, resolver) -> None:
        self.resolved_with = resolver

    def endpoint(self):
        return ResolvedEndpoint("http", "127.0.0.1", self._port) if self._port else None

    def runtime_handle(self):
        return {"kind": "native", "pid": 4242}


class FakeHarness:
    """Records prepare/materialize/resolve calls into a shared log for ordering."""

    def __init__(self, entry, log: list):
        self.name = entry.name
        self.dependencies = entry.dependencies
        self._log = log
        self.resolved_with = None
        self.prepare_count = 0
        self.materialize_count = 0

    def prepare_environment(self) -> None:
        self.prepare_count += 1
        self._log.append((self.name, "prepare_environment"))

    def resolve(self, resolver) -> None:
        self.resolved_with = resolver

    def materialize(self) -> None:
        self.materialize_count += 1
        self._log.append((self.name, "materialize"))


def _fm(orch: Orchestrator, name: str) -> FakeManager:
    """Narrow back to the concrete fake so tests can reach its recording internals
    (the orchestrator's public dict is typed against the ``ServiceManager`` Protocol)."""
    return cast(FakeManager, orch.managers[name])


def _fh(orch: Orchestrator, name: str) -> FakeHarness:
    return cast(FakeHarness, orch.harnesses[name])


def _config(
    services: list[dict],
    version: str = "1.1",
    resources: dict | None = None,
    harnesses: list[dict] | None = None,
) -> SovereignConfig:
    return SovereignConfig.model_validate(
        {
            "version": version,
            "resources": resources or {"max_unified_memory_gb": 64, "safety_margin_gb": 4},
            "services": services,
            "harnesses": harnesses or [],
        }
    )


def _orch(
    config: SovereignConfig, log: list | None = None, *, harness_log: list | None = None, **kwargs
) -> Orchestrator:
    log = log if log is not None else []
    healthy = kwargs.pop("healthy", True)
    ports = kwargs.pop("ports", {})
    mems = kwargs.pop("mems", {})
    prepare_delays = kwargs.pop("prepare_delays", {})
    no_prepare_model = kwargs.pop("no_prepare_model", set())
    prepare_model_raises = kwargs.pop("prepare_model_raises", set())
    estimate_sources = kwargs.pop("estimate_sources", {})

    def factory(entry: ServiceEntry) -> FakeManager:
        return FakeManager(
            entry,
            log,
            healthy=healthy,
            port=ports.get(entry.name, 0),
            mem_bytes=mems.get(entry.name, 0),
            prepare_delay=prepare_delays.get(entry.name, 0.0),
            has_prepare_model=entry.name not in no_prepare_model,
            prepare_model_raises=entry.name in prepare_model_raises,
            estimate_source=estimate_sources.get(entry.name),
        )

    if harness_log is not None and "harness_factory" not in kwargs:
        kwargs["harness_factory"] = lambda entry: FakeHarness(entry, harness_log)

    kwargs.setdefault("health_interval", 0.01)
    kwargs.setdefault("metrics_interval", 0.01)
    return Orchestrator(config, manager_factory=factory, **kwargs)


# --- topological sort ---
def test_topological_order_linear_chain() -> None:
    cfg = _config(
        [
            {"name": "c", "base_type": "x", "dependencies": ["b"]},
            {"name": "b", "base_type": "x", "dependencies": ["a"]},
            {"name": "a", "base_type": "x"},
        ]
    )
    orch = _orch(cfg)
    orch._build()
    assert orch.topological_order() == ["a", "b", "c"]


def test_cycle_raises() -> None:
    cfg = _config(
        [
            {"name": "a", "base_type": "x", "dependencies": ["b"]},
            {"name": "b", "base_type": "x", "dependencies": ["a"]},
        ]
    )
    orch = _orch(cfg)
    orch._build()
    with pytest.raises(CircularDependencyError, match="cycle"):
        orch.topological_order()


# --- boot ordering & concurrency ---
def test_boot_respects_dependency_order() -> None:
    log: list = []
    cfg = _config(
        [
            {"name": "engine", "base_type": "x"},
            {"name": "frontend", "base_type": "x", "dependencies": ["engine"]},
        ]
    )
    orch = _orch(cfg, log)
    asyncio.run(orch.boot())
    # frontend must not start before engine is ready (started).
    assert log.index(("engine", "start")) < log.index(("frontend", "prepare"))
    assert orch.states == {"engine": ServiceState.READY, "frontend": ServiceState.READY}


def test_boot_waits_for_all_dependencies() -> None:
    log: list = []
    cfg = _config(
        [
            {"name": "a", "base_type": "x"},
            {"name": "b", "base_type": "x"},
            {"name": "c", "base_type": "x", "dependencies": ["a", "b"]},
        ]
    )
    asyncio.run(_orch(cfg, log).boot())
    c_prepare = log.index(("c", "prepare"))
    assert log.index(("a", "start")) < c_prepare
    assert log.index(("b", "start")) < c_prepare


def test_endpoint_registered_when_ready() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg, ports={"engine": 11435})
    asyncio.run(orch.boot())
    assert "engine" in orch.registry
    assert orch.registry.get("engine").port == 11435


def test_resolve_called_during_boot() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg)
    asyncio.run(orch.boot())
    assert _fm(orch, "engine").resolved_with is orch.resolver


def test_unhealthy_service_fails_boot() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg, healthy=False, health_interval=0.01)
    # give a tiny timeout via health_check
    orch._entries["engine"].health_check = None  # default 60s is too long; patch timeout
    orch._health_timeout = lambda name: 0.05  # type: ignore[method-assign]
    with pytest.raises(BootError, match="did not become healthy"):
        asyncio.run(orch.boot())
    assert orch.states["engine"] is ServiceState.FAILED


# --- DOWNLOADING state (pre-download) ---
def _transition_recorder(name_filter: str | None = None):
    seen: list = []

    def hook(name, old, new) -> None:
        if name_filter is None or name == name_filter:
            seen.append(new)

    return seen, hook


def test_boot_transitions_through_downloading() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    seen, hook = _transition_recorder("engine")
    orch = _orch(cfg, on_transition=hook)
    asyncio.run(orch.boot())
    assert seen == [
        ServiceState.PROVISIONING,
        ServiceState.DOWNLOADING,
        ServiceState.STARTING,
        ServiceState.READY,
    ]


def test_prepare_model_called_between_provision_and_start() -> None:
    log: list = []
    cfg = _config([{"name": "engine", "base_type": "x"}])
    asyncio.run(_orch(cfg, log).boot())
    assert log.index(("engine", "prepare")) < log.index(("engine", "prepare_model"))
    assert log.index(("engine", "prepare_model")) < log.index(("engine", "start"))


def test_download_failure_marks_failed() -> None:
    log: list = []
    cfg = _config([{"name": "engine", "base_type": "x"}])
    seen, hook = _transition_recorder("engine")
    orch = _orch(cfg, log, on_transition=hook, prepare_model_raises={"engine"})
    with pytest.raises(BootError, match="download failed"):
        asyncio.run(orch.boot())
    assert orch.states["engine"] is ServiceState.FAILED
    assert ServiceState.DOWNLOADING in seen
    assert ("engine", "start") not in log  # never launched


def test_manager_without_prepare_model_skips_downloading() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    seen, hook = _transition_recorder("engine")
    orch = _orch(cfg, on_transition=hook, no_prepare_model={"engine"})
    asyncio.run(orch.boot())
    assert ServiceState.DOWNLOADING not in seen
    assert orch.states["engine"] is ServiceState.READY


# --- auto base_type routing (M4) ---
def test_build_routes_auto_base_type(monkeypatch) -> None:
    import sovereign.runtime.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "route_entry", lambda entry, state_dir: "mlx_lm")
    cfg = _config([{"name": "engine", "base_type": "auto", "config": {"model": "org/m"}}])
    seen: list[str] = []
    orch = _orch(cfg)
    # Capture the base_type the manager factory actually receives.
    orig = orch._manager_factory

    def factory(entry):
        seen.append(entry.base_type)
        return orig(entry)

    orch._manager_factory = factory
    orch.build()
    assert seen == ["mlx_lm"]  # resolved before instantiation
    assert orch.requested_base_types == {"engine": "auto"}


def test_build_auto_routing_uses_cache_offline(tmp_path, monkeypatch) -> None:
    from sovereign.services.inference import hf as models_mod
    from sovereign.services.inference.hf import RoutingCache

    RoutingCache(tmp_path / "models.json").put(
        "org/m", base_type="llama_cpp", weight_bytes=None
    )
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)  # offline
    cfg = _config([{"name": "engine", "base_type": "auto", "config": {"model": "org/m"}}])
    orch = _orch(cfg, state_dir=tmp_path)
    orch.build()
    assert orch.entry("engine").base_type == "llama_cpp"  # from the cache


# --- reconciliation ---
def test_reconcile_detects_health_loss() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg, health_interval=0.02, metrics_interval=0.02)

    async def scenario() -> None:
        await orch.boot()
        assert orch.states["engine"] is ServiceState.READY
        _fm(orch, "engine")._healthy = False  # simulate a crash
        stop = asyncio.Event()
        task = asyncio.create_task(orch.reconcile(stop))
        await asyncio.sleep(0.1)
        assert orch.states["engine"] is ServiceState.DEGRADED
        stop.set()
        await task

    asyncio.run(scenario())


def test_reconcile_collects_metrics() -> None:
    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg, health_interval=0.02, metrics_interval=0.02)

    async def scenario() -> None:
        await orch.boot()
        stop = asyncio.Event()
        task = asyncio.create_task(orch.reconcile(stop))
        await asyncio.sleep(0.1)
        assert orch.metrics["engine"]["status"] == "running"
        stop.set()
        await task

    asyncio.run(scenario())


def test_health_loop_survives_probe_exception() -> None:
    """One manager's is_healthy() raising must degrade that service and keep
    the loop (and the other service's supervision) alive."""
    cfg = _config([{"name": "flaky", "base_type": "x"}, {"name": "solid", "base_type": "x"}])
    orch = _orch(cfg, health_interval=0.02, metrics_interval=0.02)

    async def scenario() -> None:
        await orch.boot()

        def boom() -> bool:
            raise RuntimeError("probe blew up")

        orch.managers["flaky"].is_healthy = boom  # type: ignore[method-assign]
        stop = asyncio.Event()
        task = asyncio.create_task(orch.reconcile(stop))
        await asyncio.sleep(0.1)
        assert orch.states["flaky"] is ServiceState.DEGRADED
        assert orch.states["solid"] is ServiceState.READY  # still supervised
        stop.set()
        await task  # loop never crashed

    asyncio.run(scenario())


def test_metrics_loop_survives_metrics_exception() -> None:
    cfg = _config([{"name": "flaky", "base_type": "x"}, {"name": "solid", "base_type": "x"}])
    orch = _orch(cfg, health_interval=0.02, metrics_interval=0.02)

    async def scenario() -> None:
        await orch.boot()

        def boom() -> dict:
            raise RuntimeError("docker stats went weird")

        orch.managers["flaky"].get_metrics = boom  # type: ignore[method-assign]
        stop = asyncio.Event()
        task = asyncio.create_task(orch.reconcile(stop))
        await asyncio.sleep(0.1)
        assert orch.metrics["flaky"] == {"status": "error"}
        assert orch.metrics["solid"]["status"] == "running"  # unaffected
        stop.set()
        await task

    asyncio.run(scenario())


# --- shutdown ---
def test_shutdown_reverse_order() -> None:
    log: list = []
    cfg = _config(
        [
            {"name": "engine", "base_type": "x"},
            {"name": "frontend", "base_type": "x", "dependencies": ["engine"]},
        ]
    )
    orch = _orch(cfg, log)

    async def scenario() -> None:
        await orch.boot()
        log.clear()
        await orch.shutdown()

    asyncio.run(scenario())
    assert log == [("frontend", "stop"), ("engine", "stop")]
    assert orch.states["engine"] is ServiceState.STOPPED


# --- persistence ---
def test_manifest_and_state_written(tmp_path) -> None:
    cfg = _config(
        [
            {"name": "engine", "base_type": "x"},
            {"name": "frontend", "base_type": "x", "dependencies": ["engine"]},
        ]
    )
    orch = _orch(cfg, state_dir=tmp_path, ports={"engine": 11435})
    asyncio.run(orch.boot())

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == "1.1"
    names = [s["name"] for s in manifest["services"]]
    assert names == ["engine", "frontend"]  # boot order
    engine = manifest["services"][0]
    assert engine["endpoint"] == {"scheme": "http", "host": "127.0.0.1", "port": 11435}
    assert engine["co_resident"] == ["frontend"]

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["services"] == {"engine": "ready", "frontend": "ready"}


def test_runtime_handles_persisted_as_each_service_starts(tmp_path) -> None:
    """A crash mid-boot must leave already-started PIDs on disk so `down` works.

    The second service fails to boot; the first is READY — its runtime handle
    must already be in state.json even though boot() as a whole raised.
    """
    cfg = _config(
        [
            {"name": "engine", "base_type": "x"},
            {"name": "frontend", "base_type": "x", "dependencies": ["engine"]},
        ]
    )
    orch = _orch(cfg, state_dir=tmp_path, prepare_model_raises={"frontend"})
    with pytest.raises(BootError, match="download failed"):
        asyncio.run(orch.boot())

    state = json.loads((tmp_path / "state.json").read_text())
    assert state["runtime"]["engine"] == {"kind": "native", "pid": 4242}
    assert state["services"]["engine"] == "ready"


def test_status_snapshot_shape() -> None:
    cfg = _config(
        [
            {"name": "engine", "base_type": "x"},
            {"name": "frontend", "base_type": "x", "dependencies": ["engine"]},
        ]
    )
    orch = _orch(cfg, ports={"engine": 11435})
    asyncio.run(orch.boot())
    snapshot = orch.status_snapshot()
    assert snapshot["budget"]["system_total_bytes"] > 0  # machine RAM, via psutil
    assert snapshot["budget"]["system_used_bytes"] >= 0
    assert set(snapshot["services"]) == {"engine", "frontend"}
    frontend = snapshot["services"]["frontend"]
    assert frontend["state"] == "ready"
    assert "dependencies" not in frontend
    assert "metrics" in frontend

    engine = snapshot["services"]["engine"]
    assert engine["endpoint"] == "http://127.0.0.1:11435"
    assert engine["engine"] == "x"
    assert frontend["endpoint"] is None  # no port configured for this fake manager

    from datetime import datetime

    assert engine["since"] is not None
    assert datetime.fromisoformat(engine["since"])


def test_status_snapshot_descriptor_by_base_type() -> None:
    cfg = _config(
        [
            {
                "name": "webui",
                "base_type": "docker",
                "config": {"image": "ghcr.io/open-webui/open-webui:main"},
            },
            {
                "name": "heavy",
                "base_type": "mlx_lm",
                "config": {"model": "mlx-community/Qwen3.6-27B-8bit"},
            },
            {
                "name": "cline_local",
                "base_type": "cline_cli",
                "config": {"config_dir": "~/.sovereign/harnesses/cline_local"},
            },
        ]
    )
    orch = _orch(cfg)
    orch.build()
    snap = orch.status_snapshot()
    assert snap["services"]["webui"]["descriptor"] == "ghcr.io/open-webui/open-webui:main"
    assert snap["services"]["heavy"]["descriptor"] == "mlx-community/Qwen3.6-27B-8bit"
    assert snap["services"]["cline_local"]["descriptor"] is None
    assert snap["services"]["webui"]["engine"] == "docker"
    assert snap["services"]["heavy"]["engine"] == "mlx_lm"
    assert snap["services"]["cline_local"]["engine"] == "cline_cli"


def test_status_snapshot_includes_activity() -> None:
    orch = _orch(_config([{"name": "a", "base_type": "x"}]))
    orch.build()
    orch.managers["a"].activity = ("pulling foo — 2/5 layers",)
    snap = orch.status_snapshot()
    assert snap["services"]["a"]["activity"] == {"lines": ["pulling foo — 2/5 layers"]}


def test_status_snapshot_since_present_immediately_after_build() -> None:
    from datetime import datetime

    orch = _orch(_config([{"name": "a", "base_type": "x"}]))
    orch.build()
    snap = orch.status_snapshot()
    since = snap["services"]["a"]["since"]
    assert since is not None
    assert datetime.fromisoformat(since)


def test_set_state_updates_since_only_on_real_transition() -> None:
    orch = _orch(_config([{"name": "a", "base_type": "x"}]))
    orch.build()
    first = orch.state_since["a"]
    orch._set_state("a", ServiceState.PENDING)  # no-op: same state
    assert orch.state_since["a"] == first
    orch._set_state("a", ServiceState.PROVISIONING)  # real transition
    assert orch.state_since["a"] != first


def test_boot_is_watchable_live() -> None:
    """An observer running alongside boot sees PROVISIONING before READY."""
    orch = _orch(_config([{"name": "a", "base_type": "x"}]), prepare_delays={"a": 0.05})

    async def scenario() -> None:
        orch.build()  # PENDING visible immediately
        seen: list[str] = []
        stop = asyncio.Event()

        async def observer() -> None:
            while not stop.is_set():
                seen.append(orch.status_snapshot()["services"]["a"]["state"])
                await asyncio.sleep(0.005)

        obs = asyncio.create_task(observer())
        await orch.boot()
        stop.set()
        await obs
        assert "provisioning" in seen  # boot progress was observable live
        assert orch.states["a"] is ServiceState.READY  # and it finished

    asyncio.run(scenario())


# --- admission control (§7) ---
def test_over_budget_boot_refused_with_actionable_error() -> None:
    cfg = _config(
        [
            {"name": "comfyui", "base_type": "x"},
            {"name": "llama_heavy", "base_type": "x", "dependencies": ["comfyui"]},
        ],
        resources={"max_unified_memory_gb": 64, "safety_margin_gb": 8},
    )
    orch = _orch(cfg, mems={"comfyui": 25 * 10**9, "llama_heavy": 40 * 10**9})
    with pytest.raises(BootError) as exc:
        asyncio.run(orch.boot())
    msg = str(exc.value)
    assert "Cannot start 'llama_heavy'" in msg
    assert "comfyui (~25.0 GB)" in msg  # tells you what to stop
    assert orch.states["llama_heavy"] is ServiceState.FAILED
    assert orch.states["comfyui"] is ServiceState.READY  # it fit and booted


def test_fitting_services_reserve_budget() -> None:
    cfg = _config([{"name": "a", "base_type": "x"}, {"name": "b", "base_type": "x"}])
    orch = _orch(cfg, mems={"a": 20 * 10**9, "b": 10 * 10**9})
    asyncio.run(orch.boot())
    assert orch.budgeter.reserved_bytes == 30 * 10**9
    assert orch.budgeter.reservations() == {"a": 20 * 10**9, "b": 10 * 10**9}


def test_shutdown_releases_budget() -> None:
    cfg = _config([{"name": "a", "base_type": "x"}])
    orch = _orch(cfg, mems={"a": 20 * 10**9})

    async def scenario() -> None:
        await orch.boot()
        assert orch.budgeter.reserved_bytes == 20 * 10**9
        await orch.shutdown()

    asyncio.run(scenario())
    assert orch.budgeter.reserved_bytes == 0


def test_manifest_records_memory_budget(tmp_path) -> None:
    cfg = _config(
        [{"name": "a", "base_type": "x"}],
        resources={"max_unified_memory_gb": 64, "safety_margin_gb": 8},
    )
    orch = _orch(cfg, state_dir=tmp_path, mems={"a": 20 * 10**9})
    asyncio.run(orch.boot())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["memory_budget"] == {
        "total_bytes": 64 * 10**9,
        "safety_margin_bytes": 8 * 10**9,
        "reserved_bytes": 20 * 10**9,
        "available_bytes": 36 * 10**9,
    }
    assert manifest["services"][0]["estimated_memory_bytes"] == 20 * 10**9


def test_boot_warns_on_unknown_memory_footprint(caplog) -> None:
    """The unknown->admit policy stays, but it must be loud at boot."""
    import logging

    cfg = _config([{"name": "mystery", "base_type": "x"}])
    orch = _orch(cfg)  # FakeManager reports no estimate source -> unknown
    with caplog.at_level(logging.WARNING, logger="sovereign.runtime.orchestrator"):
        asyncio.run(orch.boot())
    assert any("UNKNOWN memory footprint" in r.message for r in caplog.records)


def test_boot_no_unknown_warning_when_source_known(caplog) -> None:
    import logging

    cfg = _config([{"name": "engine", "base_type": "x"}])
    orch = _orch(cfg, mems={"engine": 8 * 10**9}, estimate_sources={"engine": "local"})
    orch.build()
    with caplog.at_level(logging.WARNING, logger="sovereign.runtime.orchestrator"):
        asyncio.run(orch.boot())
    assert not any("UNKNOWN memory footprint" in r.message for r in caplog.records)


# --- harness materialization (H1) ---
def test_harness_materialized_after_deps_ready() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log)
    asyncio.run(orch.boot())
    assert ("h", "materialize") in log
    assert _fh(orch, "h").resolved_with is orch.resolver
    assert _fh(orch, "h").materialize_count == 1


def test_harness_provisioned_before_materialize() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log)
    asyncio.run(orch.boot())
    assert log.index(("h", "prepare_environment")) < log.index(("h", "materialize"))
    assert _fh(orch, "h").prepare_count == 1


def test_harness_reprovisioned_on_endpoint_change() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, ports={"engine": 11435})

    async def scenario() -> None:
        await orch.boot()
        _fm(orch, "engine")._port = 11999
        await orch._restart("engine")
        # Re-materialization re-runs the (idempotent) provisioning hook too.
        assert _fh(orch, "h").prepare_count == 2
        assert _fh(orch, "h").materialize_count == 2

    asyncio.run(scenario())


def test_harness_not_materialized_when_deps_not_ready() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, healthy=False, health_interval=0.01)
    orch._entries["engine"].health_check = None
    orch._health_timeout = lambda name: 0.05  # type: ignore[method-assign]
    with pytest.raises(BootError):
        asyncio.run(orch.boot())
    assert _fh(orch, "h").materialize_count == 0


def test_harness_remateralized_when_endpoint_changes() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, ports={"engine": 11435})

    async def scenario() -> None:
        await orch.boot()
        assert _fh(orch, "h").materialize_count == 1
        # Simulate a restart landing on a new port.
        _fm(orch, "engine")._port = 11999
        await orch._restart("engine")
        assert orch.registry.get("engine").port == 11999
        assert _fh(orch, "h").materialize_count == 2

    asyncio.run(scenario())


def test_harness_not_remateralized_when_endpoint_unchanged() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, ports={"engine": 11435})

    async def scenario() -> None:
        await orch.boot()
        assert _fh(orch, "h").materialize_count == 1
        await orch._restart("engine")  # same port
        assert _fh(orch, "h").materialize_count == 1

    asyncio.run(scenario())


def test_unknown_harness_base_type_fails_boot() -> None:
    """A typo'd harness base_type is a config error, not a silent no-op."""
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "not_a_real_harness"}],
    )
    orch = _orch(cfg)  # default harness factory -> registry lookup
    with pytest.raises(BootError, match="not_a_real_harness"):
        orch.build()


def test_harness_missing_optional_dep_warns_but_boots(caplog) -> None:
    """A harness whose optional package isn't installed logs a clear warning
    and the services still boot."""
    import logging

    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y"}],
    )

    def factory(entry):
        raise ImportError("No module named 'minisweagent'")

    orch = _orch(cfg, harness_factory=factory)
    with caplog.at_level(logging.WARNING, logger="sovereign.runtime.orchestrator"):
        asyncio.run(orch.boot())
    assert orch.states["engine"] is ServiceState.READY
    messages = [r.getMessage() for r in caplog.records]
    assert any("harness 'h'" in m and "minisweagent" in m for m in messages)


def test_manifest_includes_harnesses(tmp_path) -> None:
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, state_dir=tmp_path, harness_log=[], ports={"engine": 11435})
    asyncio.run(orch.boot())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["harnesses"] == [
        {"name": "h", "base_type": "y", "dependencies": ["engine"]}
    ]


# --- telemetry integration (§5) ---
def test_status_snapshot_includes_telemetry_block() -> None:
    orch = _orch(_config([{"name": "a", "base_type": "x"}]))
    orch.build()
    snap = orch.status_snapshot()
    assert snap["services"]["a"]["telemetry"] == {
        "worker_state": None,
        "last_heartbeat": None,
        "prefill": [],
        "generation_tps": None,
        "prompt_tps": None,
        "tps_history": [],
    }


def test_status_snapshot_telemetry_reflects_cache_state() -> None:
    from sovereign.workers.protocol import EventType

    orch = _orch(_config([{"name": "a", "base_type": "x"}]))
    orch.build()
    orch.telemetry.apply_local("a", EventType.STATE_CHANGE, {"state": "serving"})
    orch.telemetry.apply_local(
        "a", EventType.GENERATION_STATS, {"generation_tps": 12.5, "prompt_tps": 400.0}
    )
    snap = orch.status_snapshot()
    telemetry = snap["services"]["a"]["telemetry"]
    assert telemetry["worker_state"] == "serving"
    assert telemetry["generation_tps"] == 12.5
    assert telemetry["prompt_tps"] == 400.0


def test_effective_metrics_prefers_fresh_cache_over_manager() -> None:
    from sovereign.workers.protocol import EventType

    cfg = _config([{"name": "a", "base_type": "x"}])
    orch = _orch(cfg, mems={"a": 0})
    orch.build()
    orch.telemetry.apply_local("a", EventType.MEMORY, {"memory_bytes": 555})
    metrics = orch._effective_metrics("a", orch.managers["a"])
    assert metrics["memory_bytes"] == 555
    assert metrics["status"] == "running"


def test_effective_metrics_falls_back_when_stale() -> None:
    from sovereign.workers.protocol import EventType

    cfg = _config([{"name": "a", "base_type": "x"}])
    orch = _orch(cfg)
    orch.build()
    # A stale event (older than the freshness window) must not win over the
    # manager's own get_metrics() fallback.
    orch.telemetry.apply_local("a", EventType.MEMORY, {"memory_bytes": 555}, ts=0.0)
    metrics = orch._effective_metrics("a", orch.managers["a"])
    assert metrics.get("memory_bytes") != 555
    assert metrics["status"] == "running"
    assert metrics["name"] == "a"  # from FakeManager.get_metrics()


def test_effective_metrics_merges_generation_stats() -> None:
    from sovereign.workers.protocol import EventType

    cfg = _config([{"name": "a", "base_type": "x"}])
    orch = _orch(cfg)
    orch.build()
    orch.telemetry.apply_local("a", EventType.GENERATION_STATS, {"generation_tps": 9.0})
    metrics = orch._effective_metrics("a", orch.managers["a"])
    assert metrics["tokens_per_second"] == 9.0


def test_telemetry_hub_lifecycle_via_serve_forever(socket_path) -> None:
    """The hub binds .sovereign/telemetry.sock during serve_forever and unlinks
    it in the finally block, exercised via the real FakeManager factory.

    The state dir comes from the short-path socket_path fixture (not tmp_path)
    so the hub's AF_UNIX bind stays under macOS's ~104-byte sun_path cap.
    """
    state_dir = socket_path.parent
    log: list = []
    cfg = _config([{"name": "a", "base_type": "x"}])
    socket_seen = asyncio.Event()

    async def watcher(orch: Orchestrator, stop: asyncio.Event) -> None:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not (state_dir / "telemetry.sock").exists():
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.01)
        socket_seen.set()
        stop.set()

    def factory(entry: ServiceEntry) -> FakeManager:
        return FakeManager(entry, log)

    from sovereign.runtime.orchestrator import serve_forever

    asyncio.run(
        serve_forever(
            cfg,
            state_dir=state_dir,
            manager_factory=factory,
            extra_tasks=[watcher],
        )
    )
    assert socket_seen.is_set()
    assert not (state_dir / "telemetry.sock").exists()


def test_sigint_fires_on_stop_once_after_extra_tasks_exit(socket_path) -> None:
    """A real first SIGINT invokes the on_stop hook (the CLI's "Stopping
    stack…" notice) exactly once — and only after the extra tasks (i.e. the
    dashboard's Live) have wound down, so the notice prints below the panels."""
    import os
    import signal

    state_dir = socket_path.parent
    log: list = []
    cfg = _config([{"name": "a", "base_type": "x"}])
    order: list[str] = []

    async def press_ctrl_c(orch: Orchestrator, stop: asyncio.Event) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        await stop.wait()
        order.append("extra-task-exited")

    from sovereign.runtime.orchestrator import serve_forever

    try:
        asyncio.run(
            serve_forever(
                cfg,
                state_dir=state_dir,
                manager_factory=lambda entry: FakeManager(entry, log),
                extra_tasks=[press_ctrl_c],
                on_stop=lambda: order.append("stopping"),
            )
        )
    finally:
        # serve_forever's loop-level handler dies with the loop; restore the
        # default so a later Ctrl+C still interrupts the test run.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    assert order == ["extra-task-exited", "stopping"]
