"""Phase 6: Orchestrator — topo boot, concurrency, reconciliation, shutdown."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from sovereign.config import ServiceEntry, SovereignConfig
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.orchestrator import (
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
        mem_gb: float = 0.0,
        prepare_delay: float = 0.0,
        has_prepare_model: bool = True,
        prepare_model_raises: bool = False,
    ):
        self.name = entry.name
        self.dependencies = entry.dependencies
        self._log = log
        self._healthy = healthy
        self._port = port
        self._mem_gb = mem_gb
        self._prepare_delay = prepare_delay
        self._prepare_model_raises = prepare_model_raises
        self.resolved_with = None
        self.activity = ()
        # A native engine exposes prepare_model (pre-download); managers without
        # the capability (docker, older fakes) simply lack the attribute,
        # which is what the SupportsModelPreparation isinstance check keys on.
        if has_prepare_model:
            self.prepare_model = self._prepare_model

    def estimated_memory_gb(self) -> float:
        return self._mem_gb

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

    def adjust_resources(self, memory_limit_mb: int) -> None: ...

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

    def factory(entry: ServiceEntry) -> FakeManager:
        return FakeManager(
            entry,
            log,
            healthy=healthy,
            port=ports.get(entry.name, 0),
            mem_gb=mems.get(entry.name, 0.0),
            prepare_delay=prepare_delays.get(entry.name, 0.0),
            has_prepare_model=entry.name not in no_prepare_model,
            prepare_model_raises=entry.name in prepare_model_raises,
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
    assert orch.managers["engine"].resolved_with is orch.resolver


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
    import sovereign.orchestrator as orch_mod

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
        orch.managers["engine"]._healthy = False  # simulate a crash
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
    assert set(snapshot["services"]) == {"engine", "frontend"}
    frontend = snapshot["services"]["frontend"]
    assert frontend["state"] == "ready"
    assert "dependencies" not in frontend
    assert "metrics" in frontend

    engine = snapshot["services"]["engine"]
    assert engine["endpoint"] == "http://127.0.0.1:11435"
    assert frontend["endpoint"] is None  # no port configured for this fake manager

    from datetime import datetime

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
    assert datetime.fromisoformat(snap["services"]["a"]["since"])


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
    orch = _orch(cfg, mems={"comfyui": 25, "llama_heavy": 40})
    with pytest.raises(BootError) as exc:
        asyncio.run(orch.boot())
    msg = str(exc.value)
    assert "Cannot start 'llama_heavy'" in msg
    assert "comfyui (~25.0GB)" in msg  # tells you what to stop
    assert orch.states["llama_heavy"] is ServiceState.FAILED
    assert orch.states["comfyui"] is ServiceState.READY  # it fit and booted


def test_fitting_services_reserve_budget() -> None:
    cfg = _config([{"name": "a", "base_type": "x"}, {"name": "b", "base_type": "x"}])
    orch = _orch(cfg, mems={"a": 20, "b": 10})
    asyncio.run(orch.boot())
    assert orch.budgeter.reserved_gb == 30
    assert orch.budgeter.reservations() == {"a": 20.0, "b": 10.0}


def test_shutdown_releases_budget() -> None:
    cfg = _config([{"name": "a", "base_type": "x"}])
    orch = _orch(cfg, mems={"a": 20})

    async def scenario() -> None:
        await orch.boot()
        assert orch.budgeter.reserved_gb == 20
        await orch.shutdown()

    asyncio.run(scenario())
    assert orch.budgeter.reserved_gb == 0


def test_manifest_records_memory_budget(tmp_path) -> None:
    cfg = _config(
        [{"name": "a", "base_type": "x"}],
        resources={"max_unified_memory_gb": 64, "safety_margin_gb": 8},
    )
    orch = _orch(cfg, state_dir=tmp_path, mems={"a": 20})
    asyncio.run(orch.boot())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["memory_budget"] == {
        "total_gb": 64.0,
        "safety_margin_gb": 8.0,
        "reserved_gb": 20.0,
        "available_gb": 36.0,
    }
    assert manifest["services"][0]["estimated_memory_gb"] == 20.0


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
    assert orch.harnesses["h"].resolved_with is orch.resolver
    assert orch.harnesses["h"].materialize_count == 1


def test_harness_provisioned_before_materialize() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log)
    asyncio.run(orch.boot())
    assert log.index(("h", "prepare_environment")) < log.index(("h", "materialize"))
    assert orch.harnesses["h"].prepare_count == 1


def test_harness_reprovisioned_on_endpoint_change() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, ports={"engine": 11435})

    async def scenario() -> None:
        await orch.boot()
        orch.managers["engine"]._port = 11999
        await orch._restart("engine")
        # Re-materialization re-runs the (idempotent) provisioning hook too.
        assert orch.harnesses["h"].prepare_count == 2
        assert orch.harnesses["h"].materialize_count == 2

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
    assert orch.harnesses["h"].materialize_count == 0


def test_harness_remateralized_when_endpoint_changes() -> None:
    log: list = []
    cfg = _config(
        [{"name": "engine", "base_type": "x"}],
        harnesses=[{"name": "h", "base_type": "y", "dependencies": ["engine"]}],
    )
    orch = _orch(cfg, log, harness_log=log, ports={"engine": 11435})

    async def scenario() -> None:
        await orch.boot()
        assert orch.harnesses["h"].materialize_count == 1
        # Simulate a restart landing on a new port.
        orch.managers["engine"]._port = 11999
        await orch._restart("engine")
        assert orch.registry.get("engine").port == 11999
        assert orch.harnesses["h"].materialize_count == 2

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
        assert orch.harnesses["h"].materialize_count == 1
        await orch._restart("engine")  # same port
        assert orch.harnesses["h"].materialize_count == 1

    asyncio.run(scenario())


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
