"""Phase 4: llama_cpp manager — mocked unit tests + Protocol/registry checks.

The real llama-cpp-python binding and a GGUF model are not required here; the
subprocess, HTTP probe, import-probe, and psutil calls are all mocked (via the
shared inference.base module, where the process/health/metrics lifecycle now
lives). Flag-mapping assertions live on ``engine_kwargs()``; ``get_start_args()``
is asserted only to produce the shared worker-launch argv + dumped JSON.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from typing import cast

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.inference import base as native_mod
from sovereign.services.inference import hf as models_mod
from sovereign.services.inference.hf import RepoInfo
from sovereign.services.inference.llama_cpp import manager as llama_manager_mod
from sovereign.services.inference.llama_cpp.manager import LlamaCppManager
from sovereign.workers.worker_config import load_worker_config


def _repo_info(repo_id: str, siblings: list[tuple[str, int | None]], tags=()) -> RepoInfo:
    return RepoInfo(repo_id=repo_id, tags=tuple(tags), siblings=tuple(siblings))


@pytest.fixture(autouse=True)
def _offline_metadata(monkeypatch):
    """Default HF metadata fetch to offline (None) so no test hits the network via
    the prepare_environment prefetch or repo-id estimation; specific tests override."""
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check=(
            {"type": "http", "endpoint": "/health", "port": 11435}
            if with_health
            else None
        ),
        config=config or {"model": "/models/x.gguf"},
    )


def _manager(config: dict | None = None) -> LlamaCppManager:
    return LlamaCppManager(_entry(config))


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """get_start_args() now writes a worker-config JSON under
    ``<state_dir>/workers/`` (derived from log_dir's parent) — run every test
    from an isolated tmp cwd so the default ``.sovereign/logs`` relative path
    doesn't touch the real repo tree."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _passthrough_download(monkeypatch):
    """Neutralise the real HF download: resolve local refs in place and treat a repo
    id as if it downloaded to a path equal to the ref, so argv assertions stay
    readable. Applied suite-wide so no test reaches the network on prepare_model()."""
    from pathlib import Path

    def _fake(ref, kind, *, progress=None):
        return ref.local_path if ref.is_local else Path(ref.raw)

    monkeypatch.setattr(models_mod, "download_model", _fake)


def _prepared(config: dict | None = None) -> LlamaCppManager:
    """A manager with prepare_model() already run (model paths resolved)."""
    m = _manager(config)
    m.prepare_model()
    return m


class FakeProc:
    def __init__(self, pid: int = 4242, poll_value: int | None = None):
        self.pid = pid
        self._poll = poll_value
        self.terminated = False
        self.killed = False
        self.wait_raises: Exception | None = None

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0

    def kill(self):
        self.killed = True
        self._poll = -9

    def wait(self, timeout=None):
        if self.wait_raises is not None:
            exc, self.wait_raises = self.wait_raises, None
            raise exc
        return self._poll


# --- construction / protocol / registry ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("llama_cpp") is LlamaCppManager


def test_requires_health_check() -> None:
    with pytest.raises(ValueError, match="requires a health_check"):
        LlamaCppManager(_entry(with_health=False))


def test_port_and_path_taken_from_health_check() -> None:
    m = _manager()
    assert m.port == 11435
    assert m.health_path == "/health"


# --- worker argv + dumped WorkerConfig (get_start_args is now shared/final) ---
def test_get_start_args_before_prepare_raises() -> None:
    with pytest.raises(RuntimeError, match="prepare_model"):
        _manager({"model": "/models/x.gguf"}).get_start_args()


def test_prepare_model_sets_paths() -> None:
    m = _prepared({"model": "org/repo:Q4_K_M", "draft_model": "org/tiny-draft"})
    assert str(m.model_path) == "org/repo:Q4_K_M"
    assert str(m.draft_model_path) == "org/tiny-draft"


def test_get_start_args_launches_generic_engine_worker(tmp_path) -> None:
    m = _prepared({"model": "/models/x.gguf", "log_dir": str(tmp_path / "logs")})
    args = m.get_start_args()
    assert args[0].endswith(("python", "python3")) or "python" in args[0].lower()
    assert args[1:4] == ["-m", "sovereign.workers.engine_worker", "--config"]
    config_path = args[4]
    assert config_path == str(tmp_path / "workers" / "llama_heavy_v1.json")


def test_get_start_args_dumps_worker_config(tmp_path) -> None:
    m = _prepared(
        {
            "model": "/models/llama3-70b.gguf",
            "gpu_layers": 48,
            "threads": 8,
            "context_size": 32768,
            "served_model_name": "llama3-70b",
            "log_dir": str(tmp_path / "logs"),
        }
    )
    args = m.get_start_args()
    cfg = load_worker_config(args[4])
    assert cfg.service == "llama_heavy_v1"
    assert cfg.engine == "llama_cpp"
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11435
    assert cfg.health_path == "/health"
    assert cfg.model_path == "/models/llama3-70b.gguf"
    assert cfg.served_model_name == "llama3-70b"
    assert cfg.telemetry_socket == str(tmp_path / "telemetry.sock")
    assert cfg.engine_kwargs == {
        "gpu_layers": 48,
        "threads": 8,
        "context_size": 32768,
    }


def test_get_start_args_json_file_mode_0600(tmp_path) -> None:
    import stat

    m = _prepared({"model": "/models/x.gguf", "log_dir": str(tmp_path / "logs")})
    m.get_start_args()
    mode = stat.S_IMODE((tmp_path / "workers" / "llama_heavy_v1.json").stat().st_mode)
    assert mode == 0o600


def test_get_start_args_api_key_never_in_config_json(tmp_path) -> None:
    m = _prepared(
        {"model": "/models/x.gguf", "api_key": "secret", "log_dir": str(tmp_path / "logs")}
    )
    m.get_start_args()
    raw = json.loads((tmp_path / "workers" / "llama_heavy_v1.json").read_text())
    assert "secret" not in json.dumps(raw)


# --- engine_kwargs (Sovereign config -> worker adapter mapping) ---
def test_engine_kwargs_full_mapping() -> None:
    m = _prepared(
        {
            "model": "/models/llama3-70b.gguf",
            "gpu_layers": 48,
            "threads": 8,
            "context_size": 32768,
            "max_parallel": 4,
            "num_draft_tokens": 2,
        }
    )
    assert m.engine_kwargs() == {
        "gpu_layers": 48,
        "threads": 8,
        "context_size": 32768,
        "max_parallel": 4,
        "num_draft_tokens": 2,
    }


def test_engine_kwargs_minimal_omits_unset() -> None:
    assert _prepared().engine_kwargs() == {}


def test_engine_kwargs_kv_cache_type_from_enabled_caching() -> None:
    m = _caching_manager({"enabled": True, "cache_path": "/tmp/c", "kv_cache_type": "q8_0"})
    assert m.engine_kwargs()["kv_cache_type"] == "q8_0"


def test_engine_kwargs_no_kv_cache_type_when_caching_disabled() -> None:
    m = _caching_manager({"enabled": False, "cache_path": "/tmp/c"})
    assert "kv_cache_type" not in m.engine_kwargs()


def test_engine_kwargs_config_override_wins_last() -> None:
    m = _prepared({"model": "/x.gguf", "gpu_layers": 10, "engine_kwargs": {"gpu_layers": 99}})
    assert m.engine_kwargs()["gpu_layers"] == 99


# --- API key via environment (never argv/config JSON) ---
def test_start_env_carries_api_key() -> None:
    m = _manager({"model": "/models/x.gguf", "api_key": "secret"})
    assert m.start_env() == {"SOVEREIGN_API_KEY": "secret"}


def test_start_env_empty_without_api_key() -> None:
    assert _manager().start_env() == {}


def test_start_passes_api_key_in_subprocess_env(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def fake_popen(args, **kwargs):
        captured["env"] = kwargs.get("env")
        return FakeProc(poll_value=None)

    monkeypatch.setattr(native_mod.subprocess, "Popen", fake_popen)
    m = _prepared({"model": "/models/x.gguf", "api_key": "secret", "log_dir": str(tmp_path)})
    m.start()
    assert captured["env"]["SOVEREIGN_API_KEY"] == "secret"
    assert "PATH" in captured["env"]  # inherits the parent environment


def test_start_env_none_when_no_extra_env(tmp_path, monkeypatch) -> None:
    """Without engine-contributed env, Popen gets env=None (full inheritance)."""
    captured: dict = {}

    def fake_popen(args, **kwargs):
        captured["env"] = kwargs.get("env", "missing")
        return FakeProc(poll_value=None)

    monkeypatch.setattr(native_mod.subprocess, "Popen", fake_popen)
    m = _prepared({"model": "/models/x.gguf", "log_dir": str(tmp_path)})
    m.start()
    assert captured["env"] is None


def test_get_start_args_expands_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    m = _prepared({"model": "~/models/x.gguf", "log_dir": str(tmp_path / "logs")})
    cfg = load_worker_config(m.get_start_args()[4])
    assert cfg.model_path == str(m.model_path)  # resolution happens in prepare_model()


def test_get_start_args_hf_repo_resolves_to_local_path(tmp_path) -> None:
    m = _prepared(
        {"model": "ggml-org/gemma-3-1b-it-GGUF", "log_dir": str(tmp_path / "logs")}
    )
    cfg = load_worker_config(m.get_start_args()[4])
    assert cfg.model_path == "ggml-org/gemma-3-1b-it-GGUF"  # resolved path (fake download)


# --- served_model_name / api_model_name (harness+bench wiring) ---
def test_api_model_name_defaults_to_model() -> None:
    m = _manager({"model": "/models/x.gguf"})
    assert m.api_model_name() == "/models/x.gguf"


def test_api_model_name_prefers_served_model_name() -> None:
    m = _manager({"model": "/models/x.gguf", "served_model_name": "llama3-70b"})
    assert m.api_model_name() == "llama3-70b"


def test_endpoint_carries_api_model_name() -> None:
    m = _manager({"model": "/models/x.gguf", "served_model_name": "llama3-70b"})
    assert m.endpoint().model == "llama3-70b"


# --- provisioning ---
def test_provisioning_declaration() -> None:
    assert LlamaCppManager.import_probe_modules == ("llama_cpp", "llama_cpp.server.app")
    assert LlamaCppManager.provisioning_brewfile() is None  # no more Brewfile/binary
    assert any(
        "llama-cpp-python[server]" in " ".join(cmd)
        for cmd in LlamaCppManager.provisioning_commands
    )


def test_provisioning_satisfied_uses_import_probe(monkeypatch) -> None:
    calls: list[str] = []

    def fake_probe(module: str) -> bool:
        calls.append(module)
        return True

    monkeypatch.setattr(llama_manager_mod, "probe_import", fake_probe)
    assert LlamaCppManager.provisioning_satisfied() is True
    assert set(calls) == {"llama_cpp", "llama_cpp.server.app"}


def test_provisioning_not_satisfied_when_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        llama_manager_mod, "probe_import", lambda module: module != "llama_cpp"
    )
    assert LlamaCppManager.provisioning_satisfied() is False


def test_prepare_environment_provisions_first(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    order: list[str] = []

    def _record_probe(_m: str) -> bool:
        order.append("probe")
        return True

    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: order.append("provision"))
    )
    monkeypatch.setattr(native_mod, "probe_import", _record_probe)
    m = _manager({"model": "org/some-hf-repo"})  # repo id: no local file check
    m.prepare_environment()
    assert order[0] == "provision"  # install the toolchain before validating it


# --- prepare_environment ---
def test_prepare_environment_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    with pytest.raises(FileNotFoundError, match="model for 'llama_heavy_v1' not found"):
        _manager({"model": "/nope/missing.gguf"}).prepare_environment()


def test_prepare_environment_ok(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    _manager({"model": str(model)}).prepare_environment()  # must not raise


def test_prepare_environment_missing_binding(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod, "probe_import", lambda m: False)
    with pytest.raises(FileNotFoundError, match="llama_cpp.*not importable"):
        _manager({"model": str(model)}).prepare_environment()


def test_prepare_environment_repo_id_ok(monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    # A repo id that isn't local must NOT raise (the worker downloads it on start).
    _manager({"model": "ggml-org/gemma-3-1b-it-GGUF"}).prepare_environment()


def test_prepare_environment_draft_model_raises(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    with pytest.raises(ValueError, match="no longer supports GGUF draft models"):
        _manager(
            {"model": str(model), "draft_model": "/nope/missing-draft.gguf"}
        ).prepare_environment()


def test_prepare_environment_max_parallel_warns(tmp_path, monkeypatch, caplog) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    with caplog.at_level("WARNING", logger="sovereign"):
        _manager({"model": str(model), "max_parallel": 4}).prepare_environment()
    assert any("max_parallel=4" in r.message for r in caplog.records)


# --- health ---
def test_is_healthy_false_when_no_process() -> None:
    assert _manager().is_healthy() is False


def test_is_healthy_false_when_process_exited() -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=0))
    assert m.is_healthy() is False


def test_is_healthy_true_on_http_200(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=None))

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(native_mod.urllib.request, "urlopen", lambda url, timeout=None: FakeResp())
    assert m.is_healthy() is True


def test_is_healthy_false_on_connection_error(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=None))

    def boom(url, timeout=None):
        raise native_mod.urllib.error.URLError("refused")

    monkeypatch.setattr(native_mod.urllib.request, "urlopen", boom)
    assert m.is_healthy() is False


# --- lifecycle ---
def test_start_launches_process_with_argv(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(native_mod.subprocess, "Popen", fake_popen)
    m = _prepared({"model": "/models/x.gguf", "log_dir": str(tmp_path / "logs")})
    m.start()
    assert captured["args"] == m.get_start_args()
    assert m.process is not None
    assert (tmp_path / "logs" / "llama_heavy_v1.log").exists()
    m.stop()


def test_stop_terminates_running_process(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    monkeypatch.setattr(native_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _prepared({"model": "/x.gguf", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.terminated is True
    assert proc.killed is False
    assert m.process is None


def test_stop_kills_on_timeout(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    proc.wait_raises = subprocess.TimeoutExpired(cmd="llama-server", timeout=_stop())
    monkeypatch.setattr(native_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _prepared({"model": "/x.gguf", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.terminated is True
    assert proc.killed is True


def _stop() -> float:
    return native_mod.STOP_TIMEOUT


# --- runtime handle (cross-process teardown identity) ---
def test_runtime_handle_records_pid_and_create_time() -> None:
    import os

    import psutil

    m = _manager()
    m.process = cast(
        "subprocess.Popen[bytes]", FakeProc(pid=os.getpid(), poll_value=None)
    )  # a real, live PID
    handle = m.runtime_handle()
    assert handle is not None
    assert handle["kind"] == "native"
    assert handle["pid"] == os.getpid()
    assert handle["create_time"] == psutil.Process(os.getpid()).create_time()


def test_runtime_handle_omits_create_time_when_process_vanished(monkeypatch) -> None:
    import psutil

    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(pid=4242, poll_value=None))

    def raise_no_such(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(native_mod.psutil, "Process", raise_no_such)
    handle = m.runtime_handle()
    assert handle == {"kind": "native", "pid": 4242}  # PID still recorded


def test_runtime_handle_none_when_not_running() -> None:
    assert _manager().runtime_handle() is None


# --- metrics ---
def test_get_metrics_stopped_when_no_process() -> None:
    assert _manager().get_metrics() == {"status": "stopped"}


def test_get_metrics_running(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(pid=4242, poll_value=None))

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            assert pid == 4242

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    assert m.get_metrics() == {
        "memory_bytes": 14500 * 1024**2,
        "status": "running",
    }


def test_get_metrics_uses_phys_footprint_when_available(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(pid=4242, poll_value=None))

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            pass

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    monkeypatch.setattr(native_mod, "macos_phys_footprint", lambda pid: 999 * 1024**2)
    metrics = m.get_metrics()
    assert metrics["memory_bytes"] == 999 * 1024**2  # footprint wins over the rss stub


def test_get_metrics_falls_back_to_rss_when_footprint_unavailable(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(pid=4242, poll_value=None))

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            pass

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    monkeypatch.setattr(native_mod, "macos_phys_footprint", lambda pid: None)
    assert m.get_metrics()["memory_bytes"] == 14500 * 1024**2


# --- Phase 7: resource estimation ---
def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check={"type": "http", "endpoint": "/health", "port": 11435},
        config={"model": "/x.gguf"},
        memory_gb=40,
    )
    assert LlamaCppManager(entry).estimated_memory_bytes() == 40 * 10**9


def test_estimated_memory_from_model_file_plus_kv(tmp_path, sparse_file) -> None:
    model = tmp_path / "m.gguf"
    sparse_file(model, 2 * 1024**3)  # 2 GiB (logical size only)
    m = _manager(
        {"model": str(model), "context_size": 4096, "kv_bytes_per_token": 1024**2}
    )
    # 2 GiB model + 4096 tokens * 1 MiB = 4 GiB KV -> exact byte sum
    assert m.estimated_memory_bytes() == 2 * 1024**3 + 4096 * 1024**2


def test_estimated_memory_repo_id_is_kv_only(monkeypatch) -> None:
    # Offline + uncached: weight bytes unknown → only the KV-cache term counts.
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)
    m = _manager(
        {"model": "org/repo", "context_size": 4096, "kv_bytes_per_token": 1024**2}
    )
    assert m.estimated_memory_bytes() == m.estimated_kv_cache_bytes()


def test_estimated_memory_repo_id_from_metadata(monkeypatch) -> None:
    info = _repo_info(
        "org/repo",
        [("model.Q4_K_M.gguf", 2 * 1024**3), ("model.Q8_0.gguf", 4 * 1024**3)],
    )
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    # Bare repo, two quants: prefers Q4_K_M (2 GiB) + no KV (context unset) → exact 2 GiB.
    m = _manager({"model": "org/repo"})
    assert m.estimated_memory_bytes() == 2 * 1024**3


def test_estimated_memory_excludes_draft_weights(tmp_path, sparse_file) -> None:
    # llama_cpp has no second-GGUF speculative decoding (supports_draft_model =
    # False) — a configured draft_model's weights never load, so they must not
    # inflate admission control's estimate even though prepare_environment()
    # hard-errors on this config before boot.
    model = tmp_path / "m.gguf"
    sparse_file(model, 2 * 1024**3)  # 2 GiB
    draft = tmp_path / "d.gguf"
    sparse_file(draft, 1 * 1024**3)  # 1 GiB
    m = _manager({"model": str(model), "draft_model": str(draft)})
    assert m.estimated_memory_bytes() == 2 * 1024**3


def test_per_slot_context_divides_across_slots() -> None:
    m = _manager({"model": "/x.gguf", "context_size": 32768, "max_parallel": 4})
    assert m.per_slot_context() == 8192


def test_per_slot_context_none_without_context() -> None:
    assert _manager({"model": "/x.gguf"}).per_slot_context() is None


# --- Phase 7: prompt caching ---
def _caching_manager(caching: dict, config: dict | None = None) -> LlamaCppManager:
    entry = ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check={"type": "http", "endpoint": "/health", "port": 11435},
        config=config or {"model": "/models/x.gguf"},
        policy={"prompt_caching": caching},
    )
    return LlamaCppManager(entry)


def test_prompt_caching_kv_cache_type_in_engine_kwargs_when_enabled() -> None:
    m = _caching_manager({"enabled": True, "cache_path": "/tmp/c", "kv_cache_type": "q8_0"})
    assert m.engine_kwargs()["kv_cache_type"] == "q8_0"


def test_prompt_caching_no_kv_cache_type_when_disabled() -> None:
    m = _caching_manager({"enabled": False, "cache_path": "/tmp/c"})
    assert "kv_cache_type" not in m.engine_kwargs()


def test_prompt_caching_cache_path_inert_warning(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    cache = tmp_path / "cache"
    m = _caching_manager({"enabled": True, "cache_path": str(cache)}, {"model": str(model)})
    with caplog.at_level("WARNING", logger="sovereign"):
        m.prepare_environment()
    assert any("inert" in r.message for r in caplog.records)


def test_validate_prompt_caching_creates_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    cache = tmp_path / "cache" / "llama"
    m = _caching_manager(
        {"enabled": True, "cache_path": str(cache)}, {"model": str(model)}
    )
    m.prepare_environment()
    assert cache.is_dir()


def test_validate_prompt_caching_rejects_bad_kv_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager(
        {"enabled": True, "cache_path": str(tmp_path / "c"), "kv_cache_type": "bogus"},
        {"model": str(model)},
    )
    with pytest.raises(ValueError, match="invalid kv_cache_type 'bogus'"):
        m.prepare_environment()


def test_validate_prompt_caching_requires_cache_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager({"enabled": True}, {"model": str(model)})
    with pytest.raises(ValueError, match="no cache_path is set"):
        m.prepare_environment()
