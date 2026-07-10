"""Phase 4: llama_cpp manager — mocked unit tests + Protocol/registry checks.

The real llama-server binary and a GGUF model are not required here; the
subprocess, HTTP probe, and psutil calls are all mocked (via the shared
inference.base module, where the process/health/metrics lifecycle now lives).
"""

from __future__ import annotations

import contextlib
import subprocess

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.inference import base as native_mod
from sovereign.services.inference import hf as models_mod
from sovereign.services.inference.hf import RepoInfo
from sovereign.services.inference.llama_cpp.manager import LlamaCppManager


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


# --- flag generation ---
def test_get_start_args_before_prepare_raises() -> None:
    with pytest.raises(RuntimeError, match="prepare_model"):
        _manager({"model": "/models/x.gguf"}).get_start_args()


def test_prepare_model_sets_paths() -> None:
    m = _prepared({"model": "org/repo:Q4_K_M", "draft_model": "org/tiny-draft"})
    assert str(m.model_path) == "org/repo:Q4_K_M"
    assert str(m.draft_model_path) == "org/tiny-draft"


def test_get_start_args_full_flag_mapping() -> None:
    m = _prepared(
        {
            "model": "/models/llama3-70b.gguf",
            "gpu_layers": 48,
            "threads": 8,
            "context_size": 32768,
            "max_parallel": 4,
            "api_key": "secret",
        }
    )
    args = m.get_start_args()
    assert args[0] == "llama-server"
    assert "--model" in args and "/models/llama3-70b.gguf" in args
    for flag, value in [
        ("--host", "127.0.0.1"),
        ("--port", "11435"),
        ("-ngl", "48"),
        ("-t", "8"),
        ("-c", "32768"),
        ("-np", "4"),
    ]:
        assert args[args.index(flag) + 1] == value
    # The API key must never be on the ps-visible command line.
    assert "--api-key" not in args
    assert "secret" not in args


def test_get_start_args_minimal_omits_optional_flags() -> None:
    args = _prepared().get_start_args()
    for flag in ["-ngl", "-t", "-c", "-np", "--api-key"]:
        assert flag not in args


# --- API key via environment (never argv) ---
def test_start_env_carries_api_key() -> None:
    m = _manager({"model": "/models/x.gguf", "api_key": "secret"})
    assert m.start_env() == {"LLAMA_API_KEY": "secret"}


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
    assert captured["env"]["LLAMA_API_KEY"] == "secret"
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


def test_get_start_args_expands_home(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    args = _prepared({"model": "~/models/x.gguf"}).get_start_args()
    assert "/home/tester/models/x.gguf" in args


def test_get_start_args_hf_repo_resolves_to_local_path() -> None:
    args = _prepared({"model": "ggml-org/gemma-3-1b-it-GGUF"}).get_start_args()
    assert args[args.index("--model") + 1] == "ggml-org/gemma-3-1b-it-GGUF"  # resolved path


def test_argv_never_contains_hf_repo() -> None:
    # HF repo ids are pre-downloaded and launched from the resolved path — the
    # server never sees --hf-repo / --hf-repo-draft.
    args = _prepared(
        {"model": "ggml-org/gemma-3-1b-it-GGUF", "draft_model": "org/tiny-draft"}
    ).get_start_args()
    assert "--hf-repo" not in args
    assert "--hf-repo-draft" not in args


def test_get_start_args_local_draft_model(tmp_path) -> None:
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"x")
    args = _prepared({"model": "/models/x.gguf", "draft_model": str(draft)}).get_start_args()
    assert args[args.index("--model-draft") + 1] == str(draft)


def test_get_start_args_hf_draft_model() -> None:
    args = _prepared(
        {"model": "/models/x.gguf", "draft_model": "org/tiny-draft"}
    ).get_start_args()
    assert args[args.index("--model-draft") + 1] == "org/tiny-draft"  # resolved path


def test_get_start_args_num_draft_tokens() -> None:
    args = _prepared({"model": "/models/x.gguf", "num_draft_tokens": 2}).get_start_args()
    assert args[args.index("--draft-max") + 1] == "2"


def test_get_start_args_draft_flags_absent_when_unset() -> None:
    args = _prepared().get_start_args()
    assert "--model-draft" not in args
    assert "--hf-repo-draft" not in args
    assert "--draft-max" not in args


# --- served_model_name / api_model_name (harness+bench wiring) ---
def test_served_model_name_emits_alias_flag() -> None:
    args = _prepared(
        {"model": "/models/x.gguf", "served_model_name": "llama3-70b"}
    ).get_start_args()
    assert args[args.index("--alias") + 1] == "llama3-70b"


def test_alias_flag_absent_when_unset() -> None:
    args = _prepared().get_start_args()
    assert "--alias" not in args


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
    assert LlamaCppManager.provisioning_binary == "llama-server"
    brewfile = LlamaCppManager.provisioning_brewfile()
    assert brewfile is not None
    assert 'brew "llama.cpp"' in brewfile.read_text()


def test_prepare_environment_provisions_first(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    order: list[str] = []
    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: order.append("provision"))
    )
    monkeypatch.setattr(
        native_mod.shutil,
        "which",
        lambda _b: order.append("which") or "/opt/homebrew/bin/llama-server",
    )
    m = _manager({"model": "org/some-hf-repo"})  # repo id: no local file check
    m.prepare_environment()
    assert order[0] == "provision"  # install the toolchain before validating it


# --- prepare_environment ---
def test_prepare_environment_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    with pytest.raises(FileNotFoundError, match="model for 'llama_heavy_v1' not found"):
        _manager({"model": "/nope/missing.gguf"}).prepare_environment()


def test_prepare_environment_ok(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    _manager({"model": str(model)}).prepare_environment()  # must not raise


def test_prepare_environment_missing_binary(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: None)
    with pytest.raises(FileNotFoundError, match="binary 'llama-server' not found"):
        _manager({"model": str(model)}).prepare_environment()


def test_prepare_environment_repo_id_ok(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    # A repo id that isn't local must NOT raise (llama-server downloads it on start).
    _manager({"model": "ggml-org/gemma-3-1b-it-GGUF"}).prepare_environment()


def test_prepare_environment_missing_local_draft_raises(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    with pytest.raises(FileNotFoundError, match="draft_model"):
        _manager(
            {"model": str(model), "draft_model": "/nope/missing-draft.gguf"}
        ).prepare_environment()


def test_prepare_environment_hf_draft_ok(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    _manager(
        {"model": str(model), "draft_model": "org/tiny-draft"}
    ).prepare_environment()  # must not raise


# --- health ---
def test_is_healthy_false_when_no_process() -> None:
    assert _manager().is_healthy() is False


def test_is_healthy_false_when_process_exited() -> None:
    m = _manager()
    m.process = FakeProc(poll_value=0)
    assert m.is_healthy() is False


def test_is_healthy_true_on_http_200(monkeypatch) -> None:
    m = _manager()
    m.process = FakeProc(poll_value=None)

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
    m.process = FakeProc(poll_value=None)

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
    m.process = FakeProc(pid=os.getpid(), poll_value=None)  # a real, live PID
    handle = m.runtime_handle()
    assert handle["kind"] == "native"
    assert handle["pid"] == os.getpid()
    assert handle["create_time"] == psutil.Process(os.getpid()).create_time()


def test_runtime_handle_omits_create_time_when_process_vanished(monkeypatch) -> None:
    import psutil

    m = _manager()
    m.process = FakeProc(pid=4242, poll_value=None)

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
    m.process = FakeProc(pid=4242, poll_value=None)

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            assert pid == 4242

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

        def cpu_percent(self, interval=None):
            return 12.4

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    assert m.get_metrics() == {"memory_mb": 14500.0, "cpu_percent": 12.4, "status": "running"}


def test_get_metrics_uses_phys_footprint_when_available(monkeypatch) -> None:
    m = _manager()
    m.process = FakeProc(pid=4242, poll_value=None)

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            pass

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

        def cpu_percent(self, interval=None):
            return 12.4

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    monkeypatch.setattr(native_mod, "macos_phys_footprint", lambda pid: 999 * 1024**2)
    metrics = m.get_metrics()
    assert metrics["memory_mb"] == 999.0  # footprint wins over the rss stub
    assert metrics["cpu_percent"] == 12.4


def test_get_metrics_falls_back_to_rss_when_footprint_unavailable(monkeypatch) -> None:
    m = _manager()
    m.process = FakeProc(pid=4242, poll_value=None)

    class FakeMem:
        rss = 14500 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            pass

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

        def cpu_percent(self, interval=None):
            return 12.4

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    monkeypatch.setattr(native_mod, "macos_phys_footprint", lambda pid: None)
    assert m.get_metrics()["memory_mb"] == 14500.0


# --- Phase 7: resource estimation ---
def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check={"type": "http", "endpoint": "/health", "port": 11435},
        config={"model": "/x.gguf"},
        memory_gb=40,
    )
    assert LlamaCppManager(entry).estimated_memory_gb() == 40.0


def test_estimated_memory_from_model_file_plus_kv(tmp_path, sparse_file) -> None:
    model = tmp_path / "m.gguf"
    sparse_file(model, 2 * 1024**3)  # 2 GiB (logical size only)
    m = _manager(
        {"model": str(model), "context_size": 4096, "kv_bytes_per_token": 1024**2}
    )
    # 2 GiB model + 4096 tokens * 1 MiB = 4 GiB KV -> ~6.0 GB
    assert m.estimated_memory_gb() == pytest.approx(6.0, abs=0.05)


def test_estimated_memory_repo_id_is_kv_only(monkeypatch) -> None:
    # Offline + uncached: weight bytes unknown → only the KV-cache term counts.
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)
    m = _manager(
        {"model": "org/repo", "context_size": 4096, "kv_bytes_per_token": 1024**2}
    )
    assert m.estimated_memory_gb() == pytest.approx(m.estimated_kv_cache_gb(), abs=0.001)


def test_estimated_memory_repo_id_from_metadata(monkeypatch) -> None:
    info = _repo_info(
        "org/repo",
        [("model.Q4_K_M.gguf", 2 * 1024**3), ("model.Q8_0.gguf", 4 * 1024**3)],
    )
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    # Bare repo, two quants: prefers Q4_K_M (2 GiB) + no KV (context unset) → ~2.0 GB.
    m = _manager({"model": "org/repo"})
    assert m.estimated_memory_gb() == pytest.approx(2.0, abs=0.05)


def test_estimated_memory_includes_local_draft(tmp_path, sparse_file) -> None:
    model = tmp_path / "m.gguf"
    sparse_file(model, 2 * 1024**3)  # 2 GiB
    draft = tmp_path / "d.gguf"
    sparse_file(draft, 1 * 1024**3)  # 1 GiB
    m = _manager({"model": str(model), "draft_model": str(draft)})
    # 2 GiB + 1 GiB model bytes + 0 KV (no context_size set)
    assert m.estimated_memory_gb() == pytest.approx(3.0, abs=0.05)


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


def test_prompt_caching_flags_added_when_enabled() -> None:
    m = _caching_manager({"enabled": True, "cache_path": "/tmp/c", "kv_cache_type": "q8_0"})
    m.prepare_model()
    args = m.get_start_args()
    assert args[args.index("--slot-save-path") + 1] == "/tmp/c"
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q8_0"


def test_prompt_caching_no_flags_when_disabled() -> None:
    m = _caching_manager({"enabled": False, "cache_path": "/tmp/c"})
    m.prepare_model()
    assert "--slot-save-path" not in m.get_start_args()


def test_validate_prompt_caching_creates_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    cache = tmp_path / "cache" / "llama"
    m = _caching_manager(
        {"enabled": True, "cache_path": str(cache)}, {"model": str(model)}
    )
    m.prepare_environment()
    assert cache.is_dir()


def test_validate_prompt_caching_rejects_bad_kv_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager(
        {"enabled": True, "cache_path": str(tmp_path / "c"), "kv_cache_type": "bogus"},
        {"model": str(model)},
    )
    with pytest.raises(ValueError, match="invalid kv_cache_type 'bogus'"):
        m.prepare_environment()


def test_validate_prompt_caching_requires_cache_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager({"enabled": True}, {"model": str(model)})
    with pytest.raises(ValueError, match="no cache_path is set"):
        m.prepare_environment()
