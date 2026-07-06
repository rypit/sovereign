"""Phase 4: llama_cpp manager — mocked unit tests + Protocol/registry checks.

The real llama-server binary and a GGUF model are not required here; the
subprocess, HTTP probe, and psutil calls are all mocked. A real-process
end-to-end check lives in the Phase 4 smoke test.
"""

from __future__ import annotations

import contextlib
import subprocess

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.llama_cpp import manager as mgr_mod
from sovereign.services.llama_cpp.manager import LlamaCppManager


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check=(
            {"type": "http", "endpoint": "/health", "port": 11435}
            if with_health
            else None
        ),
        config=config or {"model_path": "/models/x.gguf"},
    )


def _manager(config: dict | None = None) -> LlamaCppManager:
    return LlamaCppManager(_entry(config))


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
def test_get_start_args_full_flag_mapping() -> None:
    m = _manager(
        {
            "model_path": "/models/llama3-70b.gguf",
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
        ("--api-key", "secret"),
    ]:
        assert args[args.index(flag) + 1] == value


def test_get_start_args_minimal_omits_optional_flags() -> None:
    args = _manager().get_start_args()
    for flag in ["-ngl", "-t", "-c", "-np", "--api-key"]:
        assert flag not in args


def test_get_start_args_expands_home(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    args = _manager({"model_path": "~/models/x.gguf"}).get_start_args()
    assert "/home/tester/models/x.gguf" in args


# --- prepare_environment ---
def test_prepare_environment_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    with pytest.raises(FileNotFoundError, match="model for 'llama_heavy_v1' not found"):
        _manager({"model_path": "/nope/missing.gguf"}).prepare_environment()


def test_prepare_environment_ok(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    _manager({"model_path": str(model)}).prepare_environment()  # must not raise


def test_prepare_environment_missing_binary(tmp_path, monkeypatch) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: None)
    with pytest.raises(FileNotFoundError, match="binary 'llama-server' not found"):
        _manager({"model_path": str(model)}).prepare_environment()


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

    monkeypatch.setattr(mgr_mod.urllib.request, "urlopen", lambda url, timeout=None: FakeResp())
    assert m.is_healthy() is True


def test_is_healthy_false_on_connection_error(monkeypatch) -> None:
    m = _manager()
    m.process = FakeProc(poll_value=None)

    def boom(url, timeout=None):
        raise mgr_mod.urllib.error.URLError("refused")

    monkeypatch.setattr(mgr_mod.urllib.request, "urlopen", boom)
    assert m.is_healthy() is False


# --- lifecycle ---
def test_start_launches_process_with_argv(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(mgr_mod.subprocess, "Popen", fake_popen)
    m = _manager({"model_path": "/models/x.gguf", "log_dir": str(tmp_path / "logs")})
    m.start()
    assert captured["args"] == m.get_start_args()
    assert m.process is not None
    assert (tmp_path / "logs" / "llama_heavy_v1.log").exists()
    m.stop()


def test_stop_terminates_running_process(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    monkeypatch.setattr(mgr_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _manager({"model_path": "/x.gguf", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.terminated is True
    assert proc.killed is False
    assert m.process is None


def test_stop_kills_on_timeout(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    proc.wait_raises = subprocess.TimeoutExpired(cmd="llama-server", timeout=_stop())
    monkeypatch.setattr(mgr_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _manager({"model_path": "/x.gguf", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.terminated is True
    assert proc.killed is True


def _stop() -> float:
    return mgr_mod._STOP_TIMEOUT


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

    monkeypatch.setattr(mgr_mod.psutil, "Process", FakePsProc)
    assert m.get_metrics() == {"memory_mb": 14500.0, "cpu_percent": 12.4, "status": "running"}


# --- Phase 7: resource estimation ---
def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check={"type": "http", "endpoint": "/health", "port": 11435},
        config={"model_path": "/x.gguf"},
        memory_gb=40,
    )
    assert LlamaCppManager(entry).estimated_memory_gb() == 40.0


def test_estimated_memory_from_model_file_plus_kv(tmp_path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * (2 * 1024**3))  # 2 GiB on disk
    m = _manager(
        {"model_path": str(model), "context_size": 4096, "kv_bytes_per_token": 1024**2}
    )
    # 2 GiB model + 4096 tokens * 1 MiB = 4 GiB KV -> ~6.0 GB
    assert m.estimated_memory_gb() == pytest.approx(6.0, abs=0.05)


def test_per_slot_context_divides_across_slots() -> None:
    m = _manager({"model_path": "/x.gguf", "context_size": 32768, "max_parallel": 4})
    assert m.per_slot_context() == 8192


def test_per_slot_context_none_without_context() -> None:
    assert _manager({"model_path": "/x.gguf"}).per_slot_context() is None


# --- Phase 7: prompt caching ---
def _caching_manager(caching: dict, config: dict | None = None) -> LlamaCppManager:
    entry = ServiceEntry(
        name="llama_heavy_v1",
        base_type="llama_cpp",
        health_check={"type": "http", "endpoint": "/health", "port": 11435},
        config=config or {"model_path": "/models/x.gguf"},
        policy={"prompt_caching": caching},
    )
    return LlamaCppManager(entry)


def test_prompt_caching_flags_added_when_enabled() -> None:
    m = _caching_manager({"enabled": True, "cache_path": "/tmp/c", "kv_cache_type": "q8_0"})
    args = m.get_start_args()
    assert args[args.index("--slot-save-path") + 1] == "/tmp/c"
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q8_0"


def test_prompt_caching_no_flags_when_disabled() -> None:
    m = _caching_manager({"enabled": False, "cache_path": "/tmp/c"})
    assert "--slot-save-path" not in m.get_start_args()


def test_validate_prompt_caching_creates_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    cache = tmp_path / "cache" / "llama"
    m = _caching_manager(
        {"enabled": True, "cache_path": str(cache)}, {"model_path": str(model)}
    )
    m.prepare_environment()
    assert cache.is_dir()


def test_validate_prompt_caching_rejects_bad_kv_type(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager(
        {"enabled": True, "cache_path": str(tmp_path / "c"), "kv_cache_type": "bogus"},
        {"model_path": str(model)},
    )
    with pytest.raises(ValueError, match="invalid kv_cache_type 'bogus'"):
        m.prepare_environment()


def test_validate_prompt_caching_requires_cache_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/opt/homebrew/bin/llama-server")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    m = _caching_manager({"enabled": True}, {"model_path": str(model)})
    with pytest.raises(ValueError, match="no cache_path is set"):
        m.prepare_environment()
