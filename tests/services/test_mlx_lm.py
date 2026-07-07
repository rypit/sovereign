"""Phase 11: mlx_lm manager — mocked unit tests + Protocol/registry checks.

The real mlx_lm.server binary and an MLX model are not required here; the
subprocess, HTTP probe, and psutil calls are mocked (via the shared base_native
module, where the process/health/metrics lifecycle now lives).
"""

from __future__ import annotations

import contextlib
import subprocess
import threading

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core import base_native as native_mod
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.mlx_lm.manager import MlxLmManager


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="mlx_fast",
        base_type="mlx_lm",
        health_check=(
            {"type": "http", "endpoint": "/health", "port": 8080} if with_health else None
        ),
        config=config or {"model": "mlx-community/Llama-3.2-1B-Instruct-4bit"},
    )


def _manager(config: dict | None = None) -> MlxLmManager:
    return MlxLmManager(_entry(config))


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
    assert get_service_manager("mlx_lm") is MlxLmManager


def test_requires_health_check() -> None:
    with pytest.raises(ValueError, match="requires a health_check"):
        MlxLmManager(_entry(with_health=False))


def test_port_and_path_from_health_check() -> None:
    m = _manager()
    assert m.port == 8080
    assert m.health_path == "/health"


# --- flag generation ---
def test_get_start_args_full_flag_mapping() -> None:
    m = _manager(
        {
            "model": "mlx-community/foo-4bit",
            "max_tokens": 1024,
            "temp": 0.7,
            "top_p": 0.95,
            "decode_concurrency": 4,
            "prompt_cache_size": 2048,
            "prompt_cache_bytes": 8 * 1024**3,
            "adapter_path": "/adapters/a",
            "draft_model": "mlx-community/draft-4bit",
            "num_draft_tokens": 5,
            "trust_remote_code": True,
        }
    )
    args = m.get_start_args()
    assert args[0] == "mlx_lm.server"
    assert args[args.index("--model") + 1] == "mlx-community/foo-4bit"
    for flag, value in [
        ("--host", "127.0.0.1"),
        ("--port", "8080"),
        ("--max-tokens", "1024"),
        ("--temp", "0.7"),
        ("--top-p", "0.95"),
        ("--decode-concurrency", "4"),
        ("--prompt-cache-size", "2048"),
        ("--prompt-cache-bytes", str(8 * 1024**3)),
        ("--adapter-path", "/adapters/a"),
        ("--draft-model", "mlx-community/draft-4bit"),
        ("--num-draft-tokens", "5"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--trust-remote-code" in args


def test_get_start_args_minimal_omits_optional_flags() -> None:
    args = _manager().get_start_args()
    for flag in [
        "--max-tokens", "--temp", "--top-p", "--decode-concurrency",
        "--prompt-cache-bytes", "--adapter-path", "--draft-model", "--num-draft-tokens",
    ]:
        assert flag not in args
    assert "--trust-remote-code" not in args


def test_prompt_cache_bytes_flag() -> None:
    m = _manager({"model": "mlx-community/foo", "prompt_cache_bytes": 4 * 1024**3})
    args = m.get_start_args()
    assert args[args.index("--prompt-cache-bytes") + 1] == str(4 * 1024**3)


def test_draft_model_flag(tmp_path) -> None:
    draft = tmp_path / "draft"
    draft.mkdir()
    m = _manager({"model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                  "draft_model": str(draft)})
    args = m.get_start_args()
    assert args[args.index("--draft-model") + 1] == str(draft)


def test_num_draft_tokens_flag() -> None:
    m = _manager({"model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                  "num_draft_tokens": 5})
    args = m.get_start_args()
    assert args[args.index("--num-draft-tokens") + 1] == "5"


def test_draft_flags_absent_when_unset() -> None:
    args = _manager().get_start_args()
    assert "--draft-model" not in args
    assert "--num-draft-tokens" not in args


def test_get_start_args_repo_id_passes_through() -> None:
    args = _manager({"model": "mlx-community/foo-4bit"}).get_start_args()
    assert "mlx-community/foo-4bit" in args  # not mangled into a path


def test_get_start_args_expands_home(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    args = _manager({"model": "~/models/mlx-foo"}).get_start_args()
    assert "/home/tester/models/mlx-foo" in args


# --- prepare_environment ---
def test_prepare_environment_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: None)
    with pytest.raises(FileNotFoundError, match="mlx_lm.server"):
        _manager().prepare_environment()


def test_prepare_environment_repo_id_ok(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    # A repo id that isn't local must NOT raise (mlx downloads it on start).
    _manager({"model": "mlx-community/foo-4bit"}).prepare_environment()


def test_prepare_environment_local_path_missing(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    with pytest.raises(FileNotFoundError, match="model for 'mlx_fast' not found"):
        _manager({"model": "/nope/missing-mlx-model"}).prepare_environment()


def test_prepare_environment_local_path_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    _manager({"model": str(model_dir)}).prepare_environment()  # must not raise


def test_prepare_environment_missing_local_draft_raises(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    with pytest.raises(FileNotFoundError, match="draft_model"):
        _manager(
            {"model": "mlx-community/foo-4bit", "draft_model": "/nope/missing-draft"}
        ).prepare_environment()


def test_prepare_environment_hf_draft_ok(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    _manager(
        {"model": "mlx-community/foo-4bit", "draft_model": "mlx-community/draft-4bit"}
    ).prepare_environment()  # must not raise


def test_prepare_environment_missing_adapter_raises(monkeypatch) -> None:
    monkeypatch.setattr(native_mod.shutil, "which", lambda _b: "/usr/bin/mlx_lm.server")
    with pytest.raises(FileNotFoundError, match="adapter_path"):
        _manager(
            {"model": "mlx-community/foo-4bit", "adapter_path": "/nope/missing-adapter"}
        ).prepare_environment()


# --- resource estimation ---
def test_estimated_memory_override() -> None:
    entry = ServiceEntry(
        name="mlx_fast",
        base_type="mlx_lm",
        health_check={"type": "http", "endpoint": "/health", "port": 8080},
        config={"model": "mlx-community/foo"},
        memory_gb=6,
    )
    assert MlxLmManager(entry).estimated_memory_gb() == 6.0


def test_estimated_memory_from_local_dir(tmp_path) -> None:
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x" * (2 * 1024**3))  # 2 GiB
    m = _manager({"model": str(model_dir)})
    assert m.estimated_memory_gb() == pytest.approx(2.0, abs=0.05)


def test_estimated_memory_repo_id_unknown() -> None:
    assert _manager({"model": "mlx-community/foo-4bit"}).estimated_memory_gb() == 0.0


def test_estimated_memory_includes_prompt_cache_bytes(tmp_path) -> None:
    model_dir = tmp_path / "main"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x" * (4 * 1024**3))  # 4 GiB
    m = _manager({"model": str(model_dir), "prompt_cache_bytes": 2 * 1024**3})
    assert m.estimated_memory_gb() == pytest.approx(6.0, abs=0.05)


def test_estimated_memory_includes_local_draft_model(tmp_path) -> None:
    model_dir = tmp_path / "main"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x" * (2 * 1024**3))  # 2 GiB
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "weights.safetensors").write_bytes(b"x" * (1 * 1024**3))  # 1 GiB
    m = _manager({"model": str(model_dir), "draft_model": str(draft_dir)})
    assert m.estimated_memory_gb() == pytest.approx(3.0, abs=0.05)


def test_estimated_memory_repo_id_draft_contributes_zero(tmp_path) -> None:
    model_dir = tmp_path / "main"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"x" * (2 * 1024**3))  # 2 GiB
    m = _manager({"model": str(model_dir), "draft_model": "mlx-community/draft-4bit"})
    assert m.estimated_memory_gb() == pytest.approx(2.0, abs=0.05)


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
        return FakeProc()

    monkeypatch.setattr(native_mod.subprocess, "Popen", fake_popen)
    m = _manager({"model": "mlx-community/foo", "log_dir": str(tmp_path / "logs")})
    m.start()
    assert captured["args"] == m.get_start_args()
    assert (tmp_path / "logs" / "mlx_fast.log").exists()
    m.stop()


def test_stop_terminates_running_process(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    monkeypatch.setattr(native_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _manager({"model": "mlx-community/foo", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.terminated is True
    assert proc.killed is False
    assert m.process is None


def test_stop_kills_on_timeout(tmp_path, monkeypatch) -> None:
    proc = FakeProc(poll_value=None)
    proc.wait_raises = subprocess.TimeoutExpired(cmd="mlx_lm.server", timeout=10)
    monkeypatch.setattr(native_mod.subprocess, "Popen", lambda *a, **k: proc)
    m = _manager({"model": "mlx-community/foo", "log_dir": str(tmp_path)})
    m.start()
    m.stop()
    assert proc.killed is True


# --- log tailer ---
def test_tail_log_reports_download_progress(tmp_path) -> None:
    log = tmp_path / "mlx_fast.log"
    log.write_text(
        "Starting httpd...\n"
        "Fetching 8 files:  38%|███▊      | 3/8 [00:03<00:06,  1.22s/it]\n"
    )
    m = _manager()
    stop = threading.Event()
    t = threading.Thread(target=m._tail_log_for_activity, args=(log, stop), daemon=True)
    t.start()
    t.join(timeout=1.0)
    stop.set()
    assert m.activity == "downloading model: 3/8 files (38%)"


def test_tail_log_clears_activity_at_100_percent(tmp_path) -> None:
    log = tmp_path / "mlx_fast.log"
    log.write_text(
        "Fetching 8 files: 100%|██████████| 8/8 [00:03<00:00,  2.35it/s]\n"
    )
    m = _manager()
    stop = threading.Event()
    t = threading.Thread(target=m._tail_log_for_activity, args=(log, stop), daemon=True)
    t.start()
    t.join(timeout=1.0)
    stop.set()
    assert m.activity == ""


def test_start_launches_tailer_thread(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_mod.subprocess, "Popen", lambda *a, **k: FakeProc())
    m = _manager({"model": "mlx-community/foo", "log_dir": str(tmp_path / "logs")})
    m.start()
    assert m._tailer is not None
    assert m._tailer.is_alive()
    m._tailer_stop.set()
    m._tailer.join(timeout=1.0)
    m.stop()


# --- metrics ---
def test_get_metrics_stopped_when_no_process() -> None:
    assert _manager().get_metrics() == {"status": "stopped"}


def test_get_metrics_running(monkeypatch) -> None:
    m = _manager()
    m.process = FakeProc(pid=4242, poll_value=None)

    class FakeMem:
        rss = 6000 * 1024**2

    class FakePsProc:
        def __init__(self, pid):
            assert pid == 4242

        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return FakeMem()

        def cpu_percent(self, interval=None):
            return 9.9

    monkeypatch.setattr(native_mod.psutil, "Process", FakePsProc)
    assert m.get_metrics() == {"memory_mb": 6000.0, "cpu_percent": 9.9, "status": "running"}
