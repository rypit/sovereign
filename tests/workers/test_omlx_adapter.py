"""Tests for the omlx worker adapter (ADR 0007 subprocess pattern: ``omlx
serve`` child, no in-process binding, no telemetry translator).

``build_server_argv`` and ``prepare_model_dir`` are pure/near-pure and
unit-testable with no ``omlx`` binary installed. ``run()`` is exercised by
patching ``subprocess.Popen`` (a fake child) and ``urllib.request.urlopen``
(health).
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, cast

import pytest

from sovereign.workers import omlx_adapter as adapter
from sovereign.workers.omlx_adapter import (
    build_server_argv,
    prepare_model_dir,
    supervise,
)
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient
from sovereign.workers.worker_config import WorkerConfig


# --- build_server_argv ---
def test_build_server_argv_basics():
    argv = build_server_argv({}, model_dir="/state/omlx/svc/models", host="127.0.0.1", port=18000)
    assert argv[:7] == [
        "serve",
        "--model-dir",
        "/state/omlx/svc/models",
        "--host",
        "127.0.0.1",
        "--port",
        "18000",
    ]


def test_build_server_argv_maps_renamed_kwargs():
    argv = build_server_argv(
        {
            "max_concurrent_requests": 4,
            "memory_guard_gb": 24.5,
            "paged_ssd_cache_dir": "/cache/ssd",
        },
        model_dir="/m",
        host="h",
        port=1,
    )
    assert argv[argv.index("--max-concurrent-requests") + 1] == "4"
    assert argv[argv.index("--memory-guard-gb") + 1] == "24.5"
    assert argv[argv.index("--paged-ssd-cache-dir") + 1] == "/cache/ssd"


def test_build_server_argv_renders_gb_sizes_as_size_strings():
    argv = build_server_argv(
        {"paged_ssd_cache_max_gb": 50, "hot_cache_gb": 8}, model_dir="/m", host="h", port=1
    )
    assert argv[argv.index("--paged-ssd-cache-max-size") + 1] == "50GB"
    assert argv[argv.index("--hot-cache-max-size") + 1] == "8GB"


def test_build_server_argv_consumes_model_dir_and_name_keys():
    argv = build_server_argv(
        {"model_dir": "/state/models", "model_name": "org/m"}, model_dir="/m", host="h", port=1
    )
    assert "--model-name" not in argv
    assert argv[argv.index("--model-dir") + 1] == "/m"  # the prepared dir, not the kwarg


def test_build_server_argv_passthrough_escape_hatch():
    argv = build_server_argv({"embedding_batch_size": 16}, model_dir="/m", host="h", port=1)
    assert argv[argv.index("--embedding-batch-size") + 1] == "16"


def test_build_server_argv_bool_kwargs_become_bare_flags():
    argv = build_server_argv(
        {"no_cache": True, "verbose": False}, model_dir="/m", host="h", port=1
    )
    assert "--no-cache" in argv
    assert "--verbose" not in argv


def test_build_server_argv_api_key_never_included():
    argv = build_server_argv({}, model_dir="/m", host="h", port=1)
    assert "--api-key" not in argv


# --- prepare_model_dir: the single-model symlink layout ---
def test_prepare_model_dir_creates_nested_symlink(tmp_path):
    snapshot = tmp_path / "hf" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    root = tmp_path / "models"

    returned = prepare_model_dir(str(root), "mlx-community/m-4bit", str(snapshot))

    assert returned == str(root)
    link = root / "mlx-community" / "m-4bit"
    assert link.is_symlink()
    assert link.resolve() == snapshot.resolve()


def test_prepare_model_dir_idempotent(tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    root = tmp_path / "models"
    prepare_model_dir(str(root), "org/m", str(snapshot))
    prepare_model_dir(str(root), "org/m", str(snapshot))  # must not raise
    assert (root / "org" / "m").resolve() == snapshot.resolve()


def test_prepare_model_dir_repoints_stale_symlink(tmp_path):
    old = tmp_path / "old-snap"
    old.mkdir()
    new = tmp_path / "new-snap"
    new.mkdir()
    root = tmp_path / "models"
    prepare_model_dir(str(root), "org/m", str(old))
    prepare_model_dir(str(root), "org/m", str(new))
    assert (root / "org" / "m").resolve() == new.resolve()


def test_prepare_model_dir_leaves_real_directory_alone(tmp_path):
    root = tmp_path / "models"
    real = root / "org" / "m"
    real.mkdir(parents=True)
    marker = real / "config.json"
    marker.write_text("{}")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    prepare_model_dir(str(root), "org/m", str(snapshot))
    assert not real.is_symlink()
    assert marker.exists()


# --- supervise: child-exit detection ---
class _FakeProcess:
    def __init__(self) -> None:
        self._exited = False
        self.returncode: int | None = None

    def poll(self):
        return self.returncode if self._exited else None

    def exit(self, code: int = 1) -> None:
        self._exited = True
        self.returncode = code

    def terminate(self) -> None:
        self.exit(0)

    def kill(self) -> None:
        self.exit(-9)

    def wait(self, timeout=None):
        return self.returncode


def _as_popen(process: _FakeProcess) -> subprocess.Popen[Any]:
    return cast("subprocess.Popen[Any]", process)


def test_supervise_stops_cleanly_on_stop_event():
    stop_event = threading.Event()
    stop_event.set()
    assert supervise(_as_popen(_FakeProcess()), stop_event) is False


def test_supervise_reports_crash_on_unexpected_exit():
    process = _FakeProcess()
    process.exit(1)
    assert supervise(_as_popen(process), threading.Event()) is True


def test_supervise_returns_false_when_stopped_mid_flight():
    stop_event = threading.Event()

    def _stop_soon():
        time.sleep(0.05)
        stop_event.set()

    threading.Thread(target=_stop_soon, daemon=True).start()
    assert supervise(_as_popen(_FakeProcess()), stop_event, interval=0.01) is False


# --- run(): subprocess supervision ---
class _FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []

    def emit(self, event: Any, payload: Any) -> None:
        self.events.append((event, payload))

    def as_client(self) -> TelemetryClient:
        return cast(TelemetryClient, self)


class _FakeController:
    def __init__(self) -> None:
        self.shutdown_callback = None


class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _cfg(tmp_path, **overrides: Any) -> WorkerConfig:
    snapshot = tmp_path / "snap"
    snapshot.mkdir(exist_ok=True)
    cfg = WorkerConfig(
        service="omlx_coder_v1",
        engine="omlx",
        host="127.0.0.1",
        port=18000,
        health_path="/v1/models",
        telemetry_socket="/tmp/does-not-matter.sock",
        model_path=str(snapshot),
        engine_kwargs={
            "model_dir": str(tmp_path / "models"),
            # Flat `org--name` — what OmlxManager.api_model_name() always
            # passes (omlx's directory-derived id convention).
            "model_name": "org--m",
        },
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_run_prepares_model_dir_waits_for_health_then_emits_serving(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return _FakeProcess()

    monkeypatch.setattr(adapter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "supervise", lambda *a, **k: False)

    telemetry = _FakeTelemetry()
    controller = _FakeController()
    adapter.run(_cfg(tmp_path), telemetry.as_client(), controller)

    argv = captured["argv"]
    assert argv[:2] == ["omlx", "serve"]
    assert argv[argv.index("--model-dir") + 1] == str(tmp_path / "models")
    assert (tmp_path / "models" / "org--m").is_symlink()
    state_changes = [p for e, p in telemetry.events if e == EventType.STATE_CHANGE]
    assert {"state": "serving"} in state_changes
    assert controller.shutdown_callback is not None


def test_run_raises_when_health_never_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: _FakeProcess())
    monkeypatch.setattr(adapter, "_HEALTH_TIMEOUT", 0.05)
    monkeypatch.setattr(adapter, "_HEALTH_POLL_INTERVAL", 0.01)

    def boom(url, timeout=None):
        raise adapter.urllib.error.URLError("refused")

    monkeypatch.setattr(adapter.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="failed to become healthy"):
        adapter.run(_cfg(tmp_path), _FakeTelemetry().as_client(), _FakeController())


def test_run_raises_on_child_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: _FakeProcess())
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "supervise", lambda *a, **k: True)  # crashed
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        adapter.run(_cfg(tmp_path), _FakeTelemetry().as_client(), _FakeController())


def test_run_shutdown_callback_terminates_child(tmp_path, monkeypatch):
    process = _FakeProcess()
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda argv, **k: process)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "supervise", lambda *a, **k: False)

    controller = _FakeController()
    adapter.run(_cfg(tmp_path), _FakeTelemetry().as_client(), controller)

    assert controller.shutdown_callback is not None
    controller.shutdown_callback()
    assert process.returncode == 0  # terminated


def test_run_passes_api_key_via_env_not_config(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return _FakeProcess()

    monkeypatch.setattr(adapter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(adapter.urllib.request, "urlopen", lambda url, timeout=None: _FakeResp())
    monkeypatch.setattr(adapter, "supervise", lambda *a, **k: False)
    monkeypatch.setenv("SOVEREIGN_API_KEY", "s3cr3t")

    adapter.run(_cfg(tmp_path), _FakeTelemetry().as_client(), _FakeController())
    argv = captured["argv"]
    assert argv[argv.index("--api-key") + 1] == "s3cr3t"
