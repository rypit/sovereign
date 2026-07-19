"""Tests for the ComfyUI worker adapter (ADR 0007 subprocess pattern:
``comfy … launch`` child, no in-process binding, no telemetry translator).

``build_server_argv`` and ``prepare_checkpoint_dir`` are pure/near-pure and
unit-testable with no comfy-cli installed. ``run()`` is exercised by patching
``subprocess.Popen`` (a fake child) and ``urllib.request.urlopen`` (health).
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any, cast

import pytest

from sovereign.workers import comfyui_adapter as adapter
from sovereign.workers.comfyui_adapter import (
    build_server_argv,
    prepare_checkpoint_dir,
    supervise,
)
from sovereign.workers.protocol import EventType
from sovereign.workers.telemetry import TelemetryClient
from sovereign.workers.worker_config import WorkerConfig


# --- build_server_argv ---
def test_build_server_argv_basics():
    argv = build_server_argv(
        {}, extra_model_paths="/state/comfyui/svc/models/extra_model_paths.yaml",
        host="127.0.0.1", port=8188,
    )
    assert argv[0] == "--skip-prompt"
    assert argv[argv.index("launch") + 1] == "--"
    assert argv[argv.index("--listen") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8188"
    assert argv[argv.index("--extra-model-paths-config") + 1] == (
        "/state/comfyui/svc/models/extra_model_paths.yaml"
    )


def test_build_server_argv_workspace_before_launch():
    argv = build_server_argv(
        {"workspace_dir": "/home/u/.sovereign/comfyui"},
        extra_model_paths="/y", host="h", port=1,
    )
    assert argv[argv.index("--workspace") + 1] == "/home/u/.sovereign/comfyui"
    assert argv.index("--workspace") < argv.index("launch")


def test_build_server_argv_maps_output_dir():
    argv = build_server_argv(
        {"output_dir": "/stacks/a/outputs"}, extra_model_paths="/y", host="h", port=1
    )
    assert argv[argv.index("--output-directory") + 1] == "/stacks/a/outputs"
    assert argv.index("--output-directory") > argv.index("--")


def test_build_server_argv_consumes_models_root_and_checkpoint_name():
    argv = build_server_argv(
        {"models_root": "/state/models", "checkpoint_name": "sd.safetensors"},
        extra_model_paths="/y", host="h", port=1,
    )
    assert "--models-root" not in argv
    assert "--checkpoint-name" not in argv


def test_build_server_argv_passthrough_escape_hatch():
    argv = build_server_argv({"preview_method": "auto"}, extra_model_paths="/y", host="h", port=1)
    assert argv[argv.index("--preview-method") + 1] == "auto"
    assert argv.index("--preview-method") > argv.index("--")


def test_build_server_argv_bool_kwargs_become_bare_flags():
    argv = build_server_argv(
        {"force_fp16": True, "cpu": False}, extra_model_paths="/y", host="h", port=1
    )
    assert "--force-fp16" in argv
    assert "--cpu" not in argv


# --- prepare_checkpoint_dir: the single-checkpoint symlink layout ---
def test_prepare_checkpoint_dir_creates_symlink_and_yaml(tmp_path):
    ckpt = tmp_path / "hf" / "sd_xl_base_1.0.safetensors"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    root = tmp_path / "models"

    yaml_path = prepare_checkpoint_dir(str(root), "sd_xl_base_1.0.safetensors", str(ckpt))

    link = root / "checkpoints" / "sd_xl_base_1.0.safetensors"
    assert link.is_symlink()
    assert link.resolve() == ckpt.resolve()
    content = (root / "extra_model_paths.yaml").read_text()
    assert yaml_path == str(root / "extra_model_paths.yaml")
    assert f"base_path: {root}" in content
    assert "checkpoints: checkpoints" in content


def test_prepare_checkpoint_dir_idempotent(tmp_path):
    ckpt = tmp_path / "sd.safetensors"
    ckpt.write_bytes(b"w")
    root = tmp_path / "models"
    prepare_checkpoint_dir(str(root), "sd.safetensors", str(ckpt))
    prepare_checkpoint_dir(str(root), "sd.safetensors", str(ckpt))  # must not raise
    assert (root / "checkpoints" / "sd.safetensors").resolve() == ckpt.resolve()


def test_prepare_checkpoint_dir_repoints_stale_symlink(tmp_path):
    old = tmp_path / "old.safetensors"
    old.write_bytes(b"old")
    new = tmp_path / "new.safetensors"
    new.write_bytes(b"new")
    root = tmp_path / "models"
    prepare_checkpoint_dir(str(root), "sd.safetensors", str(old))
    prepare_checkpoint_dir(str(root), "sd.safetensors", str(new))
    assert (root / "checkpoints" / "sd.safetensors").resolve() == new.resolve()


def test_prepare_checkpoint_dir_leaves_real_file_alone(tmp_path):
    root = tmp_path / "models"
    real = root / "checkpoints" / "sd.safetensors"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"user-materialised")
    ckpt = tmp_path / "resolved.safetensors"
    ckpt.write_bytes(b"resolved")
    prepare_checkpoint_dir(str(root), "sd.safetensors", str(ckpt))
    assert not real.is_symlink()
    assert real.read_bytes() == b"user-materialised"


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
    ckpt = tmp_path / "sd_xl_base_1.0.safetensors"
    if not ckpt.exists():
        ckpt.write_bytes(b"weights")
    cfg = WorkerConfig(
        service="sdxl",
        engine="comfyui",
        host="127.0.0.1",
        port=8188,
        health_path="/system_stats",
        telemetry_socket="/tmp/does-not-matter.sock",
        model_path=str(ckpt),
        engine_kwargs={
            "workspace_dir": str(tmp_path / "workspace"),
            "models_root": str(tmp_path / "models"),
            "checkpoint_name": "sd_xl_base_1.0.safetensors",
        },
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_run_prepares_checkpoint_waits_for_health_then_emits_serving(tmp_path, monkeypatch):
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
    assert argv[0] == "comfy"
    assert argv[argv.index("--workspace") + 1] == str(tmp_path / "workspace")
    assert argv[argv.index("--extra-model-paths-config") + 1] == str(
        tmp_path / "models" / "extra_model_paths.yaml"
    )
    assert (tmp_path / "models" / "checkpoints" / "sd_xl_base_1.0.safetensors").is_symlink()
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
