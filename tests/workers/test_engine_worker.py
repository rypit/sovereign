"""Exercises the REAL ``sovereign.workers.engine_worker`` entrypoint as a
subprocess, against a fake adapter (no real engine bindings) shipped under
``tests/fixtures/fake_worker_adapters``. Verifies health readiness,
end-to-end telemetry delivery to a real ``TelemetryHub``, graceful SIGTERM
shutdown, and the crash path.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from sovereign.runtime.telemetry import TelemetryHub, TelemetryStateCache
from sovereign.workers.worker_config import WorkerConfig, dump_worker_config

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _child_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = str(FIXTURES_DIR)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    env["SOVEREIGN_WORKER_ADAPTER_PACKAGE"] = "fake_worker_adapters"
    return env


def _spawn(config_path: Path, tmp_path: Path, extra_env: dict[str, str] | None = None):
    env = _child_env(tmp_path)
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "sovereign.workers.engine_worker", "--config", str(config_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_health(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.1)
    raise AssertionError(f"health endpoint never came up: {last_err}")


@pytest.fixture
def hub(tmp_path: Path):
    cache = TelemetryStateCache()
    sock_path = tmp_path / "telemetry.sock"
    hub = TelemetryHub(sock_path, cache)
    hub.start()
    yield hub, cache, sock_path
    hub.stop()


def _make_config(tmp_path: Path, sock_path: Path, port: int) -> Path:
    cfg = WorkerConfig(
        service="fake-svc",
        engine="fake",
        host="127.0.0.1",
        port=port,
        health_path="/health",
        telemetry_socket=str(sock_path),
        model_path="/dev/null",
    )
    config_path = tmp_path / "worker_config.json"
    dump_worker_config(cfg, config_path)
    return config_path


def test_health_and_telemetry_end_to_end(hub, tmp_path: Path) -> None:
    _hub, cache, sock_path = hub
    port = _free_port()
    config_path = _make_config(tmp_path, sock_path, port)
    proc = _spawn(config_path, tmp_path)
    try:
        _wait_for_health(f"http://127.0.0.1:{port}/health")

        deadline = time.time() + 10.0
        state = None
        while time.time() < deadline:
            state = cache.snapshot("fake-svc")
            if state["worker_state"] == "serving" and state["last_heartbeat"] is not None:
                break
            time.sleep(0.1)
        assert state is not None
        assert state["worker_state"] == "serving"
        assert state["last_heartbeat"] is not None
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_sigterm_triggers_graceful_shutdown(hub, tmp_path: Path) -> None:
    _hub, cache, sock_path = hub
    port = _free_port()
    config_path = _make_config(tmp_path, sock_path, port)
    proc = _spawn(config_path, tmp_path)
    try:
        _wait_for_health(f"http://127.0.0.1:{port}/health")

        proc.send_signal(signal.SIGTERM)
        exit_code = proc.wait(timeout=10)
        assert exit_code == 0

        deadline = time.time() + 5.0
        state = cache.snapshot("fake-svc")
        while time.time() < deadline and state["worker_state"] != "stopping":
            state = cache.snapshot("fake-svc")
            time.sleep(0.05)
        assert state["worker_state"] == "stopping"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_crashing_adapter_exits_nonzero_and_reports_crash(hub, tmp_path: Path) -> None:
    _hub, cache, sock_path = hub
    port = _free_port()
    config_path = _make_config(tmp_path, sock_path, port)
    proc = _spawn(config_path, tmp_path, extra_env={"SOVEREIGN_FAKE_ADAPTER_CRASH": "1"})
    try:
        exit_code = proc.wait(timeout=10)
        assert exit_code == 1

        deadline = time.time() + 5.0
        state = cache.snapshot("fake-svc")
        while time.time() < deadline and state["worker_state"] != "crashed":
            state = cache.snapshot("fake-svc")
            time.sleep(0.05)
        assert state["worker_state"] == "crashed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
