"""Phase 3: docker_engine manager — mocked unit tests + Protocol/registry checks."""

from __future__ import annotations

import subprocess

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.docker_engine import manager as mgr_mod
from sovereign.services.docker_engine.manager import DockerEngineManager


def _make_manager(config: dict | None = None) -> DockerEngineManager:
    entry = ServiceEntry(
        name="docker_engine",
        base_type="docker_engine",
        config=config or {},
    )
    return DockerEngineManager(entry)


def _fake_run(returncode: int = 0, stdout: str = "", stderr: str = "", raises=None):
    def run(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return run


def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_make_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("docker_engine") is DockerEngineManager


def test_is_healthy_true_when_daemon_reachable(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=0, stdout="29.6.1\n"))
    assert _make_manager().is_healthy() is True


def test_is_healthy_false_when_daemon_down(monkeypatch) -> None:
    monkeypatch.setattr(
        mgr_mod.subprocess,
        "run",
        _fake_run(returncode=1, stderr="Cannot connect to the Docker daemon"),
    )
    assert _make_manager().is_healthy() is False


def test_is_healthy_false_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(raises=FileNotFoundError()))
    assert _make_manager().is_healthy() is False


def test_is_healthy_false_on_timeout(monkeypatch) -> None:
    timeout = subprocess.TimeoutExpired(cmd="docker", timeout=10)
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(raises=timeout))
    assert _make_manager().is_healthy() is False


def test_start_raises_when_daemon_down(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=1))
    with pytest.raises(RuntimeError, match="not reachable"):
        _make_manager().start()


def test_start_ok_when_daemon_up(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=0, stdout="29.6.1"))
    _make_manager().start()  # must not raise


def test_prepare_environment_raises_when_binary_absent(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _binary: None)
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        _make_manager().prepare_environment()


def test_prepare_environment_ok_when_binary_present(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _binary: "/usr/local/bin/docker")
    _make_manager().prepare_environment()  # must not raise


def test_get_metrics_reports_running_with_container_count(monkeypatch) -> None:
    calls = {"n": 0}

    def run(cmd, **kwargs):
        calls["n"] += 1
        if "version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "29.6.1\n", "")
        return subprocess.CompletedProcess(cmd, 0, "abc123\ndef456\n", "")

    monkeypatch.setattr(mgr_mod.subprocess, "run", run)
    metrics = _make_manager().get_metrics()
    assert metrics == {"status": "running", "containers": 2}


def test_get_metrics_reports_stopped_when_down(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=1))
    assert _make_manager().get_metrics() == {"status": "stopped"}


def test_custom_binary_is_used(monkeypatch) -> None:
    seen = {}

    def run(cmd, **kwargs):
        seen["binary"] = cmd[0]
        return subprocess.CompletedProcess(cmd, 0, "29.6.1", "")

    monkeypatch.setattr(mgr_mod.subprocess, "run", run)
    _make_manager({"binary": "podman"}).is_healthy()
    assert seen["binary"] == "podman"


# --- shared Docker helpers (reused by all Docker services) ---
def test_run_docker_builds_argv(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mgr_mod.subprocess, "run", fake_run)
    mgr_mod.run_docker(["ps", "-q"], binary="podman")
    assert seen["cmd"] == ["podman", "ps", "-q"]


def test_pull_activity_formats_layer_count() -> None:
    assert mgr_mod.pull_activity("img", 0, 0) == "pulling img"
    assert mgr_mod.pull_activity("img", 3, 8) == "pulling img — 3/8 layers"


class _FakePullProc:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self._returncode = returncode

    def wait(self):
        return self._returncode


def test_stream_pull_reports_layer_progress(monkeypatch) -> None:
    lines = [
        "latest: Pulling from open-webui\n",
        "a1b2c3d4e5f6: Pulling fs layer\n",
        "0f1e2d3c4b5a: Pulling fs layer\n",
        "a1b2c3d4e5f6: Download complete\n",
        "a1b2c3d4e5f6: Pull complete\n",
        "0f1e2d3c4b5a: Already exists\n",
        "Status: Downloaded newer image\n",
    ]
    monkeypatch.setattr(mgr_mod.subprocess, "Popen", lambda *a, **k: _FakePullProc(lines))
    progress: list[str] = []
    mgr_mod.stream_pull("open-webui:latest", on_progress=progress.append)
    # Progressed through partial and full layer completion.
    assert any("1/2 layers" in p for p in progress)
    assert any("2/2 layers" in p for p in progress)


def test_stream_pull_raises_on_failure(monkeypatch) -> None:
    proc = _FakePullProc(["Error: pull access denied\n"], returncode=1)
    monkeypatch.setattr(mgr_mod.subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(RuntimeError, match="docker pull failed"):
        mgr_mod.stream_pull("nope:latest", on_progress=lambda _s: None)


# --- shared container metrics helpers ---
@pytest.mark.parametrize(
    ("text", "expected"),
    [("15.5MiB", 15.5), ("1GiB", 1024.0), ("512KiB", 0.5), ("100MB", 95.37)],
)
def test_parse_mem_to_mb(text: str, expected: float) -> None:
    assert mgr_mod.parse_mem_to_mb(text) == pytest.approx(expected, abs=0.05)


def test_container_metrics_running(monkeypatch) -> None:
    monkeypatch.setattr(
        mgr_mod, "run_docker",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, "12.34%;15.5MiB / 2GiB\n", ""),
    )
    m = mgr_mod.container_metrics("c1")
    assert m["status"] == "running"
    assert m["cpu_percent"] == 12.34
    assert m["memory_mb"] == pytest.approx(15.5, abs=0.1)


def test_container_metrics_stopped(monkeypatch) -> None:
    monkeypatch.setattr(
        mgr_mod, "run_docker",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "no such container"),
    )
    assert mgr_mod.container_metrics("c1") == {"status": "stopped"}
