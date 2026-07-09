"""``docker`` — the generic Docker container service manager + shared helpers."""

from __future__ import annotations

import subprocess

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.core.resolver import (
    ConsumerKind,
    ResolutionError,
    ResolvedEndpoint,
    Resolver,
    ServiceRegistry,
)
from sovereign.services.docker import manager as mgr_mod
from sovereign.services.docker.config import FileSpec
from sovereign.services.docker.manager import DockerManager, materialize_file


def _entry(config: dict | None = None, env: dict | None = None, deps=None) -> ServiceEntry:
    return ServiceEntry(
        name="open_webui",
        base_type="docker",
        health_check={"type": "http", "endpoint": "/health", "port": 3000},
        config=config or {"image": "ghcr.io/open-webui/open-webui:main", "port": 3000},
        env_overrides=env or {},
        dependencies=deps or [],
    )


def _manager(config: dict | None = None, env: dict | None = None, deps=None) -> DockerManager:
    return DockerManager(_entry(config, env, deps))


def _resolver_with_llama() -> Resolver:
    reg = ServiceRegistry()
    reg.register("llama_heavy_v1", ResolvedEndpoint("http", "127.0.0.1", 11435))
    return Resolver(reg, env={})


def _fake_run(returncode: int = 0, stdout: str = "", stderr: str = "", raises=None):
    def run(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return run


# --- protocol / registry / wiring ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("docker") is DockerManager


def test_consumer_kind_is_docker() -> None:
    assert DockerManager.consumer_kind is ConsumerKind.DOCKER


def test_resolve_rewrites_loopback_to_host_gateway() -> None:
    m = _manager(env={"OPENAI_API_BASE_URL": "{{ llama_heavy_v1.endpoint }}"})
    m.resolve(_resolver_with_llama())
    assert m.resolved_env == {"OPENAI_API_BASE_URL": "http://host.docker.internal:11435"}


# --- run_args ---
def test_run_args_port_mapping_defaults_container_port() -> None:
    m = _manager({"image": "img:latest", "port": 8888})
    args = m.run_args()
    assert "-p" in args and "8888:8888" in args


def test_run_args_port_mapping_explicit_container_port() -> None:
    m = _manager({"image": "img:latest", "port": 8888, "container_port": 8080})
    args = m.run_args()
    assert "8888:8080" in args


def test_run_args_include_resolved_env() -> None:
    m = _manager(env={"OPENAI_API_BASE_URL": "{{ llama_heavy_v1.endpoint }}"})
    m.resolve(_resolver_with_llama())
    args = m.run_args()
    i = args.index("-e")
    assert args[i + 1] == "OPENAI_API_BASE_URL=http://host.docker.internal:11435"


def test_run_args_expands_bind_mount_host_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    volumes = ["~/.sovereign/searxng:/etc/searxng"]
    m = _manager({"image": "img:latest", "port": 8888, "volumes": volumes})
    args = m.run_args()
    vi = args.index("-v")
    assert args[vi + 1] == f"{tmp_path}/.sovereign/searxng:/etc/searxng"


def test_run_args_leaves_named_volume_untouched() -> None:
    volumes = ["sovereign_open_webui:/app/backend/data"]
    m = _manager({"image": "img:latest", "port": 3000, "volumes": volumes})
    args = m.run_args()
    vi = args.index("-v")
    assert args[vi + 1] == "sovereign_open_webui:/app/backend/data"


def test_run_args_image_last() -> None:
    m = _manager({"image": "ghcr.io/open-webui/open-webui:main", "port": 3000})
    args = m.run_args()
    assert args[-1] == "ghcr.io/open-webui/open-webui:main"


# --- endpoint / runtime handle ---
def test_endpoint_reports_host_port() -> None:
    ep = _manager({"image": "img:latest", "port": 3000}).endpoint()
    assert ep.url_for(ConsumerKind.NATIVE) == "http://127.0.0.1:3000"


def test_runtime_handle_reports_container_name() -> None:
    m = _manager({"image": "img:latest", "port": 3000})
    assert m.runtime_handle() == {"kind": "docker", "container": "open_webui"}


def test_runtime_handle_respects_container_name_override() -> None:
    m = _manager({"image": "img:latest", "port": 3000, "container_name": "my-webui"})
    assert m.runtime_handle() == {"kind": "docker", "container": "my-webui"}


# --- lifecycle (mocked docker) ---
def test_start_clears_then_runs(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        DockerManager, "_run_docker", lambda self, args, **kw: calls.append(args)
    )
    _manager({"image": "img:latest", "port": 3000}).start()
    assert calls[0][:2] == ["rm", "-f"]
    assert calls[1][:2] == ["run", "-d"]


def test_stop_removes_container(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        DockerManager, "_run_docker", lambda self, args, **kw: calls.append(args)
    )
    _manager({"image": "img:latest", "port": 3000}).stop()
    assert calls == [["rm", "-f", "open_webui"]]


# --- health ---
def test_is_healthy_true_on_2xx(monkeypatch) -> None:
    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mgr_mod.urllib.request, "urlopen", lambda url, timeout=None: FakeResp())
    assert _manager({"image": "img:latest", "port": 3000}).is_healthy() is True


def test_is_healthy_false_on_error(monkeypatch) -> None:
    def boom(url, timeout=None):
        raise mgr_mod.urllib.error.URLError("refused")

    monkeypatch.setattr(mgr_mod.urllib.request, "urlopen", boom)
    assert _manager({"image": "img:latest", "port": 3000}).is_healthy() is False


# --- metrics ---
def test_get_metrics_delegates_to_container_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        mgr_mod, "container_metrics",
        lambda container, **kw: {"status": "running", "container": container},
    )
    m = _manager({"image": "img:latest", "port": 3000})
    assert m.get_metrics() == {"status": "running", "container": "open_webui"}


# --- provisioning ---
def test_provisioning_declaration() -> None:
    from sovereign.services.docker.manager import DockerManager

    assert DockerManager.provisioning_binary == "docker"
    brewfile = DockerManager.provisioning_brewfile()
    assert brewfile is not None
    assert 'cask "docker-desktop"' in brewfile.read_text()


def test_prepare_environment_provisions_first(monkeypatch) -> None:
    from sovereign.core.provisioning import Provisioner

    order: list[str] = []
    monkeypatch.setattr(
        Provisioner, "provision", classmethod(lambda cls: order.append("provision"))
    )
    monkeypatch.setattr(
        mgr_mod.shutil, "which", lambda _b: order.append("which") or "/usr/local/bin/docker"
    )
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=0, stdout="29.6.1"))
    m = _manager({"image": "img:latest", "port": 3000, "auto_pull": False})
    m.prepare_environment()
    assert order[0] == "provision"  # install Docker before probing it


# --- prepare_environment (daemon probe moved here from the old engine service) ---
def test_prepare_environment_raises_when_binary_absent(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _binary: None)
    m = _manager({"image": "img:latest", "port": 3000})
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        m.prepare_environment()


def test_prepare_environment_raises_when_daemon_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _binary: "/usr/local/bin/docker")
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=1))
    m = _manager({"image": "img:latest", "port": 3000, "auto_pull": False})
    with pytest.raises(RuntimeError, match="not reachable"):
        m.prepare_environment()


def test_prepare_environment_materializes_files_and_pulls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/usr/local/bin/docker")
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=0, stdout="29.6.1"))
    pulled: list[str] = []
    monkeypatch.setattr(mgr_mod, "stream_pull", lambda image, **kw: pulled.append(image))

    settings_file = tmp_path / "settings.yml"
    m = _manager(
        {
            "image": "searxng/searxng:latest",
            "port": 8888,
            "files": [{"path": str(settings_file), "content": "secret_key: ${RANDOM_HEX:8}"}],
        }
    )
    m.prepare_environment()
    assert settings_file.exists()
    assert pulled == ["searxng/searxng:latest"]


def test_prepare_environment_skips_pull_when_auto_pull_false(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/usr/local/bin/docker")
    monkeypatch.setattr(mgr_mod.subprocess, "run", _fake_run(returncode=0, stdout="29.6.1"))
    pulled: list[str] = []
    monkeypatch.setattr(mgr_mod, "stream_pull", lambda image, **kw: pulled.append(image))

    m = _manager({"image": "img:latest", "port": 3000, "auto_pull": False})
    m.prepare_environment()
    assert pulled == []


# --- materialize_file ---
def test_materialize_file_writes_content(tmp_path) -> None:
    path = tmp_path / "sub" / "settings.yml"
    written = materialize_file(FileSpec(path=str(path), content="hello: world\n"))
    assert written is True
    assert path.read_text() == "hello: world\n"


def test_materialize_file_is_idempotent(tmp_path) -> None:
    path = tmp_path / "settings.yml"
    path.write_text("keepme: true\n")
    written = materialize_file(FileSpec(path=str(path), content="hello: world\n"))
    assert written is False
    assert path.read_text() == "keepme: true\n"


def test_materialize_file_random_hex_length() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.yml"
        materialize_file(FileSpec(path=str(path), content="secret_key: ${RANDOM_HEX:64}"))
        text = path.read_text()
        hexval = text.split("secret_key: ")[1].strip()
        assert len(hexval) == 64
        int(hexval, 16)  # valid hex


def test_materialize_file_random_hex_differs_across_files(tmp_path) -> None:
    path_a = tmp_path / "a.yml"
    path_b = tmp_path / "b.yml"
    materialize_file(FileSpec(path=str(path_a), content="secret_key: ${RANDOM_HEX:64}"))
    materialize_file(FileSpec(path=str(path_b), content="secret_key: ${RANDOM_HEX:64}"))
    assert path_a.read_text() != path_b.read_text()


def test_materialize_file_substitutes_env(tmp_path) -> None:
    path = tmp_path / "a.yml"
    spec = FileSpec(path=str(path), content="key: ${ENV:MY_VAR}")
    materialize_file(spec, env={"MY_VAR": "abc123"})
    assert path.read_text() == "key: abc123"


def test_materialize_file_missing_env_raises(tmp_path) -> None:
    path = tmp_path / "a.yml"
    with pytest.raises(ResolutionError):
        materialize_file(FileSpec(path=str(path), content="key: ${ENV:MISSING_VAR}"), env={})


# --- shared Docker helpers (reused by all docker instances) ---
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
    assert any("1/2 layers" in p for p in progress)
    assert any("2/2 layers" in p for p in progress)


def test_stream_pull_raises_on_failure(monkeypatch) -> None:
    proc = _FakePullProc(["Error: pull access denied\n"], returncode=1)
    monkeypatch.setattr(mgr_mod.subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(RuntimeError, match="docker pull failed"):
        mgr_mod.stream_pull("nope:latest", on_progress=lambda _s: None)


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


# --- expand_volume ---
def test_expand_volume_expands_tilde_host_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert mgr_mod.expand_volume("~/x:/y") == f"{tmp_path}/x:/y"


def test_expand_volume_leaves_named_volume_untouched() -> None:
    assert mgr_mod.expand_volume("sovereign_open_webui:/app/backend/data") == (
        "sovereign_open_webui:/app/backend/data"
    )
