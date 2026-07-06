"""Phase 5: open_webui manager — mocked docker + dynamic wiring."""

from __future__ import annotations

import subprocess

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.core.resolver import (
    ConsumerKind,
    ResolvedEndpoint,
    Resolver,
    ServiceRegistry,
)
from sovereign.services.open_webui import manager as mgr_mod
from sovereign.services.open_webui.manager import OpenWebUIManager


def _entry(config: dict | None = None, env: dict | None = None) -> ServiceEntry:
    return ServiceEntry(
        name="open_webui",
        base_type="open_webui",
        health_check={"type": "http", "endpoint": "/health", "port": 3000},
        config=config or {"image": "ghcr.io/open-webui/open-webui:main", "port": 3000},
        env_overrides=env or {"OLLAMA_API_BASE_URL": "{{ llama_heavy_v1.endpoint }}"},
        dependencies=["docker_engine", "llama_heavy_v1"],
    )


def _manager(config: dict | None = None, env: dict | None = None) -> OpenWebUIManager:
    return OpenWebUIManager(_entry(config, env))


def _resolver_with_llama() -> Resolver:
    reg = ServiceRegistry()
    reg.register("llama_heavy_v1", ResolvedEndpoint("http", "127.0.0.1", 11435))
    return Resolver(reg, env={})


# --- protocol / registry / wiring ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("open_webui") is OpenWebUIManager


def test_consumer_kind_is_docker() -> None:
    assert OpenWebUIManager.consumer_kind is ConsumerKind.DOCKER


def test_resolve_rewrites_loopback_to_host_gateway() -> None:
    m = _manager()
    m.resolve(_resolver_with_llama())
    assert m.resolved_env == {"OLLAMA_API_BASE_URL": "http://host.docker.internal:11435"}


def test_run_args_include_resolved_env_and_port_mapping() -> None:
    m = _manager()
    m.resolve(_resolver_with_llama())
    args = m._run_args()
    assert args[:2] == ["run", "-d"]
    assert "-p" in args and "3000:8080" in args
    i = args.index("-e")
    assert args[i + 1] == "OLLAMA_API_BASE_URL=http://host.docker.internal:11435"
    assert args[-1] == "ghcr.io/open-webui/open-webui:main"


def test_endpoint_reports_host_port() -> None:
    ep = _manager().endpoint()
    assert ep.url_for(ConsumerKind.NATIVE) == "http://127.0.0.1:3000"


# --- lifecycle (mocked docker) ---
def test_start_clears_then_runs(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kw):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(
        OpenWebUIManager, "_run_docker", lambda self, args, **kw: fake_run(args, **kw)
    )
    m = _manager()
    m.resolve(_resolver_with_llama())
    m.start()
    assert calls[0][:2] == ["rm", "-f"]  # idempotent cleanup first
    assert calls[1][:2] == ["run", "-d"]


def test_stop_removes_container(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        OpenWebUIManager, "_run_docker", lambda self, args, **kw: calls.append(args)
    )
    _manager().stop()
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
    assert _manager().is_healthy() is True


def test_is_healthy_false_on_error(monkeypatch) -> None:
    def boom(url, timeout=None):
        raise mgr_mod.urllib.error.URLError("refused")

    monkeypatch.setattr(mgr_mod.urllib.request, "urlopen", boom)
    assert _manager().is_healthy() is False


# --- metrics ---
def test_get_metrics_delegates_to_container_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        mgr_mod, "container_metrics",
        lambda container, **kw: {"status": "running", "container": container},
    )
    assert _manager().get_metrics() == {"status": "running", "container": "open_webui"}


# --- prepare_environment ---
def test_prepare_environment_missing_docker(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: None)
    with pytest.raises(FileNotFoundError, match="Docker CLI 'docker' not found"):
        _manager().prepare_environment()


def test_prepare_environment_streams_pull(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/usr/local/bin/docker")
    pulled: list[str] = []

    def fake_stream_pull(image, *, binary, on_progress):
        pulled.append(image)
        on_progress("pulling — 1/2 layers")  # exercise the activity callback

    monkeypatch.setattr(mgr_mod, "stream_pull", fake_stream_pull)
    m = _manager()
    m.prepare_environment()
    assert pulled == ["ghcr.io/open-webui/open-webui:main"]
    assert m.activity == ""  # cleared after the pull


def test_prepare_environment_skips_pull_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/usr/local/bin/docker")
    pulled: list[str] = []
    monkeypatch.setattr(
        mgr_mod, "stream_pull", lambda image, **kw: pulled.append(image)
    )
    _manager({"auto_pull": False, "port": 3000}).prepare_environment()
    assert pulled == []
