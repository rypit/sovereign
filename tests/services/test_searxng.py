"""Phase 11: searxng manager — mocked docker + settings materialization."""

from __future__ import annotations

import pytest
import yaml

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.core.resolver import ConsumerKind
from sovereign.services.searxng import manager as mgr_mod
from sovereign.services.searxng.manager import SearxngManager


def _entry(config: dict | None = None) -> ServiceEntry:
    return ServiceEntry(
        name="searxng",
        base_type="searxng",
        health_check={"type": "http", "endpoint": "/", "port": 8888},
        config=config or {"image": "searxng/searxng:latest", "port": 8888},
        dependencies=["docker_engine"],
    )


def _manager(config: dict | None = None) -> SearxngManager:
    return SearxngManager(_entry(config))


# --- protocol / registry ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("searxng") is SearxngManager


def test_consumer_kind_is_docker() -> None:
    assert SearxngManager.consumer_kind is ConsumerKind.DOCKER


def test_endpoint_reports_host_port() -> None:
    ep = _manager().endpoint()
    assert ep.url_for(ConsumerKind.NATIVE) == "http://127.0.0.1:8888"


# --- run args ---
def test_run_args_include_port_and_settings_volume(tmp_path) -> None:
    m = _manager({"port": 8888, "container_port": 8080, "config_dir": str(tmp_path)})
    args = m._run_args()
    assert args[:2] == ["run", "-d"]
    assert "-p" in args and "8888:8080" in args
    vi = args.index("-v")
    assert args[vi + 1] == f"{tmp_path}:/etc/searxng"
    assert args[-1] == "searxng/searxng:latest"


def test_run_args_include_base_url(tmp_path) -> None:
    m = _manager({"config_dir": str(tmp_path), "base_url": "http://localhost:8888/"})
    args = m._run_args()
    i = args.index("-e")
    assert args[i + 1] == "SEARXNG_BASE_URL=http://localhost:8888/"


# --- settings materialization ---
def test_materialize_settings_writes_json_format_and_secret(tmp_path) -> None:
    m = _manager({"config_dir": str(tmp_path)})
    m._materialize_settings()
    data = yaml.safe_load((tmp_path / "settings.yml").read_text())
    assert data["search"]["formats"] == ["html", "json"]
    assert data["server"]["limiter"] is False
    assert len(data["server"]["secret_key"]) >= 32


def test_materialize_settings_is_idempotent(tmp_path) -> None:
    settings = tmp_path / "settings.yml"
    settings.write_text("use_default_settings: true\nserver: {secret_key: keepme}\n")
    _manager({"config_dir": str(tmp_path)})._materialize_settings()
    assert "keepme" in settings.read_text()  # existing file preserved


def test_materialize_settings_uses_declared_secret(tmp_path) -> None:
    m = _manager({"config_dir": str(tmp_path), "secret": "mysecret123456789012345678901234"})
    m._materialize_settings()
    data = yaml.safe_load((tmp_path / "settings.yml").read_text())
    assert data["server"]["secret_key"] == "mysecret123456789012345678901234"


# --- lifecycle (mocked docker) ---
def test_start_clears_then_runs(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        SearxngManager, "_run_docker", lambda self, args, **kw: calls.append(args)
    )
    _manager({"config_dir": str(tmp_path)}).start()
    assert calls[0][:2] == ["rm", "-f"]
    assert calls[1][:2] == ["run", "-d"]


def test_stop_removes_container(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        SearxngManager, "_run_docker", lambda self, args, **kw: calls.append(args)
    )
    _manager().stop()
    assert calls == [["rm", "-f", "searxng"]]


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
    assert _manager().get_metrics() == {"status": "running", "container": "searxng"}


# --- prepare_environment ---
def test_prepare_environment_materializes_and_pulls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: "/usr/local/bin/docker")
    pulled: list[str] = []
    monkeypatch.setattr(
        mgr_mod, "stream_pull",
        lambda image, **kw: pulled.append(image),
    )
    m = _manager({"config_dir": str(tmp_path)})
    m.prepare_environment()
    assert (tmp_path / "settings.yml").exists()
    assert pulled == ["searxng/searxng:latest"]


def test_prepare_environment_missing_docker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mgr_mod.shutil, "which", lambda _b: None)
    with pytest.raises(FileNotFoundError, match="Docker CLI 'docker' not found"):
        _manager({"config_dir": str(tmp_path)}).prepare_environment()
