"""omlx manager — mocked unit tests + Protocol/registry checks.

The real ``omlx`` CLI is not required here; the shared process/health/metrics
lifecycle lives in (and is covered by) inference.base, so this suite focuses
on what omlx adds: flag/kwarg mapping, the routing abstention, the memory
guard derived from Sovereign's own admission estimate, and the ADR 0006 gap
surfacing (no draft_model, no telemetry scrape surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.inference import hf as models_mod
from sovereign.services.inference.hf import RepoInfo, parse_model_ref
from sovereign.services.inference.mlx_lm.manager import MlxLmManager
from sovereign.services.inference.omlx import manager as omlx_manager_mod
from sovereign.services.inference.omlx.manager import OmlxManager
from sovereign.services.inference.routing import _claim_engine
from sovereign.workers.worker_config import load_worker_config


def _repo_info(repo_id: str, siblings: list[tuple[str, int | None]], tags=()) -> RepoInfo:
    return RepoInfo(repo_id=repo_id, tags=tuple(tags), siblings=tuple(siblings))


@pytest.fixture(autouse=True)
def _offline_metadata(monkeypatch):
    """Default HF metadata fetch to offline (None) so no test hits the network via
    the prepare_environment prefetch or repo-id estimation; specific tests override."""
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """get_start_args() writes a worker-config JSON under ``<state_dir>/workers/``
    (derived from log_dir's parent) — run every test from an isolated tmp cwd so
    the default ``.sovereign/logs`` relative path doesn't touch the real repo tree."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _passthrough_download(monkeypatch):
    """Neutralise the real HF download: resolve local refs in place and treat a repo
    id as if it downloaded to a path equal to the ref, so argv assertions stay
    readable. Applied suite-wide so no test reaches the network on prepare_model()."""

    def _fake(ref, kind, *, progress=None):
        return ref.local_path if ref.is_local else Path(ref.raw)

    monkeypatch.setattr(models_mod, "download_model", _fake)


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="omlx_coder_v1",
        base_type="omlx",
        health_check=(
            {"type": "http", "endpoint": "/v1/models", "port": 18000}
            if with_health
            else None
        ),
        config=config or {"model": "mlx-community/some-model-4bit"},
    )


def _manager(config: dict | None = None) -> OmlxManager:
    return OmlxManager(_entry(config))


def _prepared(config: dict | None = None) -> OmlxManager:
    m = _manager(config)
    m.prepare_model()
    return m


# --- construction / protocol / registry ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("omlx") is OmlxManager


def test_requires_health_check() -> None:
    with pytest.raises(ValueError, match="requires a health_check"):
        OmlxManager(_entry(with_health=False))


# --- routing: deliberate abstention from auto ---
def test_claim_route_abstains_for_mlx_repo() -> None:
    info = _repo_info(
        "mlx-community/some-model-4bit",
        [("model.safetensors", 4 * 1024**3)],
        tags=("mlx",),
    )
    ref = parse_model_ref("mlx-community/some-model-4bit")
    assert OmlxManager.claim_route(ref, info) is None


def test_claim_route_abstains_for_local_mlx_dir(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    ref = parse_model_ref(str(tmp_path))
    assert OmlxManager.claim_route(ref, None) is None


def test_auto_sweep_prefers_mlx_lm_over_omlx() -> None:
    """With both engines registered, an mlx repo still routes to mlx_lm —
    omlx joining the registry must not perturb the precedence contract."""
    info = _repo_info(
        "mlx-community/some-model-4bit",
        [("model.safetensors", 4 * 1024**3)],
        tags=("mlx",),
    )
    ref = parse_model_ref("mlx-community/some-model-4bit")
    winner = _claim_engine([OmlxManager, MlxLmManager], ref, info)
    assert winner is MlxLmManager


# --- engine_kwargs (Sovereign config -> worker adapter mapping) ---
def test_engine_kwargs_always_carries_model_dir_and_name(tmp_path) -> None:
    m = _prepared(
        {"model": "mlx-community/some-model-4bit", "log_dir": str(tmp_path / "logs")}
    )
    kwargs = m.engine_kwargs()
    assert kwargs["model_dir"] == str(tmp_path / "omlx" / "omlx_coder_v1" / "models")
    assert kwargs["model_name"] == "mlx-community--some-model-4bit"


def test_engine_kwargs_full_mapping(tmp_path) -> None:
    m = _prepared(
        {
            "model": "mlx-community/some-model-4bit",
            "log_dir": str(tmp_path / "logs"),
            "max_concurrent_requests": 4,
            "memory_guard_gb": 24.5,
            "paged_ssd_cache_dir": str(tmp_path / "ssd"),
            "paged_ssd_cache_max_gb": 50,
            "hot_cache_gb": 8,
        }
    )
    kwargs = m.engine_kwargs()
    assert kwargs["max_concurrent_requests"] == 4
    assert kwargs["memory_guard_gb"] == 24.5
    assert kwargs["paged_ssd_cache_dir"] == str(tmp_path / "ssd")
    assert kwargs["paged_ssd_cache_max_gb"] == 50
    assert kwargs["hot_cache_gb"] == 8


def test_engine_kwargs_ssd_cache_on_by_default(tmp_path) -> None:
    kwargs = _prepared({"model": "org/m", "log_dir": str(tmp_path / "logs")}).engine_kwargs()
    assert kwargs["paged_ssd_cache_dir"] == str(
        tmp_path / "omlx" / "omlx_coder_v1" / "kv-cache"
    )


def test_engine_kwargs_ssd_cache_disabled(tmp_path) -> None:
    kwargs = _prepared(
        {"model": "org/m", "log_dir": str(tmp_path / "logs"), "paged_ssd_cache": False}
    ).engine_kwargs()
    assert "paged_ssd_cache_dir" not in kwargs
    assert "paged_ssd_cache_max_gb" not in kwargs


def test_engine_kwargs_config_override_wins_last(tmp_path) -> None:
    m = _prepared(
        {
            "model": "org/m",
            "log_dir": str(tmp_path / "logs"),
            "max_concurrent_requests": 4,
            "engine_kwargs": {"max_concurrent_requests": 99},
        }
    )
    assert m.engine_kwargs()["max_concurrent_requests"] == 99


# --- memory guard pinned from Sovereign's admission estimate ---
def test_memory_guard_derived_from_estimate_plus_headroom(tmp_path, monkeypatch) -> None:
    info = _repo_info("org/m", [("model.safetensors", 20 * 10**9)], tags=("mlx",))
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    m = _prepared({"model": "org/m", "log_dir": str(tmp_path / "logs"), "hot_cache_gb": 4})
    # weights (20 GB) + hot tier (4 GB) + 2 GB runtime headroom -> the guard
    # omlx's enforcer gets. Headroom covers the interpreter/MLX runtime the
    # admission estimate deliberately doesn't model (a weights-only guard
    # produced a fatal 0.3 GB ceiling on a >1 GB process in the smoke run).
    assert m.engine_kwargs()["memory_guard_gb"] == 26.0


def test_memory_guard_explicit_override_wins(tmp_path, monkeypatch) -> None:
    info = _repo_info("org/m", [("model.safetensors", 20 * 10**9)], tags=("mlx",))
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    m = _prepared(
        {"model": "org/m", "log_dir": str(tmp_path / "logs"), "memory_guard_gb": 30}
    )
    assert m.engine_kwargs()["memory_guard_gb"] == 30


def test_memory_guard_omitted_when_estimate_unknown(tmp_path) -> None:
    # Offline + uncached: estimate is 0 -> no guard flag; omlx falls back to
    # its own default rather than getting a nonsense 0-GB ceiling.
    kwargs = _prepared({"model": "org/m", "log_dir": str(tmp_path / "logs")}).engine_kwargs()
    assert "memory_guard_gb" not in kwargs


# --- resource estimation (§7) ---
def test_extra_memory_bytes_is_hot_cache() -> None:
    assert _manager({"model": "org/m", "hot_cache_gb": 8}).extra_memory_bytes() == 8 * 10**9


def test_extra_memory_bytes_zero_by_default() -> None:
    assert _manager().extra_memory_bytes() == 0


def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="omlx_coder_v1",
        base_type="omlx",
        health_check={"type": "http", "endpoint": "/v1/models", "port": 18000},
        config={"model": "org/m"},
        memory_gb=40,
    )
    assert OmlxManager(entry).estimated_memory_bytes() == 40 * 10**9


# --- api_model_name (must match the adapter's symlink layout AND omlx's
# --- directory-derived id convention: nested segments join with `--`) ---
def test_api_model_name_repo_id_flattened_to_omlx_convention() -> None:
    assert _manager({"model": "mlx-community/m-4bit"}).api_model_name() == (
        "mlx-community--m-4bit"
    )


def test_api_model_name_served_name_with_slash_flattened() -> None:
    m = _manager({"model": "org/m", "served_model_name": "team/coder"})
    assert m.api_model_name() == "team--coder"


def test_api_model_name_local_path_uses_basename(tmp_path) -> None:
    model = tmp_path / "my-mlx-model"
    model.mkdir()
    assert _manager({"model": str(model)}).api_model_name() == "my-mlx-model"


def test_api_model_name_prefers_served_model_name() -> None:
    m = _manager({"model": "org/m", "served_model_name": "coder"})
    assert m.api_model_name() == "coder"
    assert m.endpoint().model == "coder"


# --- worker config handoff ---
def test_get_start_args_dumps_worker_config(tmp_path) -> None:
    m = _prepared(
        {
            "model": "mlx-community/some-model-4bit",
            "log_dir": str(tmp_path / "logs"),
            "max_concurrent_requests": 2,
        }
    )
    args = m.get_start_args()
    assert args[1:4] == ["-m", "sovereign.workers.engine_worker", "--config"]
    cfg = load_worker_config(args[4])
    assert cfg.engine == "omlx"
    assert cfg.service == "omlx_coder_v1"
    assert cfg.port == 18000
    assert cfg.health_path == "/v1/models"
    assert cfg.engine_kwargs["max_concurrent_requests"] == 2
    assert cfg.engine_kwargs["model_name"] == "mlx-community--some-model-4bit"


def test_api_key_never_in_config_json(tmp_path) -> None:
    import json

    m = _prepared(
        {"model": "org/m", "api_key": "secret", "log_dir": str(tmp_path / "logs")}
    )
    m.get_start_args()
    raw = (tmp_path / "workers" / "omlx_coder_v1.json").read_text()
    assert "secret" not in json.dumps(json.loads(raw))


def test_start_env_carries_api_key() -> None:
    assert _manager({"model": "org/m", "api_key": "secret"}).start_env() == {
        "SOVEREIGN_API_KEY": "secret"
    }


def test_start_env_empty_without_api_key() -> None:
    assert _manager().start_env() == {}


# --- provisioning ---
def test_provisioning_declaration() -> None:
    brewfile = OmlxManager.provisioning_brewfile()
    assert brewfile is not None
    assert brewfile.name == "Brewfile"


def test_provisioning_satisfied_uses_binary_probe(monkeypatch) -> None:
    calls: list[str] = []

    def fake_probe(binary: str) -> bool:
        calls.append(binary)
        return True

    monkeypatch.setattr(omlx_manager_mod, "probe_binary", fake_probe)
    assert OmlxManager.provisioning_satisfied() is True
    assert calls == ["omlx"]


# --- prepare_environment ---
def test_prepare_environment_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(omlx_manager_mod, "probe_binary", lambda b: False)
    with pytest.raises(FileNotFoundError, match="omlx.*not found on PATH"):
        _manager({"model": "org/m"}).prepare_environment()


def test_prepare_environment_ok(monkeypatch) -> None:
    monkeypatch.setattr(omlx_manager_mod, "probe_binary", lambda b: True)
    _manager({"model": "org/m"}).prepare_environment()  # must not raise


def test_prepare_environment_rejects_draft_model(monkeypatch) -> None:
    monkeypatch.setattr(omlx_manager_mod, "probe_binary", lambda b: True)
    with pytest.raises(ValueError, match="draft_model is not supported"):
        _manager({"model": "org/m", "draft_model": "org/tiny"}).prepare_environment()


def test_prepare_environment_surfaces_telemetry_gap(monkeypatch, caplog) -> None:
    # ADR 0006: the missing scrape surface is announced, not silently absent.
    monkeypatch.setattr(omlx_manager_mod, "probe_binary", lambda b: True)
    with caplog.at_level("INFO", logger="sovereign"):
        _manager({"model": "org/m"}).prepare_environment()
    assert any("no telemetry scrape surface" in r.message for r in caplog.records)
