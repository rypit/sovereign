"""comfyui manager — mocked unit tests + Protocol/registry checks.

The real comfy-cli is not required here; the shared process/health/metrics
lifecycle lives in (and is covered by) inference.base, so this suite focuses
on what comfyui adds: the checkpoint artifact kind, kwarg mapping for the
worker adapter, the routing abstention, the workspace pre-flight install, and
the ADR 0006 gap surfacing (no draft_model, no telemetry scrape surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.inference import hf as models_mod
from sovereign.services.inference.comfyui import manager as comfyui_manager_mod
from sovereign.services.inference.comfyui.manager import ComfyUIManager
from sovereign.services.inference.hf import RepoInfo, parse_model_ref
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
    ref as if it downloaded to a path equal to the ref, so kwarg assertions stay
    readable. Applied suite-wide so no test reaches the network on prepare_model()."""

    def _fake(ref, kind, *, progress=None):
        return ref.local_path if ref.is_local else Path(ref.raw)

    monkeypatch.setattr(models_mod, "download_model", _fake)


_MODEL_REF = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="sdxl",
        base_type="comfyui",
        health_check=(
            {"type": "http", "endpoint": "/system_stats", "port": 8188}
            if with_health
            else None
        ),
        config=config or {"model": _MODEL_REF},
    )


def _manager(config: dict | None = None) -> ComfyUIManager:
    return ComfyUIManager(_entry(config))


def _prepared(config: dict | None = None) -> ComfyUIManager:
    m = _manager(config)
    m.prepare_model()
    return m


# --- construction / protocol / registry ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("comfyui") is ComfyUIManager


def test_requires_health_check() -> None:
    with pytest.raises(ValueError, match="requires a health_check"):
        ComfyUIManager(_entry(with_health=False))


def test_model_artifact_kind_is_checkpoint() -> None:
    assert ComfyUIManager.model_artifact_kind == "checkpoint"


# --- routing: deliberate abstention from auto ---
def test_claim_route_abstains_for_diffusion_repo() -> None:
    info = _repo_info(
        "stabilityai/stable-diffusion-xl-base-1.0",
        [("sd_xl_base_1.0.safetensors", 7 * 10**9)],
        tags=("text-to-image",),
    )
    ref = parse_model_ref("stabilityai/stable-diffusion-xl-base-1.0")
    assert ComfyUIManager.claim_route(ref, info) is None


def test_claim_route_abstains_for_local_checkpoint(tmp_path) -> None:
    ckpt = tmp_path / "sd.safetensors"
    ckpt.write_bytes(b"w")
    ref = parse_model_ref(str(ckpt))
    assert ComfyUIManager.claim_route(ref, None) is None


# --- engine_kwargs (Sovereign config -> worker adapter mapping) ---
def test_engine_kwargs_carries_workspace_models_root_and_checkpoint(tmp_path) -> None:
    m = _prepared({"model": _MODEL_REF, "log_dir": str(tmp_path / "logs")})
    kwargs = m.engine_kwargs()
    assert kwargs["workspace_dir"].endswith(".sovereign/comfyui")
    assert kwargs["models_root"] == str(tmp_path / "comfyui" / "sdxl" / "models")
    assert kwargs["checkpoint_name"] == "sd_xl_base_1.0.safetensors"


def test_engine_kwargs_expands_workspace_dir(tmp_path) -> None:
    m = _prepared(
        {"model": _MODEL_REF, "log_dir": str(tmp_path / "logs"), "workspace_dir": "~/comfy"}
    )
    assert "~" not in m.engine_kwargs()["workspace_dir"]


def test_engine_kwargs_maps_output_dir(tmp_path) -> None:
    m = _prepared(
        {
            "model": _MODEL_REF,
            "log_dir": str(tmp_path / "logs"),
            "output_dir": str(tmp_path / "outputs"),
        }
    )
    assert m.engine_kwargs()["output_dir"] == str(tmp_path / "outputs")


def test_engine_kwargs_omits_output_dir_when_unset(tmp_path) -> None:
    kwargs = _prepared({"model": _MODEL_REF, "log_dir": str(tmp_path / "logs")}).engine_kwargs()
    assert "output_dir" not in kwargs


def test_engine_kwargs_config_override_wins_last(tmp_path) -> None:
    m = _prepared(
        {
            "model": _MODEL_REF,
            "log_dir": str(tmp_path / "logs"),
            "engine_kwargs": {"checkpoint_name": "override.safetensors"},
        }
    )
    assert m.engine_kwargs()["checkpoint_name"] == "override.safetensors"


# --- resource estimation (§7): checkpoint file size from repo metadata ---
def test_estimated_memory_uses_checkpoint_size(monkeypatch) -> None:
    info = _repo_info(
        "stabilityai/stable-diffusion-xl-base-1.0",
        [
            ("sd_xl_base_1.0.safetensors", 7 * 10**9),
            ("vae/diffusion_pytorch_model.safetensors", 1 * 10**9),
        ],
        tags=("text-to-image",),
    )
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    assert _manager().estimated_memory_bytes() == 7 * 10**9


def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="sdxl",
        base_type="comfyui",
        health_check={"type": "http", "endpoint": "/system_stats", "port": 8188},
        config={"model": _MODEL_REF},
        memory_gb=16,
    )
    assert ComfyUIManager(entry).estimated_memory_bytes() == 16 * 10**9


def test_extra_memory_bytes_zero() -> None:
    assert _manager().extra_memory_bytes() == 0


# --- api_model_name (must match the adapter's symlink name: the checkpoint
# --- filename workflows put in CheckpointLoaderSimple) ---
def test_api_model_name_uses_ref_filename() -> None:
    assert _manager().api_model_name() == "sd_xl_base_1.0.safetensors"


def test_api_model_name_local_path_uses_basename(tmp_path) -> None:
    ckpt = tmp_path / "my-model.safetensors"
    ckpt.write_bytes(b"w")
    assert _manager({"model": str(ckpt)}).api_model_name() == "my-model.safetensors"


def test_api_model_name_prefers_served_model_name() -> None:
    m = _manager({"model": _MODEL_REF, "served_model_name": "sdxl.safetensors"})
    assert m.api_model_name() == "sdxl.safetensors"
    assert m.endpoint().model == "sdxl.safetensors"


def test_api_model_name_bare_repo_falls_back_to_repo_id() -> None:
    assert _manager({"model": "org/one-checkpoint-repo"}).api_model_name() == (
        "org/one-checkpoint-repo"
    )


# --- worker config handoff ---
def test_get_start_args_dumps_worker_config(tmp_path) -> None:
    m = _prepared({"model": _MODEL_REF, "log_dir": str(tmp_path / "logs")})
    args = m.get_start_args()
    assert args[1:4] == ["-m", "sovereign.workers.engine_worker", "--config"]
    cfg = load_worker_config(args[4])
    assert cfg.engine == "comfyui"
    assert cfg.service == "sdxl"
    assert cfg.port == 8188
    assert cfg.health_path == "/system_stats"
    assert cfg.engine_kwargs["checkpoint_name"] == "sd_xl_base_1.0.safetensors"


# --- provisioning ---
def test_provisioning_installs_comfy_cli() -> None:
    assert ["uv", "tool", "install", "comfy-cli"] in ComfyUIManager.provisioning_commands
    assert ComfyUIManager.provisioning_brewfile() is None  # pip tool, not a formula


def test_provisioning_satisfied_uses_binary_probe(monkeypatch) -> None:
    calls: list[str] = []

    def fake_probe(binary: str) -> bool:
        calls.append(binary)
        return True

    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", fake_probe)
    assert ComfyUIManager.provisioning_satisfied() is True
    assert calls == ["comfy"]


# --- prepare_environment ---
@pytest.fixture()
def _installed_workspace(tmp_path):
    """A workspace dir that already contains a ComfyUI checkout (comfy-cli layout)."""
    workspace = tmp_path / "workspace"
    (workspace / "ComfyUI").mkdir(parents=True)
    (workspace / "ComfyUI" / "main.py").write_text("")
    return workspace


def test_prepare_environment_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: False)
    with pytest.raises(FileNotFoundError, match="comfy.*not found on PATH"):
        _manager().prepare_environment()


def test_prepare_environment_ok(monkeypatch, _installed_workspace) -> None:
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: True)
    m = _manager({"model": _MODEL_REF, "workspace_dir": str(_installed_workspace)})
    m.prepare_environment()  # must not raise


def test_prepare_environment_rejects_draft_model(monkeypatch) -> None:
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: True)
    with pytest.raises(ValueError, match="draft_model is not supported"):
        _manager({"model": _MODEL_REF, "draft_model": "org/tiny"}).prepare_environment()


def test_prepare_environment_installs_missing_workspace(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: True)
    installed: list[str] = []
    monkeypatch.setattr(comfyui_manager_mod, "_install_workspace", installed.append)
    workspace = tmp_path / "fresh-workspace"
    _manager({"model": _MODEL_REF, "workspace_dir": str(workspace)}).prepare_environment()
    assert installed == [str(workspace)]


def test_prepare_environment_skips_install_when_workspace_present(
    monkeypatch, _installed_workspace
) -> None:
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: True)
    installed: list[str] = []
    monkeypatch.setattr(comfyui_manager_mod, "_install_workspace", installed.append)
    m = _manager({"model": _MODEL_REF, "workspace_dir": str(_installed_workspace)})
    m.prepare_environment()
    assert installed == []


def test_prepare_environment_surfaces_telemetry_gap(
    monkeypatch, caplog, _installed_workspace
) -> None:
    # ADR 0006: the missing scrape surface is announced, not silently absent.
    monkeypatch.setattr(comfyui_manager_mod, "probe_binary", lambda b: True)
    m = _manager({"model": _MODEL_REF, "workspace_dir": str(_installed_workspace)})
    with caplog.at_level("INFO", logger="sovereign"):
        m.prepare_environment()
    assert any("no telemetry scrape surface" in r.message for r in caplog.records)
