"""mlx_vlm manager — mocked unit tests + Protocol/registry checks.

The real ``mlx_vlm.server`` module and an MLX VLM model are not required
here; the shared process/health/metrics lifecycle lives in (and is covered
by) inference.base, so this suite focuses on what mlx_vlm adds: flag/kwarg
mapping (incl. the MTP draft trio), the routing abstention, the env-only API
key, the bearer-authed health probe, and the pre-flight rejections
(num_draft_tokens; draft_kind/draft_block_size without draft_model).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

import sovereign.services  # noqa: F401 - ensure registration side effect
from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ServiceManager
from sovereign.core.registry import get_service_manager
from sovereign.services.inference import base as native_mod
from sovereign.services.inference import hf as models_mod
from sovereign.services.inference.hf import RepoInfo, parse_model_ref
from sovereign.services.inference.mlx_lm.manager import MlxLmManager
from sovereign.services.inference.mlx_vlm.manager import MlxVlmManager
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


@pytest.fixture
def _importable(monkeypatch):
    monkeypatch.setattr(native_mod, "probe_import", lambda m: True)


def _entry(config: dict | None = None, with_health: bool = True) -> ServiceEntry:
    return ServiceEntry(
        name="qwen_vlm",
        base_type="mlx_vlm",
        health_check=(
            {"type": "http", "endpoint": "/health", "port": 8080} if with_health else None
        ),
        config=config or {"model": "mlx-community/some-vlm-4bit"},
    )


def _manager(config: dict | None = None) -> MlxVlmManager:
    return MlxVlmManager(_entry(config))


def _prepared(config: dict | None = None) -> MlxVlmManager:
    m = _manager(config)
    m.prepare_model()
    return m


class FakeProc:
    def __init__(self, pid: int = 4242, poll_value: int | None = None):
        self.pid = pid
        self._poll = poll_value

    def poll(self):
        return self._poll


# --- construction / protocol / registry ---
def test_satisfies_service_manager_protocol() -> None:
    assert isinstance(_manager(), ServiceManager)


def test_registered_under_base_type() -> None:
    assert get_service_manager("mlx_vlm") is MlxVlmManager


def test_requires_health_check() -> None:
    with pytest.raises(ValueError, match="requires a health_check"):
        MlxVlmManager(_entry(with_health=False))


# --- config schema ---
def test_config_rejects_unknown_draft_kind() -> None:
    with pytest.raises(ValueError, match="draft_kind"):
        _manager({"model": "org/m", "draft_kind": "medusa"})


def test_config_rejects_typoed_key() -> None:
    with pytest.raises(ValueError, match="extra_forbidden|draft_blocksize"):
        _manager({"model": "org/m", "draft_blocksize": 4})


# --- routing: deliberate abstention from auto ---
def test_claim_route_abstains_for_vision_mlx_repo() -> None:
    info = _repo_info(
        "mlx-community/some-vlm-4bit",
        [("model.safetensors", 4 * 1024**3)],
        tags=("mlx", "image-text-to-text"),
    )
    ref = parse_model_ref("mlx-community/some-vlm-4bit")
    assert MlxVlmManager.claim_route(ref, info) is None


def test_claim_route_abstains_for_local_mlx_dir(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")
    ref = parse_model_ref(str(tmp_path))
    assert MlxVlmManager.claim_route(ref, None) is None


def test_auto_sweep_prefers_mlx_lm_over_mlx_vlm() -> None:
    """With both engines registered, an mlx repo still routes to mlx_lm —
    mlx_vlm joining the registry must not perturb the precedence contract."""
    info = _repo_info(
        "mlx-community/some-vlm-4bit",
        [("model.safetensors", 4 * 1024**3)],
        tags=("mlx", "image-text-to-text"),
    )
    ref = parse_model_ref("mlx-community/some-vlm-4bit")
    winner = _claim_engine([MlxVlmManager, MlxLmManager], ref, info)
    assert winner is MlxLmManager


# --- engine_kwargs (Sovereign config -> worker adapter mapping) ---
def test_engine_kwargs_full_mapping() -> None:
    m = _prepared(
        {
            "model": "mlx-community/some-vlm-4bit",
            "draft_model": "mlx-community/some-mtp-4bit",
            "max_tokens": 1024,
            "prefill_step_size": 256,
            "vision_cache_size": 8,
            "kv_bits": 4,
            "kv_quant_scheme": "turboquant",
            "kv_group_size": 64,
            "max_kv_size": 131072,
            "quantized_kv_start": 512,
            "draft_kind": "mtp",
            "draft_block_size": 4,
            "thinking_budget": 2048,
            "adapter_path": "/adapters/a",
            "trust_remote_code": True,
            "enable_thinking": True,
        }
    )
    assert m.engine_kwargs() == {
        "max_tokens": 1024,
        "prefill_step_size": 256,
        "vision_cache_size": 8,
        "kv_bits": 4,
        "kv_quant_scheme": "turboquant",
        "kv_group_size": 64,
        "max_kv_size": 131072,
        "quantized_kv_start": 512,
        "draft_kind": "mtp",
        "draft_block_size": 4,
        "thinking_budget": 2048,
        "adapter_path": "/adapters/a",
        "trust_remote_code": True,
        "enable_thinking": True,
    }


def test_engine_kwargs_minimal_omits_unset() -> None:
    assert _prepared().engine_kwargs() == {}


def test_engine_kwargs_config_override_wins_last() -> None:
    m = _prepared(
        {"model": "org/m", "max_tokens": 10, "engine_kwargs": {"max_tokens": 99}}
    )
    assert m.engine_kwargs()["max_tokens"] == 99


def test_engine_kwargs_adapter_path_expands_home(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    m = _prepared({"model": "org/m", "adapter_path": "~/adapters/a"})
    assert m.engine_kwargs()["adapter_path"] == "/home/tester/adapters/a"


# --- worker config handoff ---
def test_get_start_args_dumps_worker_config(tmp_path) -> None:
    m = _prepared(
        {
            "model": "mlx-community/some-vlm-4bit",
            "draft_model": "mlx-community/some-mtp-4bit",
            "draft_kind": "mtp",
            "draft_block_size": 4,
            "log_dir": str(tmp_path / "logs"),
        }
    )
    args = m.get_start_args()
    assert args[1:4] == ["-m", "sovereign.workers.engine_worker", "--config"]
    cfg = load_worker_config(args[4])
    assert cfg.engine == "mlx_vlm"
    assert cfg.service == "qwen_vlm"
    assert cfg.port == 8080
    assert cfg.health_path == "/health"
    assert cfg.model_path == "mlx-community/some-vlm-4bit"
    assert cfg.draft_model_path == "mlx-community/some-mtp-4bit"
    assert cfg.engine_kwargs["draft_kind"] == "mtp"
    assert cfg.engine_kwargs["draft_block_size"] == 4


def test_api_key_never_in_config_json(tmp_path) -> None:
    m = _prepared({"model": "org/m", "api_key": "secret", "log_dir": str(tmp_path / "logs")})
    m.get_start_args()
    raw = (tmp_path / "workers" / "qwen_vlm.json").read_text()
    assert "secret" not in json.dumps(json.loads(raw))


def test_start_env_carries_api_key_in_native_env_var() -> None:
    assert _manager({"model": "org/m", "api_key": "secret"}).start_env() == {
        "MLX_VLM_SERVER_API_KEY": "secret"
    }


def test_start_env_empty_without_api_key() -> None:
    assert _manager().start_env() == {}


# --- prepare_environment ---
def test_prepare_environment_missing_binding(monkeypatch) -> None:
    monkeypatch.setattr(native_mod, "probe_import", lambda m: False)
    with pytest.raises(FileNotFoundError, match="mlx_vlm.server.*not importable"):
        _manager().prepare_environment()


def test_prepare_environment_repo_id_ok(_importable) -> None:
    # A repo id that isn't local must NOT raise (downloaded on start).
    _manager({"model": "mlx-community/some-vlm-4bit"}).prepare_environment()


def test_prepare_environment_missing_local_draft_raises(_importable) -> None:
    with pytest.raises(FileNotFoundError, match="draft_model"):
        _manager(
            {"model": "org/m", "draft_model": "/nope/missing-draft"}
        ).prepare_environment()


def test_prepare_environment_missing_adapter_raises(_importable) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_path"):
        _manager(
            {"model": "org/m", "adapter_path": "/nope/missing-adapter"}
        ).prepare_environment()


def test_prepare_environment_rejects_num_draft_tokens(_importable) -> None:
    # ADR 0006: surfaced loudly — mlx-vlm sizes drafts with draft_block_size.
    with pytest.raises(ValueError, match="num_draft_tokens is not supported"):
        _manager(
            {"model": "org/m", "draft_model": "org/mtp", "num_draft_tokens": 3}
        ).prepare_environment()


def test_prepare_environment_rejects_draft_kind_without_draft_model(_importable) -> None:
    with pytest.raises(ValueError, match="require.*draft_model"):
        _manager({"model": "org/m", "draft_kind": "mtp"}).prepare_environment()


def test_prepare_environment_rejects_draft_block_size_without_draft_model(_importable) -> None:
    with pytest.raises(ValueError, match="require.*draft_model"):
        _manager({"model": "org/m", "draft_block_size": 4}).prepare_environment()


def test_prepare_environment_mtp_trio_ok(_importable) -> None:
    _manager(
        {
            "model": "mlx-community/some-vlm-4bit",
            "draft_model": "mlx-community/some-mtp-4bit",
            "draft_kind": "mtp",
            "draft_block_size": 4,
        }
    ).prepare_environment()  # must not raise


def test_prepare_environment_surfaces_telemetry_gap(_importable, caplog) -> None:
    # ADR 0006: the missing translator is announced, not silently absent.
    with caplog.at_level("INFO", logger="sovereign"):
        _manager({"model": "org/m"}).prepare_environment()
    assert any("no telemetry translator" in r.message for r in caplog.records)


# --- resource estimation (§7): drafter weights count toward admission ---
def test_estimated_memory_includes_local_draft_model(tmp_path, sparse_file) -> None:
    model_dir = tmp_path / "main"
    model_dir.mkdir()
    sparse_file(model_dir / "weights.safetensors", 2 * 1024**3)  # 2 GiB
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    sparse_file(draft_dir / "weights.safetensors", 1 * 1024**3)  # 1 GiB
    m = _manager({"model": str(model_dir), "draft_model": str(draft_dir)})
    assert m.estimated_memory_bytes() == 3 * 1024**3


def test_estimated_memory_uses_declared_override() -> None:
    entry = ServiceEntry(
        name="qwen_vlm",
        base_type="mlx_vlm",
        health_check={"type": "http", "endpoint": "/health", "port": 8080},
        config={"model": "org/m"},
        memory_gb=40,
    )
    assert MlxVlmManager(entry).estimated_memory_bytes() == 40 * 10**9


# --- health: bearer auth when api_key is set ---
class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_is_healthy_plain_probe_without_api_key(monkeypatch) -> None:
    m = _manager()
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=None))
    seen: list = []

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return _FakeResp()

    monkeypatch.setattr(native_mod.urllib.request, "urlopen", fake_urlopen)
    assert m.is_healthy() is True
    assert seen == ["http://127.0.0.1:8080/health"]  # plain URL, no Request wrapper


def test_is_healthy_sends_bearer_header_with_api_key(monkeypatch) -> None:
    import sovereign.services.inference.mlx_vlm.manager as vlm_manager_mod

    m = _manager({"model": "org/m", "api_key": "secret"})
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=None))
    seen: list = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return _FakeResp()

    monkeypatch.setattr(vlm_manager_mod.urllib.request, "urlopen", fake_urlopen)
    assert m.is_healthy() is True
    (request,) = seen
    assert request.get_full_url() == "http://127.0.0.1:8080/health"
    assert request.get_header("Authorization") == "Bearer secret"


def test_is_healthy_false_on_connection_error_with_api_key(monkeypatch) -> None:
    import sovereign.services.inference.mlx_vlm.manager as vlm_manager_mod

    m = _manager({"model": "org/m", "api_key": "secret"})
    m.process = cast("subprocess.Popen[bytes]", FakeProc(poll_value=None))

    def boom(request, timeout=None):
        raise native_mod.urllib.error.URLError("refused")

    monkeypatch.setattr(vlm_manager_mod.urllib.request, "urlopen", boom)
    assert m.is_healthy() is False


def test_is_healthy_false_when_no_process_with_api_key() -> None:
    assert _manager({"model": "org/m", "api_key": "secret"}).is_healthy() is False
