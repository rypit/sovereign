"""Engine routing: per-engine ``claim_route`` rules + the ``auto`` routing sweep.

The routing *decision* moved out of the HF pipeline into the engines (each
declares what it claims) plus a generic sweep. These tests exercise both halves:
the claim rules directly, and the orchestration through the public
``route_entry`` seam boot and ``plan`` share.
"""

from __future__ import annotations

import pytest

from sovereign.config import ServiceEntry
from sovereign.core.errors import RoutingError
from sovereign.core.registry import route_entry
from sovereign.services.inference_engines import hf as models_mod
from sovereign.services.inference_engines.hf import RepoInfo, RoutingCache, parse_model_ref
from sovereign.services.inference_engines.llama_cpp.manager import LlamaCppManager
from sovereign.services.inference_engines.mlx_lm.manager import MlxLmManager
from sovereign.services.inference_engines.routing import _claim_engine

# The two native engines, swept exactly as ``route_entry`` sweeps the registry.
_ENGINES = [LlamaCppManager, MlxLmManager]


def _repo(
    repo_id: str = "org/model",
    tags: tuple[str, ...] = (),
    siblings: tuple[tuple[str, int | None], ...] = (),
) -> RepoInfo:
    return RepoInfo(repo_id=repo_id, tags=tags, siblings=siblings)


def _svc(model: str, base_type: str = "auto") -> ServiceEntry:
    return ServiceEntry(name="svc", base_type=base_type, config={"model": model})


def _route(ref, info) -> str | None:
    """The base_type the highest-confidence engine claims, or None if none claim."""
    engine = _claim_engine(_ENGINES, ref, info)
    return engine.base_type if engine is not None else None


# ---------------------------------------------------------------------------
# claim_route — local refs (no network)
# ---------------------------------------------------------------------------


def test_claim_local_gguf_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.touch()
    assert _route(parse_model_ref(str(f)), None) == "llama_cpp"


def test_claim_local_dir_with_gguf(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.gguf").touch()
    assert _route(parse_model_ref(str(d)), None) == "llama_cpp"


def test_claim_local_dir_with_config_json(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").touch()
    assert _route(parse_model_ref(str(d)), None) == "mlx_lm"


def test_claim_local_dir_with_safetensors(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.safetensors").touch()
    assert _route(parse_model_ref(str(d)), None) == "mlx_lm"


def test_claim_local_dir_gguf_beats_config(tmp_path):
    # A dir with both signals routes to llama_cpp (GGUF outranks safetensors/config).
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.gguf").touch()
    (d / "config.json").touch()
    assert _route(parse_model_ref(str(d)), None) == "llama_cpp"


def test_claim_local_unknown_none(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    assert _route(parse_model_ref(str(d)), None) is None


# ---------------------------------------------------------------------------
# claim_route — ref hints and hub metadata
# ---------------------------------------------------------------------------


def test_claim_quant_set_implies_llama_cpp():
    assert _route(parse_model_ref("org/model:Q4_K_M"), None) == "llama_cpp"


def test_claim_filename_set_implies_llama_cpp():
    assert _route(parse_model_ref("org/model/file.gguf"), None) == "llama_cpp"


def test_claim_mlx_tag():
    info = _repo("org/model", tags=("mlx",))
    assert _route(parse_model_ref("org/model"), info) == "mlx_lm"


def test_claim_mlx_community_org():
    info = _repo("mlx-community/SmolLM")
    assert _route(parse_model_ref("mlx-community/SmolLM"), info) == "mlx_lm"


def test_claim_gguf_siblings():
    info = _repo(siblings=(("model.gguf", 1024),))
    assert _route(parse_model_ref("org/model"), info) == "llama_cpp"


def test_claim_safetensors_siblings():
    info = _repo(siblings=(("model.safetensors", 1024),))
    assert _route(parse_model_ref("org/model"), info) == "mlx_lm"


def test_claim_mlx_tag_beats_gguf_sibling():
    # mlx-community org + gguf sibling → mlx_lm wins (the mlx signal outranks it).
    info = _repo("mlx-community/model", siblings=(("model.gguf", 1024),))
    assert _route(parse_model_ref("mlx-community/model"), info) == "mlx_lm"


def test_claim_gguf_beats_safetensors_sibling():
    # A repo shipping both formats routes to llama_cpp (GGUF outranks safetensors).
    info = _repo(siblings=(("model.gguf", 1024), ("model.safetensors", 1024)))
    assert _route(parse_model_ref("org/model"), info) == "llama_cpp"


def test_claim_no_gguf_no_safetensors_none():
    info = _repo(siblings=(("config.json", 100),))
    assert _route(parse_model_ref("org/model"), info) is None


# ---------------------------------------------------------------------------
# route_entry — the auto-routing orchestration (sweep + cache + offline)
# ---------------------------------------------------------------------------


def test_route_entry_explicit_base_type_untouched(tmp_path):
    assert route_entry(_svc("org/model", base_type="llama_cpp"), tmp_path) == "llama_cpp"


def test_route_entry_online_routes_and_writes_cache(tmp_path, monkeypatch):
    info = _repo("mlx-community/foo", tags=("mlx",), siblings=(("model.safetensors", 100),))
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    assert route_entry(_svc("mlx-community/foo"), tmp_path) == "mlx_lm"
    # Persisted for deterministic offline restarts.
    cached = RoutingCache(tmp_path / "models.json").get("mlx-community/foo")
    assert cached is not None
    assert cached["base_type"] == "mlx_lm"


def test_route_entry_offline_uses_cache(tmp_path, monkeypatch):
    RoutingCache(tmp_path / "models.json").put(
        "org/model", base_type="llama_cpp", weight_bytes=None
    )
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)  # offline
    assert route_entry(_svc("org/model"), tmp_path) == "llama_cpp"


def test_route_entry_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: None)  # offline
    with pytest.raises(RoutingError, match="offline"):
        route_entry(_svc("org/model"), tmp_path)


def test_route_entry_online_unclaimed_raises(tmp_path, monkeypatch):
    info = _repo(siblings=(("config.json", 100),))  # no gguf, no safetensors
    monkeypatch.setattr(models_mod, "fetch_repo_info", lambda repo_id: info)
    with pytest.raises(RoutingError, match="no engine claims"):
        route_entry(_svc("org/model"), tmp_path)


def test_route_entry_local_never_needs_network(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"gguf")
    assert route_entry(_svc(str(tmp_path / "m.gguf")), tmp_path) == "llama_cpp"


def test_route_entry_local_unknown_raises(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    with pytest.raises(RoutingError, match="no engine claims"):
        route_entry(_svc(str(d)), tmp_path)
