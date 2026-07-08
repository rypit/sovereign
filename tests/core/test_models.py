"""Phase M1: unit tests for sovereign.core.models.

Pure unit — no real HF network calls.  HfApi.model_info is monkeypatched or
RepoInfo is built directly.  Mirrors the mock style in tests/services/test_mlx_lm.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sovereign.core.models import (
    ModelAccessError,
    ModelNotFoundError,
    ModelResolutionError,
    RepoInfo,
    RoutingCache,
    RoutingError,
    _DownloadProgressSampler,
    _repo_info_cache,
    estimate_model_bytes,
    fetch_repo_info,
    parse_model_ref,
    route_base_type,
    select_gguf_files,
    weight_bytes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo(
    repo_id: str = "org/model",
    tags: tuple[str, ...] = (),
    siblings: tuple[tuple[str, int | None], ...] = (),
) -> RepoInfo:
    return RepoInfo(repo_id=repo_id, tags=tags, siblings=siblings)


def _siblings(*entries: tuple[str, int | None]) -> tuple[tuple[str, int | None], ...]:
    return entries


@pytest.fixture(autouse=True)
def _clear_repo_cache():
    """Each test gets a fresh metadata cache so monkeypatches don't bleed."""
    _repo_info_cache.clear()
    yield
    _repo_info_cache.clear()


# ---------------------------------------------------------------------------
# parse_model_ref
# ---------------------------------------------------------------------------


def test_parse_repo_id():
    ref = parse_model_ref("org/model")
    assert not ref.is_local
    assert ref.repo_id == "org/model"
    assert ref.quant is None
    assert ref.filename is None


def test_parse_repo_with_quant():
    ref = parse_model_ref("org/model:Q4_K_M")
    assert not ref.is_local
    assert ref.repo_id == "org/model"
    assert ref.quant == "Q4_K_M"
    assert ref.filename is None


def test_parse_repo_with_filename_gguf():
    ref = parse_model_ref("org/model/sub/file.gguf")
    assert not ref.is_local
    assert ref.repo_id == "org/model"
    assert ref.filename == "sub/file.gguf"
    assert ref.quant is None


def test_parse_repo_with_bare_gguf_filename():
    ref = parse_model_ref("org/model/file.gguf")
    assert not ref.is_local
    assert ref.repo_id == "org/model"
    assert ref.filename == "file.gguf"


def test_parse_local_absolute():
    ref = parse_model_ref("/tmp/some-model")
    assert ref.is_local
    assert ref.local_path == Path("/tmp/some-model")


def test_parse_local_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    ref = parse_model_ref("~/models/mlx-foo")
    assert ref.is_local
    assert ref.local_path == Path("/home/tester/models/mlx-foo")


def test_parse_local_relative(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.chdir(tmp_path)
    ref = parse_model_ref("./model")
    assert ref.is_local


def test_parse_existing_path_treated_as_local(tmp_path):
    model = tmp_path / "weights.gguf"
    model.touch()
    ref = parse_model_ref(str(model))
    assert ref.is_local
    assert ref.local_path == model


def test_parse_raw_preserved():
    raw = "org/name:Q8_0"
    ref = parse_model_ref(raw)
    assert ref.raw == raw


# ---------------------------------------------------------------------------
# select_gguf_files
# ---------------------------------------------------------------------------


def _gguf_repo(*filenames: str, sizes: list[int | None] | None = None) -> RepoInfo:
    if sizes is None:
        sizes = [1024] * len(filenames)
    siblings = tuple(zip(filenames, sizes, strict=True))
    return _repo("org/model", siblings=siblings)


def test_select_single_quant():
    info = _gguf_repo("model-Q4_K_M.gguf")
    assert select_gguf_files(info, quant=None, filename=None) == ["model-Q4_K_M.gguf"]


def test_select_quant_arg_matching():
    info = _gguf_repo("model-Q4_K_M.gguf", "model-Q8_0.gguf")
    assert select_gguf_files(info, quant="Q8_0", filename=None) == ["model-Q8_0.gguf"]


def test_select_quant_case_insensitive():
    info = _gguf_repo("model-Q4_K_M.gguf", "model-Q8_0.gguf")
    assert select_gguf_files(info, quant="q4_k_m", filename=None) == ["model-Q4_K_M.gguf"]


def test_select_quant_ambiguity_error_lists_quants():
    info = _gguf_repo("model-Q4_K_M.gguf", "model-Q4_K_S.gguf")
    with pytest.raises(ModelResolutionError, match="Q4_K"):
        select_gguf_files(info, quant="Q4_K", filename=None)


def test_select_auto_prefers_q4_k_m():
    info = _gguf_repo("model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-IQ2_XXS.gguf")
    assert select_gguf_files(info, quant=None, filename=None) == ["model-Q4_K_M.gguf"]


def test_select_auto_multiple_no_q4_raises():
    info = _gguf_repo("model-Q8_0.gguf", "model-IQ2_XXS.gguf")
    with pytest.raises(ModelResolutionError, match="Multiple quants"):
        select_gguf_files(info, quant=None, filename=None)


def test_select_shard_grouping_and_order():
    shards = [
        "model-Q4_K_M-00002-of-00003.gguf",
        "model-Q4_K_M-00001-of-00003.gguf",
        "model-Q4_K_M-00003-of-00003.gguf",
    ]
    info = _gguf_repo(*shards)
    result = select_gguf_files(info, quant=None, filename=None)
    assert result == sorted(shards)


def test_select_shard_collapse_to_one_quant():
    shards = [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q8_0.gguf",
    ]
    info = _gguf_repo(*shards)
    result = select_gguf_files(info, quant="Q4_K_M", filename=None)
    assert result == sorted(s for s in shards if "Q4_K_M" in s)


def test_select_mmproj_excluded():
    info = _gguf_repo("model-Q4_K_M.gguf", "mmproj-Q4_K_M.gguf")
    result = select_gguf_files(info, quant=None, filename=None)
    assert "mmproj-Q4_K_M.gguf" not in result


def test_select_explicit_filename_hit():
    info = _gguf_repo("model-Q4_K_M.gguf", "model-Q8_0.gguf")
    assert select_gguf_files(info, quant=None, filename="model-Q8_0.gguf") == ["model-Q8_0.gguf"]


def test_select_explicit_filename_miss():
    info = _gguf_repo("model-Q4_K_M.gguf")
    with pytest.raises(ModelResolutionError, match="not found"):
        select_gguf_files(info, quant=None, filename="nope.gguf")


def test_select_no_gguf_files_raises():
    info = _repo(siblings=(("model.safetensors", 1024),))
    with pytest.raises(ModelResolutionError, match="No GGUF"):
        select_gguf_files(info, quant=None, filename=None)


# ---------------------------------------------------------------------------
# weight_bytes
# ---------------------------------------------------------------------------


def test_weight_bytes_safetensors_sum():
    info = _repo(siblings=(
        ("model-00001-of-00002.safetensors", 1_000_000),
        ("model-00002-of-00002.safetensors", 500_000),
        ("tokenizer.json", 10_000),
    ))
    assert weight_bytes(info, "snapshot") == 1_500_000


def test_weight_bytes_consolidated_dedup():
    # When both model* and consolidated* present, use only model*
    info = _repo(siblings=(
        ("model.safetensors", 1_000_000),
        ("consolidated.safetensors", 1_000_000),
    ))
    assert weight_bytes(info, "snapshot") == 1_000_000


def test_weight_bytes_bin_fallback():
    info = _repo(siblings=(
        ("pytorch_model.bin", 2_000_000),
        ("config.json", 1_000),
    ))
    assert weight_bytes(info, "snapshot") == 2_000_000


def test_weight_bytes_gguf_shard_sum():
    info = _repo(siblings=(
        ("model-Q4_K_M-00001-of-00002.gguf", 1_000_000),
        ("model-Q4_K_M-00002-of-00002.gguf", 800_000),
    ))
    assert weight_bytes(info, "gguf") == 1_800_000


def test_weight_bytes_missing_size_returns_none():
    info = _repo(siblings=(
        ("model.safetensors", None),
    ))
    assert weight_bytes(info, "snapshot") is None


def test_weight_bytes_gguf_missing_size_returns_none():
    info = _repo(siblings=(
        ("model-Q4_K_M.gguf", None),
    ))
    assert weight_bytes(info, "gguf") is None


def test_weight_bytes_no_safetensors_no_bin_returns_none():
    info = _repo(siblings=(
        ("config.json", 1000),
    ))
    assert weight_bytes(info, "snapshot") is None


# ---------------------------------------------------------------------------
# estimate_model_bytes
# ---------------------------------------------------------------------------


def test_estimate_local_dir(tmp_path, sparse_file):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sparse_file(model_dir / "weights.safetensors", 2 * 1024**3)
    ref = parse_model_ref(str(model_dir))
    result = estimate_model_bytes(ref, "snapshot")
    assert result == 2 * 1024**3


def test_estimate_local_file(tmp_path, sparse_file):
    model_file = tmp_path / "model.gguf"
    sparse_file(model_file, 1 * 1024**3)
    ref = parse_model_ref(str(model_file))
    result = estimate_model_bytes(ref, "gguf")
    assert result == 1 * 1024**3


def test_estimate_local_beats_api(tmp_path, sparse_file, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sparse_file(model_dir / "weights.safetensors", 4 * 1024**3)
    # Even if API would return something, local should be used
    monkeypatch.setattr("sovereign.core.models.fetch_repo_info", lambda _: None)
    ref = parse_model_ref(str(model_dir))
    assert estimate_model_bytes(ref, "snapshot") == 4 * 1024**3


def test_estimate_offline_uncached_returns_none(monkeypatch):
    monkeypatch.setattr("sovereign.core.models.fetch_repo_info", lambda _: None)
    ref = parse_model_ref("org/model")
    assert estimate_model_bytes(ref, "snapshot") is None


def test_estimate_from_api(monkeypatch):
    fake_info = _repo(
        "org/model",
        siblings=(("model.safetensors", 3 * 1024**3),),
    )
    monkeypatch.setattr("sovereign.core.models.fetch_repo_info", lambda _: fake_info)
    ref = parse_model_ref("org/model")
    assert estimate_model_bytes(ref, "snapshot") == 3 * 1024**3


# ---------------------------------------------------------------------------
# route_base_type
# ---------------------------------------------------------------------------


def test_route_local_gguf_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.touch()
    ref = parse_model_ref(str(f))
    assert route_base_type(ref, None) == "llama_cpp"


def test_route_local_dir_with_gguf(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.gguf").touch()
    ref = parse_model_ref(str(d))
    assert route_base_type(ref, None) == "llama_cpp"


def test_route_local_dir_with_config_json(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").touch()
    ref = parse_model_ref(str(d))
    assert route_base_type(ref, None) == "mlx_lm"


def test_route_local_dir_with_safetensors(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.safetensors").touch()
    ref = parse_model_ref(str(d))
    assert route_base_type(ref, None) == "mlx_lm"


def test_route_local_unknown_raises(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    ref = parse_model_ref(str(d))
    with pytest.raises(RoutingError):
        route_base_type(ref, None)


def test_route_quant_set_implies_llama_cpp():
    ref = parse_model_ref("org/model:Q4_K_M")
    assert route_base_type(ref, None) == "llama_cpp"


def test_route_filename_set_implies_llama_cpp():
    ref = parse_model_ref("org/model/file.gguf")
    assert route_base_type(ref, None) == "llama_cpp"


def test_route_mlx_tag():
    info = _repo("org/model", tags=("mlx",))
    ref = parse_model_ref("org/model")
    assert route_base_type(ref, info) == "mlx_lm"


def test_route_mlx_community_org():
    info = _repo("mlx-community/SmolLM", tags=())
    ref = parse_model_ref("mlx-community/SmolLM")
    assert route_base_type(ref, info) == "mlx_lm"


def test_route_gguf_siblings():
    info = _repo(siblings=(("model.gguf", 1024),))
    ref = parse_model_ref("org/model")
    assert route_base_type(ref, info) == "llama_cpp"


def test_route_safetensors_siblings():
    info = _repo(siblings=(("model.safetensors", 1024),))
    ref = parse_model_ref("org/model")
    assert route_base_type(ref, info) == "mlx_lm"


def test_route_mlx_tag_beats_gguf_sibling():
    # mlx-community org + gguf sibling → mlx_lm wins (mlx signal takes priority)
    info = _repo("mlx-community/model", tags=(), siblings=(("model.gguf", 1024),))
    ref = parse_model_ref("mlx-community/model")
    assert route_base_type(ref, info) == "mlx_lm"


def test_route_offline_no_info_raises():
    ref = parse_model_ref("org/model")
    with pytest.raises(RoutingError, match="offline"):
        route_base_type(ref, None)


def test_route_no_gguf_no_safetensors_raises():
    info = _repo(siblings=(("config.json", 100),))
    ref = parse_model_ref("org/model")
    with pytest.raises(RoutingError):
        route_base_type(ref, info)


# ---------------------------------------------------------------------------
# RoutingCache
# ---------------------------------------------------------------------------


def test_routing_cache_round_trip(tmp_path):
    path = tmp_path / "models.json"
    cache = RoutingCache(path)
    cache.put("org/model", base_type="mlx_lm", weight_bytes=3 * 1024**3)
    # Reload from disk
    cache2 = RoutingCache(path)
    entry = cache2.get("org/model")
    assert entry is not None
    assert entry["base_type"] == "mlx_lm"
    assert entry["weight_bytes"] == 3 * 1024**3
    assert "resolved_at" in entry


def test_routing_cache_miss_returns_none(tmp_path):
    cache = RoutingCache(tmp_path / "models.json")
    assert cache.get("org/missing") is None


def test_routing_cache_missing_file_ok(tmp_path):
    cache = RoutingCache(tmp_path / "nonexistent.json")
    assert cache.get("anything") is None


# ---------------------------------------------------------------------------
# fetch_repo_info error mapping
# ---------------------------------------------------------------------------


def _make_hfapi_info(tags, siblings):
    """Build a minimal mock for what HfApi().model_info returns."""
    mock_info = MagicMock()
    mock_info.tags = list(tags)
    mock_info.siblings = [
        MagicMock(rfilename=name, size=size) for name, size in siblings
    ]
    return mock_info


def test_fetch_gated_repo_raises_model_access_error():
    from huggingface_hub.errors import GatedRepoError as _GRE

    fake_resp = MagicMock()
    fake_resp.headers = {}
    exc = _GRE("gated", response=fake_resp)
    with patch("sovereign.core.models.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = exc
        with pytest.raises(ModelAccessError, match="gated"):
            fetch_repo_info("org/gated-model")


def test_fetch_not_found_raises_model_not_found_error():
    from huggingface_hub.errors import RepositoryNotFoundError as _RNFE

    fake_resp = MagicMock()
    fake_resp.headers = {}
    exc = _RNFE("missing", response=fake_resp)
    with patch("sovereign.core.models.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = exc
        with pytest.raises(ModelNotFoundError):
            fetch_repo_info("org/missing")


def test_fetch_connection_error_returns_none():
    with patch("sovereign.core.models.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = ConnectionError("offline")
        result = fetch_repo_info("org/model")
    assert result is None


def test_fetch_connection_error_not_cached():
    with patch("sovereign.core.models.HfApi") as MockApi:
        MockApi.return_value.model_info.side_effect = ConnectionError("offline")
        fetch_repo_info("org/model")
        MockApi.return_value.model_info.side_effect = ConnectionError("offline again")
        # Second call must reach the API (not return a cached None)
        result = fetch_repo_info("org/model")
    assert result is None
    assert MockApi.return_value.model_info.call_count == 2


def test_fetch_success_is_cached():
    fake = _make_hfapi_info(("mlx",), (("model.safetensors", 1024),))
    with patch("sovereign.core.models.HfApi") as MockApi:
        MockApi.return_value.model_info.return_value = fake
        r1 = fetch_repo_info("org/model")
        r2 = fetch_repo_info("org/model")
    assert r1 is r2
    assert MockApi.return_value.model_info.call_count == 1


# ---------------------------------------------------------------------------
# _DownloadProgressSampler
# ---------------------------------------------------------------------------


def test_progress_sampler_basic(tmp_path, sparse_file):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    expected = 10 * 1024**3  # 10 GB
    sparse_file(blobs / "abc123", 2 * 1024**3)  # 2 GB downloaded

    sampler = _DownloadProgressSampler(blobs, expected, "org/model")
    msg = sampler.sample()
    assert msg is not None
    assert "20%" in msg
    assert "2.0/10.0 GB" in msg


def test_progress_sampler_zero_expected_returns_none(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    sampler = _DownloadProgressSampler(blobs, 0, "org/model")
    assert sampler.sample() is None


def test_progress_sampler_includes_speed_after_two_samples(tmp_path, sparse_file):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    expected = 4 * 1024**3
    sparse_file(blobs / "part1", 1 * 1024**3)

    sampler = _DownloadProgressSampler(blobs, expected, "org/model")
    sampler.sample()  # first sample — no speed yet
    # Grow the file so second sample shows progress
    sparse_file(blobs / "part2", 1 * 1024**3)
    msg = sampler.sample()
    assert msg is not None
    # Speed and ETA present once window has 2+ samples and bytes grew
    assert "MB/s" in msg


def test_progress_sampler_missing_blobs_dir(tmp_path):
    blobs = tmp_path / "blobs_missing"
    sampler = _DownloadProgressSampler(blobs, 1 * 1024**3, "org/model")
    msg = sampler.sample()
    # Should not crash; 0 bytes downloaded
    assert msg is not None
    assert "0%" in msg
