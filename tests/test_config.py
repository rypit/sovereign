"""Phase 2 exit: the fixture stack loads/validates; bad stacks raise clearly."""

from __future__ import annotations

from pathlib import Path

import pytest

from sovereign.config import ConfigError, Priority, SovereignConfig, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "sample.yaml"


def test_sample_fixture_loads() -> None:
    cfg = load_config(FIXTURE)
    assert cfg.version == "1.1"
    assert cfg.resources.max_unified_memory_gb == 128
    assert cfg.resources.default_priority is Priority.MEDIUM

    names = {s.name for s in cfg.services}
    assert names == {"llama_heavy_v1", "open_webui"}

    webui = next(s for s in cfg.services if s.name == "open_webui")
    assert webui.dependencies == ["llama_heavy_v1"]
    # Templates are carried through unresolved at this layer.
    assert webui.env_overrides is not None
    assert webui.env_overrides["OLLAMA_API_BASE_URL"] == "{{ llama_heavy_v1.endpoint }}"

    assert [h.name for h in cfg.harnesses] == ["cline_local"]


def _base_stack(**overrides) -> dict:
    data = {
        "version": "1.1",
        "resources": {"max_unified_memory_gb": 64, "safety_margin_gb": 4},
        "services": [
            {"name": "a", "base_type": "llama_cpp"},
            {"name": "b", "base_type": "docker", "dependencies": ["a"]},
        ],
    }
    data.update(overrides)
    return data


def test_valid_inline_stack() -> None:
    cfg = SovereignConfig.model_validate(_base_stack())
    assert len(cfg.services) == 2


def test_duplicate_name_rejected() -> None:
    data = _base_stack(
        services=[
            {"name": "a", "base_type": "llama_cpp"},
            {"name": "a", "base_type": "docker"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate entry name"):
        SovereignConfig.model_validate(data)


def test_unknown_dependency_rejected() -> None:
    data = _base_stack(
        services=[{"name": "a", "base_type": "llama_cpp", "dependencies": ["ghost"]}]
    )
    with pytest.raises(ValueError, match="unknown entry 'ghost'"):
        SovereignConfig.model_validate(data)


def test_self_dependency_rejected() -> None:
    data = _base_stack(
        services=[{"name": "a", "base_type": "llama_cpp", "dependencies": ["a"]}]
    )
    with pytest.raises(ValueError, match="cannot depend on itself"):
        SovereignConfig.model_validate(data)


def test_duplicate_across_service_and_harness_rejected() -> None:
    data = _base_stack(
        services=[{"name": "shared", "base_type": "llama_cpp"}],
        harnesses=[{"name": "shared", "base_type": "cline_cli"}],
    )
    with pytest.raises(ValueError, match="duplicate entry name"):
        SovereignConfig.model_validate(data)


def test_extra_field_rejected() -> None:
    data = _base_stack()
    data["services"][0]["typo_field"] = True
    with pytest.raises(ValueError):
        SovereignConfig.model_validate(data)


def test_load_config_wraps_missing_file() -> None:
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(Path("/nonexistent/sovereign.yaml"))


# --- auto base_type routing (M4) ---
def test_auto_base_type_with_model_ok() -> None:
    data = _base_stack(
        services=[{"name": "a", "base_type": "auto", "config": {"model": "org/repo"}}]
    )
    cfg = SovereignConfig.model_validate(data)
    assert cfg.services[0].base_type == "auto"


def test_omitted_base_type_defaults_to_auto() -> None:
    data = _base_stack(services=[{"name": "a", "config": {"model": "org/repo"}}])
    cfg = SovereignConfig.model_validate(data)
    assert cfg.services[0].base_type == "auto"


def test_auto_base_type_without_model_rejected() -> None:
    data = _base_stack(services=[{"name": "a", "base_type": "auto"}])
    with pytest.raises(ValueError, match="requires config.model"):
        SovereignConfig.model_validate(data)


def test_explicit_base_type_needs_no_model() -> None:
    data = _base_stack(services=[{"name": "a", "base_type": "llama_cpp"}])
    cfg = SovereignConfig.model_validate(data)  # must not raise
    assert cfg.services[0].base_type == "llama_cpp"
