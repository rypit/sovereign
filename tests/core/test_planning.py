"""core/planning.py — the shared dry-run used by `sovereign plan`.

Routing/estimation seams are the same ones the boot path uses; these tests focus
on what planning adds: config validation at dry-run time and verdict mapping.
"""

from __future__ import annotations

from sovereign.config import SovereignConfig
from sovereign.core.planning import (
    VERDICT_CONFIG_ERROR,
    VERDICT_OK,
    VERDICT_REFUSED,
    plan_stack,
)


def _config(services: list[dict], *, mem: int = 64) -> SovereignConfig:
    return SovereignConfig.model_validate(
        {
            "version": "1.1",
            "resources": {"max_unified_memory_gb": mem, "safety_margin_gb": 0},
            "services": services,
        }
    )


def test_plan_flags_bad_config_key_at_dry_run(tmp_path) -> None:
    cfg = _config(
        [
            {
                "name": "engine",
                "base_type": "llama_cpp",
                "health_check": {"type": "http", "endpoint": "/health", "port": 8080},
                "config": {"model": "/m.gguf", "not_a_real_key": 1},
            }
        ]
    )
    plan = plan_stack(cfg, tmp_path)
    assert plan.services[0].verdict == VERDICT_CONFIG_ERROR
    assert "not_a_real_key" in (plan.services[0].error or "")
    assert not plan.ok


def test_plan_flags_unknown_base_type(tmp_path) -> None:
    cfg = _config([{"name": "svc", "base_type": "no_such_engine", "config": {}}])
    plan = plan_stack(cfg, tmp_path)
    assert plan.services[0].verdict == VERDICT_CONFIG_ERROR
    assert "no_such_engine" in (plan.services[0].error or "")


def test_plan_declared_memory_admits_and_labels_source(tmp_path) -> None:
    cfg = _config(
        [
            {
                "name": "webui",
                "base_type": "docker",
                "memory_gb": 4,
                "config": {"image": "img:latest", "port": 3000},
            }
        ],
        mem=8,
    )
    plan = plan_stack(cfg, tmp_path)
    svc = plan.services[0]
    assert (svc.verdict, svc.source, svc.estimated_gb) == (VERDICT_OK, "declared", 4.0)
    assert plan.ok


def test_plan_refuses_over_budget_with_actionable_error(tmp_path) -> None:
    cfg = _config(
        [
            {
                "name": "big",
                "base_type": "docker",
                "memory_gb": 32,
                "config": {"image": "img:latest", "port": 3000},
            }
        ],
        mem=8,
    )
    plan = plan_stack(cfg, tmp_path)
    assert plan.services[0].verdict == VERDICT_REFUSED
    assert "Cannot start 'big'" in (plan.services[0].error or "")
    assert not plan.ok
