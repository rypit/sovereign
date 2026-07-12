"""Unit tests for the ``ARCH_RULES`` rule engine in ``scripts/depgraph.py``.

Exercises the checker against synthetic edge lists (so a rule regression
here can never depend on the current shape of the real graph) and, once,
against the actual repo graph — fast, in-process, no subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_depgraph():
    spec = importlib.util.spec_from_file_location(
        "sovereign_depgraph_test", REPO_ROOT / "scripts" / "depgraph.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dg():
    return _load_depgraph()


def _graph(edges: dict[str, set[str]]) -> dict[str, tuple[set[str], set[str]]]:
    """Build an EdgeGraph (runtime edges only, no type-only edges) from a plain dict."""
    return {name: (targets, set()) for name, targets in edges.items()}


class TestArchRulesClean:
    def test_empty_graph_is_clean(self, dg):
        assert dg.check_arch_rules(_graph({})) == []

    def test_workers_importing_only_workers_and_procmem_is_clean(self, dg):
        graph = _graph(
            {
                "sovereign.workers.engine_worker": {
                    "sovereign.workers.protocol",
                    "sovereign.core.procmem",
                },
                "sovereign.workers.protocol": set(),
                "sovereign.core.procmem": set(),
            }
        )
        assert dg.check_arch_rules(graph) == []

    def test_registry_may_import_services_and_harnesses(self, dg):
        graph = _graph(
            {
                "sovereign.core.registry": {
                    "sovereign.services.docker.manager",
                    "sovereign.harnesses.cline_cli.manager",
                },
                "sovereign.services.docker.manager": set(),
                "sovereign.harnesses.cline_cli.manager": set(),
            }
        )
        assert dg.check_arch_rules(graph) == []

    def test_config_importing_units_and_base_config_is_clean(self, dg):
        graph = _graph(
            {
                "sovereign.services.docker.config": {
                    "sovereign.core.base_config",
                },
                "sovereign.core.base_config": {"sovereign.core.units"},
                "sovereign.core.units": set(),
            }
        )
        assert dg.check_arch_rules(graph) == []


class TestArchRulesViolations:
    def test_workers_leaf_violation(self, dg):
        graph = _graph(
            {
                "sovereign.workers.llama_cpp_adapter": {
                    "sovereign.services.inference.hf",
                },
                "sovereign.services.inference.hf": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("workers-leaf:")

    def test_hf_leaf_violation(self, dg):
        graph = _graph(
            {
                "sovereign.services.inference.hf": {"sovereign.core.resources"},
                "sovereign.core.resources": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("hf-leaf:")

    def test_config_golden_rule_violation(self, dg):
        graph = _graph(
            {
                "sovereign.services.docker.config": {"sovereign.core.resources"},
                "sovereign.core.resources": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("config-golden-rule:")

    def test_runtime_no_bench_violation(self, dg):
        graph = _graph(
            {
                "sovereign.runtime.orchestrator": {"sovereign.bench.cleanroom"},
                "sovereign.bench.cleanroom": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("runtime-no-bench:")

    def test_bench_single_door_violation_for_non_cleanroom_module(self, dg):
        graph = _graph(
            {
                "sovereign.bench.runner": {"sovereign.runtime.orchestrator"},
                "sovereign.runtime.orchestrator": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("bench-single-door:")

    def test_bench_cleanroom_is_exempt_from_bench_single_door(self, dg):
        graph = _graph(
            {
                "sovereign.bench.cleanroom": {"sovereign.runtime.orchestrator"},
                "sovereign.runtime.orchestrator": set(),
            }
        )
        assert dg.check_arch_rules(graph) == []

    def test_core_single_door_violation(self, dg):
        graph = _graph(
            {
                "sovereign.core.resources": {"sovereign.services.docker.manager"},
                "sovereign.services.docker.manager": set(),
            }
        )
        violations = dg.check_arch_rules(graph)
        assert len(violations) == 1
        assert violations[0].startswith("core-single-door:")

    def test_core_registry_is_exempt_from_core_single_door(self, dg):
        graph = _graph(
            {
                "sovereign.core.registry": {"sovereign.services.docker.manager"},
                "sovereign.services.docker.manager": set(),
            }
        )
        assert dg.check_arch_rules(graph) == []


class TestGrandfathered:
    def test_grandfathered_edge_is_skipped(self, dg, monkeypatch):
        graph = _graph(
            {
                "sovereign.workers.foo": {"sovereign.services.inference.hf"},
                "sovereign.services.inference.hf": set(),
            }
        )
        # Sanity: without the allowlist entry, this is a violation.
        assert dg.check_arch_rules(graph) != []

        monkeypatch.setattr(
            dg,
            "GRANDFATHERED",
            frozenset(
                {
                    (
                        "workers-leaf",
                        "sovereign.workers.foo",
                        "sovereign.services.inference.hf",
                    )
                }
            ),
        )
        assert dg.check_arch_rules(graph) == []


class TestCycles:
    def test_no_cycle_is_clean(self, dg):
        graph = _graph({"a": {"b"}, "b": set()})
        assert dg.check_cycles(graph) == []

    def test_two_module_cycle_detected(self, dg):
        graph = _graph({"a": {"b"}, "b": {"a"}})
        messages = dg.check_cycles(graph)
        assert len(messages) == 1
        assert messages[0].startswith("import-cycle:")


class TestRealRepoGraph:
    def test_real_repo_graph_passes_check(self, dg):
        """The actual repo, evaluated in-process (no subprocess) — fast."""
        modules = dg.discover_modules(dg.DEFAULT_ROOT)
        graph = dg.build_graph(modules)
        labels = dg.build_labels(modules, dg.DEFAULT_ROOT)

        cycle_violations = dg.check_cycles(graph)
        assert cycle_violations == [], cycle_violations

        rule_violations = dg.check_arch_rules(graph)
        assert rule_violations == [], rule_violations

        freshness = dg.check_freshness(graph, labels, dg.DEFAULT_OUT)
        assert freshness is None, freshness
