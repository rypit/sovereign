"""Both concrete harnesses (H2 + H3) build together via the real registry."""

from __future__ import annotations

import asyncio

from sovereign.config import SovereignConfig
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.orchestrator import Orchestrator


class FakeEngineManager:
    consumer_kind = ConsumerKind.NATIVE

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.activity = ()

    def prepare_environment(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_healthy(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"status": "running"}

    def adjust_resources(self, memory_limit_mb: int) -> None: ...
    def resolve(self, resolver) -> None: ...
    def endpoint(self):
        return ResolvedEndpoint("http", "127.0.0.1", 11435, model="llama3-70b")


def _config() -> SovereignConfig:
    return SovereignConfig.model_validate(
        {
            "version": "1.1",
            "resources": {"max_unified_memory_gb": 64, "safety_margin_gb": 4},
            "services": [{"name": "engine", "base_type": "fake_engine"}],
            "harnesses": [
                {
                    "name": "cline_local",
                    "base_type": "cline_cli",
                    "dependencies": ["engine"],
                    "config": {
                        "base_url": "{{ engine.endpoint }}/v1",
                        "model": "{{ engine.model }}",
                    },
                },
                {
                    "name": "mini_swe_local",
                    "base_type": "mini_swe_agent",
                    "dependencies": ["engine"],
                    "config": {
                        "base_url": "{{ engine.endpoint }}/v1",
                        "model": "{{ engine.model }}",
                    },
                },
            ],
        }
    )


def test_both_harnesses_build_and_materialize_together(tmp_path) -> None:
    from sovereign.harnesses.cline_cli.manager import ClineCliHarness
    from sovereign.harnesses.mini_swe_agent.manager import MiniSweAgentHarness

    orch = Orchestrator(
        _config(),
        manager_factory=lambda entry: FakeEngineManager(entry),
        state_dir=tmp_path,
        health_interval=0.01,
        metrics_interval=0.01,
    )
    asyncio.run(orch.boot())

    assert isinstance(orch.harnesses["cline_local"], ClineCliHarness)
    assert isinstance(orch.harnesses["mini_swe_local"], MiniSweAgentHarness)
    assert orch.harnesses["cline_local"].resolved_config["base_url"] == "http://127.0.0.1:11435/v1"
    assert orch.harnesses["mini_swe_local"].resolved_config["model"] == "llama3-70b"

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    names = {h["name"] for h in manifest["harnesses"]}
    assert names == {"cline_local", "mini_swe_local"}
