"""Every concrete harness builds together via the real registry."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sovereign.config import SovereignConfig
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint
from sovereign.runtime.orchestrator import Orchestrator


class FakeEngineManager:
    consumer_kind = ConsumerKind.NATIVE

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.activity: tuple[str, ...] = ()

    def prepare_environment(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_healthy(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return {"status": "running"}

    def adjust_resources(self, memory_limit_bytes: int) -> None: ...
    def resolve(self, resolver) -> None: ...
    def endpoint(self):
        return ResolvedEndpoint("http", "127.0.0.1", 11435, model="llama3-70b")


#: Every registered harness base_type, and the instance name it boots under.
_HARNESSES = {
    "cline_local": "cline_cli",
    "mini_swe_local": "mini_swe_agent",
    "pi_local": "pi",
    "opencode_local": "opencode",
}


def _config(config_root: Path) -> SovereignConfig:
    """The stack, with every harness materializing into ``config_root``."""
    return SovereignConfig.model_validate(
        {
            "version": "1.1",
            "resources": {"max_unified_memory_gb": 64, "safety_margin_gb": 4},
            "services": [{"name": "engine", "base_type": "fake_engine"}],
            "harnesses": [
                {
                    "name": name,
                    "base_type": base_type,
                    "dependencies": ["engine"],
                    "config": {
                        "base_url": "{{ engine.endpoint }}/v1",
                        "model": "{{ engine.model }}",
                        "config_dir": str(config_root / name),
                    },
                }
                for name, base_type in _HARNESSES.items()
            ],
        }
    )


def test_all_harnesses_build_and_materialize_together(tmp_path) -> None:
    from sovereign.harnesses.cline_cli.manager import ClineCliHarness
    from sovereign.harnesses.mini_swe_agent.manager import MiniSweAgentHarness
    from sovereign.harnesses.opencode.manager import OpencodeHarness
    from sovereign.harnesses.pi.manager import PiHarness

    orch = Orchestrator(
        _config(tmp_path / "harness-config"),
        manager_factory=lambda entry: FakeEngineManager(entry),
        state_dir=tmp_path,
        health_interval=0.01,
        metrics_interval=0.01,
    )
    asyncio.run(orch.boot())

    assert isinstance(orch.harnesses["cline_local"], ClineCliHarness)
    assert isinstance(orch.harnesses["mini_swe_local"], MiniSweAgentHarness)
    assert isinstance(orch.harnesses["pi_local"], PiHarness)
    assert isinstance(orch.harnesses["opencode_local"], OpencodeHarness)
    assert orch.harnesses["cline_local"].resolved_config["base_url"] == "http://127.0.0.1:11435/v1"
    assert orch.harnesses["mini_swe_local"].resolved_config["model"] == "llama3-70b"
    assert orch.harnesses["pi_local"].resolved_config["base_url"] == "http://127.0.0.1:11435/v1"
    assert orch.harnesses["opencode_local"].resolved_config["model"] == "llama3-70b"

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    names = {h["name"] for h in manifest["harnesses"]}
    assert names == {"cline_local", "mini_swe_local", "pi_local", "opencode_local"}
