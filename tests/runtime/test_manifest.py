"""runtime/manifest.py — secret redaction in the persisted stack manifest."""

from __future__ import annotations

import asyncio
import json

from sovereign.config import ServiceEntry, SovereignConfig
from sovereign.runtime.manifest import redact_start_args
from sovereign.runtime.orchestrator import Orchestrator


# --- redact_start_args ---
def test_redact_separate_flag_value() -> None:
    assert redact_start_args(["--api-key", "hunter2", "-c", "4096"]) == [
        "--api-key",
        "***",
        "-c",
        "4096",
    ]


def test_redact_equals_form() -> None:
    assert redact_start_args(["--api-key=hunter2"]) == ["--api-key=***"]


def test_redact_underscore_and_case_variants() -> None:
    assert redact_start_args(["--API_KEY", "s3cret"]) == ["--API_KEY", "***"]


def test_redact_leaves_ordinary_args_alone() -> None:
    args = ["llama-server", "--model", "/models/x.gguf", "--port", "11435"]
    assert redact_start_args(args) == args


def test_redact_non_flag_value_containing_api_key_untouched() -> None:
    # Only flag *values* are secrets; a path that mentions api-key is not.
    args = ["--model", "/models/api-key-benchmark.gguf"]
    assert redact_start_args(args) == args


# --- end-to-end: the manifest on disk never contains the key ---
def test_manifest_start_args_redacted(tmp_path) -> None:
    class KeyedManager:
        def __init__(self, entry: ServiceEntry) -> None:
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

        def get_start_args(self) -> list[str]:
            return ["llama-server", "--api-key", "hunter2", "--port", "11435"]

    config = SovereignConfig.model_validate(
        {
            "version": "1.1",
            "resources": {"max_unified_memory_gb": 64, "safety_margin_gb": 4},
            "services": [{"name": "engine", "base_type": "x"}],
        }
    )
    orch = Orchestrator(
        config,
        manager_factory=KeyedManager,
        state_dir=tmp_path,
        health_interval=0.01,
    )
    asyncio.run(orch.boot())

    raw = (tmp_path / "manifest.json").read_text()
    assert "hunter2" not in raw  # the secret never touches disk
    manifest = json.loads(raw)
    assert manifest["services"][0]["start_args"] == [
        "llama-server",
        "--api-key",
        "***",
        "--port",
        "11435",
    ]
