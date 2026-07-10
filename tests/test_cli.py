"""Phase 8: CLI surface — help tree, status + drift, down, logs."""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from sovereign.cli import app
from sovereign.cli import stack as main
from sovereign.cli._common import _load_dotenv
from sovereign.core.state import file_hash, write_json

runner = CliRunner()


# --- _load_dotenv ---
def test_load_dotenv_sets_missing_vars(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_abc123\nOTHER_VAR=hello\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _load_dotenv(env_file)
    assert os.environ["HF_TOKEN"] == "hf_abc123"
    assert os.environ["OTHER_VAR"] == "hello"


def test_load_dotenv_does_not_override_existing_vars(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from_file\n")
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    _load_dotenv(env_file)
    assert os.environ["HF_TOKEN"] == "from_shell"


def test_load_dotenv_strips_quotes(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('HF_TOKEN="hf_quoted"\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _load_dotenv(env_file)
    assert os.environ["HF_TOKEN"] == "hf_quoted"


def test_load_dotenv_skips_comments_and_blanks(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nVALID=yes\n")
    monkeypatch.delenv("VALID", raising=False)
    _load_dotenv(env_file)
    assert os.environ["VALID"] == "yes"


def test_load_dotenv_no_file_is_silent(tmp_path) -> None:
    _load_dotenv(tmp_path / "nonexistent.env")  # must not raise


def _write_variant(tmp_path) -> tuple:
    variant = tmp_path / "stack.yaml"
    variant.write_text("version: '1.1'\n")
    return variant, file_hash(variant)


def _write_state(state_dir, variant, variant_hash, *, services=None, runtime=None) -> None:
    write_json(
        state_dir / "state.json",
        {
            "variant_file": str(variant),
            "variant_hash": variant_hash,
            "services": services or {"engine": "ready", "frontend": "ready"},
            "runtime": runtime or {},
            "updated_at": "2026-07-05T00:00:00+00:00",
        },
    )


# --- help tree (exit criterion) ---
def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("up", "down", "status", "logs", "serve", "monitor", "bench"):
        assert cmd in result.stdout


def test_each_command_help() -> None:
    for cmd in ("up", "down", "status", "logs", "serve"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, cmd


# --- status ---
def test_status_no_state(tmp_path) -> None:
    result = runner.invoke(app, ["status", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No running stack" in result.stdout


def test_status_shows_services_and_in_sync(tmp_path) -> None:
    variant, h = _write_variant(tmp_path)
    _write_state(tmp_path, variant, h)
    result = runner.invoke(app, ["status", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "engine" in result.stdout
    assert "frontend" in result.stdout
    assert "in sync" in result.stdout


def test_status_flags_drift_when_variant_changed(tmp_path) -> None:
    variant, h = _write_variant(tmp_path)
    _write_state(tmp_path, variant, h)
    variant.write_text("version: '1.1'\n# edited since boot\n")  # hash now differs
    result = runner.invoke(app, ["status", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "drift" in result.stdout


def test_status_flags_missing_variant(tmp_path) -> None:
    variant, h = _write_variant(tmp_path)
    _write_state(tmp_path, variant, h)
    variant.unlink()
    result = runner.invoke(app, ["status", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no longer exists" in result.stdout


# --- down ---
def test_down_no_state(tmp_path) -> None:
    result = runner.invoke(app, ["down", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No recorded stack" in result.stdout


def test_down_stops_handles_and_updates_state(tmp_path, monkeypatch) -> None:
    variant, h = _write_variant(tmp_path)
    runtime = {
        "engine": {"kind": "native", "pid": 111},
        "frontend": {"kind": "docker", "container": "frontend"},
    }
    _write_state(tmp_path, variant, h, runtime=runtime)

    stopped: list = []
    monkeypatch.setattr(
        main, "stop_service_handle", lambda handle, **kw: stopped.append(handle) or "stopped"
    )
    result = runner.invoke(app, ["down", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    # reverse order: frontend before engine
    assert [h["kind"] for h in stopped] == ["docker", "native"]

    # state.json updated
    from sovereign.core.state import read_json

    state = read_json(tmp_path / "state.json")
    assert state["services"] == {"engine": "stopped", "frontend": "stopped"}
    assert state["runtime"] == {}


def test_down_corrupt_state_json_actionable_error(tmp_path) -> None:
    (tmp_path / "state.json").write_text('{"runtime": {"engine"')  # torn write
    result = runner.invoke(app, ["down", "--state-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "corrupt" in result.stdout
    assert "Traceback" not in result.stdout


def test_status_corrupt_state_json_actionable_error(tmp_path) -> None:
    (tmp_path / "state.json").write_text("garbage not json")
    result = runner.invoke(app, ["status", "--state-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "corrupt" in result.stdout
    assert "Traceback" not in result.stdout


# --- logs ---
def test_logs_reads_native_log_file(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "engine.log").write_text("line1\nline2\nline3\n")
    result = runner.invoke(app, ["logs", "engine", "-n", "2", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "line2" in result.stdout
    assert "line3" in result.stdout
    assert "line1" not in result.stdout


def test_logs_missing_service(tmp_path) -> None:
    result = runner.invoke(app, ["logs", "ghost", "--state-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No logs found" in result.stdout


# --- harness ---
def _write_harness_stack(tmp_path) -> Path:
    variant = tmp_path / "stack.yaml"
    variant.write_text(
        """
version: "1.1"
resources:
  max_unified_memory_gb: 64
  safety_margin_gb: 4
services:
  - name: engine
    base_type: llama_cpp
    health_check: {type: http, endpoint: /health, port: 11435}
    config: {model: /models/x.gguf}
harnesses:
  - name: h
    base_type: fake_test_harness
    dependencies: [engine]
    config: {base_url: "{{ engine.endpoint }}/v1"}
"""
    )
    return variant


def _write_manifest_with_ready_engine(state_dir) -> None:
    write_json(
        state_dir / "manifest.json",
        {
            "services": [
                {
                    "name": "engine",
                    "endpoint": {"scheme": "http", "host": "127.0.0.1", "port": 11435},
                }
            ]
        },
    )


class _FakeTestHarness:
    """Registered under a throwaway base_type just for CLI tests."""

    invoked_with = None
    prepared_count = 0

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.entry = entry
        self.materialized = False

    def prepare_environment(self) -> None:
        _FakeTestHarness.prepared_count += 1

    def resolve(self, resolver) -> None:
        from sovereign.core.resolver import ConsumerKind

        self.resolved_config = resolver.resolve_mapping(self.entry.config, ConsumerKind.NATIVE)

    def materialize(self) -> None:
        self.materialized = True

    def invoke(self, task):
        from sovereign.core.base_harness import RunResult

        _FakeTestHarness.invoked_with = task
        return RunResult(task_id=task.id, success=True, exit_code=0, output="did the thing")


def _register_fake_test_harness(monkeypatch) -> None:
    from sovereign.core.registry import _HARNESSES

    monkeypatch.setitem(_HARNESSES, "fake_test_harness", _FakeTestHarness)


def test_harness_list_shows_configured_harnesses(tmp_path) -> None:
    variant = _write_harness_stack(tmp_path)
    result = runner.invoke(app, ["harness", "list", "-f", str(variant)])
    assert result.exit_code == 0
    assert "h" in result.stdout
    assert "fake_test_harness" in result.stdout
    assert "engine" in result.stdout


def test_harness_list_no_harnesses(tmp_path) -> None:
    variant = tmp_path / "stack.yaml"
    variant.write_text(
        'version: "1.1"\nresources: {max_unified_memory_gb: 64, safety_margin_gb: 4}\n'
    )
    result = runner.invoke(app, ["harness", "list", "-f", str(variant)])
    assert result.exit_code == 0
    assert "No harnesses configured" in result.stdout


def test_harness_materialize_no_manifest(tmp_path) -> None:
    variant = _write_harness_stack(tmp_path)
    result = runner.invoke(
        app, ["harness", "materialize", "h", "-f", str(variant), "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "No running stack found" in result.stdout


def test_harness_materialize_runs_against_manifest(tmp_path, monkeypatch) -> None:
    _register_fake_test_harness(monkeypatch)
    variant = _write_harness_stack(tmp_path)
    _write_manifest_with_ready_engine(tmp_path)
    result = runner.invoke(
        app, ["harness", "materialize", "h", "-f", str(variant), "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "Materialized 'h'" in result.stdout


def test_harness_invoke_prints_result(tmp_path, monkeypatch) -> None:
    _register_fake_test_harness(monkeypatch)
    variant = _write_harness_stack(tmp_path)
    _write_manifest_with_ready_engine(tmp_path)
    _FakeTestHarness.prepared_count = 0
    result = runner.invoke(
        app,
        [
            "harness",
            "invoke",
            "h",
            "--prompt",
            "do the thing",
            "-f",
            str(variant),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "success=True" in result.stdout
    assert "did the thing" in result.stdout
    assert _FakeTestHarness.invoked_with.prompt == "do the thing"
    assert _FakeTestHarness.prepared_count == 1  # CLI provisions like the Orchestrator


def test_harness_invoke_unknown_dependency_not_ready(tmp_path, monkeypatch) -> None:
    _register_fake_test_harness(monkeypatch)
    variant = _write_harness_stack(tmp_path)
    write_json(tmp_path / "manifest.json", {"services": []})  # engine not registered
    result = runner.invoke(
        app,
        [
            "harness",
            "invoke",
            "h",
            "--prompt",
            "x",
            "-f",
            str(variant),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "Dependencies not ready" in result.stdout


def test_harness_unknown_name(tmp_path) -> None:
    variant = _write_harness_stack(tmp_path)
    _write_manifest_with_ready_engine(tmp_path)
    result = runner.invoke(
        app, ["harness", "materialize", "ghost", "-f", str(variant), "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "Unknown harness 'ghost'" in result.stdout


# --- bench ---
def test_bench_run_invalid_spec(tmp_path) -> None:
    result = runner.invoke(app, ["bench", "run", "-f", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "cannot read" in result.stdout


def test_bench_run_without_executor_reports_failed_cells(tmp_path) -> None:
    spec = tmp_path / "bench.yaml"
    spec.write_text("stacks: [stack.yaml]\ntrials: 1\n")
    result = runner.invoke(
        app, ["bench", "run", "-f", str(spec), "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "failed" in result.stdout


def test_bench_ls_no_runs(tmp_path) -> None:
    result = runner.invoke(app, ["bench", "ls", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No benchmark runs recorded" in result.stdout


def test_bench_ls_shows_run_summary(tmp_path) -> None:
    spec = tmp_path / "bench.yaml"
    spec.write_text("stacks: [stack.yaml]\ntrials: 1\n")
    runner.invoke(app, ["bench", "run", "-f", str(spec), "--state-dir", str(tmp_path)])
    result = runner.invoke(app, ["bench", "ls", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "1" in result.stdout  # cell count


def test_bench_ls_counts_gated_and_skipped(tmp_path) -> None:
    write_json(
        tmp_path / "benchmarks" / "runs" / "run1.json",
        {
            "run_id": "run1",
            "cells": [
                {"state": "completed", "skipped": True},
                {"state": "completed", "skipped": False},
                {"state": "failed", "error": "gated: stack needs ~10GB, only 2GB available"},
                {"state": "failed", "error": "some other failure"},
            ],
        },
    )
    result = runner.invoke(app, ["bench", "ls", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    line = next(line for line in result.stdout.splitlines() if "run1" in line)
    fields = [f.strip() for f in line.split("│") if f.strip()]
    # RUN_ID CELLS COMPLETED SKIPPED FAILED GATED
    assert fields == ["run1", "4", "2", "1", "2", "1"]


def test_bench_run_attach_mode_success(tmp_path, monkeypatch) -> None:
    import sys
    import types

    from sovereign.core.state import write_json

    fake_httpx = types.ModuleType("httpx")

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield 'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":3}}'
            yield "data: [DONE]"

    class FakeAsyncClient:
        def __init__(self, headers=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *a, **kw):
            return FakeResponse()

    fake_httpx.AsyncClient = FakeAsyncClient
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    write_json(
        tmp_path / "manifest.json",
        {
            "variant_hash": "abc",
            "memory_budget": {"available_gb": 10.0},
            "services": [
                {
                    "name": "engine",
                    "endpoint": {
                        "scheme": "http",
                        "host": "127.0.0.1",
                        "port": 11435,
                        "model": "llama3-70b",
                    },
                    "co_resident": [],
                }
            ],
        },
    )
    spec = tmp_path / "bench.yaml"
    spec.write_text("stacks: [stack.yaml]\ntrials: 1\n")
    result = runner.invoke(app, ["bench", "run", "-f", str(spec), "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "completed" in result.stdout


def test_up_refuses_while_cleanroom_bench_lock_held(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / ".sovereign" / "bench.lock", {"pid": 1, "run_id": "x"})
    variant, _ = _write_variant(tmp_path)
    result = runner.invoke(app, ["up", "-f", str(variant)])
    assert result.exit_code == 1
    assert "clean-room bench run holds" in result.stdout


# --- provision ---
class _FakeProvisionable:
    """Stands in for a registered integration class in provision tests."""

    satisfied = False
    provisioned: list[str] = []

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        return cls.satisfied

    @classmethod
    def provision(cls) -> None:
        _FakeProvisionable.provisioned.append(cls.__name__)


def test_provision_scoped_to_stack_file(tmp_path, monkeypatch) -> None:
    from sovereign.core.registry import _SERVICE_MANAGERS

    class FakeEngine(_FakeProvisionable):
        pass

    class FakeOther(_FakeProvisionable):
        pass

    monkeypatch.setitem(_SERVICE_MANAGERS, "prov_fake_engine", FakeEngine)
    monkeypatch.setitem(_SERVICE_MANAGERS, "prov_fake_other", FakeOther)
    _FakeProvisionable.provisioned = []

    stack = tmp_path / "stack.yaml"
    stack.write_text(
        'version: "1.1"\n'
        "resources: {max_unified_memory_gb: 64, safety_margin_gb: 4}\n"
        "services:\n"
        "  - name: engine\n"
        "    base_type: prov_fake_engine\n"
    )
    result = runner.invoke(app, ["provision", "-f", str(stack)])
    assert result.exit_code == 0, result.stdout
    # Only the declared base_type provisioned; the other registered fake untouched.
    assert _FakeProvisionable.provisioned == ["FakeEngine"]
    assert "installed" in result.stdout


def test_provision_reports_satisfied(tmp_path, monkeypatch) -> None:
    from sovereign.core.registry import _SERVICE_MANAGERS

    class FakeSatisfied(_FakeProvisionable):
        satisfied = True

    monkeypatch.setitem(_SERVICE_MANAGERS, "prov_fake_satisfied", FakeSatisfied)
    _FakeProvisionable.provisioned = []

    stack = tmp_path / "stack.yaml"
    stack.write_text(
        'version: "1.1"\n'
        "resources: {max_unified_memory_gb: 64, safety_margin_gb: 4}\n"
        "services:\n"
        "  - name: engine\n"
        "    base_type: prov_fake_satisfied\n"
    )
    result = runner.invoke(app, ["provision", "-f", str(stack)])
    assert result.exit_code == 0
    assert "satisfied" in result.stdout
    assert _FakeProvisionable.provisioned == []  # never installed


def test_provision_failure_exits_nonzero(tmp_path, monkeypatch) -> None:
    from sovereign.core.provisioning import ProvisioningError
    from sovereign.core.registry import _SERVICE_MANAGERS

    class FakeBroken(_FakeProvisionable):
        @classmethod
        def provision(cls) -> None:
            raise ProvisioningError("registry unreachable")

    monkeypatch.setitem(_SERVICE_MANAGERS, "prov_fake_broken", FakeBroken)

    stack = tmp_path / "stack.yaml"
    stack.write_text(
        'version: "1.1"\n'
        "resources: {max_unified_memory_gb: 64, safety_margin_gb: 4}\n"
        "services:\n"
        "  - name: engine\n"
        "    base_type: prov_fake_broken\n"
    )
    result = runner.invoke(app, ["provision", "-f", str(stack)])
    assert result.exit_code == 1
    assert "registry unreachable" in result.stdout


def test_provision_unknown_base_type_errors(tmp_path) -> None:
    stack = tmp_path / "stack.yaml"
    stack.write_text(
        'version: "1.1"\n'
        "resources: {max_unified_memory_gb: 64, safety_margin_gb: 4}\n"
        "services:\n"
        "  - name: engine\n"
        "    base_type: no_such_thing\n"
    )
    result = runner.invoke(app, ["provision", "-f", str(stack)])
    assert result.exit_code == 1
    assert "no_such_thing" in result.stdout


def test_provision_unscoped_covers_all_registered() -> None:
    # The suite-wide fixture no-ops real provision(), so this exercises the
    # full walk over every registered service + harness without side effects.
    result = runner.invoke(app, ["provision"])
    assert result.exit_code == 0, result.stdout
    for base_type in ("llama_cpp", "docker", "cline_cli", "mini_swe_agent"):
        assert base_type in result.stdout


# --- bench compare ---
def test_bench_compare_no_cells(tmp_path) -> None:
    result = runner.invoke(app, ["bench", "compare", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No completed bench cells found" in result.stdout


def test_bench_compare_renders_table(tmp_path) -> None:
    write_json(
        tmp_path / "benchmarks" / "runs" / "run1.json",
        {
            "run_id": "run1",
            "cells": [
                {
                    "stack": "a.yaml",
                    "harness": "_none",
                    "suite": "_none",
                    "state": "completed",
                    "result": {
                        "engine": "engine",
                        "tok_s": {"mean": 25.0},
                        "ttft_ms": {"mean": 120.0},
                    },
                },
                {
                    "stack": "a.yaml",
                    "harness": "h1",
                    "suite": "suite",
                    "state": "completed",
                    "result": {"pass_rate": 0.75, "false_completion_rate": 0.0},
                },
            ],
        },
    )
    result = runner.invoke(app, ["bench", "compare", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "h1" in result.stdout
    assert "engine" in result.stdout


def test_bench_compare_json_output(tmp_path) -> None:
    write_json(
        tmp_path / "benchmarks" / "runs" / "run1.json",
        {
            "run_id": "run1",
            "cells": [
                {
                    "stack": "a.yaml",
                    "harness": "_none",
                    "suite": "_none",
                    "state": "completed",
                    "result": {"engine": "engine", "tok_s": {"mean": 25.0}, "ttft_ms": {}},
                }
            ],
        },
    )
    result = runner.invoke(app, ["bench", "compare", "--state-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert '"engine": "engine"' in result.stdout


# --- sovereign plan (M5) ---
import types  # noqa: E402

from sovereign.core.errors import RoutingError  # noqa: E402
from sovereign.services.inference import hf as models_mod  # noqa: E402

_PLAN_YAML = """
version: "1.1"
resources:
  max_unified_memory_gb: {mem}
  safety_margin_gb: 0
services:
  - name: engine
    base_type: {base_type}
    health_check: {{type: http, endpoint: /health, port: 8080}}
    config:
      model: mlx-community/foo-4bit
"""


def _write_plan_config(tmp_path, *, mem=64, base_type="auto") -> Path:
    p = tmp_path / "stack.yaml"
    p.write_text(_PLAN_YAML.format(mem=mem, base_type=base_type))
    return p


def test_plan_fits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sovereign.core.planning.route_entry", lambda e, s: "mlx_lm")
    monkeypatch.setattr(
        models_mod, "estimate_model_bytes_with_source", lambda ref, kind: (8 * 1024**3, "hub")
    )
    result = runner.invoke(app, ["plan", "-f", str(_write_plan_config(tmp_path, mem=64))])
    assert result.exit_code == 0
    assert "mlx_lm (auto)" in result.stdout  # routed type flagged
    assert "OK" in result.stdout
    assert "headroom" in result.stdout  # budget footer


def test_plan_refused_when_over_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sovereign.core.planning.route_entry", lambda e, s: "mlx_lm")
    monkeypatch.setattr(
        models_mod, "estimate_model_bytes_with_source", lambda ref, kind: (200 * 1024**3, "hub")
    )
    result = runner.invoke(app, ["plan", "-f", str(_write_plan_config(tmp_path, mem=16))])
    assert result.exit_code == 1
    assert "REFUSED" in result.stdout


def test_plan_routing_error(tmp_path, monkeypatch) -> None:
    def boom(entry, state_dir):
        raise RoutingError("cannot route offline")

    monkeypatch.setattr("sovereign.core.planning.route_entry", boom)
    result = runner.invoke(app, ["plan", "-f", str(_write_plan_config(tmp_path))])
    assert result.exit_code == 1
    assert "ROUTING ERROR" in result.stdout


def test_plan_warns_on_unknown_footprint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sovereign.core.planning.route_entry", lambda e, s: "mlx_lm")
    monkeypatch.setattr(
        models_mod, "estimate_model_bytes_with_source", lambda ref, kind: (None, "unknown")
    )
    result = runner.invoke(
        app, ["plan", "-f", str(_write_plan_config(tmp_path, mem=64))], env={"COLUMNS": "220"}
    )
    assert result.exit_code == 0  # still admitted (fail-open policy unchanged)
    assert "UNKNOWN memory footprint" in result.stdout  # ...but loudly


# --- sovereign models (M5) ---
def _fake_repo(repo_id: str, size: int, nb_files: int = 3):
    return types.SimpleNamespace(
        repo_id=repo_id,
        size_on_disk=size,
        nb_files=nb_files,
        last_accessed=0.0,
        revisions=[types.SimpleNamespace(commit_hash="abc123")],
    )


def test_models_list(monkeypatch) -> None:
    import huggingface_hub

    cache = types.SimpleNamespace(
        repos=[_fake_repo("org/big", 20 * 1024**3), _fake_repo("org/small", 1 * 1024**3)],
        size_on_disk=21 * 1024**3,
    )
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: cache)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "org/big" in result.stdout
    assert "org/small" in result.stdout
    assert "Total:" in result.stdout


def test_models_prune_confirms_and_deletes(monkeypatch) -> None:
    import huggingface_hub

    strategy = types.SimpleNamespace(
        expected_freed_size=20 * 1024**3, execute=lambda: None
    )
    deleted: list = []
    cache = types.SimpleNamespace(
        repos=[_fake_repo("org/big", 20 * 1024**3)],
        size_on_disk=20 * 1024**3,
        delete_revisions=lambda *hashes: deleted.append(hashes) or strategy,
    )
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: cache)
    result = runner.invoke(app, ["models", "prune", "org/big"], input="y\n")
    assert result.exit_code == 0
    assert deleted == [("abc123",)]
    assert "Freed" in result.stdout


def test_models_prune_unknown_repo(monkeypatch) -> None:
    import huggingface_hub

    cache = types.SimpleNamespace(repos=[], size_on_disk=0)
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: cache)
    result = runner.invoke(app, ["models", "prune", "org/ghost"])
    assert result.exit_code == 1
    assert "No cached repo" in result.stdout
