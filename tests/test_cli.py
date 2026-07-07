"""Phase 8: CLI surface — help tree, status + drift, down, logs."""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from sovereign import main
from sovereign.main import _load_dotenv, app
from sovereign.utils.state import file_hash, write_json

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
    from sovereign.utils.state import read_json

    state = read_json(tmp_path / "state.json")
    assert state["services"] == {"engine": "stopped", "frontend": "stopped"}
    assert state["runtime"] == {}


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

    def __init__(self, entry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.entry = entry
        self.materialized = False

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


def test_bench_run_attach_mode_success(tmp_path, monkeypatch) -> None:
    import sys
    import types

    from sovereign.utils.state import write_json

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
