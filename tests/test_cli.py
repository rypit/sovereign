"""Phase 8: CLI surface — help tree, status + drift, down, logs."""

from __future__ import annotations

import os

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
