"""Phase 10: the `sovereign monitor` dashboard."""

from __future__ import annotations

import asyncio
import io
import types

from rich.console import Console
from typer.testing import CliRunner

from sovereign import __version__, main
from sovereign.main import (
    _dashboard,
    _dashboard_task_factory,
    _duration_cell,
    _format_duration,
    _load_dashboard_status,
    app,
)
from sovereign.utils.state import write_json

runner = CliRunner()


# --- duration formatting ---
def test_format_duration_seconds() -> None:
    assert _format_duration(42) == "42s"


def test_format_duration_minutes() -> None:
    assert _format_duration(3 * 60 + 12) == "3m 12s"


def test_format_duration_hours() -> None:
    assert _format_duration(3600 + 4 * 60) == "1h 04m"


def test_duration_cell_none_renders_dash() -> None:
    assert _duration_cell(None) == "-"


def test_duration_cell_unparseable_renders_dash() -> None:
    assert _duration_cell("not-a-timestamp") == "-"


def test_duration_cell_future_timestamp_clamps_to_zero() -> None:
    from datetime import UTC, datetime, timedelta

    future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    assert _duration_cell(future) == "0s"

_STATUS = {
    "updated_at": "2026-07-05T00:00:00+00:00",
    "services": {
        "llama_heavy_v1": {
            "state": "ready",
            "since": "2026-07-05T00:00:00+00:00",
            "endpoint": "http://127.0.0.1:11435",
            "metrics": {"cpu_percent": 12.4, "memory_mb": 14500.0, "status": "running"},
        },
        "open_webui": {
            "state": "starting",
            "since": "2026-07-05T00:03:12+00:00",
            "endpoint": None,
            "metrics": {},
        },
    },
}


def _render(table) -> str:
    console = Console(record=True, width=120)
    console.print(table)
    return console.export_text()


def test_dashboard_matches_mockup_shape() -> None:
    text = _render(_dashboard(_STATUS))
    assert f"Sovereign Control Plane v{__version__}" in text
    for header in ("SERVICE", "STATUS", "DURATION", "CPU %", "MEM (MB)", "ENDPOINT"):
        assert header in text
    assert "DEPENDENCIES" not in text
    # ready -> RUNNING label; metrics rendered; endpoint rendered
    assert "RUNNING" in text
    assert "12.4%" in text
    assert "14500" in text
    assert "STARTING" in text
    assert "http://127.0.0.1:11435" in text
    # missing endpoint/metrics render as "-"
    assert "-" in text


def test_dashboard_renders_activity_area() -> None:
    status = {
        "services": {
            "open_webui": {
                "state": "provisioning",
                "metrics": {},
                "activity": "pulling open-webui — 3/8 layers",
            }
        }
    }
    text = _render(_dashboard(status))
    assert "Activity:" in text
    assert "pulling open-webui — 3/8 layers" in text


def test_dashboard_activity_shown_for_ready_service() -> None:
    # a READY service with activity (e.g. background model download) should also show
    status = {
        "services": {
            "mlx_heavy": {
                "state": "ready",
                "metrics": {},
                "activity": "downloading model: 3/8 files (38%)",
            }
        }
    }
    text = _render(_dashboard(status))
    assert "Activity:" in text
    assert "downloading model: 3/8 files (38%)" in text


def test_dashboard_no_activity_area_when_idle() -> None:
    # ready service with no activity should not show an Activity area
    status = {
        "services": {
            "engine": {"state": "ready", "metrics": {}, "activity": ""}
        }
    }
    text = _render(_dashboard(status))
    assert "Provisioning:" not in text


def test_monitor_once_from_status_file(tmp_path) -> None:
    write_json(tmp_path / "status.json", _STATUS)
    result = runner.invoke(app, ["monitor", "--once", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "llama_heavy_v1" in result.stdout
    assert "RUNNING" in result.stdout


def test_monitor_no_state(tmp_path) -> None:
    result = runner.invoke(app, ["monitor", "--once", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No running stack" in result.stdout


def test_monitor_falls_back_to_state_json(tmp_path) -> None:
    write_json(
        tmp_path / "state.json",
        {"services": {"docker_engine": "ready"}, "variant_file": None, "variant_hash": None},
    )
    status = _load_dashboard_status(tmp_path)
    assert status["services"]["docker_engine"]["state"] == "ready"
    result = runner.invoke(app, ["monitor", "--once", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "docker_engine" in result.stdout


# --- foreground `up` dashboard ---
def test_dashboard_task_renders_from_snapshot_and_exits() -> None:
    calls = {"n": 0}

    def snapshot() -> dict:
        calls["n"] += 1
        return _STATUS

    fake_orch = types.SimpleNamespace(status_snapshot=snapshot)
    rec = Console(file=io.StringIO(), width=200, force_terminal=True)
    stop = asyncio.Event()
    stop.set()  # exit immediately after the initial frame

    task = _dashboard_task_factory(interval=0.01, live_console=rec)
    asyncio.run(asyncio.wait_for(task(fake_orch, stop), timeout=2))
    assert calls["n"] >= 1  # rendered at least the initial frame from live state


def _valid_variant(tmp_path):
    variant = tmp_path / "s.yaml"
    variant.write_text(
        "version: '1.1'\n"
        "resources: {max_unified_memory_gb: 8, safety_margin_gb: 1}\n"
        "services: []\n"
    )
    return variant


def _stub_serve(monkeypatch) -> dict:
    captured: dict = {}

    async def fake_serve(
        config, *, variant_file=None, state_dir=".sovereign", extra_tasks=(), on_transition=None
    ):
        captured["extra_tasks"] = list(extra_tasks)
        captured["on_transition"] = on_transition
        return None

    monkeypatch.setattr(main, "serve_forever", fake_serve)
    return captured


def test_up_shows_dashboard_when_tty(monkeypatch, tmp_path) -> None:
    captured = _stub_serve(monkeypatch)
    monkeypatch.setattr(main, "_stdout_is_tty", lambda: True)
    result = runner.invoke(app, ["up", "-f", str(_valid_variant(tmp_path))])
    assert result.exit_code == 0
    assert len(captured["extra_tasks"]) == 1  # dashboard task attached
    assert captured["on_transition"] is None  # dashboard shows transitions


def test_up_headless_when_not_tty(monkeypatch, tmp_path) -> None:
    captured = _stub_serve(monkeypatch)
    monkeypatch.setattr(main, "_stdout_is_tty", lambda: False)
    result = runner.invoke(app, ["up", "-f", str(_valid_variant(tmp_path))])
    assert result.exit_code == 0
    assert captured["extra_tasks"] == []
    assert captured["on_transition"] is main._print_transition  # prints progress lines


def test_serve_always_headless_even_in_tty(monkeypatch, tmp_path) -> None:
    captured = _stub_serve(monkeypatch)
    monkeypatch.setattr(main, "_stdout_is_tty", lambda: True)
    result = runner.invoke(app, ["serve", "-f", str(_valid_variant(tmp_path))])
    assert result.exit_code == 0
    assert captured["extra_tasks"] == []
    assert captured["on_transition"] is main._print_transition
