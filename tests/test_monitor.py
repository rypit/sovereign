"""Phase 10: the `sovereign monitor` dashboard."""

from __future__ import annotations

import asyncio
import io
import time
import types

from rich.console import Console
from rich.spinner import Spinner
from typer.testing import CliRunner

from sovereign import __version__, main
from sovereign.dashboard import (
    MetricHistory,
    dashboard,
    dashboard_task_factory,
    duration_cell,
    format_duration,
    load_dashboard_status,
    sparkline,
    status_cell,
)
from sovereign.main import app
from sovereign.utils.state import write_json

runner = CliRunner()


# --- duration formatting ---
def test_format_duration_seconds() -> None:
    assert format_duration(42) == "42s"


def test_format_duration_minutes() -> None:
    assert format_duration(3 * 60 + 12) == "3m 12s"


def test_format_duration_hours() -> None:
    assert format_duration(3600 + 4 * 60) == "1h 04m"


def test_duration_cell_none_renders_dash() -> None:
    assert duration_cell(None) == "-"


def test_duration_cell_unparseable_renders_dash() -> None:
    assert duration_cell("not-a-timestamp") == "-"


def test_duration_cell_future_timestamp_clamps_to_zero() -> None:
    from datetime import UTC, datetime, timedelta

    future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    assert duration_cell(future) == "0s"

_STATUS = {
    "updated_at": "2026-07-05T00:00:00+00:00",
    "services": {
        "llama_heavy_v1": {
            "state": "ready",
            "since": "2026-07-05T00:00:00+00:00",
            "endpoint": "http://127.0.0.1:11435",
            "descriptor": "mlx-community/Qwen3.6-27B-8bit",
            "metrics": {"cpu_percent": 12.4, "memory_mb": 14500.0, "status": "running"},
        },
        "open_webui": {
            "state": "starting",
            "since": "2026-07-05T00:03:12+00:00",
            "endpoint": None,
            "descriptor": None,
            "metrics": {},
        },
    },
}


def _render(table) -> str:
    console = Console(record=True, width=160)
    console.print(table)
    return console.export_text()


def test_dashboard_matches_mockup_shape() -> None:
    text = _render(dashboard(_STATUS))
    assert f"Sovereign Control Plane v{__version__}" in text
    for header in (
        "SERVICE", "DESCRIPTOR", "STATUS", "DURATION", "CPU %", "MEM (MB)", "ENDPOINT",
    ):
        assert header in text
    assert "DEPENDENCIES" not in text
    # ready -> RUNNING label; metrics rendered; endpoint rendered
    assert "RUNNING" in text
    assert "12.4%" in text
    assert "14500" in text
    assert "STARTING" in text
    assert "http://127.0.0.1:11435" in text
    assert "mlx-community/Qwen3.6-27B-8bit" in text
    # missing endpoint/metrics/descriptor render as "-"
    assert "-" in text


# --- sparklines ---
def test_sparkline_empty_for_fewer_than_two_points() -> None:
    assert sparkline([]) == ""
    assert sparkline([1.0]) == ""


def test_sparkline_flat_when_all_equal() -> None:
    assert sparkline([5.0, 5.0, 5.0]) == "▅▅▅"


def test_sparkline_scales_min_to_max() -> None:
    spark = sparkline([0.0, 100.0])
    assert spark[0] == "▁"
    assert spark[-1] == "█"


def test_sparkline_capped_to_render_width() -> None:
    values = list(range(30))
    result = sparkline(values)
    assert len(result) == 12
    assert result == sparkline(values[-12:])


# --- MetricHistory ---
def test_metric_history_prunes_old_samples_by_age() -> None:
    history = MetricHistory(window_seconds=0.05)
    history.record({"services": {"a": {"metrics": {"cpu_percent": 1.0}}}})
    time.sleep(0.1)
    history.record({"services": {"a": {"metrics": {"cpu_percent": 2.0}}}})
    assert history.values("a", "cpu_percent") == [2.0]


def test_metric_history_keeps_recent_samples() -> None:
    history = MetricHistory(window_seconds=5.0)
    history.record({"services": {"a": {"metrics": {"cpu_percent": 1.0}}}})
    history.record({"services": {"a": {"metrics": {"cpu_percent": 2.0}}}})
    assert history.values("a", "cpu_percent") == [1.0, 2.0]


def test_metric_history_prunes_services_no_longer_present() -> None:
    history = MetricHistory()
    history.record({"services": {"a": {"metrics": {"cpu_percent": 1.0}}}})
    history.record({"services": {"b": {"metrics": {"cpu_percent": 2.0}}}})
    assert history.values("a", "cpu_percent") == []
    assert history.values("b", "cpu_percent") == [2.0]


def test_metric_history_per_metric_independence() -> None:
    history = MetricHistory()
    history.record({"services": {"a": {"metrics": {"cpu_percent": 1.0}}}})
    history.record({"services": {"a": {"metrics": {"cpu_percent": 2.0, "memory_mb": 100.0}}}})
    assert history.values("a", "cpu_percent") == [1.0, 2.0]
    assert history.values("a", "memory_mb") == [100.0]


def test_metric_history_instances_share_no_state() -> None:
    a = MetricHistory()
    b = MetricHistory()
    a.record({"services": {"x": {"metrics": {"cpu_percent": 1.0}}}})
    assert b.values("x", "cpu_percent") == []


# --- _status_cell ---
def test_status_cell_spinner_for_transitional_states() -> None:
    assert isinstance(status_cell("provisioning"), Spinner)
    assert isinstance(status_cell("starting"), Spinner)


def test_status_cell_plain_markup_for_steady_states() -> None:
    assert status_cell("ready") == "[green]RUNNING[/green]"
    assert status_cell("stopped") == "[dim]STOPPED[/dim]"
    assert status_cell("failed") == "[red]FAILED[/red]"


# --- backward-compat regression: no sparkline artifacts without history ---
def test_dashboard_without_history_has_no_sparkline_artifacts() -> None:
    text = _render(dashboard(_STATUS))
    assert "RUNNING" in text
    assert "12.4%" in text
    assert "14500" in text
    assert "STARTING" in text
    assert "http://127.0.0.1:11435" in text
    assert not any(ch in text for ch in "▁▂▃▄▅▆▇█")


def test_dashboard_with_empty_history_has_no_sparkline_artifacts() -> None:
    text = _render(dashboard(_STATUS, history=MetricHistory()))
    assert not any(ch in text for ch in "▁▂▃▄▅▆▇█")


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
    text = _render(dashboard(status))
    assert "Activity:" in text
    assert "pulling open-webui — 3/8 layers" in text


# --- M5: budget footer, EST column, download progress ---
def test_dashboard_renders_est_column_and_budget_footer() -> None:
    status = {
        "budget": {"usable_gb": 120.0, "reserved_gb": 27.0, "available_gb": 93.0},
        "services": {
            "mlx_heavy": {
                "state": "downloading",
                "descriptor": "mlx-community/Qwen3.6-27B-8bit",
                "estimated_gb": 27.0,
                "metrics": {},
                # activity is huggingface_hub's own tqdm-rendered line, forwarded as-is
                "activity": "model.safetensors:  18%|█▊        | 3.20G/17.8G "
                "[01:10<05:20, 45.0MB/s]",
            },
        },
    }
    text = _render(dashboard(status))
    assert "EST (GB)" in text
    assert "27.0" in text  # estimate column value
    assert "93.0 GB headroom" in text  # budget footer
    assert "120 usable GB" in text
    # DOWNLOADING activity flows through the activity area (brackets render literally).
    assert "3.20G/17.8G" in text


def test_dashboard_tolerates_status_without_budget() -> None:
    # Old status.json (pre-M5) has no "budget" / "estimated_gb" keys.
    text = _render(dashboard(_STATUS))
    assert "headroom" not in text  # no footer without a budget
    assert "RUNNING" in text  # still renders normally


def test_dashboard_activity_shown_for_ready_service() -> None:
    # a READY service with activity (e.g. background model download) should also show
    status = {
        "services": {
            "mlx_heavy": {
                "state": "ready",
                "metrics": {},
                "activity": "Fetching 8 files:  38%|███▊      | 3/8 [00:10<00:17,  3.4s/it]",
            }
        }
    }
    text = _render(dashboard(status))
    assert "Activity:" in text
    assert "Fetching 8 files:" in text


def test_dashboard_no_activity_area_when_idle() -> None:
    # ready service with no activity should not show an Activity area
    status = {
        "services": {
            "engine": {"state": "ready", "metrics": {}, "activity": ""}
        }
    }
    text = _render(dashboard(status))
    assert "Provisioning:" not in text


def test_monitor_once_from_status_file(tmp_path) -> None:
    write_json(tmp_path / "status.json", _STATUS)
    result = runner.invoke(
        app, ["monitor", "--once", "--state-dir", str(tmp_path)], env={"COLUMNS": "200"}
    )
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
        {"services": {"docker": "ready"}, "variant_file": None, "variant_hash": None},
    )
    status = load_dashboard_status(tmp_path)
    assert status["services"]["docker"]["state"] == "ready"
    result = runner.invoke(
        app, ["monitor", "--once", "--state-dir", str(tmp_path)], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    assert "docker" in result.stdout


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

    task = dashboard_task_factory(interval=0.01, live_console=rec)
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
