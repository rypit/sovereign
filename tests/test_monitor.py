"""Phase 10: the `sovereign monitor` dashboard."""

from __future__ import annotations

import asyncio
import io
import sys
import time
import types

from rich.console import Console
from rich.spinner import Spinner
from typer.testing import CliRunner

from sovereign import __version__
from sovereign.cli import app
from sovereign.cli import stack as main
from sovereign.core.state import write_json
from sovereign.runtime.dashboard import (
    MetricHistory,
    dashboard,
    dashboard_task_factory,
    duration_cell,
    format_duration,
    load_dashboard_status,
    prefill_bar,
    sparkline,
    status_cell,
    usage_bar,
    usage_color,
)

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
            "engine": "mlx_lm",
            "metrics": {"memory_bytes": 14_500_000_000, "status": "running"},
        },
        "open_webui": {
            "state": "starting",
            "since": "2026-07-05T00:03:12+00:00",
            "endpoint": None,
            "descriptor": None,
            "engine": "docker",
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
    assert "Sovereign" in text  # top panel title
    assert f"v{__version__}" in text  # version in the panel subtitle
    assert "━" not in text  # borderless table: no heavy header rule / edges
    for header in (
        "SERVICE", "ENGINE", "DESCRIPTOR", "STATUS", "DURATION", "MEM", "ENDPOINT",
    ):
        assert header in text
    assert "DEPENDENCIES" not in text
    assert "CPU %" not in text
    # ready -> RUNNING label; metrics rendered; endpoint rendered
    assert "RUNNING" in text
    assert "14.5 GB" in text
    assert "STARTING" in text
    assert "http://127.0.0.1:11435" in text
    assert "mlx-community/Qwen3.6-27B-8bit" in text
    assert "mlx_lm" in text
    # missing endpoint/metrics/descriptor/engine render as "-"
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
    history.record({"services": {"a": {"metrics": {"memory_bytes": 1.0}}}})
    time.sleep(0.1)
    history.record({"services": {"a": {"metrics": {"memory_bytes": 2.0}}}})
    assert history.values("a", "memory_bytes") == [2.0]


def test_metric_history_keeps_recent_samples() -> None:
    history = MetricHistory(window_seconds=5.0)
    history.record({"services": {"a": {"metrics": {"memory_bytes": 1.0}}}})
    history.record({"services": {"a": {"metrics": {"memory_bytes": 2.0}}}})
    assert history.values("a", "memory_bytes") == [1.0, 2.0]


def test_metric_history_prunes_services_no_longer_present() -> None:
    history = MetricHistory()
    history.record({"services": {"a": {"metrics": {"memory_bytes": 1.0}}}})
    history.record({"services": {"b": {"metrics": {"memory_bytes": 2.0}}}})
    assert history.values("a", "memory_bytes") == []
    assert history.values("b", "memory_bytes") == [2.0]


def test_metric_history_instances_share_no_state() -> None:
    a = MetricHistory()
    b = MetricHistory()
    a.record({"services": {"x": {"metrics": {"memory_bytes": 1.0}}}})
    assert b.values("x", "memory_bytes") == []


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
    assert "14.5 GB" in text
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
                "activity": {"lines": ["pulling open-webui — 3/8 layers"]},
            }
        }
    }
    text = _render(dashboard(status))
    assert "Activity" in text  # panel title
    assert "pulling open-webui — 3/8 layers" in text


# --- M5: Memory panel, EST column, download progress ---
def test_dashboard_renders_est_column_and_memory_panel() -> None:
    status = {
        "budget": {
            "usable_bytes": 120 * 10**9,
            "reserved_bytes": 27 * 10**9,
            "available_bytes": 93 * 10**9,
            "system_total_bytes": 128 * 10**9,
            "system_used_bytes": 45 * 10**9,
        },
        "services": {
            "mlx_heavy": {
                "state": "downloading",
                "descriptor": "mlx-community/Qwen3.6-27B-8bit",
                "estimated_bytes": 27 * 10**9,
                "metrics": {"memory_bytes": 24 * 10**9},
                # activity is huggingface_hub's own tqdm-rendered lines, forwarded as-is
                "activity": {
                    "lines": [
                        "model.safetensors:  18%|█▊        | 3.20G/17.8G [01:10<05:20, 45.0MB/s]"
                    ]
                },
            },
        },
    }
    text = _render(dashboard(status))
    assert "EST" in text
    assert "EST (GB)" not in text
    assert "27.0 GB" in text  # estimate column value
    # Memory panel: one row per stat, actual usage vs budget / machine RAM
    assert "Memory" in text
    for stat in ("STAT", "USAGE", "USED", "TOTAL", "PCT"):
        assert stat in text
    assert "BUDGET" in text
    assert "STACK" in text
    assert "SYSTEM" in text
    assert "24.0 GB" in text  # stack used (sum of services' memory_bytes)
    assert "120.0 GB" in text  # budget usable
    assert "128.0 GB" in text  # machine total
    assert "20%" in text  # BUDGET row: 24/120
    assert "35%" in text  # SYSTEM row: 45/128
    assert "headroom" not in text  # reserved/headroom line lives in `plan` only now
    # DOWNLOADING activity flows through the activity area (brackets render literally).
    assert "3.20G/17.8G" in text


def test_dashboard_tolerates_status_without_budget() -> None:
    # Old status.json (pre-M5) has no "budget" / "estimated_gb" keys.
    text = _render(dashboard(_STATUS))
    assert "Memory" not in text  # no Memory panel without a budget
    assert "BUDGET" not in text
    assert "RUNNING" in text  # still renders normally


def test_memory_panel_budget_row_only_without_system_fields() -> None:
    # A status.json written by an older orchestrator lacks system_*_bytes:
    # the panel degrades to just the BUDGET row.
    status = {
        "budget": {
            "usable_bytes": 100 * 10**9,
            "reserved_bytes": 0,
            "available_bytes": 100 * 10**9,
        },
        "services": {"engine": {"state": "ready", "metrics": {"memory_bytes": 90 * 10**9}}},
    }
    text = _render(dashboard(status))
    assert "BUDGET" in text
    assert "90%" in text
    assert "STACK" not in text
    assert "SYSTEM" not in text


def test_usage_color_thresholds() -> None:
    assert usage_color(0) == "green"
    assert usage_color(49.9) == "green"
    assert usage_color(50) == "yellow"
    assert usage_color(84.9) == "yellow"
    assert usage_color(85) == "red"
    assert usage_color(120) == "red"


def test_usage_bar_fills_proportionally() -> None:
    empty, full = usage_bar(0), usage_bar(100)
    assert empty.startswith("▕") and empty.endswith("▏")
    assert "█" not in empty
    assert "░" not in full
    assert len(usage_bar(50)) == len(empty) == len(full)
    assert usage_bar(50).count("█") == usage_bar(50).count("░")


def test_usage_bar_clamps_over_100_percent() -> None:
    assert usage_bar(150) == usage_bar(100)


def test_dashboard_activity_shown_for_ready_service() -> None:
    # a READY service with activity (e.g. background model download) should also show
    status = {
        "services": {
            "mlx_heavy": {
                "state": "ready",
                "metrics": {},
                "activity": {
                    "lines": ["Fetching 8 files:  38%|███▊      | 3/8 [00:10<00:17,  3.4s/it]"]
                },
            }
        }
    }
    text = _render(dashboard(status))
    assert "Activity" in text  # panel title
    assert "Fetching 8 files:" in text


def test_dashboard_renders_multiline_activity_indented() -> None:
    # huggingface_hub's concurrent download bars arrive as several activity lines:
    # a header names the service and each line indents under it, without repeating state.
    status = {
        "services": {
            "engine": {
                "state": "downloading",
                "metrics": {},
                "activity": {
                    "lines": [
                        "Fetching 8 files:  38%| 3/8",
                        "Downloading bytes:  44%| 12.9G/29.0G",
                        "Reconstructing:  44%| 12.9G/29.0G",
                    ]
                },
            }
        }
    }
    text = _render(dashboard(status))
    # Activity lines render inside the Activity panel: strip the panel border
    # ("│ " + trailing " │") to inspect the indentation of the content itself.
    lines = [
        line.removeprefix("│ ").rstrip(" │")
        for line in text.splitlines()
        if line.startswith("│")
    ]
    header = next(line for line in lines if line.strip() == "engine")
    fetching = next(line for line in lines if "Fetching 8 files" in line)
    downloading = next(line for line in lines if "Downloading bytes" in line)
    assert header.startswith("  ")  # service name header, indented in the Activity panel
    assert "[DOWNLOADING]" not in fetching  # state not repeated in the activity block
    assert fetching.startswith("    ")  # each bar line indents under the header
    assert downloading.startswith("    ")


def test_dashboard_idle_keeps_activity_panel_with_placeholder() -> None:
    # Both panels always render (stable layout); an idle Activity panel shows
    # a dim placeholder instead of disappearing.
    status = {
        "services": {
            "engine": {"state": "ready", "metrics": {}, "activity": {"lines": []}}
        }
    }
    text = _render(dashboard(status))
    assert "Sovereign" in text
    assert "Activity" in text
    assert "no activity" in text


# --- TOK/S column + prefill bars (§5/§8) ---
def test_dashboard_renders_tokens_per_second_column() -> None:
    status = {
        "services": {
            "engine": {
                "state": "ready",
                "metrics": {"memory_bytes": 1_000_000, "tokens_per_second": 42.5},
            }
        }
    }
    text = _render(dashboard(status))
    assert "TOK/S" in text
    assert "42.5" in text


def test_dashboard_tok_s_dash_when_no_generation_stats() -> None:
    text = _render(dashboard(_STATUS))
    assert "TOK/S" in text


def test_prefill_bar_determinate() -> None:
    bar = prefill_bar(1234, 4096)
    assert bar.startswith("prefill ▕")
    assert "1234/4096 tok" in bar


def test_prefill_bar_pulse_when_total_none() -> None:
    bar = prefill_bar(0, None, elapsed=12.0)
    assert bar.startswith("prefill ▕")
    assert "12s" in bar
    assert "tok" not in bar


def test_dashboard_renders_prefill_bar_determinate() -> None:
    status = {
        "services": {
            "engine": {
                "state": "ready",
                "metrics": {},
                "telemetry": {
                    "prefill": [{"request_id": "r1", "processed": 1234, "total": 4096}]
                },
            }
        }
    }
    text = _render(dashboard(status))
    assert "prefill" in text
    assert "1234/4096 tok" in text


def test_dashboard_renders_prefill_bar_pulse() -> None:
    status = {
        "services": {
            "engine": {
                "state": "ready",
                "metrics": {},
                "telemetry": {
                    "prefill": [{"request_id": "r1", "processed": 0, "total": None}]
                },
            }
        }
    }
    history = MetricHistory()
    history.record(status)
    text = _render(dashboard(status, history=history))
    assert "prefill" in text


def test_metric_history_tracks_tokens_per_second() -> None:
    history = MetricHistory()
    history.record({"services": {"a": {"metrics": {"tokens_per_second": 10.0}}}})
    history.record({"services": {"a": {"metrics": {"tokens_per_second": 20.0}}}})
    assert history.values("a", "tokens_per_second") == [10.0, 20.0]


def test_metric_history_prefill_elapsed_grows_across_records() -> None:
    history = MetricHistory()
    status = {
        "services": {
            "a": {
                "metrics": {},
                "telemetry": {"prefill": [{"request_id": "r1", "processed": 1, "total": None}]},
            }
        }
    }
    history.record(status)
    first = history.prefill_elapsed("a", "r1")
    time.sleep(0.02)
    history.record(status)
    second = history.prefill_elapsed("a", "r1")
    assert second >= first


def test_metric_history_prefill_elapsed_resets_when_request_disappears() -> None:
    history = MetricHistory()
    status_with = {
        "services": {
            "a": {
                "metrics": {},
                "telemetry": {"prefill": [{"request_id": "r1", "processed": 1, "total": None}]},
            }
        }
    }
    history.record(status_with)
    time.sleep(0.02)
    status_without: dict = {"services": {"a": {"metrics": {}, "telemetry": {"prefill": []}}}}
    history.record(status_without)
    history.record(status_with)
    assert history.prefill_elapsed("a", "r1") < 0.02


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


def test_load_dashboard_status_tolerates_garbage_status_json(tmp_path) -> None:
    """A torn/corrupt status.json must not crash a poller — it falls back to
    state.json, or returns None so the caller keeps its last good snapshot."""
    (tmp_path / "status.json").write_text('{"services": {"a"')  # torn write
    assert load_dashboard_status(tmp_path) is None

    # With a valid state.json present, the fallback still works.
    write_json(tmp_path / "state.json", {"services": {"engine": "ready"}})
    status = load_dashboard_status(tmp_path)
    assert status is not None
    assert status["services"]["engine"]["state"] == "ready"


def test_monitor_once_survives_garbage_status_json(tmp_path) -> None:
    (tmp_path / "status.json").write_text("garbage not json")
    result = runner.invoke(app, ["monitor", "--once", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0  # no traceback; treated as no running stack
    assert "No running stack" in result.stdout


def test_monitor_falls_back_to_state_json(tmp_path) -> None:
    write_json(
        tmp_path / "state.json",
        {"services": {"docker": "ready"}, "variant_file": None, "variant_hash": None},
    )
    status = load_dashboard_status(tmp_path)
    assert status is not None
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
    # _boot_and_serve ends in _fast_exit (os._exit) to skip the download-thread
    # join; swap it for a SystemExit so CliRunner records the code, not a killed
    # pytest process.
    monkeypatch.setattr(main, "_fast_exit", lambda code: sys.exit(code))
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
