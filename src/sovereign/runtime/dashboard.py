"""The live dashboard (§8): Rich rendering for `sovereign up` and `sovereign monitor`.

Pure presentation — consumes the :class:`~sovereign.runtime.status.StatusSnapshot`
shape that ``Orchestrator.status_snapshot()`` produces (and persists as
``status.json``), and renders three stacked panels — "Sovereign" (the service
table, with sparklines), "Memory" (usage bars vs budget and machine RAM), and
"Activity" (per-service activity lines). No orchestration logic lives here; no
rendering logic lives in the orchestrator.

Two cadences are deliberately decoupled (§5/§8):

- **1 Hz snapshot**: ``dashboard_task_factory``'s poll loop (and ``monitor``'s
  own poll) re-reads ``orch.status_snapshot()`` / ``status.json`` once a
  second — that's the ingest rate for MEM/TOK/S history and table content.
- **12 fps render**: ``Live(..., refresh_per_second=12)`` redraws the same
  frame far more often than the snapshot changes, purely so the STATUS
  column's ``Spinner`` (provisioning/downloading/starting) and the prefill
  pulse animate smoothly. It never causes an extra snapshot read.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from sovereign import __version__
from sovereign.core.state import read_json_or_none
from sovereign.core.units import fmt_size

if TYPE_CHECKING:
    from sovereign.runtime.orchestrator import Orchestrator

STATE_COLORS = {
    "ready": "green",
    "running": "green",
    "degraded": "yellow",
    "starting": "cyan",
    "provisioning": "cyan",
    "downloading": "cyan",
    "failed": "red",
    "stopped": "dim",
}

_STATUS_LABEL = {"ready": "RUNNING"}


def status_label(state: str) -> str:
    return _STATUS_LABEL.get(state, state.upper())


def load_dashboard_status(state_dir: Path) -> dict | None:
    """Prefer the live status.json; fall back to state.json (states only).

    Tolerant reads: a poller (``monitor``, the foreground dashboard) calls this
    on an interval against a file another process may be mid-write to. A
    missing file or a decode error (a write caught between open and replace —
    vanishingly rare now that ``write_json`` is atomic, but free to guard
    against) returns None; callers keep their last-known-good snapshot rather
    than crash or flash a torn read.
    """
    status = read_json_or_none(state_dir / "status.json")
    if status is not None:
        return status
    state = read_json_or_none(state_dir / "state.json")
    if state is not None:
        return {
            "services": {
                name: {"state": svc_state, "metrics": {}}
                for name, svc_state in state.get("services", {}).items()
            }
        }
    return None


def format_duration(seconds: float) -> str:
    """Compact elapsed time: "42s", "3m 12s", "1h 04m", "2d 05h"."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def duration_cell(since: str | None) -> str:
    """Elapsed time since an ISO timestamp, or "-" when unknown."""
    if not since:
        return "-"
    try:
        started = datetime.fromisoformat(since)
    except (TypeError, ValueError):
        return "-"
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return format_duration(max(0.0, elapsed))


_HISTORY_SECONDS = 60.0  # trailing window kept per service per metric; tune here


class MetricHistory:
    """Rolling ~60s-window per-service, per-metric history for sparklines.

    Constructed once per dashboard session (once in monitor(), once per
    dashboard_task_factory() task invocation) — never a module-level global,
    so state never leaks across unrelated sessions or test invocations. Never
    exposed as a user-facing parameter; always defaults to _HISTORY_SECONDS.
    """

    #: Metric keys recorded from each service's ``metrics`` dict into sparkline
    #: history — memory (MEM column) and generation throughput (TOK/S column).
    _RECORDED_KEYS = ("memory_bytes", "tokens_per_second")

    def __init__(self, window_seconds: float = _HISTORY_SECONDS) -> None:
        self._window = window_seconds
        self._data: dict[str, dict[str, deque[tuple[float, float]]]] = {}
        #: First-seen monotonic time per (service, request_id), so an
        #: indeterminate prefill (total=None) can render a "pulse" with a
        #: real elapsed duration — the telemetry cache/status.json don't
        #: carry a per-request start timestamp, so this is tracked here,
        #: dashboard-side, across successive record() calls.
        self._prefill_started: dict[tuple[str, str], float] = {}

    def record(self, status: Mapping[str, Any]) -> None:
        now = time.monotonic()
        services = status.get("services", {})
        for stale in set(self._data) - set(services):
            del self._data[stale]
        live_requests: set[tuple[str, str]] = set()
        for name, svc in services.items():
            metrics = svc.get("metrics") or {}
            buckets = self._data.setdefault(name, {})
            for key in self._RECORDED_KEYS:
                if key in metrics:
                    dq = buckets.setdefault(key, deque())
                    dq.append((now, metrics[key]))
                    cutoff = now - self._window
                    while dq and dq[0][0] < cutoff:
                        dq.popleft()
            for entry in (svc.get("telemetry") or {}).get("prefill", []):
                request_id = entry.get("request_id")
                if not request_id:
                    continue
                req_key = (name, request_id)
                live_requests.add(req_key)
                self._prefill_started.setdefault(req_key, now)
        # Drop bookkeeping for requests no longer active (completed/abandoned).
        for req_key in set(self._prefill_started) - live_requests:
            del self._prefill_started[req_key]

    def values(self, service: str, metric: str) -> list[float]:
        return [v for _, v in self._data.get(service, {}).get(metric, ())]

    def prefill_elapsed(self, service: str, request_id: str) -> float:
        """Seconds since this (service, request_id) prefill was first observed."""
        started = self._prefill_started.get((service, request_id), time.monotonic())
        return max(0.0, time.monotonic() - started)


_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_SPARK_WIDTH = 12  # rendered sparkline width; tune here


def sparkline(values: Sequence[float]) -> str:
    """A trailing Unicode-block sparkline, min-max scaled to the visible window."""
    values = list(values)[-_SPARK_WIDTH:]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        mid = _SPARK_CHARS[len(_SPARK_CHARS) // 2]
        return mid * len(values)
    span = hi - lo
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / span * len(_SPARK_CHARS)), len(_SPARK_CHARS) - 1)]
        for v in values
    )


def _metric_cell(text: str, spark: str) -> str:
    return f"{text} {spark}" if spark else text


_SPINNER_STATES = {"provisioning", "downloading", "starting"}

_PREFILL_BAR_WIDTH = 16


def status_cell(state: str) -> str | Spinner:
    """Plain colored label for steady states; an animated spinner while coming online."""
    color = STATE_COLORS.get(state, "white")
    markup = f"[{color}]{status_label(state)}[/{color}]"
    if state in _SPINNER_STATES:
        return Spinner("dots", text=Text.from_markup(markup))
    return markup


def prefill_bar(processed: int, total: int | None, *, elapsed: float = 0.0) -> str:
    """One prefill's activity-area line: a determinate fraction bar when the
    total token count is known, or a "pulse" annotated with elapsed time when
    it isn't (llama_cpp's start/finish-only progress — see §3a)."""
    if total:
        filled = min(_PREFILL_BAR_WIDTH, round(_PREFILL_BAR_WIDTH * processed / total))
        bar = "█" * filled + "░" * (_PREFILL_BAR_WIDTH - filled)
        return f"prefill ▕{bar}▏ {processed}/{total} tok"
    pulse = "░" * _PREFILL_BAR_WIDTH
    return f"prefill ▕{pulse}▏ {format_duration(elapsed)}"


_USAGE_BAR_WIDTH = 24


def usage_color(pct: float) -> str:
    """Traffic-light style for a usage percentage: <50 green, <85 yellow, else red."""
    if pct < 50:
        return "green"
    if pct < 85:
        return "yellow"
    return "red"


def usage_bar(pct: float) -> str:
    """A determinate ▕█░▏ usage bar (same glyphs as :func:`prefill_bar`)."""
    filled = min(_USAGE_BAR_WIDTH, round(_USAGE_BAR_WIDTH * pct / 100))
    return f"▕{'█' * filled}{'░' * (_USAGE_BAR_WIDTH - filled)}▏"


def memory_panel(status: Mapping[str, Any]) -> Panel | None:
    """The "Memory" panel: actual usage vs budget and machine RAM, one row per stat.

    Rows degrade gracefully: SYSTEM needs the ``system_*_bytes`` fields, so a
    status.json written by an older orchestrator shows only the BUDGET row;
    a status with no budget at all (pre-M5) gets no panel. Reserved/headroom
    (the admission-control view) stays on `sovereign plan` via budget_footer().
    """
    budget = status.get("budget")
    if not budget:
        return None
    stack_used = sum(
        (svc.get("metrics") or {}).get("memory_bytes", 0)
        for svc in status.get("services", {}).values()
    )
    rows: list[tuple[str, int, int]] = [("BUDGET", stack_used, budget.get("usable_bytes", 0))]
    system_total = budget.get("system_total_bytes")
    system_used = budget.get("system_used_bytes")
    if system_total and system_used is not None:
        rows.append(("SYSTEM", system_used, system_total))

    table = Table(box=box.SIMPLE_HEAD)
    for col in ("STAT", "USAGE", "USED", "TOTAL", "PCT"):
        table.add_column(col)
    for stat, used, total in rows:
        pct = 100 * used / total if total > 0 else 0.0
        color = usage_color(pct)
        table.add_row(
            Text(stat, style=color),
            Text(usage_bar(pct), style=color),
            Text(fmt_size(used), style=color),
            Text(fmt_size(total), style=color),
            Text(f"{pct:.0f}%", style=color),
        )
    return Panel(table, title="Memory", title_align="left")


def dashboard(status: Mapping[str, Any], history: MetricHistory | None = None) -> RenderableType:
    """Render the §8 dashboard: a "Sovereign" panel (service table), a "Memory"
    panel (usage bars vs budget and machine RAM), and an "Activity" panel
    (per-service activity/prefill lines)."""
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("SERVICE")
    table.add_column("ENGINE")
    table.add_column("DESCRIPTOR")
    table.add_column("STATUS")
    table.add_column("DURATION")
    table.add_column("MEM")
    table.add_column("TOK/S")
    table.add_column("EST")
    table.add_column("ENDPOINT")

    activity_lines: list[str] = []
    for name, svc in status.get("services", {}).items():
        state = svc.get("state", "unknown")
        metrics = svc.get("metrics") or {}
        mem = fmt_size(metrics["memory_bytes"]) if "memory_bytes" in metrics else "-"
        mem_spark = sparkline(history.values(name, "memory_bytes")) if history else ""
        tps = metrics.get("tokens_per_second")
        tok_s = f"{tps:.1f}" if isinstance(tps, int | float) else "-"
        tok_spark = sparkline(history.values(name, "tokens_per_second")) if history else ""
        duration = duration_cell(svc.get("since"))
        endpoint = svc.get("endpoint") or "-"
        engine = svc.get("engine") or "-"
        descriptor = svc.get("descriptor") or "-"
        estimated = svc.get("estimated_bytes")
        est = fmt_size(estimated) if estimated is not None else "-"
        table.add_row(
            name,
            engine,
            descriptor,
            status_cell(state),
            duration,
            _metric_cell(mem, mem_spark),
            _metric_cell(tok_s, tok_spark),
            est,
            endpoint,
        )

        lines = [ln for ln in (svc.get("activity") or {}).get("lines", []) if ln.strip()]
        prefill_lines = []
        for entry in (svc.get("telemetry") or {}).get("prefill", []):
            request_id = entry.get("request_id", "")
            elapsed = history.prefill_elapsed(name, request_id) if history else 0.0
            prefill_lines.append(
                prefill_bar(entry.get("processed", 0), entry.get("total"), elapsed=elapsed)
            )
        all_lines = lines + prefill_lines
        if all_lines:
            # A header naming the service, then each activity line indented under it
            # (e.g. huggingface_hub's several concurrent download bars, or an
            # in-flight prefill's progress bar). State is already in the table's
            # STATUS column, so it isn't repeated here.
            activity_lines.append(f"  {name}")
            activity_lines.extend(f"    {ln}" for ln in all_lines)

    activity_body = Text(
        "\n".join(activity_lines) if activity_lines else "no activity", style="dim"
    )
    parts: list[RenderableType] = [
        Panel(
            table,
            title="Sovereign",
            title_align="left",
            subtitle=f"v{__version__}",
            subtitle_align="right",
        ),
    ]
    memory = memory_panel(status)
    if memory is not None:
        parts.append(memory)
    parts.append(Panel(activity_body, title="Activity", title_align="left"))
    return Group(*parts)


def budget_footer(budget: dict | None) -> Text | None:
    """A one-line reserved/headroom summary, or None without a budget.

    Used by `sovereign plan` (where there is no live usage to chart); the
    dashboard itself renders actual usage via :func:`memory_panel` instead.
    """
    if not budget:
        return None
    reserved = budget.get("reserved_bytes", 0)
    usable = budget.get("usable_bytes", 0)
    available = budget.get("available_bytes", 0)
    return Text(
        f"Memory: {fmt_size(reserved)} reserved / {fmt_size(usable)} usable "
        f"— {fmt_size(available)} headroom",
        style="bold",
    )


def dashboard_task_factory(interval: float = 1.0, live_console: Console | None = None):
    """An extra serve task that renders the live dashboard from in-process state."""

    async def task(orch: Orchestrator, stop: asyncio.Event) -> None:
        history = MetricHistory()
        snapshot = orch.status_snapshot()
        history.record(snapshot)
        with Live(
            dashboard(snapshot, history=history),
            console=live_console or Console(),
            refresh_per_second=12,
        ) as live:
            while not stop.is_set():
                snapshot = orch.status_snapshot()
                history.record(snapshot)
                live.update(dashboard(snapshot, history=history))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    pass

    return task
