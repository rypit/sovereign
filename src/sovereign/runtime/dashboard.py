"""The live dashboard (§8): Rich rendering for `sovereign up` and `sovereign monitor`.

Pure presentation — consumes the :class:`~sovereign.runtime.status.StatusSnapshot`
shape that ``Orchestrator.status_snapshot()`` produces (and persists as
``status.json``), and renders the service table, sparklines, activity lines, and
budget footer. No orchestration logic lives here; no rendering logic lives in the
orchestrator.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from sovereign import __version__
from sovereign.core.state import read_json_or_none

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

    def __init__(self, window_seconds: float = _HISTORY_SECONDS) -> None:
        self._window = window_seconds
        self._data: dict[str, dict[str, deque[tuple[float, float]]]] = {}

    def record(self, status: Mapping[str, Any]) -> None:
        now = time.monotonic()
        services = status.get("services", {})
        for stale in set(self._data) - set(services):
            del self._data[stale]
        for name, svc in services.items():
            metrics = svc.get("metrics") or {}
            buckets = self._data.setdefault(name, {})
            for key in ("cpu_percent", "memory_mb"):
                if key in metrics:
                    dq = buckets.setdefault(key, deque())
                    dq.append((now, metrics[key]))
                    cutoff = now - self._window
                    while dq and dq[0][0] < cutoff:
                        dq.popleft()

    def values(self, service: str, metric: str) -> list[float]:
        return [v for _, v in self._data.get(service, {}).get(metric, ())]


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


def status_cell(state: str) -> str | Spinner:
    """Plain colored label for steady states; an animated spinner while coming online."""
    color = STATE_COLORS.get(state, "white")
    markup = f"[{color}]{status_label(state)}[/{color}]"
    if state in _SPINNER_STATES:
        return Spinner("dots", text=Text.from_markup(markup))
    return markup


def dashboard(status: Mapping[str, Any], history: MetricHistory | None = None) -> RenderableType:
    """Render the §8 dashboard table, plus a live "Provisioning" activity area."""
    table = Table(title=f"Sovereign Control Plane v{__version__}", title_justify="left")
    table.add_column("SERVICE")
    table.add_column("ENGINE")
    table.add_column("DESCRIPTOR")
    table.add_column("STATUS")
    table.add_column("DURATION")
    table.add_column("MEM (MB)")
    table.add_column("EST (GB)")
    table.add_column("ENDPOINT")

    activity_lines: list[str] = []
    for name, svc in status.get("services", {}).items():
        state = svc.get("state", "unknown")
        metrics = svc.get("metrics") or {}
        mem = f"{metrics['memory_mb']:.0f}" if "memory_mb" in metrics else "-"
        mem_spark = sparkline(history.values(name, "memory_mb")) if history else ""
        duration = duration_cell(svc.get("since"))
        endpoint = svc.get("endpoint") or "-"
        descriptor = svc.get("descriptor") or "-"
        engine = svc.get("base_type") or "-"
        estimated = svc.get("estimated_gb")
        est = f"{estimated:.1f}" if estimated is not None else "-"
        table.add_row(
            name,
            engine,
            descriptor,
            status_cell(state),
            duration,
            _metric_cell(mem, mem_spark),
            est,
            endpoint,
        )

        lines = [ln for ln in (svc.get("activity") or {}).get("lines", []) if ln.strip()]
        if lines:
            # A header naming the service, then each activity line indented under it
            # (e.g. huggingface_hub's several concurrent download bars). State is
            # already in the table's STATUS column, so it isn't repeated here.
            activity_lines.append(f"  {name}")
            activity_lines.extend(f"    {ln}" for ln in lines)

    footer = budget_footer(status.get("budget"))
    if not activity_lines and footer is None:
        return table
    parts: list[RenderableType] = [table]
    if activity_lines:
        parts += [Text("Activity:", style="bold"), Text("\n".join(activity_lines), style="dim")]
    if footer is not None:
        parts.append(footer)
    return Group(*parts)


def budget_footer(budget: dict | None) -> Text | None:
    """A one-line unified-memory summary, or None when the status predates budgets."""
    if not budget:
        return None
    reserved = budget.get("reserved_gb", 0.0)
    usable = budget.get("usable_gb", 0.0)
    available = budget.get("available_gb", 0.0)
    return Text(
        f"Memory: {reserved:.1f} reserved / {usable:.0f} usable GB "
        f"— {available:.1f} GB headroom",
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
