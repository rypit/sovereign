"""Typer entry point for the Sovereign CLI.

Commands are stubs at this stage (Phases 0–2); their real implementations land in
later phases per the roadmap (§12). The command surface is defined now so the CLI
shape stays stable as internals fill in.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from sovereign import __version__
from sovereign.config import ConfigError, load_config
from sovereign.orchestrator import BootError, serve_forever
from sovereign.utils.state import file_hash, read_json, write_json
from sovereign.utils.teardown import stop_service_handle

app = typer.Typer(
    name="sovereign",
    help="A declarative control plane for local LLM infrastructure on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_DEFAULT_CONFIG = Path("sovereign.yaml")
_DEFAULT_STATE_DIR = Path(".sovereign")

_STATE_DIR_OPTION = typer.Option(
    _DEFAULT_STATE_DIR, "--state-dir", help="Where Sovereign keeps state/logs."
)

_STATE_COLORS = {
    "ready": "green",
    "running": "green",
    "degraded": "yellow",
    "starting": "cyan",
    "provisioning": "cyan",
    "failed": "red",
    "stopped": "dim",
}


def _not_implemented(command: str) -> None:
    console.print(f"[yellow]`sovereign {command}` is not implemented yet.[/yellow]")


def _stdout_is_tty() -> bool:
    """Whether stdout is an interactive terminal (vs. a pipe or launchd log)."""
    return sys.stdout.isatty()


def _dashboard_task_factory(interval: float = 1.0, live_console: Console | None = None):
    """An extra serve task that renders the live dashboard from in-process state."""

    async def task(orch, stop: asyncio.Event) -> None:
        history = MetricHistory()
        snapshot = orch.status_snapshot()
        history.record(snapshot)
        with Live(
            _dashboard(snapshot, history=history),
            console=live_console or console,
            refresh_per_second=4,
        ) as live:
            while not stop.is_set():
                snapshot = orch.status_snapshot()
                history.record(snapshot)
                live.update(_dashboard(snapshot, history=history))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                except TimeoutError:
                    pass

    return task


def _print_transition(name: str, old, new) -> None:
    """Headless boot/runtime progress: one line per state change (for daemon logs)."""
    color = _STATE_COLORS.get(str(new), "white")
    console.print(f"  [dim]{name}[/dim]: {old} → [{color}]{new}[/{color}]")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Merge a .env file into os.environ (shell-set vars take precedence)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _boot_and_serve(file: Path, *, dashboard: bool) -> None:
    """Load a variant, boot the stack, and run until interrupted."""
    _load_dotenv()
    try:
        config = load_config(file)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Booting stack from {file}[/green]")
    # Foreground: the live dashboard shows transitions. Headless: print them instead.
    extra_tasks = [_dashboard_task_factory()] if dashboard else []
    on_transition = None if dashboard else _print_transition
    try:
        asyncio.run(
            serve_forever(
                config,
                variant_file=file,
                extra_tasks=extra_tasks,
                on_transition=on_transition,
            )
        )
    except BootError as exc:
        console.print(f"[red]Boot failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    console.print("[green]Stack stopped.[/green]")


@app.command()
def serve(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        _DEFAULT_CONFIG, "-f", "--file", help="Stack variant file to serve."
    ),
) -> None:
    """Run Sovereign as a foreground process (launchd entry point) — always headless."""
    _boot_and_serve(file, dashboard=False)


@app.command()
def up(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        _DEFAULT_CONFIG,
        "-f",
        "--file",
        help="Stack variant file to bring up.",
    ),
) -> None:
    """Boot the stack; show the live dashboard when run in a terminal."""
    _boot_and_serve(file, dashboard=_stdout_is_tty())


@app.command()
def down(
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Stop a running stack via its recorded runtime handles (reverse order)."""
    state_path = state_dir / "state.json"
    if not state_path.exists():
        console.print("[yellow]No recorded stack to stop (no state.json).[/yellow]")
        return

    state = read_json(state_path)
    handles: dict = state.get("runtime", {})
    if not handles:
        console.print("[yellow]Nothing running to stop.[/yellow]")
        return

    for name in reversed(list(handles)):
        result = stop_service_handle(handles[name])
        console.print(f"  {name}: {result}")

    # Reflect the teardown in state.json so `status` agrees.
    state["services"] = {name: "stopped" for name in state.get("services", {})}
    state["runtime"] = {}
    write_json(state_path, state)
    console.print("[green]Stack stopped.[/green]")


@app.command()
def status(
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Report current stack state, and flag drift from the recorded variant."""
    state_path = state_dir / "state.json"
    if not state_path.exists():
        console.print("No running stack (no state.json). Run [bold]sovereign up[/bold].")
        raise typer.Exit(0)

    state = read_json(state_path)

    table = Table(title="Sovereign stack")
    table.add_column("SERVICE")
    table.add_column("STATE")
    for name, service_state in state.get("services", {}).items():
        color = _STATE_COLORS.get(service_state, "white")
        table.add_row(name, f"[{color}]{service_state}[/{color}]")
    console.print(table)

    _report_drift(state)


def _report_drift(state: dict) -> None:
    variant_file = state.get("variant_file")
    recorded_hash = state.get("variant_hash")
    if not variant_file or not recorded_hash:
        return
    path = Path(variant_file)
    if not path.exists():
        console.print(
            f"[yellow]⚠ variant file {variant_file} no longer exists.[/yellow]", soft_wrap=True
        )
    elif file_hash(path) != recorded_hash:
        console.print(
            f"[yellow]⚠ drift: {variant_file} changed since boot — "
            f"run `sovereign up -f {variant_file}` to apply.[/yellow]",
            soft_wrap=True,
        )
    else:
        console.print(f"[green]✓ in sync with {variant_file}[/green]", soft_wrap=True)


@app.command()
def logs(
    service: str = typer.Argument(..., help="Service whose logs to show."),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of lines to show."),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow the log output."),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Show a service's logs (native log file, or `docker logs` for containers)."""
    log_path = state_dir / "logs" / f"{service}.log"
    if log_path.exists():
        if follow:
            os.execvp("tail", ["tail", "-n", str(lines), "-f", str(log_path)])
        for line in log_path.read_text(errors="replace").splitlines()[-lines:]:
            typer.echo(line)
        return

    state_path = state_dir / "state.json"
    handle = read_json(state_path).get("runtime", {}).get(service) if state_path.exists() else None
    if handle and handle.get("kind") == "docker":
        cmd = ["docker", "logs", "--tail", str(lines)]
        if follow:
            cmd.append("-f")
        cmd.append(handle["container"])
        subprocess.run(cmd)  # noqa: S603 - fixed argv
        return

    console.print(f"[yellow]No logs found for '{service}'.[/yellow]")
    raise typer.Exit(1)


_STATUS_LABEL = {"ready": "RUNNING"}


def _status_label(state: str) -> str:
    return _STATUS_LABEL.get(state, state.upper())


def _load_dashboard_status(state_dir: Path) -> dict | None:
    """Prefer the live status.json; fall back to state.json (states only)."""
    status_path = state_dir / "status.json"
    if status_path.exists():
        return read_json(status_path)
    state_path = state_dir / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        return {
            "services": {
                name: {"state": svc_state, "metrics": {}}
                for name, svc_state in state.get("services", {}).items()
            }
        }
    return None


def _format_duration(seconds: float) -> str:
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


def _duration_cell(since: str | None) -> str:
    """Elapsed time since an ISO timestamp, or "-" when unknown."""
    if not since:
        return "-"
    try:
        started = datetime.fromisoformat(since)
    except (TypeError, ValueError):
        return "-"
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return _format_duration(max(0.0, elapsed))


_HISTORY_SECONDS = 60.0  # trailing window kept per service per metric; tune here


class MetricHistory:
    """Rolling ~60s-window per-service, per-metric history for sparklines.

    Constructed once per dashboard session (once in monitor(), once per
    _dashboard_task_factory() task invocation) — never a module-level global,
    so state never leaks across unrelated sessions or test invocations. Never
    exposed as a user-facing parameter; always defaults to _HISTORY_SECONDS.
    """

    def __init__(self, window_seconds: float = _HISTORY_SECONDS) -> None:
        self._window = window_seconds
        self._data: dict[str, dict[str, deque[tuple[float, float]]]] = {}

    def record(self, status: dict) -> None:
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


def _sparkline(values: Sequence[float]) -> str:
    """A trailing Unicode-block sparkline, min-max scaled to the buffer's own range."""
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


_SPINNER_STATES = {"provisioning", "starting"}


def _status_cell(state: str) -> str | Spinner:
    """Plain colored label for steady states; an animated spinner while coming online."""
    color = _STATE_COLORS.get(state, "white")
    markup = f"[{color}]{_status_label(state)}[/{color}]"
    if state in _SPINNER_STATES:
        return Spinner("dots", text=Text.from_markup(markup))
    return markup


def _dashboard(status: dict, history: MetricHistory | None = None):
    """Render the §8 dashboard table, plus a live "Provisioning" activity area."""
    table = Table(title=f"Sovereign Control Plane v{__version__}", title_justify="left")
    table.add_column("SERVICE")
    table.add_column("STATUS")
    table.add_column("DURATION")
    table.add_column("CPU %")
    table.add_column("MEM (MB)")
    table.add_column("ENDPOINT")

    activity_lines: list[str] = []
    for name, svc in status.get("services", {}).items():
        state = svc.get("state", "unknown")
        metrics = svc.get("metrics") or {}
        cpu = f"{metrics['cpu_percent']:.1f}%" if "cpu_percent" in metrics else "-"
        mem = f"{metrics['memory_mb']:.0f}" if "memory_mb" in metrics else "-"
        cpu_spark = _sparkline(history.values(name, "cpu_percent")) if history else ""
        mem_spark = _sparkline(history.values(name, "memory_mb")) if history else ""
        duration = _duration_cell(svc.get("since"))
        endpoint = svc.get("endpoint") or "-"
        table.add_row(
            name,
            _status_cell(state),
            duration,
            _metric_cell(cpu, cpu_spark),
            _metric_cell(mem, mem_spark),
            endpoint,
        )

        activity = (svc.get("activity") or "").strip()
        if activity:
            activity_lines.append(f"  {name}  [{_status_label(state)}] {activity}")

    if not activity_lines:
        return table
    body = Text("\n".join(activity_lines), style="dim")
    return Group(table, Text("Activity:", style="bold"), body)


@app.command()
def monitor(
    interval: float = typer.Option(2.0, "--interval", help="Refresh interval (seconds)."),
    once: bool = typer.Option(False, "--once", help="Render a single frame and exit."),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Live, top-style view of running services (Ctrl+C to exit)."""
    status = _load_dashboard_status(state_dir)
    if status is None:
        console.print("No running stack (no status.json). Run [bold]sovereign up[/bold].")
        raise typer.Exit(0)

    history = MetricHistory()
    history.record(status)

    if once:
        console.print(_dashboard(status, history=history))
        return

    with Live(_dashboard(status, history=history), console=console, refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(interval)
                status = _load_dashboard_status(state_dir) or status
                history.record(status)
                live.update(_dashboard(status, history=history))
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass


@app.command()
def bench() -> None:
    """Run benchmark sweeps against the resolved stack."""
    _not_implemented("bench")


@app.command()
def version() -> None:
    """Print the Sovereign version."""
    console.print(f"Sovereign {__version__}")


if __name__ == "__main__":
    app()
