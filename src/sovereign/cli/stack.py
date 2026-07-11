"""Stack lifecycle commands registered on the root Typer app.

Commands: up, serve, down, status, logs, monitor, plan, provision, version.
These remain on the root app (``sovereign up``, not ``sovereign stack up``).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.live import Live
from rich.table import Table

from sovereign.cli._common import (
    _DEFAULT_CONFIG,
    _DEFAULT_STATE_DIR,
    _STATE_DIR_OPTION,
    _fast_exit,
    _load_config_or_exit,
    _load_dotenv,
    _print_transition,
    _stdout_is_tty,
    app,
    console,
)
from sovereign.core.base_manager import SupportsProvisioning
from sovereign.core.state import file_hash, mark_stack_stopped, read_json
from sovereign.runtime.dashboard import (
    MetricHistory,
    budget_footer,
    dashboard,
    dashboard_task_factory,
    load_dashboard_status,
)
from sovereign.runtime.orchestrator import BootError, serve_forever
from sovereign.runtime.teardown import stop_service_handle

_VERDICT_COLORS = {"OK": "green", "REFUSED": "red", "ROUTING ERROR": "red", "CONFIG ERROR": "red"}


def _read_state_or_exit(state_path: Path) -> dict:
    """Read ``state.json`` for a one-shot command (``status``/``down``/``logs``).

    Unlike the pollers (``monitor``, the dashboard), a one-shot command has no
    "last known good" snapshot to fall back to — a decode error (e.g. a state
    file from a version predating atomic writes, corrupted by a torn write)
    should print an actionable message and exit, not a raw traceback.
    """
    try:
        return read_json(state_path)
    except json.JSONDecodeError as exc:
        console.print(
            f"[red]{state_path} is corrupt ({exc}).[/red] If a stack is still running, "
            "stop it manually and remove the file; otherwise delete it and run "
            "[bold]sovereign up[/bold] again."
        )
        raise typer.Exit(1) from exc


def _boot_and_serve(file: Path, *, with_dashboard: bool) -> None:
    """Load a variant, boot the stack, and run until interrupted."""
    from sovereign.bench.lock import lock_path

    if lock_path(_DEFAULT_STATE_DIR).exists():
        console.print(
            f"[red]A clean-room bench run holds {lock_path(_DEFAULT_STATE_DIR)} — "
            "wait for it to finish before starting a daemon-managed stack here.[/red]"
        )
        raise typer.Exit(1)
    _load_dotenv()
    config = _load_config_or_exit(file)

    console.print(f"[green]Booting stack from {file}[/green]")
    # Foreground: the live dashboard shows transitions. Headless: print them instead.
    extra_tasks = [dashboard_task_factory(live_console=console)] if with_dashboard else []
    on_transition = None if with_dashboard else _print_transition
    # Drive the loop with an explicit Runner so we can skip its teardown: a
    # Ctrl+C mid-download leaves an un-cancellable huggingface_hub worker thread,
    # and asyncio.run()/Runner.close() would join it for minutes. serve_forever
    # has already stopped services and torn down the dashboard by the time it
    # returns, so _fast_exit terminates cleanly and lets the OS reap the thread.
    runner = asyncio.Runner()
    code = 0
    try:
        runner.run(
            serve_forever(
                config,
                variant_file=file,
                extra_tasks=extra_tasks,
                on_transition=on_transition,
            )
        )
        console.print("[green]Stack stopped.[/green]")
    except BootError as exc:
        console.print(f"[red]Boot failed:[/red] {exc}")
        code = 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    _fast_exit(code)


@app.command()
def serve(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        _DEFAULT_CONFIG, "-f", "--file", help="Stack variant file to serve."
    ),
) -> None:
    """Run Sovereign as a foreground process (launchd entry point) — always headless."""
    _boot_and_serve(file or _DEFAULT_CONFIG, with_dashboard=False)


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
    _boot_and_serve(file or _DEFAULT_CONFIG, with_dashboard=_stdout_is_tty())


@app.command()
def down(
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Stop a running stack via its recorded runtime handles (reverse order)."""
    state_path = state_dir / "state.json"
    if not state_path.exists():
        console.print("[yellow]No recorded stack to stop (no state.json).[/yellow]")
        return

    state = _read_state_or_exit(state_path)
    handles: dict = state.get("runtime", {})
    if not handles:
        console.print("[yellow]Nothing running to stop.[/yellow]")
        return

    for name in reversed(list(handles)):
        result = stop_service_handle(handles[name])
        console.print(f"  {name}: {result}")

    # Reflect the teardown in state.json so `status` agrees.
    mark_stack_stopped(state_path)
    console.print("[green]Stack stopped.[/green]")


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
def status(
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Report current stack state, and flag drift from the recorded variant."""
    from sovereign.runtime.dashboard import STATE_COLORS

    state_path = state_dir / "state.json"
    if not state_path.exists():
        console.print("No running stack (no state.json). Run [bold]sovereign up[/bold].")
        raise typer.Exit(0)

    state = _read_state_or_exit(state_path)

    table = Table(title="Sovereign stack")
    table.add_column("SERVICE")
    table.add_column("STATE")
    for name, service_state in state.get("services", {}).items():
        color = STATE_COLORS.get(service_state, "white")
        table.add_row(name, f"[{color}]{service_state}[/{color}]")
    console.print(table)

    _report_drift(state)


@app.command()
def logs(
    service: str = typer.Argument(..., help="Service whose logs to show."),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of lines to show."),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow the log output."),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Show a service's logs (native log file, or `docker logs` for containers)."""
    import os

    log_path = state_dir / "logs" / f"{service}.log"
    if log_path.exists():
        if follow:
            os.execvp("tail", ["tail", "-n", str(lines), "-f", str(log_path)])
        for line in log_path.read_text(errors="replace").splitlines()[-lines:]:
            typer.echo(line)
        return

    state_path = state_dir / "state.json"
    state = _read_state_or_exit(state_path) if state_path.exists() else {}
    handle = state.get("runtime", {}).get(service)
    if handle and handle.get("kind") == "docker":
        cmd = ["docker", "logs", "--tail", str(lines)]
        if follow:
            cmd.append("-f")
        cmd.append(handle["container"])
        subprocess.run(cmd)  # noqa: S603 - fixed argv
        return

    console.print(f"[yellow]No logs found for '{service}'.[/yellow]")
    raise typer.Exit(1)


@app.command()
def monitor(
    interval: float = typer.Option(2.0, "--interval", help="Refresh interval (seconds)."),
    once: bool = typer.Option(False, "--once", help="Render a single frame and exit."),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Live, top-style view of running services (Ctrl+C to exit)."""
    status = load_dashboard_status(state_dir)
    if status is None:
        console.print("No running stack (no status.json). Run [bold]sovereign up[/bold].")
        raise typer.Exit(0)

    history = MetricHistory()
    history.record(status)

    if once:
        console.print(dashboard(status, history=history))
        return

    with Live(dashboard(status, history=history), console=console, refresh_per_second=12) as live:
        try:
            while True:
                time.sleep(interval)
                status = load_dashboard_status(state_dir) or status
                history.record(status)
                live.update(dashboard(status, history=history))
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass


@app.command()
def plan(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        _DEFAULT_CONFIG, "-f", "--file", help="Stack variant file to plan."
    ),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Dry-run a stack: route models, estimate memory, and check the budget. No downloads."""
    from sovereign.core.planning import plan_stack

    _load_dotenv()
    config = _load_config_or_exit(file or _DEFAULT_CONFIG)

    stack_plan = plan_stack(config, state_dir)

    table = Table(title=f"Plan for {file}")
    for col in ("SERVICE", "BASE_TYPE", "MODEL", "SOURCE", "EST GB", "VERDICT"):
        table.add_column(col)
    for svc in stack_plan.services:
        routed = svc.requested_auto and svc.base_type != "auto"
        base_type = f"{svc.base_type} (auto)" if routed else svc.base_type
        color = _VERDICT_COLORS.get(svc.verdict, "white")
        est = f"{svc.estimated_gb:.1f}" if svc.estimated_gb is not None else "-"
        # Fail-open admission is deliberate but must be visible: an unknown
        # estimate is admitted at 0 GB, i.e. outside the budget's protection.
        source = f"[yellow]{svc.source}[/yellow]" if svc.source == "unknown" else svc.source
        table.add_row(
            svc.name, base_type, svc.model, source, est, f"[{color}]{svc.verdict}[/{color}]"
        )
    console.print(table)

    for svc in stack_plan.services:
        if svc.source == "unknown" and svc.verdict == "OK":
            console.print(
                f"  [yellow]⚠ {svc.name}: admitted with UNKNOWN memory footprint — "
                "not counted against the budget.[/yellow]",
                soft_wrap=True,
            )
    for svc in stack_plan.services:
        if svc.error:
            console.print(f"  [dim]{svc.name}: {svc.error}[/dim]", soft_wrap=True)

    budget = stack_plan.budget
    footer = budget_footer(
        {
            "usable_gb": budget.usable_gb,
            "reserved_gb": round(budget.reserved_gb, 2),
            "available_gb": round(budget.available_gb, 2),
        }
    )
    if footer is not None:
        console.print(footer)
    if not stack_plan.ok:
        raise typer.Exit(1)


def _provision_targets(file: Path | None) -> dict[str, type]:
    """base_type -> class to provision: a stack file's declared types, or everything."""
    from sovereign.core.registry import (
        all_harnesses,
        all_service_managers,
        populate_registries,
        route_entry,
    )
    from sovereign.core.resolver import ConsumerKind

    populate_registries()
    registered: dict[str, type] = {**all_service_managers(), **all_harnesses()}
    if file is None:
        return registered

    config = _load_config_or_exit(file)

    from sovereign.core.errors import ModelResolutionError

    # The native engines to provision when an ``auto`` entry can't be routed offline —
    # derived from the registry, so a new engine joins the fallback automatically.
    native_engine_types = {
        bt
        for bt, cls in all_service_managers().items()
        if getattr(cls, "consumer_kind", None) == ConsumerKind.NATIVE
    }

    declared: set[str] = {h.base_type for h in config.harnesses}
    for svc in config.services:
        if svc.base_type != "auto":
            declared.add(svc.base_type)
            continue
        # Route auto entries to their concrete engine; if that can't be determined
        # (offline, no cache), provision the safe superset of every native engine.
        try:
            declared.add(route_entry(svc, _DEFAULT_STATE_DIR))
        except ModelResolutionError:
            declared.update(native_engine_types)
            console.print(
                f"  [dim]{svc.name}: base_type 'auto' unresolved offline — "
                f"provisioning all native engines ({', '.join(sorted(native_engine_types))}).[/dim]"
            )
    unknown = declared - registered.keys()
    if unknown:
        console.print(f"[red]Unknown base_type(s) in {file}: {', '.join(sorted(unknown))}[/red]")
        raise typer.Exit(1)
    return {bt: cls for bt, cls in registered.items() if bt in declared}


@app.command()
def provision(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        None,
        "-f",
        "--file",
        help="Provision only the integrations this stack file declares (default: all).",
    ),
) -> None:
    """Install every declared integration's dependencies (Brewfiles + install commands)."""
    from sovereign.core.provisioning import ProvisioningError

    failed = False
    for base_type, cls in sorted(_provision_targets(file).items()):
        if not isinstance(cls, type) or not issubclass(cls, SupportsProvisioning):
            continue  # integration predates the Provisioner mixin — nothing declared
        if cls.provisioning_satisfied():
            console.print(f"  [green]✓[/green] {base_type} — satisfied")
            continue
        console.print(f"  [cyan]→[/cyan] {base_type} — installing…")
        try:
            cls.provision()
            console.print(f"  [green]✓[/green] {base_type} — installed")
        except (ProvisioningError, FileNotFoundError, ImportError) as exc:
            console.print(f"  [red]✗ {base_type}: {exc}[/red]")
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print the Sovereign version."""
    from sovereign import __version__

    console.print(f"Sovereign {__version__}")
