"""Typer entry point for the Sovereign CLI.

Thin command layer: parsing, table rendering, and exit codes live here; the real
logic lives in the orchestrator (`up`/`serve`/`down`), :mod:`sovereign.core.planning`
(`plan`), :mod:`sovereign.dashboard` (`up`/`monitor` rendering), and the bench
package (`bench`).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from sovereign import __version__
from sovereign.bench.cleanroom import make_cleanroom_executor
from sovereign.bench.lock import BenchLockError, lock_path
from sovereign.bench.perf import make_perf_attach_executor
from sovereign.bench.quality import make_quality_executor
from sovereign.bench.report import build_comparison, flag_pareto
from sovereign.bench.runner import combine_executors, run_bench
from sovereign.bench.spec import BenchMode, BenchSpecError, load_bench_spec
from sovereign.config import ConfigError, SovereignConfig, load_config
from sovereign.core.base_harness import Task
from sovereign.core.provisioning import ProvisioningError
from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.dashboard import (
    STATE_COLORS,
    MetricHistory,
    budget_footer,
    dashboard,
    dashboard_task_factory,
    load_dashboard_status,
)
from sovereign.logging_config import configure_logging
from sovereign.orchestrator import BootError, serve_forever
from sovereign.utils.state import file_hash, mark_stack_stopped, read_json
from sovereign.utils.teardown import stop_service_handle

app = typer.Typer(
    name="sovereign",
    help="A declarative control plane for local LLM infrastructure on Apple Silicon.",
    no_args_is_help=True,
)
harness_app = typer.Typer(help="Inspect and invoke configured harnesses.")
app.add_typer(harness_app, name="harness")
bench_app = typer.Typer(help="Benchmark engine x model x harness combinations.")
app.add_typer(bench_app, name="bench")
models_app = typer.Typer(help="Inspect and prune the shared HuggingFace model cache.")
app.add_typer(models_app, name="models")
console = Console()


@app.callback()
def _configure(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Debug logging (state transitions, admission, HF cache)."
    ),
) -> None:
    configure_logging(verbose)

_DEFAULT_BENCH_SPEC = Path("bench.yaml")
_BENCH_STATE_COLORS = {
    "completed": "green",
    "failed": "red",
    "running": "cyan",
    "pending": "white",
}

_DEFAULT_CONFIG = Path("sovereign.yaml")
_DEFAULT_STATE_DIR = Path(".sovereign")

_STATE_DIR_OPTION = typer.Option(
    _DEFAULT_STATE_DIR, "--state-dir", help="Where Sovereign keeps state/logs."
)

def _stdout_is_tty() -> bool:
    """Whether stdout is an interactive terminal (vs. a pipe or launchd log)."""
    return sys.stdout.isatty()


def _print_transition(name: str, old, new) -> None:
    """Headless boot/runtime progress: one line per state change (for daemon logs)."""
    color = STATE_COLORS.get(str(new), "white")
    console.print(f"  [dim]{name}[/dim]: {old} → [{color}]{new}[/{color}]")


def _load_config_or_exit(file: Path) -> SovereignConfig:
    """Load a stack file, printing the ConfigError and exiting 1 on failure."""
    try:
        return load_config(file)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


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


def _fast_exit(code: int) -> None:  # pragma: no cover - interactive
    """Terminate now, skipping asyncio.Runner/executor teardown.

    A Ctrl+C mid-download leaves huggingface_hub worker threads blocked in
    un-cancellable network I/O; the normal loop/atexit teardown would *join* them
    for minutes. os._exit skips those joins — the OS reaps the threads, and HF's
    partial downloads are resumable, so nothing corrupts.
    """
    console.file.flush()
    os._exit(code)


def _boot_and_serve(file: Path, *, with_dashboard: bool) -> None:
    """Load a variant, boot the stack, and run until interrupted."""
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

    state = read_json(state_path)
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
        color = STATE_COLORS.get(service_state, "white")
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


def _config_path_for_harness_cli(file: Path | None, state_dir: Path) -> Path:
    """The stack file to read harnesses from: explicit ``-f``, else the recorded
    variant of a running stack, else the default ``sovereign.yaml``."""
    if file is not None:
        return file
    state_path = state_dir / "state.json"
    if state_path.exists():
        variant = read_json(state_path).get("variant_file")
        if variant:
            return Path(variant)
    return _DEFAULT_CONFIG


def _load_harness_config(file: Path | None, state_dir: Path) -> SovereignConfig:
    return _load_config_or_exit(_config_path_for_harness_cli(file, state_dir))


def _registry_from_manifest(manifest: dict) -> ServiceRegistry:
    """Rebuild a `ServiceRegistry` from a persisted manifest.json (a separate
    CLI invocation has no live Orchestrator to read endpoints from)."""
    registry = ServiceRegistry()
    for svc in manifest.get("services", []):
        endpoint = svc.get("endpoint")
        if not endpoint:
            continue
        registry.register(
            svc["name"],
            ResolvedEndpoint(
                scheme=endpoint["scheme"],
                host=endpoint["host"],
                port=endpoint["port"],
                model=endpoint.get("model"),
            ),
        )
    return registry


def _load_harness(name: str, file: Optional[Path], state_dir: Path):  # noqa: UP045
    """Build and resolve one harness instance against the live stack's manifest."""
    manifest_path = state_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(
            "[red]No running stack found (no manifest.json). Run `sovereign up` first.[/red]"
        )
        raise typer.Exit(1)
    manifest = read_json(manifest_path)

    config = _load_harness_config(file, state_dir)
    entry = next((h for h in config.harnesses if h.name == name), None)
    if entry is None:
        known = ", ".join(h.name for h in config.harnesses) or "(none configured)"
        console.print(f"[red]Unknown harness '{name}'; known: {known}[/red]")
        raise typer.Exit(1)

    from sovereign.core.registry import get_harness, populate_registries

    populate_registries()
    try:
        harness_cls = get_harness(entry.base_type)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    registry = _registry_from_manifest(manifest)
    missing = [dep for dep in entry.dependencies if dep not in registry]
    if missing:
        console.print(f"[red]Dependencies not ready: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    harness = harness_cls(entry)
    resolve = getattr(harness, "resolve", None)
    if callable(resolve):
        resolve(Resolver(registry))
    # Same provisioning the Orchestrator runs, so one-shot CLI use also installs
    # whatever the harness needs.
    prepare = getattr(harness, "prepare_environment", None)
    if callable(prepare):
        try:
            prepare()
        except (ProvisioningError, FileNotFoundError, ImportError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
    return harness


@harness_app.command("list")
def harness_list(
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        None, "-f", "--file", help="Stack file (defaults to the running stack's variant)."
    ),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """List configured harnesses and their dependencies."""
    config = _load_harness_config(file, state_dir)
    if not config.harnesses:
        console.print("[yellow]No harnesses configured.[/yellow]")
        return

    table = Table(title="Sovereign harnesses")
    table.add_column("NAME")
    table.add_column("BASE_TYPE")
    table.add_column("DEPENDENCIES")
    for h in config.harnesses:
        table.add_row(h.name, h.base_type, ", ".join(h.dependencies) or "-")
    console.print(table)


@harness_app.command("materialize")
def harness_materialize(
    name: str = typer.Argument(..., help="Harness instance name."),
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        None, "-f", "--file", help="Stack file (defaults to the running stack's variant)."
    ),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Re-run materialize() for a harness against the live stack."""
    harness = _load_harness(name, file, state_dir)
    harness.materialize()
    console.print(f"[green]Materialized '{name}'.[/green]")


@harness_app.command("invoke")
def harness_invoke(
    name: str = typer.Argument(..., help="Harness instance name."),
    prompt: str = typer.Option(..., "--prompt", help="Task prompt."),
    workdir: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        None, "--workdir", help="Working directory for the run."
    ),
    file: Optional[Path] = typer.Option(  # noqa: UP045 - Typer needs Optional at runtime
        None, "-f", "--file", help="Stack file (defaults to the running stack's variant)."
    ),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Run one headless session through a harness and print the result."""
    harness = _load_harness(name, file, state_dir)
    harness.materialize()
    task = Task(id=f"{name}-cli", prompt=prompt, workdir=str(workdir) if workdir else None)
    result = harness.invoke(task)
    color = "green" if result.success else "red"
    console.print(f"[{color}]success={result.success} exit_code={result.exit_code}[/{color}]")
    if result.output:
        console.print(result.output)
    if not result.success:
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
    failed = False
    for base_type, cls in sorted(_provision_targets(file).items()):
        provision_fn = getattr(cls, "provision", None)
        satisfied_fn = getattr(cls, "provisioning_satisfied", None)
        if not callable(provision_fn):
            continue  # integration predates the Provisioner mixin — nothing declared
        if callable(satisfied_fn) and satisfied_fn():
            console.print(f"  [green]✓[/green] {base_type} — satisfied")
            continue
        console.print(f"  [cyan]→[/cyan] {base_type} — installing…")
        try:
            provision_fn()
            console.print(f"  [green]✓[/green] {base_type} — installed")
        except (ProvisioningError, FileNotFoundError, ImportError) as exc:
            console.print(f"  [red]✗ {base_type}: {exc}[/red]")
            failed = True
    if failed:
        raise typer.Exit(1)


@bench_app.command("run")
def bench_run(
    file: Path = typer.Option(_DEFAULT_BENCH_SPEC, "-f", "--file", help="Bench spec file."),
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """Run (or resume) a bench sweep: enumerate cells, skip completed ones."""
    try:
        spec = load_bench_spec(file)
    except BenchSpecError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if spec.mode == BenchMode.CLEANROOM:
        # The clean-room executor dispatches perf vs. quality internally, since
        # both need the same boot/measure/teardown around a single Orchestrator.
        executor = make_cleanroom_executor(spec, state_dir)
    else:
        executor = combine_executors(
            make_perf_attach_executor(spec, state_dir), make_quality_executor(spec, state_dir)
        )

    try:
        manifest = run_bench(spec, state_dir=state_dir, executor=executor)
    except BenchLockError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title=f"Bench run {manifest['run_id']}")
    table.add_column("CELL")
    table.add_column("STATE")
    for cell in manifest["cells"]:
        color = _BENCH_STATE_COLORS.get(cell["state"], "white")
        table.add_row(cell["id"], f"[{color}]{cell['state']}[/{color}]")
    console.print(table)


@bench_app.command("ls")
def bench_ls(
    state_dir: Path = _STATE_DIR_OPTION,
) -> None:
    """List recorded bench runs: cell counts, completed/failed/gated/skipped."""
    runs_dir = state_dir / "benchmarks" / "runs"
    if not runs_dir.exists():
        console.print("[yellow]No benchmark runs recorded.[/yellow]")
        return

    table = Table(title="Sovereign benchmark runs")
    table.add_column("RUN_ID")
    table.add_column("CELLS")
    table.add_column("COMPLETED")
    table.add_column("SKIPPED")
    table.add_column("FAILED")
    table.add_column("GATED")
    for run_file in sorted(runs_dir.glob("*.json")):
        manifest = read_json(run_file)
        cells = manifest.get("cells", [])
        completed = sum(1 for c in cells if c["state"] == "completed")
        skipped = sum(1 for c in cells if c.get("skipped"))
        failed = sum(1 for c in cells if c["state"] == "failed")
        gated = sum(
            1 for c in cells if c["state"] == "failed" and "gated" in (c.get("error") or "")
        )
        table.add_row(
            manifest["run_id"],
            str(len(cells)),
            str(completed),
            str(skipped),
            str(failed),
            str(gated),
        )
    console.print(table)


@bench_app.command("compare")
def bench_compare(
    run_ids: Optional[list[str]] = typer.Argument(  # noqa: UP045 - Typer needs Optional at runtime
        None, help="Specific run IDs to compare (default: every recorded run)."
    ),
    state_dir: Path = _STATE_DIR_OPTION,
    as_json: bool = typer.Option(False, "--json", help="Print rows as JSON instead of a table."),
) -> None:
    """Join cell results across runs into a speed/quality Pareto comparison."""
    rows = build_comparison(state_dir, run_ids or None)
    if not rows:
        console.print("[yellow]No completed bench cells found.[/yellow]")
        return
    flag_pareto(rows)

    if as_json:
        console.print(json.dumps(rows, indent=2))
        return

    table = Table(title="Sovereign bench comparison")
    table.add_column("STACK")
    table.add_column("HARNESS")
    table.add_column("ENGINE")
    table.add_column("TOK/S")
    table.add_column("TTFT (ms)")
    table.add_column("PASS RATE")
    table.add_column("FALSE-COMPLETION")
    table.add_column("PARETO")
    for row in rows:
        pareto_cell = {True: "★", False: "-", None: "n/a"}[row["pareto"]]
        table.add_row(
            Path(row["stack"]).stem,
            row["harness"] or "-",
            row["engine"] or "-",
            f"{row['tok_s']:.1f}" if row["tok_s"] is not None else "-",
            f"{row['ttft_ms']:.0f}" if row["ttft_ms"] is not None else "-",
            f"{row['pass_rate']:.0%}" if row["pass_rate"] is not None else "-",
            f"{row['false_completion_rate']:.0%}"
            if row["false_completion_rate"] is not None
            else "-",
            pareto_cell,
        )
    console.print(table)


def _fmt_bytes(n: float) -> str:
    """Human-readable byte size (GB for anything model-sized)."""
    gb = n / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{n / (1024**2):.0f} MB"


@models_app.callback(invoke_without_command=True)
def models_main(ctx: typer.Context) -> None:
    """List the shared HuggingFace model cache (default) or prune a repo."""
    if ctx.invoked_subcommand is None:
        models_list()


@models_app.command("list")
def models_list() -> None:
    """List cached HuggingFace repos by size (REPO / SIZE / NFILES / LAST_ACCESSED)."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception as exc:  # noqa: BLE001 - missing/corrupt cache is not fatal
        console.print(f"[yellow]No HuggingFace cache to scan: {exc}[/yellow]")
        return
    repos = sorted(cache.repos, key=lambda r: r.size_on_disk, reverse=True)
    if not repos:
        console.print("[dim]HuggingFace cache is empty.[/dim]")
        return

    table = Table(title="HuggingFace model cache")
    table.add_column("REPO")
    table.add_column("SIZE")
    table.add_column("NFILES")
    table.add_column("LAST_ACCESSED")
    for repo in repos:
        last = datetime.fromtimestamp(repo.last_accessed, tz=UTC).strftime("%Y-%m-%d")
        table.add_row(repo.repo_id, _fmt_bytes(repo.size_on_disk), str(repo.nb_files), last)
    console.print(table)
    console.print(f"[bold]Total: {_fmt_bytes(cache.size_on_disk)}[/bold]")


@models_app.command("prune")
def models_prune(
    repo: str = typer.Argument(..., help="Repo id to delete from the cache (all revisions)."),
) -> None:
    """Delete every revision of a cached repo, freeing its disk space."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    match = next((r for r in cache.repos if r.repo_id == repo), None)
    if match is None:
        console.print(f"[red]No cached repo '{repo}'. Run `sovereign models list`.[/red]")
        raise typer.Exit(1)

    console.print(f"{repo}: {_fmt_bytes(match.size_on_disk)} across {match.nb_files} files")
    if not typer.confirm("Delete all cached revisions?"):
        raise typer.Exit(0)
    strategy = cache.delete_revisions(*[rev.commit_hash for rev in match.revisions])
    strategy.execute()
    console.print(f"[green]Freed {_fmt_bytes(strategy.expected_freed_size)}.[/green]")


_VERDICT_COLORS = {"OK": "green", "REFUSED": "red", "ROUTING ERROR": "red", "CONFIG ERROR": "red"}


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
        table.add_row(
            svc.name, base_type, svc.model, svc.source, est, f"[{color}]{svc.verdict}[/{color}]"
        )
    console.print(table)

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


@app.command()
def version() -> None:
    """Print the Sovereign version."""
    console.print(f"Sovereign {__version__}")


if __name__ == "__main__":
    app()
