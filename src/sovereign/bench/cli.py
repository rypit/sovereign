"""``sovereign bench`` sub-app: run sweeps, list runs, compare results.

Lives in bench/ so the bench subsystem is fully self-contained; the root cli/
package mounts this sub-app via ``app.add_typer(bench_app, name="bench")``.

Bench commands reach the Orchestrator only through bench/cleanroom.py and
other bench modules — they do not import sovereign.runtime.orchestrator
directly (that boundary is cleanroom.py's job, per CLAUDE.md §6b).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from sovereign.bench.cleanroom import make_cleanroom_executor
from sovereign.bench.lock import BenchLockError
from sovereign.bench.perf import make_perf_attach_executor
from sovereign.bench.quality import make_quality_executor
from sovereign.bench.report import build_comparison, flag_pareto
from sovereign.bench.runner import combine_executors, run_bench
from sovereign.bench.spec import BenchMode, BenchSpecError, load_bench_spec
from sovereign.cli._common import _STATE_DIR_OPTION, console
from sovereign.state import read_json

bench_app = typer.Typer(help="Benchmark engine x model x harness combinations.")

_DEFAULT_BENCH_SPEC = Path("bench.yaml")

_BENCH_STATE_COLORS = {
    "completed": "green",
    "failed": "red",
    "running": "cyan",
    "pending": "white",
}


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
    run_ids: list[str] | None = typer.Argument(  # noqa: UP045 - Typer needs Optional at runtime
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
