"""``sovereign harness`` sub-app: list, materialize, invoke."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from sovereign.cli._common import (
    _STATE_DIR_OPTION,
    _load_harness,
    _load_harness_config,
    console,
)

harness_app = typer.Typer(help="Inspect and invoke configured harnesses.")


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
    from sovereign.core.base_harness import SupportsInvoke, Task

    harness = _load_harness(name, file, state_dir)
    if not isinstance(harness, SupportsInvoke):
        console.print(f"[red]Harness '{name}' does not support invoke.[/red]")
        raise typer.Exit(1)
    harness.materialize()
    task = Task(id=f"{name}-cli", prompt=prompt, workdir=str(workdir) if workdir else None)
    result = harness.invoke(task)
    color = "green" if result.success else "red"
    console.print(f"[{color}]success={result.success} exit_code={result.exit_code}[/{color}]")
    if result.output:
        console.print(result.output)
    if not result.success:
        raise typer.Exit(1)
