"""``sovereign models`` sub-app: list and prune the HuggingFace model cache."""

from __future__ import annotations

from datetime import UTC, datetime

import typer

from sovereign.cli._common import console
from sovereign.core.units import fmt_size

models_app = typer.Typer(help="Inspect and prune the shared HuggingFace model cache.")


@models_app.callback(invoke_without_command=True)
def models_main(ctx: typer.Context) -> None:
    """List the shared HuggingFace model cache (default) or prune a repo."""
    if ctx.invoked_subcommand is None:
        models_list()


@models_app.command("list")
def models_list() -> None:
    """List cached HuggingFace repos by size (REPO / SIZE / NFILES / LAST_ACCESSED)."""
    from huggingface_hub import scan_cache_dir
    from rich.table import Table

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
        table.add_row(repo.repo_id, fmt_size(repo.size_on_disk), str(repo.nb_files), last)
    console.print(table)
    console.print(f"[bold]Total: {fmt_size(cache.size_on_disk)}[/bold]")


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

    console.print(f"{repo}: {fmt_size(match.size_on_disk)} across {match.nb_files} files")
    if not typer.confirm("Delete all cached revisions?"):
        raise typer.Exit(0)
    strategy = cache.delete_revisions(*[rev.commit_hash for rev in match.revisions])
    strategy.execute()
    console.print(f"[green]Freed {fmt_size(strategy.expected_freed_size)}.[/green]")
