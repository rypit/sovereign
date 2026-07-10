"""Shared constants, helpers, and the root Typer app used across CLI modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console

from sovereign.config import ConfigError, SovereignConfig, load_config
from sovereign.runtime.dashboard import STATE_COLORS

if TYPE_CHECKING:
    from sovereign.core.base_harness import Harness

# ---------------------------------------------------------------------------
# Root app — sub-apps are mounted in cli/__init__.py; commands are registered
# in cli/stack.py (root-level) and cli/harness.py / cli/models.py /
# bench/cli.py (sub-apps).
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="sovereign",
    help="A declarative control plane for local LLM infrastructure on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path("sovereign.yaml")
_DEFAULT_STATE_DIR = Path(".sovereign")
_DEFAULT_BENCH_SPEC = Path("bench.yaml")

_STATE_DIR_OPTION = typer.Option(
    _DEFAULT_STATE_DIR, "--state-dir", help="Where Sovereign keeps state/logs."
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _config_path_for_harness_cli(file: Path | None, state_dir: Path) -> Path:
    """The stack file to read harnesses from: explicit ``-f``, else the recorded
    variant of a running stack, else the default ``sovereign.yaml``."""
    from sovereign.core.state import read_json

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


def _load_harness(name: str, file: Optional[Path], state_dir: Path) -> Harness:  # noqa: UP045
    """Build and resolve one harness instance against the live stack's manifest."""
    from sovereign.core.provisioning import ProvisioningError
    from sovereign.core.resolver import ResolvedEndpoint, Resolver, ServiceRegistry
    from sovereign.core.state import read_json

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

    missing = [dep for dep in entry.dependencies if dep not in registry]
    if missing:
        console.print(f"[red]Dependencies not ready: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    harness = harness_cls(entry)
    harness.resolve(Resolver(registry))
    # Same provisioning the Orchestrator runs, so one-shot CLI use also installs
    # whatever the harness needs.
    try:
        harness.prepare_environment()
    except (ProvisioningError, FileNotFoundError, ImportError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    return harness
