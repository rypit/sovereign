"""Sovereign CLI package.

Thin wiring layer:
- Creates the root ``app`` (in ``_common``) and the ``--verbose`` callback.
- Mounts the ``harness``, ``bench``, and ``models`` sub-apps.
- Imports ``stack`` for its @app.command() registration side-effects.

Entry point: ``sovereign.cli:app`` (pyproject.toml [project.scripts]).
"""

from __future__ import annotations

import typer

from sovereign.bench.cli import bench_app
from sovereign.cli import stack as _stack  # noqa: F401 - side-effect: registers @app.command()s
from sovereign.cli._common import app, console  # noqa: F401 - re-exported for tests
from sovereign.cli.harness import harness_app
from sovereign.cli.logging_config import configure_logging
from sovereign.cli.models import models_app

app.add_typer(harness_app, name="harness")
app.add_typer(bench_app, name="bench")
app.add_typer(models_app, name="models")


@app.callback()
def _configure(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Debug logging (state transitions, admission, HF cache)."
    ),
) -> None:
    configure_logging(verbose)
