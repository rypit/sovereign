"""Logging setup for the ``sovereign`` logger hierarchy.

The CLI's user-facing output stays Rich ``console.print`` — logging is the
*diagnostic* channel: state transitions, admission decisions, HF cache hits,
provisioning commands, swallowed best-effort errors. Quiet by default
(WARNING); ``sovereign --verbose <command>`` turns on DEBUG. Handlers go to
stderr so piped stdout (e.g. ``sovereign bench compare --json``) stays clean.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(verbose: bool = False) -> None:
    """Configure the ``sovereign`` logger (idempotent; safe to call per command).

    Only the package's own hierarchy is configured — third-party debug noise
    (huggingface_hub, urllib3) stays at whatever the root logger allows.
    """
    handler = RichHandler(
        console=Console(stderr=True),
        show_time=verbose,
        show_path=verbose,
        rich_tracebacks=False,
    )
    logger = logging.getLogger("sovereign")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.handlers[:] = [handler]
    logger.propagate = False


@contextmanager
def logging_console(console: Console) -> Iterator[None]:
    """Temporarily route the ``sovereign`` logger's Rich handler through *console*.

    While a ``Live`` dashboard owns the terminal, a record written to the
    default stderr Console scrolls the screen behind ``Live``'s back — the next
    repaint lands lower and strands a stale copy of the frame above the region
    (one duplicate per record). Printing through the console that hosts the
    ``Live`` lets Rich interleave records cleanly above the live region, so
    ``up`` wraps its dashboard session in this.
    """
    logger = logging.getLogger("sovereign")
    swapped = [h for h in logger.handlers if isinstance(h, RichHandler)]
    saved = [(h, h.console) for h in swapped]
    for h in swapped:
        h.console = console
    try:
        yield
    finally:
        for h, original in saved:
            h.console = original
