"""logging_console: routing the sovereign logger through the Live dashboard's
console so a boot-time warning prints above the live region instead of tearing
stale frame copies into the terminal (the "title duplicated N times" bug)."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from sovereign.cli.logging_config import configure_logging, logging_console


def _handler() -> RichHandler:
    (handler,) = logging.getLogger("sovereign").handlers
    assert isinstance(handler, RichHandler)
    return handler


def test_logging_console_routes_records_to_given_console() -> None:
    configure_logging()
    recording = Console(record=True, width=120)
    with logging_console(recording):
        logging.getLogger("sovereign.runtime.orchestrator").warning(
            "'searxng' admitted with UNKNOWN memory footprint"
        )
    assert "UNKNOWN memory footprint" in recording.export_text()


def test_logging_console_restores_original_console_after() -> None:
    configure_logging()
    original = _handler().console
    with logging_console(Console(record=True)):
        assert _handler().console is not original
    assert _handler().console is original


def test_logging_console_restores_on_exception() -> None:
    configure_logging()
    original = _handler().console
    try:
        with logging_console(Console(record=True)):
            raise RuntimeError("boot blew up")
    except RuntimeError:
        pass
    assert _handler().console is original
