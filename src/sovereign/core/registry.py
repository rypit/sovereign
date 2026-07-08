"""``base_type`` → class factory maps (§2.6, §3 ``core/registry.py``).

Instance identity (``name``) is separate from implementation (``base_type``): the
Orchestrator looks up a ``base_type`` here to find which Manager/Harness class to
instantiate, which is what lets two ``llama_cpp`` instances run side by side.

Concrete services/harnesses register themselves via the decorators below, invoked
as a side effect of importing their package. None are registered yet — they arrive
in Phases 3+ (services) and the harness track.
"""

from __future__ import annotations

from typing import TypeVar

from sovereign.core.base_harness import Harness
from sovereign.core.base_manager import ServiceManager

_SERVICE_MANAGERS: dict[str, type[ServiceManager]] = {}
_HARNESSES: dict[str, type[Harness]] = {}

M = TypeVar("M", bound=ServiceManager)
H = TypeVar("H", bound=Harness)


def register_service(base_type: str):
    """Class decorator registering a ``ServiceManager`` under ``base_type``."""

    def decorator(cls: type[M]) -> type[M]:
        if base_type in _SERVICE_MANAGERS:
            raise ValueError(
                f"service base_type {base_type!r} already registered by "
                f"{_SERVICE_MANAGERS[base_type].__name__}"
            )
        _SERVICE_MANAGERS[base_type] = cls
        return cls

    return decorator


def register_harness(base_type: str):
    """Class decorator registering a ``Harness`` under ``base_type``."""

    def decorator(cls: type[H]) -> type[H]:
        if base_type in _HARNESSES:
            raise ValueError(
                f"harness base_type {base_type!r} already registered by "
                f"{_HARNESSES[base_type].__name__}"
            )
        _HARNESSES[base_type] = cls
        return cls

    return decorator


def get_service_manager(base_type: str) -> type[ServiceManager]:
    """Look up a registered service manager class, or raise a clear error."""
    if base_type == "auto":
        raise KeyError(
            "base_type 'auto' must be resolved by routing before manager lookup"
        )
    try:
        return _SERVICE_MANAGERS[base_type]
    except KeyError:
        known = ", ".join(sorted(_SERVICE_MANAGERS)) or "(none registered)"
        raise KeyError(
            f"unknown service base_type {base_type!r}; known: {known}"
        ) from None


def get_harness(base_type: str) -> type[Harness]:
    """Look up a registered harness class, or raise a clear error."""
    try:
        return _HARNESSES[base_type]
    except KeyError:
        known = ", ".join(sorted(_HARNESSES)) or "(none registered)"
        raise KeyError(f"unknown harness base_type {base_type!r}; known: {known}") from None


def all_service_managers() -> dict[str, type[ServiceManager]]:
    """Every registered service manager, keyed by base_type (for `sovereign provision`)."""
    return dict(_SERVICE_MANAGERS)


def all_harnesses() -> dict[str, type[Harness]]:
    """Every registered harness, keyed by base_type (for `sovereign provision`)."""
    return dict(_HARNESSES)
