"""``base_type`` → class factory maps (§2.6, §3 ``core/registry.py``).

Instance identity (``name``) is separate from implementation (``base_type``): the
Orchestrator looks up a ``base_type`` here to find which Manager/Harness class to
instantiate, which is what lets two ``llama_cpp`` instances run side by side.

Concrete services/harnesses register themselves via the decorators below, invoked
as a side effect of importing their package. Call :func:`populate_registries`
before any lookup — it imports every in-tree integration package (which
auto-discover their subpackages), so registration can never be silently skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

from sovereign.core.base_harness import Harness
from sovereign.core.base_manager import ServiceManager

if TYPE_CHECKING:
    from pathlib import Path

    from sovereign.config import ServiceEntry

_SERVICE_MANAGERS: dict[str, type[ServiceManager]] = {}
_HARNESSES: dict[str, type[Harness]] = {}

M = TypeVar("M", bound=ServiceManager)
H = TypeVar("H", bound=Harness)


class _Router(Protocol):
    """The engine-routing entry point the inference-engine package provides."""

    def __call__(
        self, entry: ServiceEntry, state_dir: Path, engines: list[type[ServiceManager]]
    ) -> str: ...


# The concrete router is registered as an import side effect of the inference-engine
# package (see ``services/inference_engines/routing.py``), the same dependency-inversion
# pattern as ``@register_service`` — core never imports the engine package by name.
_ROUTER: _Router | None = None


def register_router(router: _Router) -> None:
    """Register the engine-routing implementation (called once, on import)."""
    global _ROUTER
    _ROUTER = router


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


def populate_registries() -> None:
    """Import every in-tree integration package so its ``@register_*`` decorators run.

    The single entry point for registry population — the imports are lazy (inside
    the function) to avoid a cycle with the integration modules, which import this
    module for the decorators. Idempotent: Python caches the imports.
    """
    import sovereign.harnesses  # noqa: F401, PLC0415 - registration side effect
    import sovereign.services  # noqa: F401, PLC0415 - registration side effect


def route_entry(entry: ServiceEntry, state_dir: Path) -> str:
    """Resolve a service entry's concrete ``base_type``, routing ``"auto"`` entries.

    The single seam boot and ``sovereign plan`` share for engine routing. Delegates
    to the engine-provided router over every registered service manager; raises
    ``sovereign.core.errors.RoutingError`` when an ``auto`` entry can't be resolved.
    """
    populate_registries()
    if _ROUTER is None:  # pragma: no cover - registration is an import side effect
        raise RuntimeError(
            "no model router registered; is the inference-engine package importable?"
        )
    return _ROUTER(entry, state_dir, list(_SERVICE_MANAGERS.values()))


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
