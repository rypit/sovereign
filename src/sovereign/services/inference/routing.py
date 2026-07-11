"""Engine routing — pick the native engine that serves an ``auto`` model ref (§2.6).

The routing *decision* is dependency-inverted: each engine declares what it
claims via :meth:`RoutesModelRef.claim_route`, and this module sweeps every
registered engine and takes the highest-confidence claim. Dropping in a new
engine folder therefore extends ``auto`` routing with no central rule table.

Core reaches this through :func:`sovereign.core.registry.route_entry`; the
concrete router is registered below as an import side effect (``populate_registries``
imports this module via ``walk_packages``), the same pattern as ``@register_service``.
Metadata is read through the ``hf`` module (one patchable seam).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sovereign.core.base_manager import RoutesModelRef
from sovereign.core.errors import RoutingError
from sovereign.core.registry import register_router
from sovereign.services.inference import hf

if TYPE_CHECKING:
    from pathlib import Path

    from sovereign.config import ServiceEntry
    from sovereign.core.base_manager import ServiceManager
    from sovereign.services.inference.hf import ModelRef, RepoInfo

log = logging.getLogger(__name__)


def _claim_engine(
    engines: Sequence[type[ServiceManager]], ref: ModelRef, info: RepoInfo | None
) -> RoutesModelRef | None:
    """The engine with the highest ``claim_route`` confidence, or None if none claim."""
    best: RoutesModelRef | None = None
    best_confidence: int | None = None
    for cls in engines:
        if not isinstance(cls, RoutesModelRef):
            continue
        confidence = cls.claim_route(ref, info)
        if confidence is None:
            continue
        if best_confidence is None or confidence > best_confidence:
            best, best_confidence = cls, confidence
    return best


def resolve_entry_base_type(
    entry: ServiceEntry, state_dir: Path, engines: Sequence[type[ServiceManager]]
) -> str:
    """Resolve ``base_type`` for a ServiceEntry, routing ``"auto"`` entries.

    Returns the existing base_type unchanged for non-auto entries. For ``auto``:
    local refs and hub metadata route by engine claim; an offline miss falls back
    to the persisted routing cache, else raises :class:`RoutingError`.
    """
    if entry.base_type != "auto":
        return entry.base_type

    model: str = entry.config.get("model")  # type: ignore[assignment]
    ref = hf.parse_model_ref(model)
    cache = hf.RoutingCache(state_dir / "models.json")

    if ref.is_local:
        engine = _claim_engine(engines, ref, None)
        if engine is None:
            raise RoutingError(
                f"Cannot determine engine for local path '{ref.local_path}': "
                "no engine claims it (no .gguf files and no config.json/safetensors)"
            )
        return engine.base_type

    info = hf.fetch_repo_info(ref.repo_id)  # type: ignore[arg-type]
    if info is not None:
        engine = _claim_engine(engines, ref, info)
        if engine is None:
            raise RoutingError(
                f"Cannot route '{ref.raw}': no engine claims it "
                f"(tags={info.tags!r}, siblings={len(info.siblings)})"
            )
        base_type = engine.base_type
        kind = engine.model_artifact_kind
        wb = hf.weight_bytes(info, kind, quant=ref.quant, filename=ref.filename)
        cache.put(model, base_type=base_type, weight_bytes=wb)
        log.debug("routed %s -> %s (hub metadata)", model, base_type)
        return base_type

    # Offline: consult routing cache
    cached = cache.get(model)
    if cached:
        log.debug("routed %s -> %s (offline, routing cache)", model, cached["base_type"])
        return cached["base_type"]

    raise RoutingError(
        f"Cannot route '{model}' offline — "
        "set an explicit base_type or connect once to populate the routing cache"
    )


register_router(resolve_entry_base_type)
