"""Resolved stack manifest — the reproducible fingerprint of a running stack (§7b).

Written at boot. Captures final flags, resolved endpoints, model fingerprints
(``path + size + mtime``, not a 40GB content hash), and co-resident services. This
is also what the benchmark runner consumes — built once here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.utils.state import write_json

if TYPE_CHECKING:
    from sovereign.orchestrator import Orchestrator


def _model_fingerprint(model_path: str) -> dict[str, Any] | None:
    path = Path(model_path).expanduser()
    if path.is_dir():
        # MLX snapshots are directories: fingerprint the recursive size + newest mtime.
        files = [f for f in path.rglob("*") if f.is_file()]
        if not files:
            return None
        size = sum(f.stat().st_size for f in files)
        mtime = max(int(f.stat().st_mtime) for f in files)
        return {"path": str(path), "size": size, "mtime": mtime}
    if not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime": int(stat.st_mtime)}


def _service_entry(orch: Orchestrator, name: str) -> dict[str, Any]:
    entry = orch.entry(name)
    manager = orch.managers.get(name)
    item: dict[str, Any] = {
        "name": name,
        "base_type": entry.base_type,
        "state": str(orch.states.get(name)),
        "dependencies": list(entry.dependencies),
        "co_resident": [n for n in orch.service_names if n != name],
    }

    if name in orch.registry:
        endpoint = orch.registry.get(name)
        item["endpoint"] = {
            "scheme": endpoint.scheme,
            "host": endpoint.host,
            "port": endpoint.port,
        }
        if endpoint.model is not None:
            item["endpoint"]["model"] = endpoint.model

    # Final resolved flags/args, if the manager exposes them. A native engine's
    # get_start_args() raises RuntimeError before prepare_model() has run (e.g. when
    # persisting a FAILED boot); omit start_args rather than fail the manifest.
    if hasattr(manager, "get_start_args"):
        try:
            item["start_args"] = manager.get_start_args()
        except RuntimeError:
            pass
    elif hasattr(manager, "_run_args") and getattr(manager, "resolved_env", None) is not None:
        try:
            item["run_args"] = manager._run_args()
        except Exception:  # noqa: BLE001 - manifest detail is best-effort
            pass

    # Prefer the manager's resolved local path (a downloaded HF snapshot/gguf) so
    # repo-id models fingerprint too; fall back to the declared config value.
    resolved = getattr(manager, "model_path", None)
    model_path = str(resolved) if resolved is not None else entry.config.get("model")
    if isinstance(model_path, str):
        fingerprint = _model_fingerprint(model_path)
        if fingerprint is not None:
            item["model_fingerprint"] = fingerprint

    reserved = orch.budgeter.reservations().get(name)
    if reserved is not None:
        item["estimated_memory_gb"] = round(reserved, 2)
    if hasattr(manager, "per_slot_context"):
        per_slot = manager.per_slot_context()
        if per_slot is not None:
            item["per_slot_context"] = per_slot

    return item


def _harness_entry(orch: Orchestrator, entry) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": entry.name,
        "base_type": entry.base_type,
        "dependencies": list(entry.dependencies),
    }
    harness = orch.harnesses.get(entry.name)
    fingerprint_fn = getattr(harness, "fingerprint", None)
    if callable(fingerprint_fn):
        item["fingerprint"] = fingerprint_fn()
    return item


def build_manifest(orch: Orchestrator) -> dict[str, Any]:
    """Assemble the resolved stack manifest for the current orchestrator state."""
    return {
        "version": orch.config.version,
        "created_at": datetime.now(UTC).isoformat(),
        "variant_file": str(orch.variant_file) if orch.variant_file else None,
        "variant_hash": orch.variant_hash,
        "resources": orch.config.resources.model_dump(mode="json"),
        "memory_budget": {
            "total_gb": orch.budgeter.total_gb,
            "safety_margin_gb": orch.budgeter.safety_margin_gb,
            "reserved_gb": round(orch.budgeter.reserved_gb, 2),
            "available_gb": round(orch.budgeter.available_gb, 2),
        },
        "services": [_service_entry(orch, name) for name in orch.boot_order],
        "harnesses": [_harness_entry(orch, entry) for entry in orch.config.harnesses],
    }


def write_manifest(orch: Orchestrator, path: str | Path) -> dict[str, Any]:
    """Build and persist the manifest; returns the manifest dict."""
    manifest = build_manifest(orch)
    write_json(path, manifest)
    return manifest
