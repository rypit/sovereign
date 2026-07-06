"""Shared Pydantic base for all Sovereign config models.

Per the golden rule (§2.3), config depends on Pydantic **only** — never on
``subprocess``, ``os``, ``docker``, or any ``manager.py``. Keep it that way.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SovereignBaseModel(BaseModel):
    """Base for every config schema.

    ``extra="forbid"`` makes a typo'd YAML key a loud validation error rather than
    a silently-ignored field — the "fail fast with a field-level error" discipline
    the boot sequence relies on (§6.2).
    """

    model_config = ConfigDict(extra="forbid")


def validate_identifier(value: str) -> str:
    """Reusable validator: an instance ``name`` / ``base_type`` must be non-empty.

    Whitespace-only names are rejected and surrounding whitespace is stripped, so
    downstream registry lookups and dependency matching compare clean strings.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty identifier")
    return stripped
