"""Shared Pydantic base for all Sovereign config models.

Per the golden rule (§2.3), config depends on Pydantic **only** — never on
``subprocess``, ``os``, ``docker``, or any ``manager.py``. Keep it that way.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SovereignBaseModel(BaseModel):
    """Base for every config schema.

    ``extra="forbid"`` makes a typo'd YAML key a loud validation error rather than
    a silently-ignored field — the "fail fast with a field-level error" discipline
    the boot sequence relies on (§6.2).
    """

    model_config = ConfigDict(extra="forbid")


class NativeEngineConfig(SovereignBaseModel):
    """Fields every native engine's ``config:`` block shares.

    :class:`~sovereign.services.inference_engines.base.NativeEngineManager` programs against this
    type; concrete engines subclass it, override ``binary``'s default, and add
    their own knobs.
    """

    #: Model reference — a local path (``~`` expanded) or a HuggingFace repo id.
    model: str
    #: Server executable; a bare name is resolved on ``PATH``. No shared default —
    #: each engine sets its own.
    binary: str
    #: Address the server binds to.
    host: str = "127.0.0.1"
    #: Speculative-decoding draft model — local path or HF repo id.
    draft_model: str | None = None
    #: Max tokens to draft per step; engines add their own flag mapping/bounds.
    num_draft_tokens: int | None = None
    #: Client-facing model name — the string an OpenAI-compatible client sends as
    #: ``"model"``. Defaults to ``model`` when unset.
    served_model_name: str | None = None
    #: Escape hatch for flags Sovereign doesn't model yet.
    extra_args: list[str] = Field(default_factory=list)
    #: Directory for the captured stdout/stderr log (created on start).
    log_dir: str = ".sovereign/logs"


def validate_identifier(value: str) -> str:
    """Reusable validator: an instance ``name`` / ``base_type`` must be non-empty.

    Whitespace-only names are rejected and surrounding whitespace is stripped, so
    downstream registry lookups and dependency matching compare clean strings.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty identifier")
    return stripped
