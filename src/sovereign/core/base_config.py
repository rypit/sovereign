"""Shared Pydantic base for all Sovereign config models.

Per the golden rule (§2.3), config depends on Pydantic **only** — never on
``subprocess``, ``os``, ``docker``, or any ``manager.py``. Keep it that way.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from sovereign.core.units import gb_to_bytes


class SovereignBaseModel(BaseModel):
    """Base for every config schema.

    ``extra="forbid"`` makes a typo'd YAML key a loud validation error rather than
    a silently-ignored field — the "fail fast with a field-level error" discipline
    the boot sequence relies on (§6.2).
    """

    model_config = ConfigDict(extra="forbid")


class NativeEngineConfig(SovereignBaseModel):
    """Fields every native engine's ``config:`` block shares.

    :class:`~sovereign.services.inference.base.NativeEngineManager` programs against this
    type; concrete engines subclass it, override ``binary``'s default, and add
    their own knobs.
    """

    #: Model reference — a local path (``~`` expanded) or a HuggingFace repo id.
    model: str
    #: DEPRECATED — ignored. Engines are embedded Python workers now (loaded via
    #: their binding's API in-process), not external CLIs launched by executable
    #: name, so there is no argv to put a binary name on. Kept (rather than
    #: removed) only because ``extra="forbid"`` would otherwise turn existing
    #: ``binary:`` YAML into a hard validation error; use ``engine_kwargs`` for
    #: settings the engine doesn't model yet.
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
    #: DEPRECATED — no longer consumed. Engines ran as external CLIs and this was
    #: raw argv appended to the command line; embedded workers have no argv to
    #: extend. Non-empty is now a validation error pointing at ``engine_kwargs``,
    #: the escape hatch for settings the engine-specific ``config:`` block and
    #: ``engine_kwargs`` mapping don't model yet.
    extra_args: list[str] = Field(default_factory=list)
    #: Escape hatch for engine-specific worker settings Sovereign's config schema
    #: doesn't model yet — merged (last, so it can override) into what each
    #: engine's ``engine_kwargs()`` derives from its typed fields, then mapped
    #: onto the real binding's API by the worker's adapter.
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    #: Directory for the captured stdout/stderr log (created on start).
    log_dir: str = ".sovereign/logs"

    @field_validator("extra_args")
    @classmethod
    def _extra_args_removed(cls, value: list[str]) -> list[str]:
        if value:
            raise ValueError(
                "extra_args is no longer supported — engines are embedded Python "
                "workers with no argv to extend. Use engine_kwargs instead (a "
                "dict merged into the engine's worker settings)."
            )
        return value


def _gb_input_to_bytes(value: object) -> object:
    """YAML declares GB; internal fields hold int bytes (1 GB = 10**9)."""
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return value  # let Pydantic produce the normal type error
    return gb_to_bytes(value)


#: Field type for "GB in YAML, bytes internally". Pair with
#: Field(validation_alias="<name>_gb") so the YAML key keeps its GB name.
GbBytes = Annotated[int, BeforeValidator(_gb_input_to_bytes)]


def validate_identifier(value: str) -> str:
    """Reusable validator: an instance ``name`` / ``base_type`` must be non-empty.

    Whitespace-only names are rejected and surrounding whitespace is stripped, so
    downstream registry lookups and dependency matching compare clean strings.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("must be a non-empty identifier")
    return stripped
