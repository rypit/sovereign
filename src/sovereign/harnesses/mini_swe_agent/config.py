"""Config schema for the ``mini_swe_agent`` harness.

Pydantic-only (§2.3). Parses the ``config:`` block of a ``mini_swe_agent`` harness
entry — ``base_url``/``model`` are typically ``{{ engine.endpoint }}``/``{{
engine.model }}`` templates resolved against a local native engine (§4b).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class MiniSweAgentConfig(SovereignBaseModel):
    """Settings for one ``mini-swe-agent`` (DefaultAgent) instance."""

    #: OpenAI-compatible base URL — usually ``{{ engine.endpoint }}/v1``.
    base_url: str
    #: Model name the client sends — usually ``{{ engine.model }}``.
    model: str
    #: Bearer key sent to the local endpoint. Local engines rarely check this,
    #: but LiteLLM requires a non-empty value to be present.
    api_key: str = "sovereign"
    #: Max model calls per ``invoke()`` (mini-swe-agent's ``step_limit``).
    step_limit: int = Field(default=40, gt=0)
    #: Wall-clock budget for one ``invoke()`` call.
    timeout_seconds: int = Field(default=900, gt=0)
    #: Where the materialized settings file is written. Defaults to
    #: ``~/.sovereign/harnesses/<name>`` when unset.
    config_dir: str | None = None
    #: Escape hatch passed through to ``DefaultAgent`` (e.g. ``cost_limit``).
    extra: dict[str, Any] = Field(default_factory=dict)
