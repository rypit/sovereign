"""Config schema for the ``opencode`` harness.

Pydantic-only (§2.3). Parses the ``config:`` block of an ``opencode`` harness
entry. opencode reads its settings from the JSON file named by the
``OPENCODE_CONFIG`` environment variable, so each instance gets an isolated
config file rather than merging into the user's global
``~/.config/opencode/opencode.json`` (§4b, §11).
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class OpencodeConfig(SovereignBaseModel):
    """Settings for one isolated ``opencode`` CLI instance."""

    #: OpenAI-compatible base URL — usually ``{{ engine.endpoint }}/v1``.
    base_url: str
    #: Model name the client sends — usually ``{{ engine.model }}``.
    model: str
    #: Bearer key sent to the local endpoint.
    api_key: str = "sovereign"
    #: ``opencode`` binary; a bare name is resolved on ``PATH``.
    binary: str = "opencode"
    #: Isolated config directory (its ``opencode.json`` becomes
    #: ``OPENCODE_CONFIG``) so this harness never merges into the user's global
    #: opencode settings. Defaults to ``~/.sovereign/harnesses/<name>`` when unset.
    config_dir: str | None = None
    #: Wall-clock budget for one ``invoke()`` call.
    timeout_seconds: int = Field(default=900, gt=0)
    #: opencode agent to run with (``opencode run --agent``), when set.
    agent: str | None = None
