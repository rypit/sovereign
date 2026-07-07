"""Config schema for the ``cline_cli`` harness.

Pydantic-only (§2.3). Parses the ``config:`` block of a ``cline_cli`` harness
entry. Cline CLI keeps its provider settings in an isolated directory
(``CLINE_DIR``) rather than the user's shared global config, so running the
bench workhorse never touches a human's daily-driver Cline setup (§4b, §11).
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class ClineCliConfig(SovereignBaseModel):
    """Settings for one isolated ``cline`` CLI instance."""

    #: OpenAI-compatible base URL — usually ``{{ engine.endpoint }}/v1``.
    base_url: str
    #: Model name the client sends — usually ``{{ engine.model }}``.
    model: str
    #: Bearer key sent to the local endpoint.
    api_key: str = "sovereign"
    #: ``cline`` binary; a bare name is resolved on ``PATH``.
    binary: str = "cline"
    #: Isolated config directory (becomes ``CLINE_DIR``) so this harness never
    #: merges into the user's shared global Cline settings. Defaults to
    #: ``~/.sovereign/harnesses/<name>`` when unset.
    config_dir: str | None = None
    #: Wall-clock budget for one ``invoke()`` call.
    timeout_seconds: int = Field(default=900, gt=0)
    #: Cap on agentic turns per task, when the installed CLI supports it.
    max_turns: int | None = Field(default=None, gt=0)
