"""Config schema for the ``pi`` harness.

Pydantic-only (§2.3). Parses the ``config:`` block of a ``pi`` harness entry —
``base_url``/``model`` are typically ``{{ engine.endpoint }}/v1`` /
``{{ engine.model }}`` templates resolved against a local native engine (§4b).

pi reaches a local stack through a *custom provider* in its ``models.json``:
a provider block naming the endpoint's API dialect (``openai-completions`` for
every engine Sovereign boots) plus the models it serves.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class PiConfig(SovereignBaseModel):
    """Settings for one isolated ``pi`` CLI instance."""

    #: OpenAI-compatible base URL — usually ``{{ engine.endpoint }}/v1``.
    base_url: str
    #: Model name the client sends — usually ``{{ engine.model }}``.
    model: str
    #: Bearer key sent to the local endpoint. Local engines rarely check it, but
    #: pi hides models it considers unauthenticated, so a placeholder is required.
    api_key: str = "sovereign"
    #: Id of the generated provider block in ``models.json``.
    provider_id: str = "sovereign"
    #: API dialect pi speaks to this provider (``openai-completions``,
    #: ``openai-responses``, ``anthropic-messages``, ``google-generative-ai``).
    api: str = "openai-completions"
    #: ``pi`` binary; a bare name is resolved on ``PATH``.
    binary: str = "pi"
    #: Isolated config directory (becomes ``PI_CODING_AGENT_DIR``) so this
    #: harness never merges into the user's shared ``~/.pi/agent``. Defaults to
    #: ``~/.sovereign/harnesses/<name>`` when unset.
    config_dir: str | None = None
    #: Context window advertised for the model, when the engine's own limit is known.
    context_window: int | None = Field(default=None, gt=0)
    #: Max output tokens advertised for the model.
    max_tokens: int | None = Field(default=None, gt=0)
    #: Provider-level compatibility switches for servers that don't implement
    #: every OpenAI extension — e.g. ``{supportsDeveloperRole: false,
    #: supportsReasoningEffort: false}`` for engines that reject the
    #: ``developer`` role or ``reasoning_effort``.
    compat: dict[str, Any] = Field(default_factory=dict)
    #: Restrict the agent to these built-in tools (``read``, ``bash``, ``edit``,
    #: ``write``, ``grep``, ``find``, ``ls``); pi's full set when unset.
    tools: list[str] | None = None
    #: Reasoning level (``off``/``minimal``/``low``/``medium``/``high``/``xhigh``/``max``).
    thinking: str | None = None
    #: Suppress pi's startup network calls (update check + install telemetry) so
    #: a run stays local and reproducible.
    offline: bool = True
    #: Wall-clock budget for one ``invoke()`` call.
    timeout_seconds: int = Field(default=900, gt=0)
