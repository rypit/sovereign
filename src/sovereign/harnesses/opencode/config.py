"""Config schema for the ``opencode`` harness.

Pydantic-only (§2.3). Parses the ``config:`` block of an ``opencode`` harness
entry — ``base_url``/``model`` are typically ``{{ engine.endpoint }}/v1`` /
``{{ engine.model }}`` templates resolved against a local native engine (§4b).

opencode reaches a local stack through a *custom provider* entry in its own
``opencode.json``: an OpenAI-compatible AI-SDK provider whose ``baseURL``
points at the engine. ``provider_id`` names that entry, so the model opencode
is asked for is ``<provider_id>/<model>``.
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
    #: Id of the generated custom-provider block; opencode addresses the model
    #: as ``<provider_id>/<model>``.
    provider_id: str = "sovereign"
    #: AI-SDK package opencode loads for the provider. Pinnable (``pkg@1.2.3``)
    #: when a release regresses; the default is the generic OpenAI-compatible one.
    npm_package: str = "@ai-sdk/openai-compatible"
    #: ``opencode`` binary; a bare name is resolved on ``PATH``.
    binary: str = "opencode"
    #: Isolated config directory (its ``opencode.json`` becomes ``OPENCODE_CONFIG``)
    #: so this harness never merges into the user's global opencode settings.
    #: Defaults to ``~/.sovereign/harnesses/<name>`` when unset.
    config_dir: str | None = None
    #: opencode agent to run the task under (``--agent``); its own default when unset.
    agent: str | None = None
    #: Auto-approve tool permissions (``--auto``). Required for headless runs —
    #: without it ``opencode run`` *rejects* every permission request instead of
    #: prompting, so the agent can never touch the workspace.
    auto_approve: bool = True
    #: Wall-clock budget for one ``invoke()`` call.
    timeout_seconds: int = Field(default=900, gt=0)
