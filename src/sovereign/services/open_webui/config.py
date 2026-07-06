"""Config schema for the ``open_webui`` container service.

Pydantic-only (§2.3). Parses the ``config:`` block of an ``open_webui`` entry.
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class OpenWebUIConfig(SovereignBaseModel):
    """Settings for the Open WebUI Docker container."""

    image: str = "ghcr.io/open-webui/open-webui:main"
    #: Host port to publish (the address a browser / consumers hit).
    port: int = Field(default=3000, gt=0, lt=65536)
    #: Port Open WebUI listens on inside the container.
    container_port: int = Field(default=8080, gt=0, lt=65536)
    #: Container name; defaults to the service instance name.
    container_name: str | None = None
    #: ``host:container`` volume mounts.
    volumes: list[str] = Field(default_factory=list)
    #: Pull the image during prepare_environment (PROVISIONING).
    auto_pull: bool = True
    #: Docker CLI to invoke.
    binary: str = "docker"
