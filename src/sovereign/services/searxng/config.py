"""Config schema for the ``searxng`` metasearch Docker service (§10)."""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class SearxngConfig(SovereignBaseModel):
    """Settings for the SearXNG Docker container."""

    image: str = "searxng/searxng:latest"
    #: Host port to publish (what consumers hit).
    port: int = Field(default=8888, gt=0, lt=65536)
    #: Port SearXNG listens on inside the container.
    container_port: int = Field(default=8080, gt=0, lt=65536)
    container_name: str | None = None
    #: Host dir mounted at /etc/searxng; a settings.yml is materialized here.
    config_dir: str = "~/.sovereign/searxng"
    #: server.secret_key; a random one is generated if unset.
    secret: str | None = None
    #: Optional SEARXNG_BASE_URL env value (e.g. "http://localhost:8888/").
    base_url: str | None = None
    auto_pull: bool = True
    binary: str = "docker"
