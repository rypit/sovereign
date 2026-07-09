"""Config schema for the generic ``docker`` container service.

Pydantic-only, per the golden rule (§2.3). Parses the ``config:`` block of a
``docker`` service entry — any Docker container Sovereign should run.
The daemon itself is implicit infrastructure (Docker Desktop / OrbStack owns its
lifecycle); reachability is verified by the manager before each container boots.
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class FileSpec(SovereignBaseModel):
    """A config file materialized on the host before the container starts.

    Written only if absent (idempotent), so generated secrets survive restarts.
    Content supports ``${RANDOM_HEX:n}`` (n random hex chars, generated on first
    write only) and ``${ENV:VAR}`` (read from the environment; missing vars fail
    the boot). ``{{ service.attr }}`` templates are NOT supported here.
    """

    #: Host path (``~`` expanded at materialization time).
    path: str
    #: Inline file content.
    content: str


class DockerConfig(SovereignBaseModel):
    """Settings for a generic Docker container service."""

    #: Image to run (e.g. ``ghcr.io/open-webui/open-webui:main``).
    image: str
    #: Host port to publish — also the endpoint and health-check URL port.
    port: int = Field(gt=0, lt=65536)
    #: Port the app listens on inside the container; defaults to ``port``.
    container_port: int | None = Field(default=None, gt=0, lt=65536)
    #: Container name; defaults to the service instance name.
    container_name: str | None = None
    #: ``host:container`` bind mounts or ``named_volume:container`` mounts.
    volumes: list[str] = Field(default_factory=list)
    #: Config files materialized on the host before start (see :class:`FileSpec`).
    files: list[FileSpec] = Field(default_factory=list)
    #: Pull the image during prepare_environment (PROVISIONING).
    auto_pull: bool = True
    #: Docker CLI to invoke; a bare name is resolved on ``PATH``.
    binary: str = "docker"
    #: Timeout (seconds) for the daemon reachability probe.
    probe_timeout_seconds: int = 10
