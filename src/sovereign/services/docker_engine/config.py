"""Config schema for the ``docker_engine`` service.

Pydantic-only, per the golden rule (§2.3). Parses the ``config:`` block of a
``docker_engine`` service entry.
"""

from __future__ import annotations

from sovereign.core.base_config import SovereignBaseModel


class DockerEngineConfig(SovereignBaseModel):
    """Settings for talking to the (externally-managed) Docker daemon.

    Sovereign does not run the daemon itself — Docker Desktop / OrbStack owns that
    lifecycle. This manager only verifies reachability and shells out to the CLI.
    """

    #: Docker CLI to invoke; a bare name is resolved on ``PATH``.
    binary: str = "docker"
    #: Timeout (seconds) for quick daemon-probe commands (version, ps).
    probe_timeout_seconds: int = 10
