"""Template + secret resolution against the runtime service registry (§2.14, §6.2).

Two substitutions happen when a service's inputs are resolved just before boot:

* ``{{ service_name.attr }}`` — a reference to another service's resolved endpoint
  (``endpoint``/``url``, ``host``, ``port``, ``scheme``).
* ``${ENV:VAR}`` — a secret read from the process environment.

**Consumer-aware host** is the load-bearing subtlety: a *container* reaching a
*native* process on the host cannot use ``localhost`` — it must use
``host.docker.internal``. The resolver therefore emits the right host per
*consumer kind*, so ``open_webui`` (a container) auto-connects to ``llama_heavy_v1``
(a native process) with zero manual config.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Hosts that mean "this machine's loopback" and must be rewritten for containers.
_LOOPBACK = {"127.0.0.1", "localhost", "0.0.0.0"}
_DOCKER_HOST_GATEWAY = "host.docker.internal"

_SERVICE_RE = re.compile(r"\{\{\s*(?P<name>[A-Za-z0-9_-]+)\.(?P<attr>[A-Za-z0-9_]+)\s*\}\}")
_ENV_RE = re.compile(r"\$\{ENV:(?P<var>[A-Za-z0-9_]+)\}")


class ResolutionError(Exception):
    """Raised when a template references an unknown service, attribute, or secret."""


class ConsumerKind(StrEnum):
    """How a consumer reaches endpoints — determines host rewriting."""

    NATIVE = "native"  # a host process: loopback stays loopback
    DOCKER = "docker"  # a container: loopback -> host.docker.internal


@dataclass(frozen=True)
class ResolvedEndpoint:
    """A service's reachable address, resolved per consumer at read time."""

    scheme: str
    host: str
    port: int

    def host_for(self, consumer: ConsumerKind) -> str:
        if consumer is ConsumerKind.DOCKER and self.host in _LOOPBACK:
            return _DOCKER_HOST_GATEWAY
        return self.host

    def url_for(self, consumer: ConsumerKind) -> str:
        return f"{self.scheme}://{self.host_for(consumer)}:{self.port}"

    def attribute(self, attr: str, consumer: ConsumerKind) -> str:
        if attr in ("endpoint", "url"):
            return self.url_for(consumer)
        if attr == "host":
            return self.host_for(consumer)
        if attr == "port":
            return str(self.port)
        if attr == "scheme":
            return self.scheme
        raise ResolutionError(
            f"unknown endpoint attribute '{attr}' (expected endpoint/url/host/port/scheme)"
        )


class ServiceRegistry:
    """Runtime map of service name -> resolved endpoint.

    Populated by the Orchestrator as each service reaches ``READY`` (§6.2 step 4);
    read by the :class:`Resolver` when wiring dependents.
    """

    def __init__(self) -> None:
        self._endpoints: dict[str, ResolvedEndpoint] = {}

    def register(self, name: str, endpoint: ResolvedEndpoint) -> None:
        self._endpoints[name] = endpoint

    def get(self, name: str) -> ResolvedEndpoint:
        return self._endpoints[name]

    def __contains__(self, name: str) -> bool:
        return name in self._endpoints

    def names(self) -> list[str]:
        return list(self._endpoints)


class Resolver:
    """Substitutes ``{{ }}`` templates and ``${ENV:}`` secrets in config values."""

    def __init__(self, registry: ServiceRegistry, env: Mapping[str, str] | None = None) -> None:
        self.registry = registry
        self.env = env if env is not None else os.environ

    def resolve(self, value: Any, consumer: ConsumerKind) -> Any:
        """Resolve a single value; non-strings pass through unchanged."""
        if not isinstance(value, str):
            return value

        def _service(match: re.Match[str]) -> str:
            name, attr = match.group("name"), match.group("attr")
            if name not in self.registry:
                known = ", ".join(self.registry.names()) or "(none ready)"
                raise ResolutionError(
                    f"template references unknown service '{name}'; ready: {known}"
                )
            return self.registry.get(name).attribute(attr, consumer)

        def _secret(match: re.Match[str]) -> str:
            var = match.group("var")
            if var not in self.env:
                raise ResolutionError(f"secret ${{ENV:{var}}} is not set in the environment")
            return self.env[var]

        value = _SERVICE_RE.sub(_service, value)
        value = _ENV_RE.sub(_secret, value)
        return value

    def resolve_mapping(
        self, mapping: Mapping[str, Any], consumer: ConsumerKind
    ) -> dict[str, Any]:
        """Resolve every value in a mapping (e.g. ``env_overrides`` or ``config``)."""
        return {key: self.resolve(val, consumer) for key, val in mapping.items()}
