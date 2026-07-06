"""``open_webui`` — the first container service + dynamic wiring (§12, Phase 5).

Runs Open WebUI as a Docker container and auto-connects it to a native engine by
resolving its ``env_overrides`` through the consumer-aware :class:`Resolver`. Because
Open WebUI is a *container* (``consumer_kind = DOCKER``), a ``{{ llama_heavy_v1.endpoint }}``
that points at a loopback native process resolves to ``host.docker.internal`` — the
whole point of §2.14.

The manager reuses the shared Docker helpers from the ``docker_engine`` package
(``run_docker`` / ``stream_pull``) so every Docker service goes through one Docker
CLI path and reports pull progress uniformly (§2.4 — no other service's code changes).
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.registry import register_service
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint, Resolver
from sovereign.services.docker_engine.manager import (
    container_metrics,
    run_docker,
    stream_pull,
)
from sovereign.services.open_webui.config import OpenWebUIConfig

_HTTP_TIMEOUT = 2.0


@register_service("open_webui")
class OpenWebUIManager(ActivityMixin):
    """Supervises an Open WebUI Docker container, wired to a native engine."""

    base_type = "open_webui"
    consumer_kind = ConsumerKind.DOCKER

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = OpenWebUIConfig.model_validate(entry.config)
        self._raw_env: dict[str, Any] = dict(entry.env_overrides or {})
        self.resolved_env: dict[str, Any] = {}
        self.health_path = entry.health_check.endpoint if entry.health_check else "/"

    # --- wiring ---
    def resolve(self, resolver: Resolver) -> None:
        """Resolve templated env against the registry (§6.2 — before start)."""
        self.resolved_env = resolver.resolve_mapping(self._raw_env, self.consumer_kind)

    def endpoint(self) -> ResolvedEndpoint:
        return ResolvedEndpoint(scheme="http", host="127.0.0.1", port=self.config.port)

    def runtime_handle(self) -> dict | None:
        """A cross-process teardown handle (container name) for `down`."""
        return {"kind": "docker", "container": self._container_name()}

    # --- internal helpers ---
    def _container_name(self) -> str:
        return self.config.container_name or self.name

    def _run_docker(
        self, args: list[str], *, timeout: float = 30, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return run_docker(args, binary=self.config.binary, timeout=timeout, check=check)

    def _run_args(self) -> list[str]:
        args = [
            "run",
            "-d",
            "--name",
            self._container_name(),
            "-p",
            f"{self.config.port}:{self.config.container_port}",
        ]
        for key, val in self.resolved_env.items():
            args += ["-e", f"{key}={val}"]
        for vol in self.config.volumes:
            args += ["-v", vol]
        args.append(self.config.image)
        return args

    # --- Lifecycle ---
    def start(self) -> None:
        # Idempotent: clear any stale container of the same name, then run.
        self._run_docker(["rm", "-f", self._container_name()], check=False)
        self._run_docker(self._run_args(), timeout=60, check=True)

    def stop(self) -> None:
        self._run_docker(["rm", "-f", self._container_name()], check=False)

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        url = f"http://127.0.0.1:{self.config.port}{self.health_path}"
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 - fixed http scheme
                return 200 <= resp.status < 400
        except (urllib.error.URLError, OSError):
            return False

    def get_metrics(self) -> dict[str, Any]:
        return container_metrics(self._container_name(), binary=self.config.binary)

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        if shutil.which(self.config.binary) is None:
            raise FileNotFoundError(
                f"Docker CLI '{self.config.binary}' not found on PATH for '{self.name}'."
            )
        if self.config.auto_pull:
            try:
                stream_pull(
                    self.config.image,
                    binary=self.config.binary,
                    on_progress=self.set_activity,
                )
            finally:
                self.clear_activity()

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """No-op for now; container memory caps arrive with §7 (Phase 7)."""
