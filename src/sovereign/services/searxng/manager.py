"""``searxng`` — a SearXNG metasearch Docker service (§10, §12 Phase 11).

The 2nd Docker service. Mirrors ``open_webui`` (reusing the shared docker_engine
helpers) and adds a materialized ``settings.yml`` — a generated secret plus the
``json`` result format that Open WebUI's search API requires — mounted at
``/etc/searxng``. Its endpoint is wired into Open WebUI for web search via
env_overrides; because SearXNG is a container (``consumer_kind = DOCKER``), a native
loopback endpoint resolves to ``host.docker.internal`` (§2.14).
"""

from __future__ import annotations

import secrets
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.registry import register_service
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint, Resolver
from sovereign.services.docker_engine.manager import (
    container_metrics,
    run_docker,
    stream_pull,
)
from sovereign.services.searxng.config import SearxngConfig

_HTTP_TIMEOUT = 2.0


@register_service("searxng")
class SearxngManager(ActivityMixin):
    """Supervises a SearXNG Docker container with a materialized settings.yml."""

    base_type = "searxng"
    consumer_kind = ConsumerKind.DOCKER

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = SearxngConfig.model_validate(entry.config)
        self._raw_env: dict[str, Any] = dict(entry.env_overrides or {})
        self.resolved_env: dict[str, Any] = {}
        self.health_path = entry.health_check.endpoint if entry.health_check else "/"

    # --- wiring ---
    def resolve(self, resolver: Resolver) -> None:
        self.resolved_env = resolver.resolve_mapping(self._raw_env, self.consumer_kind)

    def endpoint(self) -> ResolvedEndpoint:
        return ResolvedEndpoint(scheme="http", host="127.0.0.1", port=self.config.port)

    def runtime_handle(self) -> dict | None:
        return {"kind": "docker", "container": self._container_name()}

    # --- internal helpers ---
    def _container_name(self) -> str:
        return self.config.container_name or self.name

    def _settings_dir(self) -> Path:
        return Path(self.config.config_dir).expanduser()

    def _run_docker(self, args: list[str], *, timeout: float = 30, check: bool = False):
        return run_docker(args, binary=self.config.binary, timeout=timeout, check=check)

    def _materialize_settings(self) -> None:
        """Write a minimal settings.yml (secret + json format) if absent (idempotent)."""
        settings_dir = self._settings_dir()
        settings_file = settings_dir / "settings.yml"
        if settings_file.exists():
            return  # preserve an existing config + its secret
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "use_default_settings": True,
            "server": {
                "secret_key": self.config.secret or secrets.token_hex(32),
                "limiter": False,
            },
            "search": {"formats": ["html", "json"]},
        }
        settings_file.write_text(yaml.safe_dump(settings, sort_keys=False))

    def _run_args(self) -> list[str]:
        args = [
            "run",
            "-d",
            "--name",
            self._container_name(),
            "-p",
            f"{self.config.port}:{self.config.container_port}",
            "-v",
            f"{self._settings_dir()}:/etc/searxng",
        ]
        if self.config.base_url:
            args += ["-e", f"SEARXNG_BASE_URL={self.config.base_url}"]
        for key, val in self.resolved_env.items():
            args += ["-e", f"{key}={val}"]
        args.append(self.config.image)
        return args

    # --- Lifecycle ---
    def start(self) -> None:
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
        self._materialize_settings()
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
        """No-op for now (matches the other Docker services)."""
