"""``docker_engine`` — the generic Docker container service (§12).

Sovereign runs auxiliary services in Docker (§2.1), but Docker Desktop / OrbStack
owns the daemon lifecycle on macOS, not Sovereign. So there is no standalone
"engine" service to declare or depend on: any service naming ``base_type:
docker_engine`` runs an arbitrary container from its own ``config:`` block, and
this manager verifies the daemon is reachable (as part of its own
``prepare_environment``) before pulling the image and starting the container.

This module also hosts the shared Docker CLI helpers reused by every
``docker_engine`` instance: ``run_docker``, ``stream_pull``, ``container_metrics``.
"""

from __future__ import annotations

import re
import secrets
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.provisioning import Provisioner
from sovereign.core.registry import register_service
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint, Resolver, ServiceRegistry
from sovereign.services.docker_engine.config import DockerEngineConfig, FileSpec

_HTTP_TIMEOUT = 2.0

# --- Shared Docker CLI helpers (reused by every docker_engine instance) ---

# A `docker pull` progress line: "<layer-id>: <status>".
_PULL_LINE = re.compile(r"^(?P<layer>[0-9a-f]{12,}): (?P<status>.+)$")
_DONE_STATUSES = ("Pull complete", "Already exists")

_RANDOM_HEX_RE = re.compile(r"\$\{RANDOM_HEX:(?P<n>\d+)\}")


def run_docker(
    args: list[str],
    *,
    binary: str = "docker",
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a ``docker`` subcommand, capturing output. The one Docker CLI entry point."""
    return subprocess.run(  # noqa: S603 - args are constructed, not shell
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def pull_activity(image: str, done: int, total: int) -> str:
    """A short progress line for a running ``docker pull``."""
    if total:
        return f"pulling {image} — {done}/{total} layers"
    return f"pulling {image}"


def stream_pull(
    image: str,
    *,
    binary: str = "docker",
    on_progress: Callable[[str], None],
) -> None:
    """Stream ``docker pull <image>``, reporting layer progress via ``on_progress``.

    Without a TTY, ``docker pull`` emits per-layer status lines (not a byte %), so
    progress is a layer count. Raises ``RuntimeError`` on a non-zero exit.
    """
    on_progress(pull_activity(image, 0, 0))
    proc = subprocess.Popen(  # noqa: S603 - args are constructed, not shell
        [binary, "pull", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    layers: set[str] = set()
    done: set[str] = set()
    last = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        last = line.strip()
        match = _PULL_LINE.match(last)
        if match:
            layers.add(match.group("layer"))
            if match.group("status").startswith(_DONE_STATUSES):
                done.add(match.group("layer"))
            on_progress(pull_activity(image, len(done), len(layers)))
        elif last:
            on_progress(last)
    if proc.wait() != 0:
        raise RuntimeError(f"docker pull failed for {image}: {last}")


def parse_mem_to_mb(value: str) -> float:
    """Parse a ``docker stats`` memory field (e.g. ``"15.5MiB"``) into MB."""
    value = value.strip()
    units = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }
    for unit in sorted(units, key=len, reverse=True):
        if value.endswith(unit):
            number = float(value[: -len(unit)])
            return round(number * units[unit] / (1024**2), 2)
    return 0.0


def container_metrics(container: str, *, binary: str = "docker") -> dict[str, Any]:
    """Point-in-time metrics for a running container via ``docker stats``.

    Shared by every Docker service. Returns ``{memory_mb, cpu_percent, status}`` when
    running, else ``{"status": "stopped"}``.
    """
    stats_args = [
        "stats",
        "--no-stream",
        "--format",
        "{{.CPUPerc}};{{.MemUsage}}",
        container,
    ]
    try:
        result = run_docker(stats_args, binary=binary)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"status": "stopped"}
    if result.returncode != 0 or not result.stdout.strip():
        return {"status": "stopped"}
    cpu_str, _, mem_str = result.stdout.strip().partition(";")
    return {
        "memory_mb": parse_mem_to_mb(mem_str.split("/")[0]),
        "cpu_percent": float(cpu_str.strip().rstrip("%")),
        "status": "running",
    }


def expand_volume(spec: str) -> str:
    """Expand ``~`` in a bind-mount's host path; leave named volumes untouched."""
    host, sep, container = spec.partition(":")
    if sep and host.startswith("~"):
        return f"{Path(host).expanduser()}{sep}{container}"
    return spec


def materialize_file(spec: FileSpec, env: Mapping[str, str] | None = None) -> bool:
    """Write ``spec.content`` to ``spec.path`` if absent (idempotent).

    Preserves anything already on disk — including a previously generated
    ``${RANDOM_HEX:...}`` secret — across restarts. Returns True iff written.
    """
    path = Path(spec.path).expanduser()
    if path.exists():
        return False

    content = _RANDOM_HEX_RE.sub(
        lambda m: secrets.token_bytes((int(m.group("n")) + 1) // 2).hex()[: int(m.group("n"))],
        spec.content,
    )
    content = Resolver(ServiceRegistry(), env).resolve(content, ConsumerKind.DOCKER)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


@register_service("docker_engine")
class DockerEngineManager(ActivityMixin, Provisioner):
    """Supervises a generic Docker container described entirely by its config."""

    base_type = "docker_engine"
    consumer_kind = ConsumerKind.DOCKER
    #: Provisioned via the package Brewfile (`cask "docker-desktop"`) when missing.
    provisioning_binary = "docker"

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = DockerEngineConfig.model_validate(entry.config)
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

    def _run_docker(
        self, args: list[str], *, timeout: float = 30, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return run_docker(args, binary=self.config.binary, timeout=timeout, check=check)

    def _daemon_reachable(self) -> bool:
        """True iff the Docker *server* answers — not just the client being present."""
        try:
            result = self._run_docker(
                ["version", "--format", "{{.Server.Version}}"],
                timeout=self.config.probe_timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _run_args(self) -> list[str]:
        args = [
            "run",
            "-d",
            "--name",
            self._container_name(),
            "-p",
            f"{self.config.port}:{self.config.container_port or self.config.port}",
        ]
        for key, val in self.resolved_env.items():
            args += ["-e", f"{key}={val}"]
        for vol in self.config.volumes:
            args += ["-v", expand_volume(vol)]
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
        # Install the Docker app first if missing (idempotent no-op once present);
        # daemon *reachability* stays a runtime check below.
        self.provision()
        if shutil.which(self.config.binary) is None:
            raise FileNotFoundError(
                f"Docker CLI '{self.config.binary}' not found on PATH for '{self.name}'. "
                "Install Docker Desktop or OrbStack."
            )
        if not self._daemon_reachable():
            raise RuntimeError(
                f"Docker daemon is not reachable via '{self.config.binary}' "
                f"(needed by '{self.name}'). Start Docker Desktop or OrbStack and retry."
            )
        for spec in self.config.files:
            materialize_file(spec)
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
        """No-op for now (container memory caps are a future phase)."""
