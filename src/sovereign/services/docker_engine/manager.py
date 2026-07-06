"""``docker_engine`` — the thinnest possible service manager (§12, Phase 3).

Sovereign runs auxiliary services in Docker (§2.1), but Docker Desktop / OrbStack
owns the daemon lifecycle on macOS. So this manager does not *start* a daemon; it
verifies the daemon is reachable, reports health, and exposes ``run_compose()`` for
the Dockerized services that arrive in later phases.

This module also establishes the manager-construction convention every service
follows: ``Manager(entry: ServiceEntry)`` parses ``entry.config`` into its own
sibling config model.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from sovereign.config import ServiceEntry
from sovereign.core.base_manager import ActivityMixin
from sovereign.core.registry import register_service
from sovereign.core.resolver import ConsumerKind
from sovereign.services.docker_engine.config import DockerEngineConfig

# --- Shared Docker CLI helpers (reused by every Docker-based service) ---

# A `docker pull` progress line: "<layer-id>: <status>".
_PULL_LINE = re.compile(r"^(?P<layer>[0-9a-f]{12,}): (?P<status>.+)$")
_DONE_STATUSES = ("Pull complete", "Already exists")


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


@register_service("docker_engine")
class DockerEngineManager(ActivityMixin):
    """Verifies and talks to the local Docker daemon."""

    base_type = "docker_engine"
    consumer_kind = ConsumerKind.NATIVE

    def __init__(self, entry: ServiceEntry) -> None:
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.config = DockerEngineConfig.model_validate(entry.config)

    # --- internal helper ---
    def _run(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return run_docker(
            args,
            binary=self.config.binary,
            timeout=timeout if timeout is not None else self.config.probe_timeout_seconds,
            check=check,
        )

    # --- Lifecycle ---
    def start(self) -> None:
        """No daemon to spawn — assert reachability so boot fails fast if it's down."""
        if not self.is_healthy():
            raise RuntimeError(
                f"Docker daemon is not reachable via '{self.config.binary}'. "
                "Start Docker Desktop or OrbStack and retry."
            )

    def stop(self) -> None:
        """No-op: Sovereign never stops the user's Docker daemon."""

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        """True iff the Docker *server* answers — not just the client being present."""
        try:
            result = self._run(["version", "--format", "{{.Server.Version}}"])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def get_metrics(self) -> dict[str, Any]:
        healthy = self.is_healthy()
        metrics: dict[str, Any] = {"status": "running" if healthy else "stopped"}
        if healthy:
            try:
                result = self._run(["ps", "--quiet"])
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return metrics
            if result.returncode == 0:
                metrics["containers"] = sum(
                    1 for line in result.stdout.splitlines() if line.strip()
                )
        return metrics

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        """Pre-flight: the Docker CLI must exist before we try to reach the daemon."""
        if shutil.which(self.config.binary) is None:
            raise FileNotFoundError(
                f"Docker CLI '{self.config.binary}' not found on PATH. "
                "Install Docker Desktop or OrbStack."
            )

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """No-op: the daemon's footprint isn't Sovereign's to shrink."""

    # --- Service-specific surface ---
    def run_compose(
        self,
        args: list[str],
        *,
        compose_file: str | None = None,
        project_name: str | None = None,
        timeout: float = 300,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``docker compose ...`` for the Dockerized services built on top of this."""
        cmd = ["compose"]
        if compose_file is not None:
            cmd += ["-f", compose_file]
        if project_name is not None:
            cmd += ["-p", project_name]
        cmd += args
        return self._run(cmd, timeout=timeout, check=check)
