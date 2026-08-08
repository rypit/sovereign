"""Per-invoke Docker sandboxes for harness execution (ADR 0009).

A sandbox is a run-to-completion throwaway, not a supervised service — hence a
module of helpers beside :class:`~sovereign.services.docker.manager.DockerManager`
rather than an extension of it. Everything a sandbox creates (network, egress
proxy, agent container) is labeled ``sovereign.sandbox=1`` so crash leftovers
can always be found and reaped by ``sweep_leftovers``.

Network model (strict engine-only egress): the agent container joins an
``--internal`` network with no route out. Its only reachable peer is a socat
proxy attached to both that network (under the fixed alias ``sovereign-llm``)
and the default bridge, forwarding one TCP port to the *native* engine on the
host via ``host.docker.internal``. A prompt-injected agent can therefore talk
to the local LLM and nothing else. ``network="none"`` skips even that.

All Docker traffic flows through :func:`run_docker` — the same single seam
tests already patch for the ``docker`` service.
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from sovereign.services.docker.manager import run_docker, stream_pull

logger = logging.getLogger("sovereign")

#: Label present on every sandbox-created container and network.
SANDBOX_LABEL = "sovereign.sandbox"
#: Label carrying the owning harness's instance name (scoped sweeps).
HARNESS_LABEL = "sovereign.harness"
#: In-network alias + listen port of the egress proxy. Fixed values are safe:
#: each sandbox gets its own namespaced network.
PROXY_ALIAS = "sovereign-llm"
PROXY_PORT = 8000

#: Hosts that mean "the engine runs on this machine" (mirrors the resolver).
_LOOPBACK = {"127.0.0.1", "localhost", "0.0.0.0"}

_PROXY_READY_TRIES = 10
_PROXY_READY_DELAY = 0.5


class SandboxError(Exception):
    """A sandbox could not be created or its proxy never came up."""


class SandboxTimeout(Exception):
    """The sandboxed agent exceeded its wall-clock budget."""

    def __init__(self, message: str, partial_stdout: str = "") -> None:
        super().__init__(message)
        self.partial_stdout = partial_stdout


def sandbox_id(harness_name: str) -> str:
    """A unique, label-independent name prefix for one sandbox instance."""
    return f"sov-sbx-{harness_name}-{uuid.uuid4().hex[:8]}"


def _labels(sid: str, harness_name: str) -> list[str]:
    return [
        "--label",
        f"{SANDBOX_LABEL}=1",
        "--label",
        f"{HARNESS_LABEL}={harness_name}",
        "--label",
        f"sovereign.sandbox_id={sid}",
    ]


def rewrite_base_url(base_url: str) -> tuple[int, str]:
    """Split a host-truth loopback base URL into (engine port, in-sandbox URL).

    The harness's resolved config stays host-truth (``http://127.0.0.1:8080/v1``);
    inside the sandbox the same path is served at the proxy alias. Rejects
    non-loopback hosts: a remote engine needs no host-gateway proxy, and
    silently tunneling to one would widen the sandbox's reach.
    """
    parts = urlsplit(base_url)
    if parts.hostname not in _LOOPBACK:
        raise SandboxError(
            f"sandbox base_url must target a local engine (loopback host), got '{base_url}'"
        )
    if parts.port is None:
        raise SandboxError(f"sandbox base_url must carry an explicit port, got '{base_url}'")
    return parts.port, f"http://{PROXY_ALIAS}:{PROXY_PORT}{parts.path}"


def create_internal_network(sid: str, *, harness_name: str, binary: str = "docker") -> None:
    """Create the sandbox's ``--internal`` network (no route to the outside)."""
    result = run_docker(
        ["network", "create", "--internal", *_labels(sid, harness_name), sid],
        binary=binary,
        timeout=30,
    )
    if result.returncode != 0:
        raise SandboxError(f"failed to create sandbox network '{sid}': {result.stderr.strip()}")


def start_egress_proxy(
    sid: str,
    engine_port: int,
    *,
    harness_name: str,
    image: str = "alpine/socat",
    binary: str = "docker",
) -> None:
    """Start the socat sidecar bridging the internal network to the engine port.

    The proxy sits on the default bridge (where ``host.docker.internal`` is
    reachable) and is then connected into the internal network under the fixed
    ``sovereign-llm`` alias — the only path out of the sandbox. Rolls back and
    raises :class:`SandboxError` if the proxy never reports running.
    """
    run_args = [
        "run",
        "-d",
        "--name",
        f"{sid}-proxy",
        *_labels(sid, harness_name),
        "--add-host",
        "host.docker.internal:host-gateway",
        image,
        f"TCP-LISTEN:{PROXY_PORT},fork,reuseaddr",
        f"TCP:host.docker.internal:{engine_port}",
    ]
    result = run_docker(run_args, binary=binary, timeout=60)
    if result.returncode != 0:
        teardown_sandbox(sid, binary=binary)
        raise SandboxError(f"failed to start egress proxy for '{sid}': {result.stderr.strip()}")

    connect = run_docker(
        ["network", "connect", "--alias", PROXY_ALIAS, sid, f"{sid}-proxy"],
        binary=binary,
        timeout=30,
    )
    if connect.returncode != 0:
        teardown_sandbox(sid, binary=binary)
        raise SandboxError(
            f"failed to attach egress proxy to sandbox network '{sid}': {connect.stderr.strip()}"
        )

    for _ in range(_PROXY_READY_TRIES):
        probe = run_docker(
            ["inspect", "-f", "{{.State.Running}}", f"{sid}-proxy"],
            binary=binary,
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return
        time.sleep(_PROXY_READY_DELAY)
    teardown_sandbox(sid, binary=binary)
    raise SandboxError(f"egress proxy for '{sid}' never reached running state")


def run_sandboxed(
    sid: str,
    *,
    harness_name: str,
    image: str,
    argv: list[str],
    mounts: list[str],
    env: Mapping[str, str],
    network: str,
    memory_bytes: int | None = None,
    workdir: str | None = None,
    timeout: float | None = None,
    binary: str = "docker",
) -> subprocess.CompletedProcess[str]:
    """Run the agent container attached, returning its CompletedProcess.

    ``env`` is the container's *entire* environment — callers pass an explicit
    dict, never ``os.environ`` (env hygiene is the point of the sandbox).
    ``network`` is the sandbox network name, or ``"none"`` for full isolation.
    On timeout the container is force-removed and :class:`SandboxTimeout`
    carries whatever stdout was captured.
    """
    args = ["run", "--rm", "--name", f"{sid}-agent", *_labels(sid, harness_name)]
    args += ["--network", network]
    for key, val in env.items():
        args += ["-e", f"{key}={val}"]
    for mount in mounts:
        args += ["-v", mount]
    if memory_bytes is not None:
        args += ["--memory", str(memory_bytes)]
    if workdir is not None:
        args += ["-w", workdir]
    args.append(image)
    args += argv
    try:
        return run_docker(args, binary=binary, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        run_docker(["rm", "-f", f"{sid}-agent"], binary=binary, timeout=30)
        raw = exc.stdout
        partial = raw.decode(errors="replace") if isinstance(raw, bytes) else (raw or "")
        raise SandboxTimeout(
            f"sandboxed agent '{sid}' exceeded {timeout}s", partial_stdout=partial
        ) from exc


def teardown_sandbox(sid: str, *, binary: str = "docker") -> None:
    """Remove one sandbox's containers and network. Never raises."""
    for name in (f"{sid}-agent", f"{sid}-proxy"):
        try:
            run_docker(["rm", "-f", name], binary=binary, timeout=30)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - best effort
            pass
    try:
        run_docker(["network", "rm", sid], binary=binary, timeout=30)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - best effort
        pass


def sweep_leftovers(*, binary: str = "docker", harness: str | None = None) -> int:
    """Reap labeled sandbox containers/networks left by crashed runs.

    Scoped to one harness when ``harness`` is given (pre-invoke hygiene);
    global otherwise (``sovereign down``). Never raises; returns how many
    objects were removed.
    """
    filters = ["--filter", f"label={SANDBOX_LABEL}"]
    if harness is not None:
        filters += ["--filter", f"label={HARNESS_LABEL}={harness}"]
    removed = 0
    try:
        ps = run_docker(["ps", "-aq", *filters], binary=binary, timeout=30)
        for cid in ps.stdout.split():
            run_docker(["rm", "-f", cid], binary=binary, timeout=30)
            removed += 1
        nets = run_docker(["network", "ls", "-q", *filters], binary=binary, timeout=30)
        for nid in nets.stdout.split():
            run_docker(["network", "rm", nid], binary=binary, timeout=30)
            removed += 1
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - best effort
        return removed
    return removed


def daemon_reachable(*, binary: str = "docker", timeout: float = 10) -> bool:
    """True iff the Docker *server* answers (mirrors DockerManager's probe)."""
    try:
        result = run_docker(
            ["version", "--format", "{{.Server.Version}}"], binary=binary, timeout=timeout
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def ensure_image(
    tag: str,
    *,
    context_dir: Path | None = None,
    binary: str = "docker",
) -> None:
    """Make ``tag`` locally available: no-op if present, else build or pull.

    A ``context_dir`` means the image is Sovereign-owned (built from a package
    Dockerfile); without one the tag is pulled like any user-supplied image.
    """
    probe = run_docker(["image", "inspect", tag], binary=binary, timeout=30)
    if probe.returncode == 0:
        return
    if context_dir is not None:
        logger.info("building sandbox image %s from %s", tag, context_dir)
        result = run_docker(
            ["build", "-t", tag, str(context_dir)], binary=binary, timeout=1800
        )
        if result.returncode != 0:
            raise SandboxError(f"docker build failed for {tag}: {result.stderr.strip()}")
        return
    stream_pull(tag, binary=binary, on_progress=lambda line: logger.debug("%s", line))
