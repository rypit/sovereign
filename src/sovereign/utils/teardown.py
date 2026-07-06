"""Cross-process teardown from persisted runtime handles (used by `sovereign down`).

Without a daemon transport (that's Phase 9), `down` can't reach an in-process
`up`. Instead it reads the runtime handles the Orchestrator recorded in
``state.json`` — a PID for native processes, a container name for Docker services —
and stops each directly.
"""

from __future__ import annotations

import subprocess

import psutil

_STOP_TIMEOUT = 10.0


def stop_service_handle(handle: dict, *, docker_binary: str = "docker") -> str:
    """Stop a single service from its persisted handle; returns a status word."""
    kind = handle.get("kind")

    if kind == "native":
        pid = handle.get("pid")
        try:
            proc = psutil.Process(pid)
            proc.terminate()  # SIGTERM so caches flush (§6.4)
            try:
                proc.wait(timeout=_STOP_TIMEOUT)
            except psutil.TimeoutExpired:
                proc.kill()
            return "stopped"
        except psutil.NoSuchProcess:
            return "already stopped"

    if kind == "docker":
        subprocess.run(  # noqa: S603 - fixed argv
            [docker_binary, "rm", "-f", handle["container"]],
            capture_output=True,
            text=True,
        )
        return "removed"

    return "unknown handle"
