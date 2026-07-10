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
# How far apart two create_time readings of the *same* process may plausibly be
# (clock granularity / float rounding across psutil versions).
_CREATE_TIME_TOLERANCE = 1.0


def _is_recorded_process(proc: psutil.Process, handle: dict) -> bool:
    """Whether ``proc`` is the exact process the handle recorded.

    PIDs are recycled: verify the recorded ``create_time`` (±1s) before
    signalling, so `down` never kills a stranger that inherited the PID.
    Handles written by older versions lack ``create_time`` — those keep the
    previous behavior (PID alone), preserving backward compatibility.
    """
    recorded = handle.get("create_time")
    if recorded is None:
        return True
    return abs(proc.create_time() - float(recorded)) <= _CREATE_TIME_TOLERANCE


def stop_service_handle(handle: dict, *, docker_binary: str = "docker") -> str:
    """Stop a single service from its persisted handle; returns a status word."""
    kind = handle.get("kind")

    if kind == "native":
        pid = handle.get("pid")
        try:
            proc = psutil.Process(pid)
            if not _is_recorded_process(proc, handle):
                return "already stopped"  # PID recycled by another process
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
