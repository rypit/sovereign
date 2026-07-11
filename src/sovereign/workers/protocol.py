"""Telemetry wire schema shared by workers (senders) and the parent hub (receiver).

NDJSON over a unix domain socket: one ``TelemetryEvent`` per line. Stdlib-only
so it stays importable everywhere — workers, the orchestrator, and hermetic
tests alike — without pulling in any engine bindings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    """The fixed vocabulary of telemetry events a worker may emit."""

    HEARTBEAT = "heartbeat"
    LOG = "log"
    STATE_CHANGE = "state_change"
    MEMORY = "memory"
    PREFILL_PROGRESS = "prefill_progress"
    GENERATION_STATS = "generation_stats"
    DOCKER_STATS = "docker_stats"


#: Wire protocol version. Bump if the envelope shape changes incompatibly.
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class TelemetryEvent:
    """One telemetry envelope: ``{v, ts, service, event, seq, payload}``.

    ``ts`` is a Unix epoch float (seconds); ``seq`` is a per-sender monotonic
    counter used to detect gaps/reordering, not to dedupe.
    """

    v: int
    ts: float
    service: str
    event: EventType
    seq: int
    payload: dict[str, Any]


def encode_event(event: TelemetryEvent) -> bytes:
    """Serialize a ``TelemetryEvent`` to one NDJSON line (including the trailing newline)."""
    data = asdict(event)
    data["event"] = str(event.event)
    return (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")


def decode_line(line: bytes | str) -> TelemetryEvent | None:
    """Parse one NDJSON line into a ``TelemetryEvent``.

    Tolerant by design — telemetry is best-effort observability, not a
    protocol a malformed/truncated line should be allowed to crash. Any
    parsing problem (bad JSON, missing/mistyped fields, unknown event name)
    returns None instead of raising.
    """
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            return None
        data = json.loads(line)
        if not isinstance(data, dict):
            return None
        return TelemetryEvent(
            v=int(data["v"]),
            ts=float(data["ts"]),
            service=str(data["service"]),
            event=EventType(data["event"]),
            seq=int(data["seq"]),
            payload=dict(data.get("payload") or {}),
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
