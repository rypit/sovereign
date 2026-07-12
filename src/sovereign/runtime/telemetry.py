"""Parent-side telemetry: bounded state cache + the unix-socket ingest hub.

Ingest (an uncapped, background accept-loop feeding ``TelemetryStateCache``)
is decoupled from render (whatever polls ``snapshot()`` at its own cadence —
today the dashboard/status-snapshot's 1 Hz tick). The cache is bounded per
service (sliding-window deques + a small TTL-pruned prefill dict) so parent
memory stays flat regardless of how chatty workers get.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from sovereign.workers.protocol import EventType, TelemetryEvent, decode_line

logger = logging.getLogger("sovereign")

#: Sliding-window length for memory/tps history (~5 minutes at a 1s cadence).
_HISTORY_MAXLEN = 300
#: Ring buffer of recent log lines kept per service.
_LOG_MAXLEN = 200
#: Prefill entries older than this are pruned as stale/abandoned requests.
_PREFILL_TTL = 30.0
#: Cap on concurrently tracked in-flight prefill requests per service.
_PREFILL_MAXLEN = 32


class _ServiceState:
    """Mutable per-service telemetry state. Guarded by the cache's single lock."""

    __slots__ = (
        "last_heartbeat_ts",
        "worker_state",
        "memory",
        "tps",
        "logs",
        "active_prefill",
        "last_generation_stats",
    )

    def __init__(self) -> None:
        self.last_heartbeat_ts: float | None = None
        self.worker_state: str | None = None
        self.memory: deque[tuple[float, int]] = deque(maxlen=_HISTORY_MAXLEN)
        self.tps: deque[tuple[float, float]] = deque(maxlen=_HISTORY_MAXLEN)
        self.logs: deque[str] = deque(maxlen=_LOG_MAXLEN)
        self.active_prefill: dict[str, tuple[int, int | None, float]] = {}
        self.last_generation_stats: dict[str, Any] | None = None


class TelemetryStateCache:
    """Thread-safe, memory-bounded aggregate of the latest telemetry per service.

    One global lock protects all state — updates are simple dict/deque
    mutations, so contention is not a concern versus the correctness/
    simplicity win of a single critical section.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._services: dict[str, _ServiceState] = {}

    def _state_for(self, service: str) -> _ServiceState:
        state = self._services.get(service)
        if state is None:
            state = _ServiceState()
            self._services[service] = state
        return state

    def apply(self, event: TelemetryEvent) -> None:
        """Fold a decoded ``TelemetryEvent`` from the wire into the cache."""
        self.apply_local(event.service, event.event, event.payload, ts=event.ts)

    def apply_local(
        self,
        service: str,
        event: EventType,
        payload: dict[str, Any],
        ts: float | None = None,
    ) -> None:
        """Fold an event into the cache without going through the wire.

        Used both by the hub (after decoding) and by in-process sources like
        ``DockerMonitorWorker`` that never touch the socket.
        """
        now = ts if ts is not None else time.time()
        with self._lock:
            state = self._state_for(service)
            self._prune_prefill(state, now)
            if event == EventType.HEARTBEAT:
                state.last_heartbeat_ts = now
            elif event == EventType.STATE_CHANGE:
                state.worker_state = payload.get("state")
            elif event == EventType.LOG:
                message = payload.get("message")
                if message:
                    level = payload.get("level", "info")
                    state.logs.append(f"[{level}] {message}")
            elif event == EventType.MEMORY:
                memory_bytes = payload.get("memory_bytes")
                if isinstance(memory_bytes, int | float):
                    state.memory.append((now, int(memory_bytes)))
            elif event == EventType.DOCKER_STATS:
                memory_bytes = payload.get("memory_bytes")
                if isinstance(memory_bytes, int | float):
                    state.memory.append((now, int(memory_bytes)))
            elif event == EventType.PREFILL_PROGRESS:
                request_id = str(payload.get("request_id", ""))
                processed = int(payload.get("processed", 0))
                total = payload.get("total")
                total_int = int(total) if isinstance(total, int | float) else None
                if (
                    request_id not in state.active_prefill
                    and len(state.active_prefill) >= _PREFILL_MAXLEN
                ):
                    # Drop the oldest tracked request to make room — a
                    # bounded cache must never grow past its cap.
                    oldest = min(state.active_prefill, key=lambda k: state.active_prefill[k][2])
                    state.active_prefill.pop(oldest, None)
                state.active_prefill[request_id] = (processed, total_int, now)
            elif event == EventType.GENERATION_STATS:
                state.last_generation_stats = dict(payload)
                gen_tps = payload.get("generation_tps")
                if isinstance(gen_tps, int | float):
                    state.tps.append((now, float(gen_tps)))
                request_id = str(payload.get("request_id", ""))
                state.active_prefill.pop(request_id, None)

    def _prune_prefill(self, state: _ServiceState, now: float) -> None:
        stale = [
            rid for rid, (_, _, ts) in state.active_prefill.items() if now - ts > _PREFILL_TTL
        ]
        for rid in stale:
            state.active_prefill.pop(rid, None)

    def snapshot(self, service: str) -> dict[str, Any]:
        """A plain-dict snapshot for one service, shaped like the plan's
        ``TelemetryStatus`` (worker_state, last_heartbeat, prefill[],
        generation_tps, prompt_tps, tps_history).
        """
        with self._lock:
            state = self._services.get(service)
            if state is None:
                return {
                    "worker_state": None,
                    "last_heartbeat": None,
                    "prefill": [],
                    "generation_tps": None,
                    "prompt_tps": None,
                    "tps_history": [],
                }
            now = time.time()
            self._prune_prefill(state, now)
            prefill = [
                {"request_id": rid, "processed": processed, "total": total}
                for rid, (processed, total, _) in state.active_prefill.items()
            ]
            last_stats = state.last_generation_stats or {}
            return {
                "worker_state": state.worker_state,
                "last_heartbeat": state.last_heartbeat_ts,
                "prefill": prefill,
                "generation_tps": last_stats.get("generation_tps"),
                "prompt_tps": last_stats.get("prompt_tps"),
                "tps_history": list(state.tps),
            }

    def fresh_memory_bytes(self, service: str, max_age: float) -> int | None:
        """The most recent memory sample for ``service`` if newer than ``max_age``."""
        with self._lock:
            state = self._services.get(service)
            if state is None or not state.memory:
                return None
            ts, value = state.memory[-1]
            if time.time() - ts > max_age:
                return None
            return value

    def has_fresh(self, service: str, event_type: EventType, max_age: float) -> bool:
        """Whether ``service`` has telemetry for ``event_type`` newer than ``max_age``.

        Used by the orchestrator's metrics-freshness merge: memory/docker_stats
        in the cache win over the psutil/docker-stats fallback poll when fresh.
        """
        with self._lock:
            state = self._services.get(service)
            if state is None:
                return False
            now = time.time()
            if event_type in (EventType.MEMORY, EventType.DOCKER_STATS):
                return bool(state.memory) and now - state.memory[-1][0] <= max_age
            if event_type == EventType.HEARTBEAT:
                return (
                    state.last_heartbeat_ts is not None
                    and now - state.last_heartbeat_ts <= max_age
                )
            return False


class TelemetryHubAlreadyOwned(RuntimeError):
    """Raised by ``TelemetryHub.start()`` when another process already owns the socket."""


class TelemetryHub:
    """Binds the telemetry unix socket and ingests worker events into a cache.

    Handles the classic UDS stale-file problem: if the socket path already
    exists, probe-connect first — a refused connection means the previous
    owner died without cleaning up (safe to unlink and rebind); a successful
    connection means another live process is already listening, so ``start()``
    raises :class:`TelemetryHubAlreadyOwned` rather than silently stealing
    ingest from it.
    """

    def __init__(self, socket_path: str | Path, cache: TelemetryStateCache) -> None:
        self._socket_path = Path(socket_path)
        self._cache = cache
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._reader_threads: list[threading.Thread] = []

    def start(self) -> None:
        """Bind the socket and begin accepting connections. Idempotent per instance."""
        self._reclaim_stale_socket()
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # AF_UNIX sun_path is capped (~104 bytes on macOS, 108 on linux) —
            # a stack rooted deep in the filesystem can make this bind fail
            # with "AF_UNIX path too long"; the orchestrator degrades to
            # running without live telemetry in that case.
            listener.bind(str(self._socket_path))
        except OSError:
            listener.close()
            raise
        listener.listen(64)
        listener.settimeout(0.5)
        self._listener = listener
        self._stop.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="telemetry-hub-accept", daemon=True
        )
        self._accept_thread.start()

    def _reclaim_stale_socket(self) -> None:
        if not self._socket_path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.5)
            probe.connect(str(self._socket_path))
        except OSError:
            # Refused/no-listener: a stale file left behind by a dead owner.
            self._socket_path.unlink(missing_ok=True)
            return
        else:
            raise TelemetryHubAlreadyOwned(
                f"telemetry socket {self._socket_path} is already owned by a live process"
            )
        finally:
            probe.close()

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            reader = threading.Thread(
                target=self._reader_loop, args=(conn,), name="telemetry-hub-reader", daemon=True
            )
            self._reader_threads.append(reader)
            reader.start()

    def _reader_loop(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    event = decode_line(line)
                    if event is not None:
                        self._cache.apply(event)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        """Stop accepting, close the listener, and unlink the socket file."""
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None
        self._socket_path.unlink(missing_ok=True)


class DockerMonitorWorker:
    """Polls ``docker stats`` for each Docker service and feeds the telemetry cache.

    Runs on its own daemon thread at ``interval`` seconds so Docker container
    metrics show up in the same cache (and eventually the same dashboard
    columns) as native-engine worker telemetry, without the Docker manager
    needing to know about telemetry at all.
    """

    def __init__(
        self,
        services: list[tuple[str, str, str]],
        cache: TelemetryStateCache,
        interval: float = 2.0,
    ) -> None:
        """``services`` is a list of ``(service_name, container_name, docker_binary)``."""
        self._services = services
        self._cache = cache
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="docker-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2.0)
            self._thread = None

    def _run(self) -> None:
        # Imported lazily so this module never forces a docker-manager import
        # at package load time for callers that don't use Docker.
        from sovereign.services.docker.manager import container_metrics

        while not self._stop.is_set():
            for service_name, container_name, docker_binary in self._services:
                try:
                    metrics = container_metrics(container_name, binary=docker_binary)
                except Exception:  # noqa: BLE001 - one bad service must not kill the poller
                    logger.debug("docker_stats poll failed for %s", service_name, exc_info=True)
                    continue
                if metrics.get("status") != "running":
                    continue
                payload: dict[str, Any] = {}
                memory_bytes = metrics.get("memory_bytes")
                if memory_bytes is not None:
                    payload["memory_bytes"] = memory_bytes
                cpu_percent = metrics.get("cpu_percent")
                if cpu_percent is not None:
                    payload["cpu_percent"] = cpu_percent
                self._cache.apply_local(service_name, EventType.DOCKER_STATS, payload)
            if self._stop.wait(self._interval):
                return
