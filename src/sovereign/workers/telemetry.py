"""Worker-side telemetry client: a non-blocking UDS sender with reconnect+drop.

Telemetry is best-effort observability, never a correctness dependency — an
engine worker must never stall or crash because nobody is listening on the
socket. ``TelemetryClient.emit()`` is safe to call from hot paths (a
streaming-generation loop): it enqueues onto a small bounded queue and returns
immediately, dropping the event if the queue is full or under active source
rate-limiting. A single daemon thread owns the actual socket and handles
connect/send/backoff.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any

from sovereign.workers.protocol import EventType, TelemetryEvent, encode_event

logger = logging.getLogger("sovereign")

#: Bound on the outgoing queue — a worker experiencing a telemetry outage drops
#: rather than growing unbounded or blocking its serving loop.
_QUEUE_MAXSIZE = 256
#: Short enough that a dead/unresponsive listener is noticed quickly without
#: stalling the sender thread for long.
_CONNECT_TIMEOUT = 1.0
#: Source-side rate limits (§2): prefill progress is chatty per in-flight
#: request; memory/heartbeat are coalesced so a slow listener isn't flooded.
_PREFILL_MIN_INTERVAL = 1.0 / 10  # <=10/s per request_id
_COALESCE_MIN_INTERVAL = 2.0  # memory + heartbeat


class TelemetryClient:
    """Non-blocking telemetry sender for one worker process.

    ``emit()`` never blocks and never raises. A daemon thread connects to
    ``socket_path`` (SOCK_STREAM), sends queued events, and on any
    connect/send failure drops whatever is currently queued and backs off
    exponentially before retrying — a full kernel send buffer counts as a
    disconnect.
    """

    def __init__(
        self,
        socket_path: str | Path,
        service: str,
        reconnect_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._socket_path = str(socket_path)
        self._service = service
        self._initial_backoff = reconnect_backoff
        self._max_backoff = max_backoff

        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._seq_lock = threading.Lock()
        self._seq = 0

        self._rate_lock = threading.Lock()
        self._last_prefill: dict[str, float] = {}
        self._last_coalesced: dict[EventType, float] = {}

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"telemetry-{service}", daemon=True
        )
        self._thread.start()

    def emit(self, event: EventType, payload: dict[str, Any]) -> None:
        """Enqueue a telemetry event. Never blocks, never raises.

        Applies source-side rate limiting before encoding: ``prefill_progress``
        is limited per ``request_id`` (>=10/s), and ``memory``/``heartbeat``
        are coalesced to at most one every 2s each.
        """
        try:
            if self._should_drop_for_rate_limit(event, payload):
                return
            with self._seq_lock:
                self._seq += 1
                seq = self._seq
            envelope = TelemetryEvent(
                v=1,
                ts=time.time(),
                service=self._service,
                event=event,
                seq=seq,
                payload=payload,
            )
            line = encode_event(envelope)
            self._queue.put_nowait(line)
        except queue.Full:
            logger.debug("telemetry queue full for %s, dropping %s", self._service, event)
        except Exception:  # noqa: BLE001 - emit must never raise
            logger.debug("telemetry emit failed for %s", self._service, exc_info=True)

    def _should_drop_for_rate_limit(self, event: EventType, payload: dict[str, Any]) -> bool:
        now = time.time()
        with self._rate_lock:
            if event == EventType.PREFILL_PROGRESS:
                request_id = str(payload.get("request_id", ""))
                last = self._last_prefill.get(request_id, 0.0)
                if now - last < _PREFILL_MIN_INTERVAL:
                    return True
                self._last_prefill[request_id] = now
                return False
            if event in (EventType.MEMORY, EventType.HEARTBEAT):
                last = self._last_coalesced.get(event, 0.0)
                if now - last < _COALESCE_MIN_INTERVAL:
                    return True
                self._last_coalesced[event] = now
                return False
            return False

    def close(self) -> None:
        """Stop the sender thread promptly, dropping anything still queued."""
        self._stop.set()
        self._thread.join(timeout=_CONNECT_TIMEOUT + 1.0)

    def _run(self) -> None:
        backoff = self._initial_backoff
        while not self._stop.is_set():
            sock = self._connect()
            if sock is None:
                self._drain_queue()
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, self._max_backoff)
                continue
            backoff = self._initial_backoff
            try:
                self._send_loop(sock)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    def _connect(self) -> socket.socket | None:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(_CONNECT_TIMEOUT)
            sock.connect(self._socket_path)
            return sock
        except OSError:
            return None

    def _send_loop(self, sock: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                line = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                sock.sendall(line)
            except OSError:
                # Full kernel buffer / broken pipe counts as a disconnect:
                # drop what's left queued and let _run() reconnect+backoff.
                return

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
