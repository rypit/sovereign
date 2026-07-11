"""Tests for the worker-side non-blocking telemetry client."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from sovereign.workers.protocol import EventType, decode_line
from sovereign.workers.telemetry import TelemetryClient


class _Listener:
    """A minimal in-thread UDS listener that collects decoded lines."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.events: list = []
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(socket_path))
        self._sock.listen(1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._sock.settimeout(0.5)
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        conn.settimeout(0.5)
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                decoded = decode_line(line)
                if decoded is not None:
                    self.events.append(decoded)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


def test_emit_delivers_to_listener(tmp_path: Path) -> None:
    sock_path = tmp_path / "telemetry.sock"
    listener = _Listener(sock_path)
    client = TelemetryClient(sock_path, "svc", reconnect_backoff=0.05, max_backoff=0.1)
    try:
        client.emit(EventType.LOG, {"level": "info", "message": "hello"})
        deadline = time.time() + 3.0
        while time.time() < deadline and not listener.events:
            time.sleep(0.05)
        assert len(listener.events) == 1
        assert listener.events[0].event == EventType.LOG
        assert listener.events[0].payload == {"level": "info", "message": "hello"}
        assert listener.events[0].service == "svc"
    finally:
        client.close()
        listener.close()


def test_emit_with_no_listener_is_fast_and_never_raises(tmp_path: Path) -> None:
    sock_path = tmp_path / "nonexistent.sock"
    client = TelemetryClient(sock_path, "svc", reconnect_backoff=1.0, max_backoff=5.0)
    try:
        start = time.time()
        for i in range(100):
            client.emit(EventType.HEARTBEAT, {"i": i})
        elapsed = time.time() - start
        assert elapsed < 0.1
    finally:
        client.close()


def test_late_listener_gets_reconnect_delivery(tmp_path: Path) -> None:
    sock_path = tmp_path / "telemetry.sock"
    client = TelemetryClient(sock_path, "svc", reconnect_backoff=0.05, max_backoff=0.1)
    try:
        # No listener yet — these get dropped, which is expected/fine.
        client.emit(EventType.HEARTBEAT, {})
        time.sleep(0.2)

        listener = _Listener(sock_path)
        try:
            client.emit(EventType.LOG, {"level": "info", "message": "after reconnect"})
            deadline = time.time() + 3.0
            while time.time() < deadline and not listener.events:
                time.sleep(0.05)
            assert len(listener.events) >= 1
            assert listener.events[-1].payload["message"] == "after reconnect"
        finally:
            listener.close()
    finally:
        client.close()


def test_close_joins_promptly(tmp_path: Path) -> None:
    sock_path = tmp_path / "nope.sock"
    client = TelemetryClient(sock_path, "svc", reconnect_backoff=0.05, max_backoff=0.1)
    client.emit(EventType.HEARTBEAT, {})
    start = time.time()
    client.close()
    assert time.time() - start < 3.0
    assert not client._thread.is_alive()
