"""Tests for the parent-side telemetry cache and unix-socket ingest hub."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest

from sovereign.runtime.telemetry import (
    DockerMonitorWorker,
    TelemetryHub,
    TelemetryHubAlreadyOwned,
    TelemetryStateCache,
)
from sovereign.workers.protocol import EventType, TelemetryEvent, encode_event


def _send(sock_path: Path, events: list[TelemetryEvent]) -> None:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(sock_path))
    try:
        for event in events:
            client.sendall(encode_event(event))
        client.shutdown(socket.SHUT_WR)
    finally:
        client.close()


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_hub_ingests_events_into_cache(socket_path: Path) -> None:
    sock_path = socket_path
    cache = TelemetryStateCache()
    hub = TelemetryHub(sock_path, cache)
    hub.start()
    try:
        _send(
            sock_path,
            [
                TelemetryEvent(
                    1, time.time(), "svc", EventType.STATE_CHANGE, 1, {"state": "serving"}
                ),
                TelemetryEvent(1, time.time(), "svc", EventType.HEARTBEAT, 2, {}),
                TelemetryEvent(
                    1, time.time(), "svc", EventType.MEMORY, 3, {"memory_bytes": 1024}
                ),
            ],
        )
        _wait_until(lambda: cache.snapshot("svc")["worker_state"] == "serving")
        snap = cache.snapshot("svc")
        assert snap["last_heartbeat"] is not None
        _wait_until(lambda: cache.fresh_memory_bytes("svc", 5.0) == 1024)
    finally:
        hub.stop()


def test_cache_memory_window_is_bounded() -> None:
    cache = TelemetryStateCache()
    for i in range(1000):
        cache.apply_local("svc", EventType.MEMORY, {"memory_bytes": i}, ts=time.time())
    cache.snapshot("svc")
    with cache._lock:  # noqa: SLF001 - whitebox check of the bound itself
        assert len(cache._services["svc"].memory) <= 300


def test_cache_prefill_ttl_prunes_stale_entries() -> None:
    cache = TelemetryStateCache()
    old_ts = time.time() - 60
    cache.apply_local(
        "svc",
        EventType.PREFILL_PROGRESS,
        {"request_id": "r1", "processed": 1, "total": 10},
        ts=old_ts,
    )
    cache.apply_local(
        "svc", EventType.PREFILL_PROGRESS, {"request_id": "r2", "processed": 2, "total": 10}
    )
    snap = cache.snapshot("svc")
    ids = {p["request_id"] for p in snap["prefill"]}
    assert ids == {"r2"}


def test_cache_prefill_max_entries_bounded() -> None:
    cache = TelemetryStateCache()
    for i in range(64):
        cache.apply_local(
            "svc",
            EventType.PREFILL_PROGRESS,
            {"request_id": f"r{i}", "processed": 1, "total": 10},
        )
    snap = cache.snapshot("svc")
    assert len(snap["prefill"]) <= 32


def test_hub_stale_socket_is_reclaimed(socket_path: Path) -> None:
    sock_path = socket_path
    # Simulate a stale file left by a dead process: bind+close without unlinking.
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(sock_path))
    dead.close()
    assert sock_path.exists()

    cache = TelemetryStateCache()
    hub = TelemetryHub(sock_path, cache)
    hub.start()
    try:
        assert sock_path.exists()
    finally:
        hub.stop()


def test_hub_refuses_to_steal_live_socket(socket_path: Path) -> None:
    sock_path = socket_path
    cache1 = TelemetryStateCache()
    hub1 = TelemetryHub(sock_path, cache1)
    hub1.start()
    try:
        cache2 = TelemetryStateCache()
        hub2 = TelemetryHub(sock_path, cache2)
        with pytest.raises(TelemetryHubAlreadyOwned):
            hub2.start()
    finally:
        hub1.stop()


def test_docker_monitor_worker_feeds_cache(monkeypatch) -> None:
    calls = []

    def fake_container_metrics(container, *, binary="docker"):
        calls.append((container, binary))
        return {"memory_bytes": 4096, "status": "running"}

    monkeypatch.setattr(
        "sovereign.services.docker.manager.container_metrics", fake_container_metrics
    )

    cache = TelemetryStateCache()
    worker = DockerMonitorWorker([("svc", "svc-container", "docker")], cache, interval=0.05)
    worker.start()
    try:
        _wait_until(lambda: cache.fresh_memory_bytes("svc", 5.0) == 4096)
        assert calls and calls[0] == ("svc-container", "docker")
    finally:
        worker.stop()


def test_docker_monitor_worker_skips_stopped_containers(monkeypatch) -> None:
    def fake_container_metrics(container, *, binary="docker"):
        return {"status": "stopped"}

    monkeypatch.setattr(
        "sovereign.services.docker.manager.container_metrics", fake_container_metrics
    )

    cache = TelemetryStateCache()
    worker = DockerMonitorWorker([("svc", "svc-container", "docker")], cache, interval=0.05)
    worker.start()
    try:
        time.sleep(0.2)
        assert cache.fresh_memory_bytes("svc", 5.0) is None
    finally:
        worker.stop()
