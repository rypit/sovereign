"""Round-trip and tolerance tests for the telemetry wire schema."""

from __future__ import annotations

from sovereign.workers.protocol import EventType, TelemetryEvent, decode_line, encode_event


def test_round_trip_encode_decode() -> None:
    event = TelemetryEvent(
        v=1,
        ts=1234.5,
        service="llm",
        event=EventType.HEARTBEAT,
        seq=7,
        payload={"ok": True},
    )
    line = encode_event(event)
    assert line.endswith(b"\n")
    decoded = decode_line(line)
    assert decoded == event


def test_round_trip_with_str_input() -> None:
    event = TelemetryEvent(
        v=1,
        ts=1.0,
        service="svc",
        event=EventType.MEMORY,
        seq=1,
        payload={"memory_bytes": 123},
    )
    line = encode_event(event).decode("utf-8")
    decoded = decode_line(line)
    assert decoded == event


def test_decode_garbage_returns_none() -> None:
    assert decode_line(b"not json at all {{{") is None
    assert decode_line("") is None
    assert decode_line(b"\n") is None
    assert decode_line(b"[1, 2, 3]") is None


def test_decode_missing_fields_returns_none() -> None:
    assert decode_line(b'{"v": 1, "ts": 1.0}') is None


def test_decode_unknown_event_returns_none() -> None:
    assert (
        decode_line(b'{"v":1,"ts":1.0,"service":"x","event":"bogus","seq":1,"payload":{}}')
        is None
    )


def test_decode_missing_payload_defaults_empty() -> None:
    decoded = decode_line(b'{"v":1,"ts":1.0,"service":"x","event":"heartbeat","seq":1}')
    assert decoded is not None
    assert decoded.payload == {}
