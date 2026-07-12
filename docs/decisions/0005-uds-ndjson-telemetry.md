# 0005. UDS NDJSON telemetry, drop-when-unobserved, bounded parent cache

Status: accepted

## Context

Embedded engine workers (ADR 0004) need to report structured live state —
heartbeats, load-state transitions, prefill progress, token throughput,
memory — to the parent process that renders the dashboard and answers
`sovereign status`. This is observability, not an audit log: nobody needs
to replay it later, and losing a few events while nothing is watching is
correct behavior, not data loss.

## Decision

Workers connect to `.sovereign/telemetry.sock` (`SOCK_STREAM`) and write
newline-delimited JSON events (`protocol.py`:
`{"v":1,"ts":…,"service":…,"event":…,"seq":…,"payload":{…}}`). The
worker-side `TelemetryClient` never blocks or raises on `emit()` — a bounded
queue (`maxsize=256`) with drop-on-full feeds one daemon sender thread that
reconnects with backoff; a full kernel send buffer counts as "disconnected"
and events are dropped, not queued unboundedly. The parent-side
`TelemetryHub` binds the socket, accepts connections, and ingests
**uncapped** — ingest and render are fully decoupled, so a burst can't stall
producers. The `TelemetryStateCache` bounds memory instead: fixed-size
deques per service (memory/tps history, logs), a TTL+max-size map for
in-flight prefills. Total parent overhead per service stays under 100 KB,
keeping the process inside its 30–60 MB budget. Rate limits apply at the
source too (prefill progress ≤10 events/s/request; heartbeat/memory every
2 s).

## Consequences

- A crashed or absent dashboard/monitor never backs up a worker or causes it
  to stall serving requests.
- Bounded, predictable parent memory regardless of how long `up` runs.
- `sovereign monitor` (a separate process) can't bind the same hub while
  `up` holds it — it falls back to `status.json` polling (1–2 s staleness,
  documented, acceptable).
- Cost: telemetry is best-effort. It is not an audit trail and must never be
  treated as one — a dropped event is invisible by design.

## Alternatives considered

- Per-worker NDJSON files rotated on disk — rejected: needs rotation and
  tail-race handling for no benefit, since nothing needs history replay;
  burns disk for observability nobody reads later.
- A blocking socket write from the worker's request-handling thread —
  rejected: would let a slow/absent observer stall inference itself, which
  is the one thing telemetry must never do.

---
Provenance: PR #20 (Embedded Python Engine Workers + Multiplexed Telemetry), §2 "Telemetry transport".
