# 0004. Embedded engine workers over external CLIs

Status: accepted

## Context

Sovereign originally booted native inference engines by shelling out to
external CLIs (`llama-server`, `mlx_lm.server`) via `subprocess.Popen(argv)`.
That coupled Sovereign to each CLI's flag surface, limited observability to
log tailing and psutil polling, and offered no structured telemetry
(prefill progress, tokens/sec) short of parsing logs.

## Decision

Replace the CLI-handoff path with detached, embedded Python engine worker
processes. Each engine runs inside its own process
(`python -m sovereign.workers.engine_worker --config <path>`), spawned with
`subprocess.Popen([sys.executable, ...], start_new_session=True)` — real
spawn semantics (fresh interpreter, nothing inherited), not
`multiprocessing.Process` (whose anonymous queues die with the parent and
can't be re-attached by a later `sovereign monitor` process). The worker
loads the model via the engine's Python bindings (`llama-cpp-python` /
`mlx_lm`), serves an OpenAI-compatible HTTP API in-process, and streams
typed telemetry to the parent over a unix domain socket. Workers are
detached: they outlive `up`, their PIDs are recorded in `state.json`, and
`sovereign down` / `runtime/teardown.py` operate unchanged. The control
plane (the `up`/`down`/`status` process) never loads tensors — a segfault or
leak in engine code kills one worker, never the orchestrator.

## Consequences

- Structured, first-class telemetry (see ADR 0005) instead of log scraping.
- A crash in engine-binding code is isolated to its worker; the control
  plane and other services are unaffected.
- Sovereign owns the config handoff (`WorkerConfig` JSON,
  `workers/worker_config.py`) instead of reconstructing a CLI's flag
  grammar — one mapping function (`engine_kwargs()`) per engine instead of
  an argv builder.
- Cost: real feature gaps where the bindings don't match the CLI server
  1:1 (see ADR 0006) — these are surfaced loudly, not silently absorbed.
- Engine bindings (`llama_cpp`, `mlx_lm`) are imported only inside
  `adapter.run()`, never at module scope, so worker modules stay importable
  (and discoverable/testable) on any platform.

## Alternatives considered

- Keep external CLI subprocesses, add a log-tailing telemetry shim —
  rejected: still coupled to each CLI's flag/log format, and prefill/tok-s
  data isn't reliably present in CLI logs.
- `multiprocessing.Process` — rejected per the spawn-axiom resolution above:
  its IPC handles don't survive being re-attached by a separate `monitor`
  invocation, which per ADR 0001 must work from a fresh process.

---
Provenance: PR #20 (Embedded Python Engine Workers + Multiplexed Telemetry).
