# 0007. llama_cpp via `llama-server`, telemetry over its HTTP surface

Status: accepted

Supersedes 0004 for the `llama_cpp` engine only (mlx_lm is unchanged).

## Context

ADR 0004 replaced external inference CLIs with embedded Python-binding
workers, motivated by **structured telemetry** — it rejected CLIs because
observability was "limited to log tailing and psutil" and the only CLI
alternative on the table was a *log-tailing telemetry shim*. For
`llama-cpp-python` this trade cost real capability: the bundled server holds a
single `Llama` behind a lock, so `max_parallel` (`-np`) multi-slot batching
has no equivalent (ADR 0006's documented gap), and two-model speculative
decoding had to be hand-bridged (`greedy_draft_tokens` in
`workers/llama_cpp_adapter.py`).

Two facts reopen the trade for `llama_cpp` specifically:

1. `llama-server` no longer forces log-scraping. It exposes **structured
   telemetry over HTTP** — streaming response `timings` (`prompt_per_second`,
   `predicted_per_second`), a `/slots` endpoint with per-slot prompt-processing
   progress, and `/metrics`. That is a different option than the log shim ADR
   0004 rejected.
2. Every field the dashboard consumes (`runtime/status.py:TelemetryStatus`:
   `worker_state`, `last_heartbeat`, `prefill` processed/total,
   `generation_tps`, `prompt_tps`, `tps_history`) is recoverable from that
   HTTP surface. `worker_state`/`last_heartbeat` come from `engine_worker`'s
   engine-agnostic heartbeat loop; memory is measured externally via psutil on
   the PID. No consumed field depends on the in-process monkeypatch hooks.

ADR 0006 Mitigation 3 (hand-roll a continuous-batching server over the
low-level `llama_batch` ctypes API) would close the gap by *adding* a large,
correctness-sensitive, permanently-owned scheduler. Reverting to `llama-server`
closes the same gap by *deleting* code.

## Decision

Run the `llama_cpp` engine as a **`llama-server` subprocess**, keeping ADR
0004's process/lifecycle model intact: the worker (`engine_worker`) stays the
detached, `state.json`-tracked child spawned with `start_new_session=True`; the
control plane still never loads tensors; `sovereign down` / `teardown.py` are
unchanged. What changes is *what the worker runs* and *where telemetry comes
from*:

- The `llama_cpp` adapter launches `llama-server` (native binary) with an argv
  built from `engine_kwargs()` — the same mapping role `mlx_lm_adapter` already
  fills with `build_server_argv`. `-np`/`--draft`/etc. pass through natively.
- A telemetry translator polls `llama-server`'s `/slots` + `/metrics` (no
  request-path proxy needed) and emits the **same** UDS NDJSON events
  (`PREFILL_PROGRESS`, `GENERATION_STATS`; ADR 0005) keyed by request. The wire
  protocol and `TelemetryStateCache` are unchanged.
- `max_parallel` becomes **honored**: `-np N` gives true N-slot continuous
  batching. `per_slot_context()`'s division is now real, and the boot-time
  "no effect" warning in `prepare_environment()` is removed.
- Two-model speculative decoding is `llama-server`-native; Sovereign's
  `greedy_draft_tokens` bridge and the `create_app`/monkeypatch server
  machinery are **deleted**.
- Provisioning installs the `llama-server` binary (Homebrew `llama.cpp`, pinned)
  instead of the `llama-cpp-python[server]` wheel; the import-probe gate becomes
  a binary-on-PATH / version probe.

mlx_lm keeps its embedded in-process server (it already has true batching via
`BatchGenerator` and maps to `mlx_lm.server` with no gaps) — so this ADR does
not touch it.

## Consequences

- The `max_parallel` gap (ADR 0006) closes with **zero owned scheduler code**;
  native speculative decoding replaces the hand-rolled draft bridge — a net
  code deletion in `workers/llama_cpp_adapter.py`.
- Prefill telemetry can improve from indeterminate to a real `/slots` fraction.
- Process isolation is at least as strong — a native binary crash is cleaner
  than a Python worker holding tensors.
- Cost: engine asymmetry (llama_cpp = CLI subprocess, mlx_lm = in-process
  bindings), already an accepted pattern under ADR 0006; a `llama-server`
  binary to provision/version-pin rather than a pip wheel; and a new telemetry
  translator (smaller than the server + draft-bridge code it replaces).
- ADR 0006's `max_parallel` section is now historical — the gap it managed via
  "accept-and-warn" no longer exists for llama_cpp.

## Alternatives considered

- **ADR 0006 Mitigation 3** (own continuous-batching server over `llama_batch`)
  — rejected: reimplements what `llama-server` already does well, and we would
  own every future bug in a large ctypes scheduler.
- **Keep the bindings, sharpen the warning** (point users at mlx_lm / scale-out)
  — rejected once it was clear `llama-server`'s HTTP telemetry preserves
  dashboard parity, making the revert strictly more capable.
- **Revert mlx_lm too** — rejected: mlx_lm has no gap and maps cleanly to its
  embedded server; there is nothing to regain.

---
Provenance: supersedes 0004 (for llama_cpp) and closes the `max_parallel` gap
in 0006; follow-up to the ADR 0004 embedded-workers migration (PR #20).
