# Architecture

The living invariants doc — agents read this *and update it* when layering
or a core contract changes (see "How to change architecture" below).
`docs/sovereign-implementation-plan-v1.1.md` stays frozen as the historical
`§N` anchor; this doc is the current, maintained picture. Decisions behind
each rule live in `docs/decisions/` (ADRs).

## Layering

```
config.py, **/config.py, core/base_config.py   (golden rule: Pydantic only)
        ↓
core/*  (registry, base_manager, base_harness, resources, planning, errors,
         state, procmem, units, provisioning, resolver)
        ↓
workers/*  (leaf: no imports outside workers/ except core/procmem)
        ↓
services/*, harnesses/*  (self-registering integrations; hf.py is a leaf)
        ↓
runtime/*  (orchestrator, telemetry, dashboard, status, manifest, teardown)
        ↓
bench/*  (only cleanroom.py may import runtime/orchestrator)
        ↑
cli/*  (thin: parsing, tables, exit codes; calls into runtime/core)
```

Each arrow is "may depend on," not "imports directly" — e.g. `runtime/`
depends on `core/` but not on `services/` internals beyond what
`core/registry` resolves for it.

## Dependency rules (machine-checked)

These are encoded verbatim as `ARCH_RULES` in `scripts/depgraph.py` and
enforced by `scripts/depgraph.py --check` (`make arch`). **This list and
`ARCH_RULES`'s rule ids must match exactly** — `scripts/check_docs.py`
asserts it.

- **`config-golden-rule`** — `config.py`, every `**/config.py`, and
  `core/base_config.py` may import only `core/units`, `core/base_config`,
  and Pydantic/stdlib. Config describes desired state; it must never own
  `subprocess`, `os` process control, or Docker. See ADR 0001, ADR 0003.
- **`workers-leaf`** — `workers/*` imports nothing outside `workers/` except
  `core/procmem`. Worker modules must stay importable (and unit-testable)
  without engine bindings or the rest of the control plane. See ADR 0004.
- **`hf-leaf`** — `services/inference/hf.py` imports only `core/errors` and
  `core/state`. The HF pipeline (ref parsing, metadata, GGUF selection,
  memory estimation, download) is a leaf: nothing above `services/` imports
  it directly — the orchestrator/planner/CLI route through
  `core.registry.route_entry` and catch `core/errors`.
- **`runtime-no-bench`** — `runtime/*` never imports `bench/*`. Benchmarking
  is a consumer of the runtime, never the reverse.
- **`bench-single-door`** — only `bench/cleanroom.py` may import
  `runtime/orchestrator`. Every other bench module reaches the orchestrator
  (if at all) through that one door, so there is exactly one place that
  knows how to boot/measure/teardown a stack for benchmarking.
- **`core-single-door`** — nothing in `core/` imports `services/` or
  `harnesses/` at runtime except `core/registry`. Registry is the one
  sanctioned door from the contract layer into concrete integrations;
  every other `core/*` module stays a pure contract/utility layer.

Rules are declared as data (`id`, `description`, scope pattern, allowed/
forbidden import patterns) and evaluated over the runtime edge list
`depgraph.py` already builds from the AST — no separate parser, no
duplicated logic between the report and the check.

`--check` also fails on:

- **Any runtime import cycle** (Tarjan SCC over the same edge list).
- **A stale `docs/dependency-graph.md`** — the graph is regenerated
  in-memory and compared against the checked-in file (ignoring the
  generated-date line). Run `make graph` to refresh it.

A `GRANDFATHERED` allowlist exists in `depgraph.py` for any violation
deliberately kept; as of this writing it is empty — the current graph is
clean against every rule above.

## Core contracts

- **`ServiceManager`** (`src/sovereign/core/base_manager.py`) — the single
  Protocol the Orchestrator programs against for every supervised,
  run-forever thing (native process or Docker container). Optional
  capabilities (`SupportsModelPreparation`, `SupportsMemoryEstimate`,
  `SupportsEstimateSource`, `RoutesModelRef`, …) are separate Protocols in
  the same module, checked with `isinstance()`. ADR 0002.
- **`Harness`** (`src/sovereign/core/base_harness.py`) — the contract for
  agent harnesses (`materialize()`, `invoke(task)`), symmetric to
  `ServiceManager` but distinct — harnesses are not supervised services.
- **`WorkerConfig`** (`src/sovereign/workers/worker_config.py`) — the JSON
  handoff from a manager to its engine worker process
  (`v, service, engine, host, port, health_path, telemetry_socket,
  model_path, draft_model_path, served_model_name, engine_kwargs`). ADR 0004.
  `mlx_lm`'s worker loads tensors in-process; the other engines' workers
  instead launch a server as a child process under ADR 0007's exception to
  ADR 0004's in-process-binding default — `llama_cpp` (`llama-server`, with
  a telemetry translator over its HTTP surface), `omlx` (`omlx serve`,
  supervise-only) and `comfyui` (`comfy … launch`, supervise-only; ADR 0008).
- **Telemetry event schema** (`src/sovereign/workers/protocol.py`) — the
  NDJSON wire format workers speak to the parent telemetry hub
  (`heartbeat`, `log`, `state_change`, `memory`, `prefill_progress`,
  `generation_stats`, `docker_stats`). ADR 0005.
- **`StatusSnapshot`** (`src/sovereign/runtime/status.py`) — the schema
  `Orchestrator.status_snapshot()` produces, persisted as `status.json` and
  rendered by `runtime/dashboard.py`; `TelemetryStatus`/`PrefillStatus`
  mirror `TelemetryStateCache.snapshot()`'s field names exactly.
- **`core/registry.py`** — `base_type` → class factory maps and
  `route_entry()` for `auto` routing; the one sanctioned door from `core/`
  into `services/`/`harnesses/`. ADR 0002.
- **`core/planning.py`** — the shared dry-run `sovereign plan` and `up` both
  use (same routing + admission math), so they can never drift. ADR 0003.

## Cross-references

| Area | Governing ADR | Plan anchor |
| --- | --- | --- |
| No daemon, per-directory state | 0001 | §2 (points 2, 10, 12), §9 |
| Registry self-registration, `base_type` | 0002 | §2 (points 4–6), §11.1 |
| Refuse-to-boot admission control | 0003 | §2 (point 8), §7, §11.5 |
| Embedded engine workers | 0004 | PR #20 |
| UDS NDJSON telemetry | 0005 | PR #20 |
| Engine-gap policy (`§3a` in code comments) | 0006 | PR #20 addendum |
| ComfyUI engine, `checkpoint` artifact kind | 0008 | — |

## How to change architecture

1. Write an ADR in `docs/decisions/` (see its `README.md` for when one is
   required and the template).
2. Update this doc — the layering diagram, the rule list, the contracts
   table, the cross-reference table — and the architecture map in
   `CLAUDE.md` if a directory's role changed.
3. If layering itself changed, update `ARCH_RULES` in `scripts/depgraph.py`
   to match, and run `make arch` (and `make graph` if the module set
   changed) before committing.
