# CLAUDE.md

Sovereign is a declarative control plane for local LLM infrastructure on
macOS/Apple Silicon: you describe a stack in `sovereign.yaml`, and it boots
inference engines natively (Metal/MLX), runs auxiliary services in Docker,
wires them together, and enforces a unified-memory budget.

## Commands

```bash
make test        # uv run pytest -q          (~500 tests, <10s, fully hermetic)
make lint        # uv run ruff check .
make typecheck   # uv run mypy               (src/sovereign, must stay clean)
make check       # all three — what CI runs
python3 scripts/setup.py   # full macOS bootstrap (brew + uv + provision)
```

Tests run on any platform: everything external (brew/npm, Docker, HF network,
subprocesses) is mocked at single seams. `mlx-lm` only installs on
macOS/arm64 (platform marker); code must never import it at module scope.

## Architecture map

```
cli/               Typer CLI (thin: parsing, tables, exit codes)
  __init__.py      root app + --verbose callback; mounts sub-apps
  _common.py       shared app/console/helpers (loaded first; no circular deps)
  stack.py         stack lifecycle commands on the root app (up/down/plan/status/…)
  harness.py       harness sub-app (list/materialize/invoke)
  models.py        models sub-app (list/prune)
config.py          sovereign.yaml schema (Pydantic)
core/
  state.py         state.json/status.json read-write helpers (file_hash, write_json, …)
  base_manager.py  ServiceManager Protocol + optional-capability Protocols
                   (SupportsModelPreparation, SupportsMemoryEstimate,
                   SupportsEstimateSource, RoutesModelRef, …)
  base_harness.py  Harness Protocol + BaseHarness
  base_config.py   SovereignBaseModel (extra="forbid") + NativeEngineConfig
  errors.py        model-resolution exceptions (the cross-layer contract)
  registry.py      base_type -> class maps; populate_registries(); route_entry()
  planning.py      shared dry-run used by `sovereign plan` (same seams as boot)
  resources.py     memory budget / admission control (refuse-to-boot)
  provisioning.py  per-integration dependency install (Brewfile + commands)
runtime/
  orchestrator.py  async DAG boot, reconcile loops, persistence; drives sync
                   managers via asyncio.to_thread — managers stay synchronous;
                   owns the TelemetryStateCache, starts/stops the TelemetryHub
                   (UDS) + DockerMonitorWorker around boot/reconcile
  telemetry.py     parent-side telemetry: TelemetryStateCache (bounded, thread-safe
                   per-service state), TelemetryHub (accepts worker UDS connections,
                   ingests uncapped into the cache), DockerMonitorWorker (polls
                   `docker stats` on its own thread, feeds the same cache)
  dashboard.py     Rich live dashboard; renders runtime/status.StatusSnapshot
                   (MEM + TOK/S sparklines, prefill progress bars); 1 Hz snapshot
                   poll decoupled from the 12fps spinner/pulse animation refresh
  status.py        StatusSnapshot TypedDicts (the schema orchestrator produces
                   and dashboard consumes), incl. TelemetryStatus/PrefillStatus
                   mirroring TelemetryStateCache.snapshot()'s field names
  manifest.py      resolved stack manifest written at boot (§7b)
  teardown.py      cross-process stop from persisted runtime handles (`sovereign down`)
services/
  docker/         auxiliary services in Docker
  inference/     embedded Python-binding engine workers + their shared base
    base.py              shared worker-lifecycle base (NativeEngineManager): dumps
                         a WorkerConfig, launches `sovereign.workers.engine_worker`,
                         HTTP health, psutil/procmem metrics fallback
    hf.py                the HuggingFace pipeline: ref parsing, metadata, GGUF
                         selection, memory estimation, download, RoutingCache
    routing.py           engine-routing sweep (each engine's claim_route); registers
                         the router core calls via registry.route_entry()
    llama_cpp/  mlx_lm/   the two native engines (auto-discovered); each supplies
                         engine_kwargs() (mapped by its workers/*_adapter.py)
workers/           embedded engine worker processes (spawned via
                   `python -m sovereign.workers.engine_worker --config <path>`)
  protocol.py      typed telemetry event schema (NDJSON) + encode/decode
  telemetry.py     worker-side TelemetryClient: non-blocking UDS sender, reconnect+drop
  worker_config.py WorkerConfig dataclass + JSON load/dump (the manager->worker handoff)
  engine_worker.py generic entrypoint: loads WorkerConfig, dispatches to
                   `sovereign.workers.<engine>_adapter` by `engine` key
  llama_cpp_adapter.py / mlx_lm_adapter.py   pure kwarg-mapping + run() (bindings
                   imported lazily, inside run(), never at module scope)
harnesses/         cline_cli, mini_swe_agent          (auto-discovered)
bench/             content-addressed bench cells; only cleanroom.py may
                   import the Orchestrator; bench sub-app CLI lives in bench/cli.py
scripts/
  depgraph.py      AST-based dep graph report + `--check` (import cycles,
                   ARCH_RULES layering violations, docs/dependency-graph.md
                   freshness) — `make graph` / `make arch`
  check_docs.py    §N citation validity, ADR well-formedness, architecture.md
                   <-> ARCH_RULES rule-id parity — `make arch`
docs/
  decisions/       ADRs (docs/decisions/README.md has the convention + template)
  architecture.md  living layering/contracts doc; rule ids mirror ARCH_RULES
```

Dependency direction: `config` depends only on Pydantic (the "golden rule" —
never subprocess/os/docker in a config module). The HF pipeline
(`services/inference/hf.py`) imports no managers; nothing above
`services/` imports it — the orchestrator/planner/CLI route through
`core.registry.route_entry` and catch `core.errors`, so the engine-domain HF
code stays a leaf. `runtime/orchestrator` imports `core/*`; nothing in
`runtime/` imports `bench`.

## Conventions and gotchas

- **Registration**: integrations self-register via `@register_service("x")` /
  `@register_harness("x")` decorators. `services/__init__.py` walks its tree
  recursively (`pkgutil.walk_packages`, so nested groupings like
  `inference/llama_cpp` register too); `harnesses/__init__.py`
  pkgutil-imports every subpackage; `core/registry.populate_registries()` is
  the one call every lookup path makes first. Adding an integration = dropping
  a folder with `__init__.py` + `config.py` + `manager.py` (plus optional
  `Brewfile`) under `services/` (or `services/inference/` for a native
  engine); no aggregator edit needed. Harness modules must import optional deps
  lazily (inside methods) — discovery imports them unconditionally.
- **Optional manager capabilities** are Protocols in `core/base_manager.py`,
  checked with `isinstance()` — don't `getattr`-probe for hooks, and add new
  hooks to a Protocol so they stay visible.
- **Test seams**: tests patch `sovereign.services.inference.hf.<fn>`
  (engines and the router call through the `hf_models`/`hf` module alias),
  `run_docker()` for Docker, and
  `urllib.request.urlopen` for health checks. Engine-binding availability is
  probed via `sovereign.services.inference.base.probe_import` (patchable,
  cached per module) rather than a binary-on-PATH check. Telemetry runs over
  a unix domain socket (`<state_dir>/telemetry.sock`) — tests exercise the
  real `TelemetryHub`/`TelemetryClient` over a real socket rather than
  mocking the transport; `SOVEREIGN_WORKER_ADAPTER_PACKAGE` overrides which
  adapter package `engine_worker` dispatches to, so its entrypoint can be
  exercised against a fake adapter. `tests/conftest.py` autouse fixtures
  disable real provisioning and stub the HF API to look offline — opt back
  in with `@pytest.mark.allow_provisioning`.
- **State is per-directory**: everything lives under `.sovereign/` relative
  to the CWD (`state.json`, `status.json`, `manifest.json`, `logs/`,
  `benchmarks/`). Separate CLI processes coordinate through these files —
  there is no daemon IPC. Run commands from the stack's directory or pass
  `--state-dir`.
- **`plan` must not drift from boot**: `core/planning.py` reuses
  `core.registry.route_entry` (engine routing) and `estimate_service_memory`,
  and reads the SOURCE label through the manager's `estimated_memory_source()`
  (`SupportsEstimateSource`) — if you change admission or routing, both paths
  pick it up; never re-implement the math.
- **Refuse-to-boot, never auto-kill** (§11.5 of the plan): admission control
  refuses services that would blow the memory budget; Sovereign never kills a
  running service.
- **`base_type` only**: instance `name` is identity, `base_type` picks the
  class. `auto` (or omitted) routes via `core.registry.route_entry`, which sweeps
  each engine's `claim_route` (`RoutesModelRef`) over the ref + HF metadata — a
  new engine joins `auto` routing by dropping in a folder, no central rule table.
- Diagnostics go to the `sovereign` logger (`--verbose` for DEBUG);
  user-facing output goes through the Rich `console`.
- Docstrings cite plan sections (§N) — they refer to
  `docs/sovereign-implementation-plan-v1.1.md`.

## Working agreements

- Write an ADR (`docs/decisions/`, see its `README.md`) before/alongside a
  change to: layering or dependency direction, a core Protocol/contract,
  `sovereign.yaml` schema *semantics*, the memory/admission model, the
  process/lifecycle model, the telemetry wire protocol, or testing seams.
  The `/adr` skill scaffolds one.
- Touch these docs when they apply: `docs/architecture.md` (layering,
  contracts, or an invariant changed), this file's architecture map (a
  directory/file's role changed), `docs/dependency-graph.md` via `make graph`
  (the module set changed).
- `make check` now runs `make arch` (`scripts/depgraph.py --check` +
  `scripts/check_docs.py`) alongside lint/typecheck/test — it fails on a
  layering violation, an import cycle, a stale dependency graph, a broken
  `§N` citation, or a malformed ADR. Run `/architecture-review` before a PR
  that touches a Protocol, the config schema, or the telemetry wire format.
- No ADR needed for a purely additive change (new optional config field,
  new integration folder following the existing pattern) — say so in the
  PR description instead of writing one.
