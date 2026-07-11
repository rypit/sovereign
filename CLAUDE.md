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
                   managers via asyncio.to_thread — managers stay synchronous
  dashboard.py     Rich live dashboard; renders runtime/status.StatusSnapshot
  status.py        StatusSnapshot TypedDicts (the schema orchestrator produces
                   and dashboard consumes)
  manifest.py      resolved stack manifest written at boot (§7b)
  teardown.py      cross-process stop from persisted runtime handles (`sovereign down`)
services/
  docker/         auxiliary services in Docker
  inference/     native engines + their shared base
    base.py              shared subprocess/health/metrics lifecycle (NativeEngineManager)
    hf.py                the HuggingFace pipeline: ref parsing, metadata, GGUF
                         selection, memory estimation, download, RoutingCache
    routing.py           engine-routing sweep (each engine's claim_route); registers
                         the router core calls via registry.route_entry()
    llama_cpp/  mlx_lm/   the two native engines (auto-discovered)
harnesses/         cline_cli, mini_swe_agent          (auto-discovered)
bench/             content-addressed bench cells; only cleanroom.py may
                   import the Orchestrator; bench sub-app CLI lives in bench/cli.py
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
  `urllib.request.urlopen` for health checks. `tests/conftest.py` autouse
  fixtures disable real provisioning and stub the HF API to look offline —
  opt back in with `@pytest.mark.allow_provisioning`.
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
