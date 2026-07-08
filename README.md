# Sovereign

[![CI](https://github.com/rypit/sovereign/actions/workflows/ci.yml/badge.svg)](https://github.com/rypit/sovereign/actions/workflows/ci.yml)

A declarative control plane for running local LLM infrastructure — engines,
frontends, coding harnesses, and benchmarks — on macOS / Apple Silicon laptops.

You describe the stack you want in a `sovereign.yaml`; Sovereign boots inference
engines **natively** (for real Metal/MLX acceleration), runs auxiliary services in
Docker, wires them together, and enforces a unified-memory budget so a second
engine can't OOM-crash the machine.

See [`sovereign-implementation-plan-v1.1.md`](./sovereign-implementation-plan-v1.1.md)
for the full design.

The default stack also runs **SearXNG**, wired into Open WebUI for web search.

## Status

The MVP orchestration spine is implemented and tested (Phases 0–8 and 10 of the
roadmap), and **both post-MVP tracks — harnesses and benchmarking — are complete**
(419 tests passing):

- Core contracts (`ServiceManager` / `Harness` Protocols, `base_type` registry),
  `sovereign.yaml` parsing, and the full Typer CLI (`up`/`down`/`status`/`logs`/
  `monitor`/`harness`/`bench`).
- The Orchestrator: DAG boot in dependency order, async reconciliation loop,
  memory admission control, resolved-stack manifest + drift detection, and
  harness materialization (including re-materialization when a dependency's
  endpoint changes, e.g. after a restart).
- Services: `docker_engine` (a generic Docker container runner — `open_webui` and
  `searxng` are just instances of it configured in YAML), `llama_cpp`, `mlx_lm`.
  The Docker daemon is implicit infrastructure: each `docker_engine` service
  verifies it's reachable on its own, so there's no separate engine entry or
  dependency to declare. (Existing YAMLs using `base_type: open_webui` or
  `searxng` need to switch to `base_type: docker_engine`.) `llama_cpp` and
  `mlx_lm` share a native-engine base and are configured consistently:
  `llama_cpp`'s config field is now `model` (renamed from `model_path`) and,
  like `mlx_lm`, accepts a local model path or a HuggingFace repo id
  (`org/name`, `org/name:Q4_K_M`, or `org/name/file.gguf`); both engines also
  support speculative-decoding `draft_model`/`num_draft_tokens`, and an optional
  `served_model_name` for the OpenAI-compatible `"model"` string.
- HF-native model management. `base_type: auto` (the default when `base_type` is
  omitted) routes a model to `llama_cpp`/`mlx_lm` from its HuggingFace metadata,
  so you only declare the `model`. Sovereign estimates a repo's memory footprint
  from its weight-file sizes for admission control, and **pre-downloads** the
  model into the shared HF cache before launch — a `DOWNLOADING` state with
  byte-level progress (MB/s + ETA, Xet-aware via `hf_xet`) in the dashboard. Both
  engines always launch from the resolved local path (no `--hf-repo`), so
  `health_check.timeout_seconds` now only needs to cover model *load*, not the
  download. `sovereign plan -f stack.yaml` is a no-download dry-run (routing +
  memory estimate + budget verdict per service); `sovereign models list/prune`
  inspects and reclaims the HF cache.
- Harnesses: `cline_cli` (subprocess, isolated `CLINE_DIR`, `--yolo`/`--json`
  headless) and `mini_swe_agent` (in-process `DefaultAgent`/`LitellmModel`, the
  optional `harness` dependency extra). Both are invocable via
  `sovereign harness list/materialize/invoke`.
- Per-integration provisioning: one shared contract (`core/provisioning.py`)
  lets every service and harness install its own dependency chain (a Brewfile
  in its folder + install commands) — run at boot, by `sovereign provision`,
  and by `scripts/setup.py`. Declaring an integration in YAML is all it takes.
- Benchmarking (`sovereign bench run/ls/compare`): a bench spec (`bench.yaml`,
  never part of `sovereign.yaml`) sweeps `stacks x harnesses x suites` into
  content-addressed cells that skip on re-run. Attach mode measures an
  already-running stack read-only (TTFT/tok-s/latency via an in-house `httpx`
  prober, the optional `bench` extra); clean-room mode boots/measures/tears
  down each stack itself behind a lockfile so it can't fight the daemon.
  Agentic-quality cells run a harness against a native task-suite format,
  grading from `git diff` + a programmatic grader rather than the harness's
  own self-report, and gate off stacks whose perf cell already failed
  its thresholds. `bench compare` joins perf + quality results into a
  Pareto (speed/quality) comparison across runs.

Still to come: `launchd` install/uninstall (Phase 9), `comfyui` (Phase 11).
See the [implementation plan](./sovereign-implementation-plan-v1.1.md) for
per-phase status, and
[`harness-bench-implementation-plan.md`](./harness-bench-implementation-plan.md)
for the harness/bench tracks' detailed phase breakdown.

## Setup

```bash
python3 scripts/setup.py
```

Bootstraps the toolchain (Homebrew `uv`), syncs the Python environment,
registers `sovereign` as a global executable, and then runs
`sovereign provision` to install every integration's own dependencies.

### Provisioning

Declaring a service or harness in `sovereign.yaml` is all it takes — Sovereign
installs its full dependency chain automatically. Each integration folder owns
its setup artifacts (e.g. `services/llama_cpp/Brewfile`,
`harnesses/cline_cli/Brewfile` + `npm install -g cline`), and one shared
contract (`core/provisioning.py`, surfaced as `prepare_environment()` on both
services and harnesses) runs them idempotently in three places:

- at boot (`sovereign up`), for whatever the stack declares;
- via `sovereign provision [-f stack.yaml]` — scoped to a stack file, or every
  registered integration when unscoped (what `setup.py` runs);
- before one-shot `sovereign harness invoke`/bench quality runs.

### MLX engine

The `mlx_lm` engine ships with the project: `mlx-lm` is a dependency (Apple Silicon
only), so `uv sync` provides the `mlx_lm.server` binary. Try it with a tiny model —
`mlx.yaml` omits `base_type`, so it's routed to `mlx_lm` automatically:

```bash
uv run sovereign plan -f mlx.yaml   # dry-run: shows the routed engine + memory estimate, no download
uv run sovereign up   -f mlx.yaml   # DOWNLOADING (byte progress) -> STARTING -> READY
uv run sovereign models list        # what's in the shared HF cache
```

## Development

```bash
uv run sovereign --help    # CLI surface
uv run pytest -q           # tests
uv run ruff check .        # lint
```

### Harnesses & benchmarking

Harness dependencies install automatically when a harness is declared (see
Provisioning above). The `harness`/`bench` extras remain as the declarative,
lockfile-friendly path (e.g. for CI):

```bash
uv sync --extra harness --extra bench
```

With a stack up (`sovereign up -f sovereign.yaml`) and harnesses declared in
its `harnesses:` section:

```bash
uv run sovereign harness list
uv run sovereign harness invoke <name> --prompt "..." --workdir /path/to/repo
```

Benchmarks live in a separate spec file, never in `sovereign.yaml`:

```bash
uv run sovereign bench run -f examples/bench/perf-attach.yaml   # attach mode, needs a live `sovereign up`
uv run sovereign bench run -f examples/bench/quality-cleanroom.yaml  # bench boots/tears down its own stack
uv run sovereign bench ls
uv run sovereign bench compare
```

## Locked decisions

Recorded here per the plan's Day-1 checklist (§13):

- **`base_type` only** (§11.1). The instance `name` is the unique ID; `base_type`
  is the factory key that selects a Manager/Harness class. There is no separate
  `type` field.
- **Refuse-to-boot, never auto-kill** (§11.5). Under memory pressure, admission
  control refuses to start a service that would blow the budget, with a specific
  actionable error. Sovereign never auto-kills a running service. (To be revisited
  after living with refusal.)
- **Strictly local — no cloud baseline** (§11, Open Decision #1). Sovereign runs
  only local engines; no paid cloud model is an allowed benchmark baseline.
  **LiteLLM and Claude Code are dropped** — the Anthropic-Messages gateway they'd
  require is out of scope. The Pareto frontier is anchored entirely by local
  engine × model × harness combinations.
