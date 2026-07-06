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
roadmap; 201 tests passing):

- Core contracts (`ServiceManager` / `Harness` Protocols, `base_type` registry),
  `sovereign.yaml` parsing, and the full Typer CLI (`up`/`down`/`status`/`logs`/
  `monitor`).
- The Orchestrator: DAG boot in dependency order, async reconciliation loop,
  memory admission control, resolved-stack manifest + drift detection.
- Services: `docker_engine` (a generic Docker container runner — `open_webui` and
  `searxng` are just instances of it configured in YAML), `llama_cpp`, `mlx_lm`.
  The Docker daemon is implicit infrastructure: each `docker_engine` service
  verifies it's reachable on its own, so there's no separate engine entry or
  dependency to declare. (Existing YAMLs using `base_type: open_webui` or
  `searxng` need to switch to `base_type: docker_engine`.)

Still to come: `launchd` install/uninstall (Phase 9), `comfyui` (Phase 11), and
the harness + benchmarking tracks. See the
[implementation plan](./sovereign-implementation-plan-v1.1.md) for per-phase status.

## Setup

```bash
python3 scripts/setup.py
```

Installs Homebrew dependencies, syncs the Python environment, and registers `sovereign` as a global executable.

### MLX engine

The `mlx_lm` engine ships with the project: `mlx-lm` is a dependency (Apple Silicon
only), so `uv sync` provides the `mlx_lm.server` binary. Try it with a tiny model:

```bash
uv run sovereign up -f mlx.yaml   # boots a tiny MLX model
```

## Development

```bash
uv run sovereign --help    # CLI surface
uv run pytest -q           # tests
uv run ruff check .        # lint
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
