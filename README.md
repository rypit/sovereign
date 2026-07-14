# Sovereign

[![CI](https://github.com/rypit/sovereign/actions/workflows/ci.yml/badge.svg)](https://github.com/rypit/sovereign/actions/workflows/ci.yml)

A declarative control plane for running local LLM infrastructure — engines,
frontends, coding harnesses, and benchmarks — on macOS / Apple Silicon laptops.

You describe the stack you want in a `sovereign.yaml`; Sovereign boots inference
engines **natively** (for real Metal/MLX acceleration), runs auxiliary services in
Docker, wires them together, and enforces a unified-memory budget so a second
engine can't OOM-crash the machine.

## How it works

- **Declare, don't script.** A stack is a list of services (and optionally
  harnesses) in YAML. Each service names a `base_type` — an inference engine,
  or `docker` for anything containerized — and Sovereign boots them in
  dependency order, health-checks them, and keeps a live dashboard
  (`sovereign monitor`).
- **Models come from HuggingFace refs.** `model` accepts a local path or an HF
  repo id (`org/name`, `org/name:Q4_K_M`, `org/name/file.gguf`). If you omit
  `base_type`, Sovereign routes the model to the right engine from its HF
  metadata. Models are pre-downloaded into the shared HF cache before launch,
  with byte-level progress in the dashboard; `sovereign models list/prune`
  inspects and reclaims the cache.
- **Memory is budgeted, not hoped for.** Sovereign estimates each model's
  footprint from its weight files and refuses to boot a service that would
  blow the machine's unified-memory budget — it never kills a running service.
  `sovereign plan` shows the routing, estimates, and budget verdict as a
  no-download dry-run.
- **Dependencies install themselves.** Declaring a service or harness in YAML
  is all it takes: each integration owns its dependency chain (Brewfile +
  install commands), run idempotently at boot or explicitly via
  `sovereign provision`.
- **Harnesses and benchmarks are first-class.** Coding agents can be pointed
  at the stack (`sovereign harness invoke`), and `sovereign bench` sweeps
  stacks × harnesses × suites into repeatable, content-addressed benchmark
  cells — performance (TTFT / tok/s / latency) and agentic quality, joined
  into a speed/quality comparison by `bench compare`.

Sovereign is strictly local: only local engines run, and no cloud model is a
benchmark baseline. See the
[implementation plan](./docs/sovereign-implementation-plan-v1.1.md) for the
full design.

## Supported integrations

| Kind | Integration | Notes |
| --- | --- | --- |
| Engine | `mlx_lm` | Apple MLX, runs embedded in-process (`mlx-lm` ships as a dependency on Apple Silicon) |
| Engine | `llama_cpp` | Runs the native `llama-server` binary as a subprocess (installed via Homebrew) |
| Service | `docker` | Generic container runner — Open WebUI and SearXNG in the examples are just `docker` instances |
| Harness | `cline_cli` | Cline in headless mode, isolated per-stack config |
| Harness | `mini_swe_agent` | mini-SWE-agent in-process (the optional `harness` extra) |

Both engines accept local model paths or HF refs, support speculative decoding
(`draft_model` / `num_draft_tokens`), and an optional `served_model_name` for
the OpenAI-compatible `"model"` string. The default example stack
(`examples/sovereign.yaml`) runs an MLX engine plus Open WebUI with SearXNG
wired in for web search.

## Getting started

```bash
python3 scripts/setup.py
```

This bootstraps the toolchain (Homebrew, `uv`), syncs the Python environment,
registers `sovereign` as a global executable, and provisions every registered
integration's dependencies.

Then try a tiny model — `examples/mlx.yaml` omits `base_type`, so it's routed
to `mlx_lm` automatically:

```bash
sovereign plan -f examples/mlx.yaml   # dry-run: routed engine + memory estimate, no download
sovereign up   -f examples/mlx.yaml   # DOWNLOADING (byte progress) -> STARTING -> READY
sovereign models list                 # what's in the shared HF cache
```

## CLI overview

| Command | What it does |
| --- | --- |
| `sovereign up` / `down` | Boot / tear down the stack in `sovereign.yaml` (or `-f <file>`) |
| `sovereign status` | One-shot service status table |
| `sovereign monitor` | Live dashboard (memory, tok/s, download progress) |
| `sovereign logs <name>` | Tail a service's logs |
| `sovereign plan` | Dry-run: engine routing + memory estimates + budget verdict |
| `sovereign provision` | Install integration dependencies (a stack's, or all) |
| `sovereign models list/prune` | Inspect / reclaim the shared HF model cache |
| `sovereign harness list/materialize/invoke` | Work with coding harnesses |
| `sovereign bench run/ls/compare` | Run and compare benchmarks |

## Harnesses & benchmarking

With a stack up and harnesses declared in its `harnesses:` section:

```bash
sovereign harness list
sovereign harness invoke <name> --prompt "..." --workdir /path/to/repo
```

Benchmarks live in a separate spec file (`bench.yaml`), never in
`sovereign.yaml`. **Attach mode** measures an already-running stack read-only;
**clean-room mode** boots, measures, and tears down each stack itself:

```bash
sovereign bench run -f examples/bench/perf-attach.yaml        # attach: needs a live `sovereign up`
sovereign bench run -f examples/bench/quality-cleanroom.yaml  # clean-room: self-contained
sovereign bench ls
sovereign bench compare
```

Harness/bench Python dependencies install automatically via provisioning; the
extras remain as the declarative, lockfile-friendly path (e.g. for CI):
`uv sync --extra harness --extra bench`.

## Runtime state

All runtime state lives under `.sovereign/` **relative to the directory you run
Sovereign from** — service states, the dashboard snapshot, the resolved-stack
manifest, `logs/`, and `benchmarks/`. Separate CLI invocations (`status`,
`monitor`, `down`, `harness invoke`) coordinate through these files, so run
them from the same directory as `sovereign up` — or pass `--state-dir`.
Downloaded models are *not* here; they live in the shared HuggingFace cache
(`sovereign models list`).

## Development

```bash
uv run sovereign --help    # CLI surface
make test                  # uv run pytest -q
make lint                  # uv run ruff check .
make typecheck             # uv run mypy
make check                 # all of the above (what CI runs)
```

CI runs the suite on a macOS arm64 runner (Python 3.12) — the product's actual
and only target platform. Tests are hermetic and run anywhere.

Contributing? See [`CONTRIBUTING.md`](./CONTRIBUTING.md),
[`docs/architecture.md`](./docs/architecture.md) for the current layering and
contracts, [`docs/decisions/`](./docs/decisions/) for the ADRs behind them,
and [`CLAUDE.md`](./CLAUDE.md) for an architecture map and codebase
conventions (useful for humans too).
