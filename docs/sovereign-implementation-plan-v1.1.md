# Sovereign — Implementation Plan (v1.1)

*A declarative control plane for running local LLM infrastructure — engines, frontends, coding harnesses, and benchmarks — on macOS/Apple Silicon laptops.*

This is the consolidated plan. v1.1 folds in the harness/benchmark/stack-capture work: it restates the decisions already locked in, adds the two new entity types the design grew (Harnesses and Jobs), corrects ComfyUI to a native service, and lays out a phased build order with concrete exit criteria for each phase.

> **Progress snapshot — updated 2026-07-07.** The MVP orchestration spine (Phases 0–8 and 10) is implemented and green, and **both post-MVP tracks (harnesses, benchmarking) are now complete** — the full suite is green (CI is the source of truth for the count). Remaining work: finish Phase 9 (`launchd` install/uninstall) and Phase 11 (`comfyui`). **Open Decision #1 is now resolved: Sovereign is strictly local — LiteLLM and Claude Code are dropped.** Per-phase status is tracked inline in §12; the service/harness catalog status is in §10; the harness/bench tracks have their own detailed phase breakdown in [`harness-bench-implementation-plan.md`](./harness-bench-implementation-plan.md).

---

## 0. The Core Taxonomy

Sovereign deals with **three** kinds of entity, distinguished by *lifecycle shape* — not by frontend-vs-backend:

| Entity | Sovereign's relationship | Lifecycle | Contract | Terminal state? |
|---|---|---|---|---|
| **Service** | Supervises | Run-forever | `ServiceManager` (§4) | No — health-checked steady state |
| **Harness** | Configures + invokes | Configure-then-run-on-demand | `Harness` (§4b) | N/A — leaf consumer |
| **Job** | Runs to completion | Run-to-completion | `Job` (§6b) | Yes — `COMPLETED` / `FAILED` |

This mirrors the split Kubernetes landed on (Deployments vs. Jobs): run-forever and run-to-completion genuinely don't share a contract. Forcing a benchmark or a coding harness into `ServiceManager` turns most of that Protocol into no-ops — `start()` with nothing to start, `is_healthy()` with nothing to poll, memory the Budgeter can't track — which is the signal it's the wrong contract. Services stay exactly as originally designed; the two new entity types get their own contracts and their own config surfaces.

---

## 1. What Sovereign Is

The local-AI ecosystem on macOS is fragmented across inference servers (llama.cpp, oMLX, vLLM-metal, mlx_lm), model formats (GGUF, MLX safetensors), frontends (Open WebUI, Cline, Aider), and coding harnesses, each with different setup steps and no shared resource awareness. Running two of them at once is how you crash a Mac.

Sovereign is a single control plane that:
- Reads a declarative `sovereign.yaml` describing the stack you want (which engines, which frontends, which harnesses, how much memory each gets).
- Boots inference engines **natively** (bare metal, so they get real Metal/MLX acceleration) and auxiliary services **in Docker** (since Docker on macOS can't pass through the GPU).
- Wires services together automatically (endpoints, ports, env vars) instead of making you hand-edit config files.
- Enforces a memory budget so you stop OOM-crashing the machine by starting a second engine on top of a first.
- Configures and invokes **coding harnesses** (Cline CLI, SWE-agent) against the local stack.
- **Captures the resolved stack** and lets you **benchmark** engine × model × harness combinations to find what's best for your hardware and use cases.
- Runs persistently via `launchd`, with a CLI (`sovereign up/down/status/monitor/bench`) as the primary interface.

Inspired by the "Zero to Hero" AI stack blueprint, adapted for single-user, single-machine, Apple Silicon constraints instead of a Docker+NVIDIA server. The goal is zero to a fully running, integrated LLM development environment.

---

## 2. Locked Architecture Decisions

Treat these as constraints, not discussion points, when writing code.

1. **Hybrid execution model.** Inference engines run as native subprocesses (bare metal). Stateless/auxiliary services (frontends, search, sandboxes, vector DBs) run in Docker. Nothing GPU-bound goes in a container.
2. **Config is declarative, orchestration is imperative.** `sovereign.yaml` describes desired state only. All "how to make it true" logic lives in Python.
3. **Config and behavior are strictly separated.** `config.py` files depend on Pydantic only — never on `subprocess`, `os`, `docker`, or a `manager.py`. `manager.py` files may import their sibling `config.py`. One-way dependency, enforced by convention (and ideally an import-linter rule later).
4. **Domain-driven service layout.** Each integration is a self-contained folder under `services/` (or `harnesses/`) with its own `config.py` + `manager.py`. Adding a new engine never requires touching another service's code.
5. **Every service satisfies one Protocol.** `ServiceManager` (a `typing.Protocol`, not an ABC) is the single contract the Orchestrator programs against. Native processes and Docker containers look identical from the Orchestrator's point of view.
6. **Instance identity vs. implementation.** `name` is the unique instance ID (e.g. `llama_heavy_v1`). `base_type` is the factory key that tells the Orchestrator which Manager class to instantiate (e.g. `llama_cpp`). This lets you run two llama.cpp instances with different models/ports side by side.
7. **Health is defined in config, executed by the manager.** The Orchestrator, not the engine, decides what "ready" means. `health_check` is a YAML block; `is_healthy()` is what actually pings it.
8. **Memory is budgeted, not partitioned.** Apple Silicon's unified memory has no hard OS-level split. Sovereign manages this "by proxy" — translating a declared budget into engine flags (`-ngl`, `-c`, `-t`) and QoS hints, plus admission control that refuses to boot a service that would blow the budget.
9. **Stack tooling:** `uv`, `Typer`, `Pydantic`, `Rich`, `psutil`, `asyncio`, `Ruff`, `pytest`.
10. **Process persistence via `launchd`.** A user-space `LaunchAgent` (not a root `LaunchDaemon`) runs Sovereign as an ordinary foreground process; launchd handles restart-on-crash and log capture.
11. **Ollama is out of scope as an engine.** It wraps llama.cpp rather than being a distinct engine; Sovereign talks to llama.cpp directly.
12. **Own repository, `src/` layout.** Not part of a monorepo.
13. **ComfyUI runs native, not in Docker.** Image generation is GPU-bound, and Docker on macOS has no Metal/MPS passthrough — a containerized ComfyUI would run CPU-only and be unusable. Its manager mirrors `llama_cpp` (subprocess + HTTP health), not `open_webui`. **Consequence:** ComfyUI is a first-class `ResourceBudgeter` citizen — a 70B GGUF (~40GB) plus Flux (~25GB) is exactly the collision refuse-to-boot admission control exists for. It is the best validation case for §7, not an afterthought.
14. **Endpoint resolution is consumer-aware.** A container reaching a native process must use `host.docker.internal`, not `localhost`. The resolver emits the right host per *consumer type* — this affects `open_webui` today and every Dockerized benchmark sandbox later.

---

## 3. Repository Structure

```
sovereign/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore                # .venv, .sovereign/, __pycache__, .DS_Store
├── src/
│   └── sovereign/
│       ├── __init__.py
│       ├── main.py               # Typer app, entry point (serve, up, down, status, monitor, bench)
│       ├── orchestrator.py       # DAG boot/shutdown + reconciliation loop
│       ├── core/
│       │   ├── base_config.py     # SovereignBaseModel, shared validators
│       │   ├── base_manager.py    # ServiceManager Protocol
│       │   ├── base_harness.py    # Harness Protocol (materialize + invoke)
│       │   ├── registry.py        # base_type -> Manager/Harness class factory map
│       │   ├── resolver.py        # {{ }} templates + ${ENV:} secrets, consumer-aware host
│       │   └── resources.py       # ResourceBudgeter (admission control)
│       ├── services/
│       │   ├── __init__.py        # imports + registers every service module
│       │   ├── docker/     # config.py + manager.py
│       │   ├── llama_cpp/
│       │   ├── open_webui/
│       │   ├── searxng/           # Docker, dynamic env wiring
│       │   └── comfyui/           # native (see §2.13)
│       ├── harnesses/
│       │   ├── __init__.py        # imports + registers every harness module
│       │   ├── cline_cli/         # config.py + manager.py (materialize + invoke)
│       │   └── swe_agent/
│       ├── bench/
│       │   ├── runner.py          # Job type, COMPLETED/FAILED, attach + clean-room modes
│       │   ├── spec.py            # bench-spec Pydantic model (sweep matrix + thresholds)
│       │   ├── cells.py           # content-addressing + skip-completed
│       │   └── report.py          # manifest join, Pareto compare
│       └── utils/
│           ├── launchd.py         # plist generation + install/uninstall
│           ├── manifest.py        # resolved stack manifest (capture)
│           └── state.py           # .sovereign/state.json read/write
└── tests/
    ├── test_orchestrator.py
    ├── test_resources.py
    ├── test_bench.py
    ├── services/
    │   ├── test_docker.py
    │   ├── test_llama_cpp.py
    │   └── test_open_webui.py
    └── harnesses/
        └── test_cline_cli.py
```

**Golden rule, restated:** `config.py` never imports `manager.py`. If you ever need a manager type inside a config file, that field belongs in the manager's runtime logic, not the schema.

---

## 4. Core Contract: `ServiceManager`

The single interface the Orchestrator programs against for **supervised, long-running** things — every engine, every container. Harnesses and Jobs do **not** implement this (see §4b, §6b).

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class ServiceManager(Protocol):
    name: str
    dependencies: list[str]

    # Lifecycle
    def start(self) -> None: ...
    def stop(self) -> None: ...

    # Readiness / observability
    def is_healthy(self) -> bool: ...
    def get_metrics(self) -> dict[str, Any]: ...

    # Resource cooperation
    def prepare_environment(self) -> None: ...
    def adjust_resources(self, memory_limit_bytes: int) -> None: ...
```

`@runtime_checkable` lets the Orchestrator do `isinstance(manager, ServiceManager)` before ever calling `.start()`, so a malformed integration fails loudly at registration time instead of mid-boot.

`prepare_environment()` is the pre-flight hook (does the model file exist? is the cache dir writable? is there disk space?) — it runs before `start()`, so failures surface as a clean error rather than a half-booted process.

---

## 4b. Core Contract: `Harness`

Harnesses are **leaf consumers** of the `ServiceRegistry`: they reuse the resolver and dependency edges, but nothing depends on them. Two separable capabilities:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Harness(Protocol):
    name: str
    dependencies: list[str]

    # Project resolved endpoints/secrets into the tool's own config format.
    # Runs only after dependencies are READY; re-runs when an endpoint changes.
    def materialize(self) -> None: ...

    # Run one headless, non-interactive session. Must guarantee completion:
    # auto-approve, bounded turns, meaningful exit codes. Not all harnesses support this.
    def invoke(self, task: "Task") -> "RunResult": ...
```

### Harness catalog (verified against current docs)

| Harness | `materialize` | `invoke` | Wiring to local engine | Role |
|---|---|---|---|---|
| **Cline CLI** | ✅ isolated config via `CLINE_DIR` | ✅ `--yolo` / `--json`, fails fast on missing auth | Direct — OpenAI-compatible, offline | **#1: daily driver + bench workhorse in one.** Isolated config dir means no shared-state merge (the `settings.json` concern applied only to the VS Code path, which is out of scope). |
| **SWE-agent** | ✅ YAML config | ✅ native headless batch runner | Direct — OpenAI-compatible | **Suite engine for "SWE-agent style."** Origin of grade-the-repo; benchmark adapters nearly free. `mini-swe-agent` for minimal surface. |

> **Claude Code dropped.** It reaches only the Anthropic Messages API and needs a LiteLLM gateway to hit local models — a cloud/proxy path that's out of scope for a strictly-local Sovereign (§11 Open Decision #1). The two local-first harnesses above anchor the Pareto frontier on their own.

**Teams are a config axis, not a framework choice.** Cline CLI has native teams (coordinator + specialists, persistent state); model this as a swept dimension. It goes **last in the benchmark funnel** because of the slots×context cost (§7).

---

## 5. `sovereign.yaml` — Canonical Schema

Three top-level entity sections: `services:`, `harnesses:`, and global `resources:`. Benchmarks are **not** here — they live in a separate bench spec, run imperatively via `sovereign bench run`, never at boot, so `sovereign.yaml` stays pure desired-state.

```yaml
version: "1.1"

resources:
  max_unified_memory_gb: 128
  safety_margin_gb: 8
  default_priority: medium

services:
  - name: docker
    base_type: docker
    priority: critical
    dependencies: []

  - name: llama_heavy_v1          # stable name + pinned port across variant files
    base_type: llama_cpp
    priority: critical
    affinity:
      hardware_type: apple_silicon
      prefer_cores: [4, 5, 6, 7]
    health_check:
      type: http
      endpoint: /health
      port: 11435
      timeout_seconds: 60
    policy:
      prompt_caching:
        enabled: true
        cache_path: "/tmp/sovereign/cache/llama_heavy"
        kv_cache_type: f16
    config:
      model_path: "~/models/llama3-70b.gguf"
      gpu_layers: 48
      threads: 8
      context_size: 32768
      max_parallel: 4              # -np; load-bearing when harness teams run (see §7)
      api_key: "${ENV:LLAMA_API_KEY}"
    dependencies: []

  - name: open_webui
    base_type: open_webui
    priority: medium
    health_check:
      type: http
      endpoint: /health
      port: 3000
    config:
      image: ghcr.io/open-webui/open-webui:main
      port: 3000
    env_overrides:
      OLLAMA_API_BASE_URL: "{{ llama_heavy_v1.endpoint }}"   # resolves to host.docker.internal for containers
    dependencies: [docker, llama_heavy_v1]

harnesses:
  - name: cline_local
    base_type: cline_cli
    config:
      config_dir: "~/.sovereign/harnesses/cline_local"       # CLINE_DIR — isolated
      provider: openai_compatible
      base_url: "{{ llama_heavy_v1.endpoint }}"
      model: "llama3-70b"
    dependencies: [llama_heavy_v1]
```

### Field reference (additions to the original)

| Field | Purpose |
|---|---|
| `config.max_parallel` → `-np` | Concurrent request slots. Load-bearing under harness teams — total context divides across slots. |
| `harnesses[].config` | Passthrough dict for the harness; `materialize()` turns it into the tool's own config format. |

**Stable names + pinned ports across variant files.** If `llama_heavy_v1` keeps its name and port in every variant, swapping the underlying model doesn't change `{{ llama_heavy_v1.endpoint }}`, so a frontend's baked env vars stay valid and engine swaps stop cascading restarts. Adopt this convention now even though diff-reconciliation is v2 (§7b).

All original fields (`priority`, `affinity`, `health_check`, `policy`, `config`, `env_overrides`, `secrets`, `dependencies`) are unchanged.

---

## 6. Orchestrator Design

### 6.1 State machine (services)

| State | Meaning | Transition trigger |
|---|---|---|
| `PENDING` | Parsed, not yet acted on | Config loaded |
| `PROVISIONING` | Installing/pulling what's needed | `prepare_environment()` running |
| `STARTING` | Process/container spawned | `start()` called |
| `READY` | Health check passing | `is_healthy()` → `True` |
| `DEGRADED` | Was ready, now failing | `is_healthy()` → `False` after `READY` |

Jobs use a separate lifecycle with terminal states — see §6b.

### 6.2 Boot sequence

1. Parse `sovereign.yaml` → Pydantic-validate → fail fast with a field-level error if invalid.
2. Build the dependency graph; topologically sort it. A cycle raises `CircularDependencyError` immediately.
3. For each service in dependency order: `prepare_environment()` → admission via `ResourceBudgeter.can_fit()` → resolve `{{ }}` templates and `${ENV:}` secrets against the runtime `ServiceRegistry` (consumer-aware host) → `start()` → poll `is_healthy()` (2s interval) until `READY` or timeout.
4. Once `READY`, register the resolved endpoint into the `ServiceRegistry`.
5. Independent DAG branches boot concurrently via `asyncio.gather`; dependents wait.
6. **After the stack is up, write the resolved stack manifest** (§7b) and materialize any harnesses whose dependencies are `READY`.

### 6.3 Reconciliation loop (steady state)

- Every **2s**: `is_healthy()` per service. A `READY → False` transition marks it `DEGRADED` and triggers the restart policy.
- Every **10s**: `get_metrics()` per service (via `psutil.oneshot()`). Feeds the Budgeter and dashboard.
- **On endpoint change: re-materialize dependent harness configs.** Otherwise a service restart on a new port silently strands every harness pointing at it.

Both loops are `asyncio`-based, keeping monitoring overhead negligible with a dozen services running.

### 6.4 Shutdown

Reverse topological order. Frontends stop before their engines, containers before the Docker daemon manager, native processes get `SIGTERM` (not `SIGKILL`) so they can flush caches cleanly.

---

## 6b. Benchmarking Subsystem

The differentiator over "just run llama-bench yourself": Sovereign is the only component that knows the *full resolved state* — exact flags, `gpu_layers`, context, co-resident services — so it can stamp every result with a reproducible config + hardware fingerprint.

**Architecture:**
- `sovereign bench` is a **second consumer of the Orchestrator-as-library**, peer to `sovereign serve`. Jobs use terminal states `COMPLETED` / `FAILED`. (Services never finish, so this is a distinct `Job` type, not a stretched service state machine.)
- Two run modes: **attach** (measure the live stack read-only, annotate what else was running) and **clean-room** (bench owns the stack, boots each matrix cell, tears down between cells — needs a lockfile so it can't fight the daemon). Clean-room sidesteps the CLI↔daemon IPC question (§11.2) entirely for v1.
- **Sweeps declarative, invocation imperative.** A bench spec defines `suite × (engine/model/quant/flags) × harness × harness-settings`, run via `bench run`. For v1 the stack axis just enumerates variant files (reusing §7b capture); inline override matrices are v2.

**Making iteration rapid (the actual goal):**
- **Funnel, not flat matrix.** Perf benchmarks (seconds/cell) run first and gate quality benchmarks (minutes–hours/cell) via thresholds in the spec — min tok/s, TTFT ceiling, memory headroom. Only survivors get agentic runs. The Budgeter pre-prunes cells that can't fit for free.
- **Content-address every cell.** Cell key = `hash(stack manifest × harness fingerprint × suite version × seed)`. Completed cells skip on re-run — change one axis, only that slice re-executes. This is *the* feature that makes iteration fast.

**Two benchmark families:**
- *Performance* (tok/s, TTFT, memory-under-load) — needs only a live endpoint. No harness work. Ships first.
- *Quality / agentic* (harness × model on task suites) — needs `invoke()` + a sandboxed workspace: a throwaway Docker container hitting the native engine via `host.docker.internal` (§2.14).

**Measurement discipline:**
- **Grade the repo, not the transcript.** Git diff + test results are ground truth; the agent's self-report is metadata. **False-completion rate** (claims success, empty diff / failing tests) is a first-class metric.
- **Don't author eval content.** Adapt existing suites (llama-bench, SWE-agent's benchmarks, Aider polyglot), pin versions in the run manifest. The highest-value suite is 10–15 of *your own* real tasks in a simple native format. Grade programmatically where possible; reserve LLM-judging for free-form output, pin the judge, never let the model under test judge itself.
- **Respect variance.** 3+ trials/cell, seeds pinned where the engine honors them, per-task time/token budgets so a looping agent can't eat the sweep. Report mean **and** spread.
- **Output is a Pareto frontier, not a winner.** `bench compare` joins manifests across axes and shows the speed/quality tradeoff surface. No auto-optimizer in v1 — you're the optimizer.
- **Results are files:** `.sovereign/benchmarks/<run_id>/` with a manifest, consistent with the `state.json` philosophy.

---

## 7. Resource & Memory Management

Unified memory can't be hard-partitioned the way VRAM can, so Sovereign manages it by proxy: translate a declared budget into the flags each engine respects, then refuse to overcommit.

| Knob | Engine | What it controls |
|---|---|---|
| `gpu_layers` → `-ngl` | llama.cpp | Metal-offloaded vs. CPU work |
| `context_size` → `-c` | llama.cpp | KV-cache size (scales memory linearly) |
| `threads` → `-t` | llama.cpp / vLLM | CPU core utilization |
| `max_parallel` → `-np` | llama.cpp | Concurrent request slots |
| Metal cache limit | mlx_lm | `mx.metal.set_cache_limit(n)` |
| `priority` | all | Mapped to `os.nice()` / `taskpolicy` QoS class |

**Admission control:** before `start()`, `ResourceBudgeter.can_fit(estimated_bytes)` checks `(total - reserved - safety_margin)`. On failure the Orchestrator refuses the boot with a specific error ("Cannot start `llama_heavy_v1`: needs ~40.0 GB, only 22.0 GB available — stop `comfyui` to free memory") instead of letting macOS swap.

**Teams make `-np` load-bearing.** An agent team issues concurrent model calls, and llama-server divides total context across slots — a 4-agent team on `-c 32768` quietly runs ~8k tokens per agent. The Budgeter and the bench spec must treat **slots × context jointly**, not as independent knobs.

**Prompt caching** is a `policy` block, not a raw flag, because it needs pre-flight validation (cache dir writable? enough disk? does an existing cache match the current model hash?). `prepare_environment()` validates it; `get_start_args()` turns the validated policy into CLI flags. Under pressure the reconciliation loop can call `adjust_resources()` on lower-priority services to shrink caches.

**Eviction policy:** refuse-to-boot with a clear message; never auto-kill a running service (see §11.5).

---

## 7b. Stack Variants & Switching

- **Variants = separate full YAML files**, invoked with `sovereign up -f heavy-70b.yaml`. Zero schema changes; Pydantic validates any file. Tolerate duplicating the `resources` block in v1.
- **Capture:** the Orchestrator writes a **resolved stack manifest** at boot — final flags, resolved ports, model fingerprints (`path + size + mtime`, *not* a 40GB content hash), and co-resident services. `state.json` records which variant file + its content hash produced the running state, so `status` can flag drift. This manifest is also what the benchmark runner consumes — build it once.
- **v1 switching = full down/up.** Model load time dominates anyway.
- **Diff-based reconciliation** (restart only services whose resolved config changed → "swap the engine while the frontend stays up") is the v2 upgrade — but the stable-names/pinned-ports rule (§5) is what makes it possible, so adopt the convention now.

---

## 8. Telemetry & Dashboard

`get_metrics()` is part of the Protocol, so the Orchestrator gathers `{name: metrics}` for every service identically:

```python
def get_metrics(self) -> dict[str, Any]:
    p = psutil.Process(self.process.pid)
    with p.oneshot():
        return {
            "memory_bytes": p.memory_info().rss,
            "cpu_percent": p.cpu_percent(interval=None),
            "status": "running",
        }
```

**`sovereign monitor` — final direction.** A clean, minimalist `rich.Table`, refreshed on an interval, styled like `top`/`htop` rather than a graphing dashboard.

```
$ sovereign monitor

  Sovereign Control Plane v1.1.0
  ──────────────────────────────

  SERVICE          STATUS      CPU %    MEM         DEPENDENCIES
  ──────────────────────────────────────────────────────────────
  llama_heavy_v1   [RUNNING]    12.4%    15.2 GB     -
  open_webui       [STARTING]    0.5%    891.3 MB    docker, llama_heavy_v1
  ──────────────────────────────────────────────────────────────
  [Press Ctrl+C to exit]
```

Panel/sparkline/plotext variants are deferred — the `get_metrics()` contract already gives you everything needed to add richer visuals later without touching the Orchestrator.

---

## 9. macOS Process Model

- **`LaunchAgent`, not `LaunchDaemon`.** Runs as your user, with access to Homebrew, Docker context, and HF token — no root headaches.
- Sovereign generates `com.local.sovereign.plist`, writes it to `~/Library/LaunchAgents/`, and runs `launchctl load` on install. `KeepAlive` + `RunAtLoad` let launchd handle crash-restart.
- The binary launchd runs is `sovereign serve` — a normal foreground process.
- **CLI ↔ daemon:** read-only commands (`status`, `monitor`) read `.sovereign/state.json`; no IPC needed for v1. A real transport (Unix socket or local FastAPI server) is added only when imperative commands like `sovereign restart <service>` must reach the *running* daemon (§11.2).

---

## 10. Initial Service & Harness Catalog

| Entity | `base_type` | Kind | Layer | Status |
|---|---|---|---|---|
| Docker daemon interface | `docker` | Service | Infrastructure | ✅ Implemented (Phase 3) |
| llama.cpp | `llama_cpp` | Service | Inference | ✅ Implemented (Phase 4) |
| Open WebUI | `open_webui` | Service | Frontend | ✅ Implemented (Phase 5) |
| SearXNG | `searxng` | Service | Convenience | ✅ Implemented — dynamic web-search env wiring into `open_webui`; 2nd Docker service, first multi-dependency test |
| MLX (oMLX/mlx_lm) | `mlx_lm` | Service | Inference | ✅ Implemented — native pattern, proven alongside `llama_cpp` |
| ComfyUI | `comfyui` | Service | Visual generation | ⬜ Not started — **native**, first-class Budgeter citizen (§2.13); best validation case for refuse-to-boot |
| Cline CLI | `cline_cli` | Harness | Coding agent | ✅ Implemented — subprocess, isolated `CLINE_DIR`, `--yolo`/`--json` headless |
| mini-swe-agent | `mini_swe_agent` | Harness | Coding agent | ✅ Implemented — in-process `DefaultAgent`/`LitellmModel`, minimal-surface suite engine (chosen over full SWE-agent) |
| Vector DB / RAG | `vector_db` | Service | Memory | Deferred (§11.3) |
| Jupyter, Playwright | — | Service | Convenience | Deferred (§11.3) |
| **Ollama** | — | — | — | **Excluded** — wraps llama.cpp; Sovereign talks to llama.cpp directly. |
| ~~LiteLLM gateway~~ | ~~`litellm`~~ | — | — | **Dropped** — strictly-local (§11 Open Decision #1); no Anthropic-Messages gateway needed |
| ~~Claude Code~~ | ~~`claude_code`~~ | — | — | **Dropped** — cloud/proxy harness, out of scope for a strictly-local Sovereign |

---

## 11. Open Decisions

**Resolved during design:**
- §11.1 `type` vs `base_type` — **drop `type`, keep only `base_type`.** Locked in Phase 1. Each manager already encapsulates native-vs-container internally.
- ComfyUI placement — **native** (§2.13).
- SearXNG scope — **clears the bar**, scheduled as 2nd Docker service. ✅ Implemented.
- Harness shared-state risk — resolved: Cline **CLI** uses isolated `CLINE_DIR`, no merge-patch (the `settings.json` concern applied to the VS Code path, which is out of scope).
- `omlx`/`mlx_lm` timing — landed early; the Adapter pattern kept it off the Orchestrator. ✅ Implemented.
- **Open Decision #1 — Cloud baseline: RESOLVED → Sovereign is strictly local.** No paid cloud model is an allowed benchmark baseline. **LiteLLM and Claude Code are dropped** from the catalog (§10) and roadmap (§12). The Pareto frontier is anchored entirely by local engine × model × harness combinations.

**Still open — defaults are included but not chosen:**

1. **Is "swap engine while frontends stay up" a daily-driver need?** If yes, diff-reconciliation moves from v2 into Phase 6 scope. If nice-to-have, it stays v2. *Default: v2 (full down/up).*
2. **Auto-kill under memory pressure (§11.5).** *Default: refuse-to-boot, never auto-kill; revisit after living with refusal.* Ratify explicitly.

---

## 12. Phased Roadmap

Each phase has a concrete exit criterion — don't move on until it's true.

**Status legend (2026-07-05 snapshot):** ✅ complete · 🟡 partial · ⬜ not started.

**Phase 0 — Repo bootstrap.** ✅ **Complete.** `uv init`, `src/` layout, `pyproject.toml` (Ruff + pytest), `.gitignore`, empty Typer app.
*Exit: `uv run sovereign --help` runs.*

**Phase 1 — Core contracts.** ✅ **Complete.** `core/base_config.py`, `core/base_manager.py` (`ServiceManager`), `core/base_harness.py` (`Harness`), `core/registry.py`. Resolve the `type`/`base_type` question (§11.1).
*Exit: a dummy manager passes `isinstance(x, ServiceManager)`; a dummy harness passes `isinstance(x, Harness)`.*

**Phase 2 — `sovereign.yaml` parsing.** ✅ **Complete.** Top-level `SovereignConfig`; `services: list[ServiceEntry]`, `harnesses: list[HarnessEntry]`; descriptive validation errors.
*Exit: a fixture YAML with services + harnesses loads and validates in a test.*

**Phase 3 — First service: `docker`.** ✅ **Complete.** Thinnest possible manager — verify the daemon is reachable, expose `run_compose()`.
*Exit: `is_healthy()` correctly reflects whether Docker Desktop/OrbStack is running.*

**Phase 4 — First native engine: `llama_cpp`.** ✅ **Complete.** Real subprocess lifecycle, HTTP `is_healthy()`, `psutil` `get_metrics()`, `prepare_environment()` validating `model_path`.
*Exit: a one-service `sovereign.yaml` boots a real `llama-server` and reports `RUNNING`.*

**Phase 5 — First container service + dynamic wiring: `open_webui`.** ✅ **Complete.** Build `core/resolver.py` for `{{ }}` templates, `${ENV:}` secrets, **and consumer-aware host** (`host.docker.internal` vs `localhost`). Catch the container-host issue here, not mid-benchmark.
*Exit: `open_webui` auto-connects to `llama_heavy_v1` with zero manual config.*

**Phase 6 — The Orchestrator.** ✅ **Complete.** DAG + topological sort, async boot, `ServiceRegistry`, reconciliation loop. **Also: write the resolved stack manifest, record source variant + hash in `state.json`.**
*Exit: a 3+ service stack boots in dependency order; killing a process is detected within ~2s; a stack manifest is written.*

**Phase 7 — Resource manager.** ✅ **Complete.** `core/resources.py` (`ResourceBudgeter`), admission control in the boot step, prompt-caching validation + flag generation, slots×context accounting.
*Exit: booting a service that would exceed the budget is refused with a specific, actionable error — not a crash.*

**Phase 8 — CLI surface.** ✅ **Complete** (the `bench` subcommand is still a stub, pending the bench track). Typer commands: `up` (with `-f <variant>`), `down`, `status` (**with drift detection** against the recorded variant hash), `logs <service>`.
*Exit: full `--help` tree; `status` reflects real state and flags drift.*

**Phase 9 — `launchd` persistence.** 🟡 **Partial** — the `sovereign serve` entrypoint and `.sovereign/state.json` are done; **`utils/launchd.py` plist generation + `sovereign install`/`uninstall` are still missing.** Resolve CLI↔daemon (§11.2).
*Exit: after `sovereign install`, a fresh terminal's `sovereign status` reflects the running daemon, surviving logout/login.*

**Phase 10 — Dashboard.** ✅ **Complete.** `sovereign monitor` — the minimalist `Table` version.
*Exit: live-updating table matches the mockup; refresh doesn't visibly steal CPU from inference.*

**Phase 11 — Catalog expansion.** 🟡 **Partial** — `searxng` ✅ and `mlx_lm` ✅ are done; **`comfyui` (native) is still needed.** `litellm` is dropped (strictly-local, §11 Open Decision #1). Remaining after ComfyUI: convenience services that clear the "real module, not a wrapper" bar.
*Exit: each addition is a new `services/<name>/` folder with zero Orchestrator changes.*

**Phase 12 — Testing & hardening.** 🟡 **Ongoing** — `tests/` mirrors `src/` and **the full suite passes** with mock `ServiceManager`/`Harness` implementations (no Docker or real models needed); Ruff clean.

### Two parallel post-MVP tracks (roughly alongside Phase 11)

**Harness track** ✅ **Complete** — see [`harness-bench-implementation-plan.md`](./harness-bench-implementation-plan.md) for the detailed phase breakdown. `materialize()`/`invoke()` hardened (endpoint-change re-materialization, manifest inclusion, `served_model_name`/`api_model_name`), then **Cline CLI** and **mini-swe-agent** landed as the first two concrete harnesses, both invocable via `sovereign harness list/materialize/invoke`. (Claude Code dropped — §11 Open Decision #1; full SWE-agent and the Cline teams axis deferred as optional follow-ups.)

**Bench track** ✅ **Complete** — see [`harness-bench-implementation-plan.md`](./harness-bench-implementation-plan.md). `Job`/`BenchSpec`/content-addressed cells → attach-mode perf prober (in-house `httpx`, TTFT/tok-s/latency with mean+spread) → clean-room sweeps (bench-owned boot/measure/teardown, `bench.lock` so it can't fight the daemon) → agentic quality runner (native task-suite format, grade-the-repo via git diff + programmatic grader, false-completion rate, perf/quality funnel gating) → `bench compare` (Pareto join across runs). Sandboxed (Docker) quality workspaces, SWE-bench subset, and a guidellm adapter are documented as optional later extensions.

---

## 13. Day 1 Checklist

1. `uv init sovereign && cd sovereign`
2. Create the `src/sovereign/` tree from §3 (empty files are fine).
3. Write `core/base_manager.py` (`ServiceManager`) and `core/base_harness.py` (`Harness`) — the two files everything else depends on.
4. Write `tests/test_protocol.py` asserting a trivial fake manager satisfies `ServiceManager` and a trivial fake harness satisfies `Harness`.
5. Write down the locked decisions in the README: §11.1 (`base_type` only), §11.5 (refuse-to-boot, no auto-kill), and Open Decision #1 (**strictly local** — LiteLLM + Claude Code dropped).
6. Start Phase 2 (`sovereign.yaml` parsing) against the canonical schema in §5.
