# Harnesses as a First-Class Concept + Native-Engine Benchmarking — Phased Plan

*Companion to [`sovereign-implementation-plan-v1.1.md`](./sovereign-implementation-plan-v1.1.md): this document turns the two post-MVP tracks it sketches (§4b Harness contract, §6b Benchmarking subsystem, §12 parallel tracks) into a concrete, phase-by-phase build order with exit criteria, sized so each phase is independently implementable and testable.*

## Context

Sovereign's MVP orchestration spine is done (Phases 0–8 + 10; the suite is green). The v1.1 design doc already specifies the two remaining post-MVP tracks — **harnesses** (coding agents configured against and invoked on the local stack) and **benchmarking** (perf + agentic-quality sweeps over engine × model × harness cells) — and the scaffolding is deliberately stubbed in.

**Already exists (build on, don't recreate):**

- `src/sovereign/core/base_harness.py` — `Task`, `RunResult` dataclasses + `Harness` Protocol (`name`, `dependencies`, `materialize()`, `invoke(task)`), `@runtime_checkable`.
- `src/sovereign/core/registry.py` — separate harness registry: `register_harness(base_type)` / `get_harness(base_type)`, symmetric to services.
- `src/sovereign/config.py` — `HarnessEntry` (name, base_type, dependencies, config dict) + `harnesses:` YAML section, cross-section name/dep validation.
- `src/sovereign/orchestrator.py` — `harness_factory` ctor arg, `_build()` instantiates harnesses, `boot()` ends by calling `_materialize_harnesses()`, which duck-types `resolve(self.resolver)` then `materialize()` once all deps are READY.
- `src/sovereign/core/resolver.py` — `{{ svc.attr }}` templates, `ConsumerKind.NATIVE/DOCKER` with `host.docker.internal` rewrite, `ResolvedEndpoint(scheme, host, port)` with `attribute()`.
- `src/sovereign/utils/manifest.py` — resolved stack manifest (`build_manifest`), explicitly documented as the bench runner's input; per-service `start_args`, `model_fingerprint {path,size,mtime}`, `co_resident`, `per_slot_context`.
- `src/sovereign/main.py` — `bench` command stub; `serve_forever()` is the Orchestrator-as-library entry.
- `core/resources.py` — `ResourceBudgeter.can_fit/admit/release`.

**Gaps this plan fills:** no concrete harness packages; `harnesses/__init__.py` imports nothing; no re-materialize on endpoint change; harnesses absent from the manifest; no OpenAI "served model name" on engines; empty `bench` command; no `Job` type, bench spec, cell content-addressing, perf prober, task suites, or reports.

**Decisions taken for this plan** (follow the v1.1 doc's own recommendations; revisitable):

1. **Harness lineup:** Cline CLI (subprocess, npm) + **mini-swe-agent** (pip dep, in-process Python API) — mini-swe-agent first since it's pure Python and proves the whole pipeline; it also covers the "SWE-agent suite engine" role with minimal surface.
2. **Perf bench:** in-house async prober (httpx + asyncio, streaming `/v1/chat/completions`) rather than guidellm — full control over metrics and manifest stamping; a guidellm adapter can be added later behind the same result schema.
3. **Quality suite v1:** custom local task-suite format (the doc's "10–15 of your own real tasks"); SWE-bench subset is a later optional phase.
4. **Sandboxing v1:** throwaway host git workspaces; Docker sandboxes are a later phase (resolver already supports `ConsumerKind.DOCKER`).

**Locked constraints (do not violate):** strictly local — no cloud baselines; `config.py` never imports `manager.py`; refuse-to-boot, never auto-kill; bench specs are NOT in `examples/sovereign.yaml` (imperative `sovereign bench run` only); grade the repo, not the transcript.

**New dependencies:** `httpx>=0.27` (bench prober); `[project.optional-dependencies] harness = ["mini-swe-agent>=1.0"]` (imported lazily in the manager so the base install stays lean). Cline CLI is an npm binary — installed via Brewfile/npm, checked by `prepare_environment()`-style validation, never a Python dep.

---

## Phase H1 — Harness plumbing hardening (core, no new harnesses yet)

Make the existing hooks production-grade before any concrete harness lands.

1. **`served_model_name` on native engines.** Add optional `served_model_name: str | None` to `services/llama_cpp/config.py` and `services/mlx_lm/config.py`.
   - llama_cpp: emit `--alias <name>` in `get_start_args()` when set.
   - Both managers get `api_model_name() -> str`: alias if set, else the `model` string (path/repo-id) — the string an OpenAI-compatible client sends as `"model"`.
   - Extend `ResolvedEndpoint` with `model: str | None = None`; populate it in `NativeEngineManager.endpoint()` (`core/base_native.py`); support `attribute("model")` in `core/resolver.py`. Now harness YAML can say `model: "{{ mlx_heavy.model }}"` and `base_url: "{{ mlx_heavy.endpoint }}/v1"`.
2. **`BaseHarness` shared base** — new `src/sovereign/core/base_harness_impl.py` (or extend `base_harness.py`; keep Protocol + dataclasses as-is). Concrete class holding `HarnessEntry`, class attr `consumer_kind = ConsumerKind.NATIVE`, `resolve(resolver)` storing the resolver and computing `self.resolved_config = resolver.resolve_mapping(entry.config, self.consumer_kind)`, plus `fingerprint() -> dict` (base_type, tool version if obtainable, resolved config hash) — consumed by the manifest and bench cell keys.
3. **Re-materialize on endpoint change.** In `orchestrator.py` `_register_endpoint`: if a service's endpoint differs from the previously registered one *after initial boot completed*, re-run `_materialize_harnesses()` for harnesses whose (transitive) deps include that service. Covers the `_restart` path. Guard: materialize only when all deps READY (existing check).
4. **Harnesses in the manifest.** Extend `utils/manifest.py::build_manifest` with `harnesses: [{name, base_type, dependencies, fingerprint}]` from `orch.harnesses`.
5. **CLI:** new `sovereign harness` Typer sub-app in `main.py`: `harness list` (from config + state), `harness materialize <name>` (one-shot against a running stack: read manifest/state for endpoints — or require the stack up and use a fresh Orchestrator in read-only resolve mode), `harness invoke <name> --prompt "..." [--workdir PATH]` printing the `RunResult`.
6. **Tests** (`tests/test_orchestrator.py` patterns — `FakeManager` + factories): fake harness via `harness_factory` asserting materialize-after-READY, re-materialize on endpoint change, manifest includes harnesses, resolver `attribute("model")`, llama_cpp `--alias` flag generation.

*Exit: a fake harness in a test stack is materialized at boot, re-materialized when its engine restarts on a new port, and appears in `manifest.json`; `sovereign harness list` works.*

## Phase H2 — First real harness: `mini_swe_agent`

New package `src/sovereign/harnesses/mini_swe_agent/` (`config.py` + `manager.py`), registered with `@register_harness("mini_swe_agent")`; add the import to `harnesses/__init__.py`.

- **`config.py`** (Pydantic only): `base_url` (templated), `model` (templated), `api_key: str = "sovereign"`, `step_limit: int = 40`, `timeout_seconds: int = 900`, `config_dir: str = "~/.sovereign/harnesses/{name}"`, `extra: dict` passthrough.
- **`materialize()`**: write resolved settings (endpoint, model, key) as a small YAML/env file under `config_dir` — serves as the durable "harness is wired" artifact and lets a human run mini-swe-agent by hand against the stack.
- **`invoke(task)`**: lazy-import `minisweagent` (clear `ImportError` message pointing at the `harness` extra). Build `LitellmModel(model_name=f"openai/{model}", model_kwargs={"api_base": base_url, "api_key": ...})` + `LocalEnvironment(cwd=task.workdir)` + `DefaultAgent(step_limit=...)`; run with a wall-clock timeout; map outcome to `RunResult` (self-reported success is metadata per the `base_harness.py` docstring); stash trajectory/steps/cost in `RunResult.metadata`.
- **pyproject:** add the `harness` optional-dependency group.
- **Tests** (`tests/harnesses/test_mini_swe_agent.py`): inject a fake `minisweagent` via `sys.modules` monkeypatching (no real dep needed in CI); assert registry lookup, `isinstance(…, Harness)`, materialize file contents, invoke result mapping, timeout handling. Mirror `tests/services/test_llama_cpp.py` structure.

*Exit: with a live `mlx_lm`/`llama_cpp` stack up, `sovereign harness invoke mini_swe_local --prompt "…" --workdir /tmp/x` completes a real headless run against the local endpoint.*

## Phase H3 — Second harness: `cline_cli`

New package `src/sovereign/harnesses/cline_cli/`, `@register_harness("cline_cli")`.

- **Config:** `config_dir` (becomes `CLINE_DIR` — isolated, per the locked decision), `base_url`/`model` templated, `api_key`, `binary: str = "cline"`, `timeout_seconds`, `max_turns`.
- **`materialize()`**: create `config_dir`, write Cline's provider settings file for the `openai_compatible` provider (verify the exact file name/schema against the installed Cline CLI 2.x at implementation time — `cline config` / docs.cline.bot "OpenAI Compatible"); never touch the user's global Cline state.
- **`invoke(task)`**: subprocess `cline --yolo --json <prompt>`-style one-shot (verify exact flags at impl time), `cwd=task.workdir`, `env={**os.environ, "CLINE_DIR": config_dir}`, kill-on-timeout, parse the NDJSON event stream for completion status; meaningful exit codes into `RunResult`.
- **Preflight:** `shutil.which(binary)` with actionable error ("npm install -g cline" / Brewfile entry); record `cline --version` in `fingerprint()`.
- **Tests:** `FakeProc` pattern from `tests/services/test_llama_cpp.py` (monkeypatch `subprocess.Popen`, `shutil.which`); NDJSON parsing fixtures.

*Exit: same invoke round-trip as H2 but through the Cline binary; both harnesses coexist in one YAML.*

## Phase B1 — Bench skeleton: `Job` type, spec, content-addressed cells

New package `src/sovereign/bench/`:

- **`runner.py`** — `JobState` (`PENDING/RUNNING/COMPLETED/FAILED`) + `Job` dataclass (id, cell_key, state, started/finished, error). Terminal states; distinct from the service state machine.
- **`spec.py`** — `BenchSpec` Pydantic model (own file, e.g. `bench.yaml`, loaded only by `bench run` — never part of `SovereignConfig`): `suites: [str]` (paths), `stacks: [str]` (variant YAML paths — reuses §7b variant capture), `harnesses: [str]` (names within those stacks), `trials: int = 3`, `seed: int`, `thresholds: {min_tok_s, max_ttft_ms, min_headroom_gb}`, `budgets: {task_timeout_s, max_tokens}`, `mode: attach|cleanroom`.
- **`cells.py`** — cell key = `sha256(canonical_json(stack-manifest slice for the engine under test × harness fingerprint × suite name+version × seed × trial))`. `is_complete(key)` checks `.sovereign/benchmarks/cells/<key>/result.json`; completed cells skip on re-run (the feature that makes iteration fast).
- **Results are files:** `.sovereign/benchmarks/runs/<run_id>/run.json` (spec hash, cell list, per-cell status) + per-cell dirs. Reuse `utils/state.py::write_json/read_json/file_hash`.
- **CLI:** replace the `bench` stub with a Typer sub-app: `bench run -f bench.yaml`, `bench ls`, `bench compare` (stub until B5).
- **Tests** (`tests/test_bench.py`): spec validation, cell-key stability (same inputs → same key; one axis change → new key), skip-completed.

*Exit: `sovereign bench run -f bench.yaml` parses a spec, enumerates cells, marks all as skipped/failed cleanly (no measurements yet), and writes a run manifest.*

## Phase B2 — Performance benchmarks, attach mode

- **`bench/perf.py`** — in-house prober (new dep `httpx`): async streaming requests to `{endpoint}/v1/chat/completions` using `ResolvedEndpoint` + `api_model_name` from the manifest. Metrics per trial: TTFT, output tok/s, end-to-end latency, tokens in/out; concurrency sweep (1, N) honoring `per_slot_context` from the manifest; memory-under-load sampled from `status.json` metrics (written by the running daemon) or psutil against the manifest's runtime handle.
- **Attach mode:** read `manifest.json` + `state.json` of the live stack (read-only — never boots/stops anything), annotate results with `co_resident` services and the variant hash. 3+ trials; report mean and spread.
- **Funnel:** evaluate `thresholds` from the spec; cells failing perf gates are recorded as `gated` so quality phases skip them.
- **Tests:** fake OpenAI-compatible SSE server (local `asyncio`/httpx MockTransport), assert TTFT/tok-s math, threshold gating, manifest stamping.

*Exit: against a live stack, `bench run` produces per-cell perf results with mean±spread, stamped with variant hash + co-residents; re-running skips completed cells.*

## Phase B3 — Clean-room sweeps

- Bench owns the stack: for each stack axis value, boot the variant via the Orchestrator as a library (`serve_forever` is interactive — add/reuse a bounded `boot() … shutdown()` context path in `orchestrator.py`), run all cells for that stack, tear down before the next. Model-load cost is why cells are grouped per stack.
- **Lockfile** `.sovereign/bench.lock` (pid + run_id): `bench run` refuses if `state.json` shows a live daemon-managed stack (and documents that `sovereign up` should refuse while the lock exists — small guard in `_boot_and_serve`). Never fight the daemon.
- **Budgeter pre-prune:** before booting a cell's stack, use `ResourceBudgeter.can_fit` with the entries' `estimated_memory_gb` to mark impossible cells `gated(memory)` for free.
- **Tests:** orchestrator-as-library boot/teardown with `FakeManager` factories; lock contention cases.

*Exit: a spec listing two variant files sweeps both, booting/tearing down each in turn, refusing to run while a daemon stack is up.*

## Phase B4 — Agentic quality runner (grade the repo)

- **`bench/suites.py`** — native suite format: a directory with `suite.yaml`: `version`, `tasks: [{id, repo: {path|git_url, rev}, prompt, grader: {type: command|pytest, cmd, expect_diff: true}, timeout_s}]`. Ship one example suite under `examples/bench/smoke_suite/` (2–3 tiny tasks against a fixture repo) so the pipeline is testable without authoring the full personal suite.
- **Workspaces:** per task-trial, `git clone`/copy the fixture repo into `.sovereign/bench/workspaces/<cell>/`, pass as `Task.workdir` to `Harness.invoke()`; delete on success, keep on failure for debugging.
- **Grading:** after invoke — `git diff --stat` (empty diff ⇒ no work), run the grader command, record pass/fail, wall time, tokens (from `RunResult.metadata`). **False-completion rate** (harness claimed success ∧ (empty diff ∨ grader failed)) is a first-class metric. The agent's self-report stays metadata.
- **Funnel enforcement:** quality cells run only for cells whose perf results passed B2 thresholds.
- **Tests:** fake harness returning scripted RunResults + fixture repo in `tests/fixtures/`; assert grading matrix (success/failure × diff/no-diff), false-completion accounting, workspace lifecycle.

*Exit: `bench run` with the smoke suite drives a real harness against a real local engine and outputs per-task pass/fail graded from the repo, not the transcript.*

## Phase B5 — Reports & Pareto compare

- **`bench/report.py`** — `bench compare [run_ids…]`: join cell results with their stack manifests across axes; Rich table (engine × model × harness rows; tok/s, TTFT, pass-rate, false-completion, mem columns) + `--json`/`--csv`. Flag Pareto-non-dominated cells on (speed, quality); no auto-optimizer.
- `bench ls` fleshed out (runs, cells, gated/skipped counts).
- **Tests:** golden-file joins over synthetic result dirs.

*Exit: `sovereign bench compare` renders the speed/quality tradeoff surface from two runs' files.*

## Phase B6 (optional, later) — extensions

Docker-sandboxed quality workspaces (throwaway container per task, engine via `ConsumerKind.DOCKER` → `host.docker.internal`; reuse `docker` manager patterns); SWE-bench Lite subset via mini-swe-agent's batch runner (pinned dataset rev); guidellm adapter behind the B2 result schema; Cline teams as a swept config axis (slots×context accounting per §7 — `-np` becomes load-bearing).

---

## Verification (end-to-end)

1. `uv run pytest -q` green after every phase (mock-based; no Docker/models needed in CI) and `uv run ruff check .` clean.
2. **H-track live check (Apple Silicon):** `uv run sovereign up -f examples/mlx.yaml`, then `sovereign harness invoke <name> --prompt "add a hello() to util.py" --workdir <throwaway repo>` → nonzero diff in the workdir; kill the engine, let it restart, confirm the harness config file re-materialized (mtime/content).
3. **B-track live check:** with the stack up, `sovereign bench run -f examples/bench/perf-attach.yaml` → results under `.sovereign/benchmarks/`; run twice, second run reports all cells skipped; `sovereign bench compare` renders the table. Then a clean-room spec with the smoke suite: full boot→invoke→grade→teardown cycle.
4. Per-phase exit criteria above are the gate for moving on (matches the repo's roadmap discipline).

## Notes for the implementer

- Follow the domain layout religiously: new behavior = new folder under `harnesses/` or `bench/`; zero Orchestrator edits beyond the ones listed in H1/B3.
- `config.py` files import Pydantic only. Register via decorators; add package imports to `harnesses/__init__.py`.
- Verify third-party CLI/API details **at implementation time** (Cline settings schema & headless flags; `minisweagent` class paths) — the plan's names are from current docs and may drift; pin what you find in `fingerprint()`.
- Reuse: `utils/state.py` for all JSON IO; `FakeProc`/`FakeManager` test patterns; `Resolver.resolve_mapping`; `estimate_service_memory`; `file_hash` for spec/suite hashing.
- Update `sovereign-implementation-plan-v1.1.md` §10/§12 status markers and the README status section as phases land.
