Fixes the findings from a full project audit ("roast"): the headline guarantees — files-as-IPC, refuse-to-boot, faithful status — were each undercut by their implementation. Ten commits so far, each green on `make check`.

## Landed

**Phase 0 — coordination-layer correctness**
- `bd59fb9` Atomic `write_json` (same-dir temp + `os.replace`) and tolerant readers — `monitor`/`status`/`down` no longer crash on a half-written state file
- `fc3f56a` `sovereign down` verifies PID identity (`create_time`) before signaling — a recycled PID is never killed
- `b21fb8e` Runtime handles persist as each service starts (no more untracked orphans after mid-boot Ctrl+C); reconcile loops fault-isolate per service, including the Docker `--%` stats crash
- `d00ac7a` Bench lock uses atomic `O_EXCL` acquisition with stale-lock (dead PID) recovery

**Phase 1 — stop lying to the operator**
- `fc00c39` Fail-open admission is surfaced: unknown memory footprints warn at boot and are flagged in `sovereign plan`
- `8d5fa6f` A repo-id-shaped model ref (`org/name`) can no longer be hijacked by a same-named directory in the CWD
- `47c80eb` llama_cpp API key moves off the `ps`-visible argv into the environment; manifest serialization redacts key-shaped values
- `cc01623` Unknown harness `base_type` is a boot error; missing optional deps warn loudly; health acceptance unified at 2xx-only

**Phase 2 / 3 (partial)**
- `247356e` Opt-in integration smoke test that boots a real GGUF through `llama-server` on a macOS runner (non-blocking CI job), plus CI restructure: lint/typecheck/hermetic tests on Linux, one macOS hermetic job, `--all-extras` installed
- `2958838` `requires-python` lowered to `>=3.12` (verified; no 3.13+ features in the tree)

## Outstanding (in progress on this branch)

- `648c6a3` P2.3 complete — add `tests/` to mypy; test fakes checked against Protocols; integration smoke test (`test_llama_stack_boots_serves_and_tears_down`) added and deselected by default (marked `integration`, excluded via `-m 'not integration'`)
- `P2.4` complete — coverage visibility (`pytest-cov` + `make coverage` + HTML report in `htmlcov/`)
- `P3.2` complete — hoisted common `estimated_memory_gb`/draft-model handling into `NativeEngineManager`; engines override for specific overhead (KV cache, prompt cache)
- [ ] P3.3 — replace remaining `getattr`-probing with Protocol `isinstance` checks (orchestrator activity, CLI provision hooks)
- [ ] P3.4 — repo-root cleanup (`sovereign.yaml`/`mlx.yaml` → `examples/`, plan docs → `docs/`)
- [ ] P3.5 — RoutingCache IO errors logged instead of `contextlib.suppress(Exception)`

---
