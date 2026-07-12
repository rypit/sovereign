# 0006. Engine-gap policy: surface loudly, bridge when feasible

Status: accepted

## Context

Moving llama.cpp to embedded Python bindings (ADR 0004) traded the
`llama-server` CLI's full feature surface for whatever `llama-cpp-python`
actually exposes. Some CLI features have no equivalent in the bindings at
all (true multi-slot concurrency); some have a partial equivalent that
needs bridging work to restore (two-model GGUF speculative decoding, where
the bindings only ship n-gram prompt-lookup decoding out of the box); some
degrade cleanly to an adjacent implementation (disk-backed prompt-cache
policy → in-process RAM cache). Silently dropping any of these — accepting
degraded behavior without saying so — would be a worse regression than the
gap itself.

## Decision

Per gap, in order of preference:

1. **Bridge it ourselves** if the underlying capability exists but the
   convenience wrapper doesn't — e.g. two-model GGUF speculative decoding:
   the bindings expose a `LlamaDraftModel` ABC callable, so Sovereign
   implements its own Python-level greedy draft loop
   (`workers/llama_cpp_adapter.py`) rather than accepting the regression.
   Both models are still counted in unified-memory admission (ADR 0003).
2. **Degrade to a clean equivalent** and say so in the config docstring —
   e.g. disk-backed slot-save prompt caching becomes `cache=True,
   cache_type="ram"`; a `cache_path` is accepted-but-inert with a `LOG`
   warning, never silently ignored.
3. **Hard-error at `prepare_environment()`** with an actionable message when
   neither of the above applies and the CLI's behavior genuinely cannot be
   approximated — e.g. `max_parallel` (`-np`) has no multi-slot equivalent
   in a single embedded `Llama` instance; boot-time `LOG` warning when
   requested `>1`, documented in `per_slot_context()`'s docstring.

Never fake a capability (e.g. reporting fractional prefill progress where
the bindings only offer start/finish) — report what's actually knowable
(llama.cpp's prefill bar is indeterminate; mlx_lm's is a true fraction,
since `mlx_lm.server`'s callback provides one).

## Consequences

- Users get an accurate mental model of what each engine can do embedded,
  not a false parity claim with the CLI server.
- Real capability work (the GGUF draft-model bridge) happens where it's
  tractable instead of being deferred indefinitely as "known regression."
- Cost: engine parity is uneven — llama_cpp has more caveats than mlx_lm
  (which maps to `mlx_lm.server` natively with no gaps). This asymmetry is
  documented per-engine (`workers/llama_cpp_adapter.py`,
  `services/inference/llama_cpp/manager.py`), not hidden.

## Alternatives considered

- Accept the CLI-to-bindings regression as-is (no bridging) — rejected: the
  draft-model gap was bridgeable and the user directive was to close gaps
  where feasible rather than settle for silent regression.
- Fake missing telemetry (synthesize a fractional prefill bar for
  llama.cpp) — rejected: reporting invented precision is worse than
  reporting an honest indeterminate state.

---
Provenance: PR #20 addendum "restore GGUF draft-model speculation for
llama_cpp"; cited in code as `§3a` (the "Engine embedding" section's
"Hard gaps" discussion in that PR's plan, not a section of
`docs/sovereign-implementation-plan-v1.1.md`).
