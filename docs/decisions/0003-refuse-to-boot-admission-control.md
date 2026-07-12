# 0003. Refuse-to-boot admission control, never auto-kill

Status: accepted

## Context

Apple Silicon's unified memory has no hard OS-level partition between
processes the way discrete VRAM does. Multiple GPU-bound services (a large
GGUF plus a Flux image model, say) can jointly exceed physical memory with
no OS-level guard rail, and macOS will swap rather than fail fast — a much
worse failure mode (thrashing, unusable machine) than an upfront refusal.

## Decision

`ResourceBudgeter.can_fit(estimated_bytes)` runs before every `start()`,
checking `(total - reserved - safety_margin)`. On failure the Orchestrator
refuses the boot with a specific, actionable error (e.g. "Cannot start
`llama_heavy_v1`: needs ~40.0 GB, only 22.0 GB available — stop `comfyui` to
free memory") instead of starting the process and letting macOS swap.
Sovereign never kills a running service to make room for a new one — the
eviction policy is refuse-to-boot only.

## Consequences

- Predictable failure mode: a refused boot is a clear message, not a stalled
  machine.
- Users control what stays up; Sovereign never makes that call for them.
- `plan`/`boot` share the exact same admission math
  (`core/planning.py` reuses `estimate_service_memory` and
  `route_entry`) so `sovereign plan` never drifts from what `up` will do.
- Cost: no automatic "make room" behavior — a user who wants the new service
  more than the old one must stop the old one manually. Revisit only after
  living with plain refusal (plan §11, Open Decision #2).

## Alternatives considered

- Auto-kill lower-priority services under pressure — rejected: killing a
  running service a user didn't ask to stop is a worse surprise than a
  refused boot, and is explicitly the trap ComfyUI + a 70B GGUF is meant to
  validate against (plan §2.13).
- Best-effort overcommit with a warning only — rejected: on unified memory
  this degrades into swap thrash, not a recoverable warning state.

---
Provenance: plan §2 (point 8), §7, §11.5.
