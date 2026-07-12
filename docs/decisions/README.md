# Architecture Decision Records

This directory records the load-bearing decisions behind Sovereign's design —
the ones that would otherwise have to be re-derived by reading commit history
or asking around. It complements, but does not replace, the frozen
`docs/sovereign-implementation-plan-v1.1.md` (the historical `§N` anchor) and
the living `docs/architecture.md` (current invariants, kept up to date).

## Convention

- Filename: `NNNN-kebab-case-title.md`, numbered sequentially, never reused.
- One ADR per decision; keep it to about one page.
- Cite provenance: the plan section (`§N`) or PR that established the
  decision, so a reader can trace it back to the discussion that produced it.

## Status

Each ADR has a `Status:` line, one of:

- `accepted` — currently in force.
- `superseded-by-NNNN` — replaced by a later ADR (which should exist and
  itself be `accepted`, forming a chain a reader can follow).

## When an ADR is REQUIRED

Write one before (or alongside) a change to any of:

- Layering / dependency direction between packages (`config` / `core` /
  `services` / `runtime` / `bench` / `workers`).
- A core Protocol or contract (`ServiceManager`, `Harness`, `WorkerConfig`,
  the telemetry wire schema, `StatusSnapshot`).
- The `sovereign.yaml` schema's *semantics* (not just adding an optional
  field — its meaning, defaults, or validation behavior).
- The memory / admission-control model (refuse-to-boot, budgeting math).
- The process / lifecycle model (boot sequence, teardown, detached workers).
- The telemetry wire protocol (event types, transport, drop semantics).
- Testing seams (what gets mocked, where the boundary between hermetic and
  integration tests sits).

If a change touches one of these areas and you decide a new ADR *isn't*
warranted, say so explicitly in the PR description ("Decision recorded? Not
needed because …") — see `.github/pull_request_template.md`.

## Template

```markdown
# NNNN. Title

Status: accepted

## Context

What situation forced a decision? What constraints applied?

## Decision

What we decided, stated plainly.

## Consequences

What this makes easier, what it makes harder, what it forecloses.

## Alternatives considered

What else was on the table and why it lost.

---
Provenance: plan §N / PR #NN.
```

## Index

| ADR | Title |
| --- | --- |
| [0001](0001-declarative-control-plane-no-daemon.md) | Declarative control plane, per-directory state, no daemon |
| [0002](0002-registry-self-registration-base-type-routing.md) | Registry self-registration + `base_type` routing |
| [0003](0003-refuse-to-boot-admission-control.md) | Refuse-to-boot admission control, never auto-kill |
| [0004](0004-embedded-engine-workers.md) | Embedded engine workers over external CLIs |
| [0005](0005-uds-ndjson-telemetry.md) | UDS NDJSON telemetry, drop-when-unobserved |
| [0006](0006-engine-gap-policy.md) | Engine-gap policy: surface loudly, bridge when feasible |
