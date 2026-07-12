# 0001. Declarative control plane, per-directory state, no daemon

Status: accepted

## Context

Sovereign manages local LLM infrastructure on a single developer machine.
Something has to own "what should be running" and "what is actually
running." A daemon (background service, root LaunchDaemon, always-on
supervisor) is the conventional answer, but it adds a persistent process to
reason about, a socket/IPC surface, upgrade/restart hazards, and a second
place state can drift from reality.

## Decision

`sovereign.yaml` describes desired state only; all "how to make it true"
logic lives in ordinary Python invoked by an ordinary CLI process
(`sovereign up`, `sovereign down`, `sovereign status`). There is no daemon.
State is per-directory: everything a stack needs (`state.json`,
`status.json`, `manifest.json`, `logs/`, `benchmarks/`) lives under
`.sovereign/` relative to the CWD. Separate CLI invocations coordinate
through these files, not through IPC. Persistence across logout is handled
by a user-space `launchd` `LaunchAgent` (not a root `LaunchDaemon`) running
Sovereign as an ordinary foreground process — launchd supplies
restart-on-crash and log capture, nothing more.

## Consequences

- Easy to reason about: `cat .sovereign/state.json` is the whole story.
- Multiple stacks (multiple directories) never collide or share state.
- No daemon upgrade/compatibility problem, no privileged process.
- Costs: no cross-directory view without a wrapper; coordination between
  concurrent CLI invocations (e.g. `up` and `monitor`) is file-based and can
  be briefly stale (documented where it matters, e.g. telemetry).

## Alternatives considered

- A root `LaunchDaemon` — rejected: unneeded privilege, harder crash-log
  ergonomics, root-owned files fighting a per-user tool.
- A resident supervisor process with an IPC/socket API — rejected for v1:
  more moving parts than the problem needs; revisit only if a driving use
  case for live cross-process control appears.

---
Provenance: plan §2 (points 2, 10, 12) and §9.
