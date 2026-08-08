# 0009. Docker-sandboxed harnesses with engine-only egress

Status: accepted

## Context

Harnesses run coding agents with every safety off: `cline_cli` executes
`cline --yolo` (auto-approves all tool calls) as a host subprocess inheriting
the full parent environment, and `mini_swe_agent` runs the agent's shell
commands in-process on the host via its `LocalEnvironment`. A misbehaving or
prompt-injected agent can modify the host or exfiltrate data over the open
internet. The engines themselves must stay native (§2.1: nothing GPU-bound in
a container), so isolation has to wrap the *consumer*. The resolver already
anticipated this (`ConsumerKind.DOCKER`, §2.14), and the harness/bench plan
deferred "Docker sandboxes" as a later phase (harness-bench plan decision #4,
Phase B6). A host-native alternative (macOS user-account + `sandbox-exec`
isolation à la SandVault) was evaluated and rejected: it cannot restrict
network egress, which is the requirement that motivates this change.

## Decision

- **Per-invoke throwaway sandboxes.** Each `invoke()` creates (and tears
  down) its own Docker network + containers. Nothing sandbox-related is
  supervised, health-polled, boot-persisted, or admission-counted; sandboxes
  are excluded from the memory budget (an optional `memory_gb` maps to
  `docker run --memory` instead) and from `DockerMonitorWorker`.
- **Strict engine-only egress.** In `engine_only` mode the agent container
  joins an `--internal` network whose only other member is a socat proxy
  (attached to the bridge with `--add-host host.docker.internal:host-gateway`)
  forwarding one TCP port to the native engine. The agent reaches the LLM at
  the fixed alias `http://sovereign-llm:8000` and nothing else. `none` mode
  is `--network none` — used by harnesses whose LLM traffic originates on the
  host (mini_swe_agent), where the command container needs no network at all.
- **Invoke-time endpoint rewrite, no new `ConsumerKind`.** Harness resolution
  stays `NATIVE` (host-truth base_url); the sandbox path parses the loopback
  port out of the resolved URL and substitutes the proxy alias only inside
  the container. `plan`/boot parity is untouched, and engine restarts on new
  ports need no reconcile: the registry is rebuilt from `manifest.json` on
  every CLI invoke and re-materialized by the orchestrator on endpoint change.
- **Label-based cleanup.** Every sandbox object carries
  `sovereign.sandbox=1` + `sovereign.harness=<name>`; teardown is
  try/finally, and `sweep_leftovers()` reaps crash leftovers pre-invoke
  (harness-scoped) and during `sovereign down` (global).
- **Mechanics live in `services/docker/sandbox.py`** (module-level helpers
  routed through the existing `run_docker` seam); the shared `SandboxConfig`
  schema lives in `core/base_config.py` so harness config modules can import
  it under the §2.3 golden rule. Harnesses import the sandbox module lazily
  inside methods.

## Consequences

- Easier: an agent can be given `--yolo`-grade autonomy with a bounded blast
  radius — writable surface is the task workdir bind mount, environment is an
  explicit allowlist, and the network is the engine or nothing.
- Easier: bench quality cells inherit sandboxing from the harness entry with
  no bench-side changes; grading stays host-side over the shared workdir.
- Harder: sandboxed cline needs a Sovereign-owned image (Node + cline) built
  on first use; image drift is a new maintenance surface.
- Harder: ~1–2 s of docker setup/teardown per invoke (negligible against
  multi-minute agent runs).
- Foreclosed: sandboxed harnesses reaching remote (non-loopback) engines —
  the rewrite refuses them rather than widening the tunnel.

## Alternatives considered

- **A new `ConsumerKind.SANDBOX`** resolving straight to the proxy alias:
  rejected — the alias/port are per-invoke runtime facts, and teaching the
  resolver about them would break `plan`/boot parity and leak sandbox
  lifecycle into the manifest.
- **Long-lived per-harness proxy/network**: rejected — goes stale when an
  engine restarts on a new port, and demands reconcile logic the per-invoke
  model gets for free.
- **Extending `DockerManager`**: rejected — it models a supervised
  long-running service (health-polled, admission-counted, boot-persisted);
  a sandbox is a run-to-completion throwaway with the opposite lifecycle.
- **Host-native isolation (separate macOS user + `sandbox-exec`, e.g.
  SandVault)**: rejected as the primary mechanism — native speed and no
  Docker dependency, but no egress control (full outbound internet), macOS-
  only, and effectively untestable in the hermetic suite. The sandbox
  primitive's surface (argv + mounts + env + timeout) deliberately leaves
  room for such a provider later.

---
Provenance: plan §2.1, §2.14, §6b; harness-bench plan decision #4 / Phase B6.
