# 0002. Registry self-registration + `base_type` routing

Status: accepted

## Context

Adding a new engine or harness should never require editing a central
dispatch table shared by every other integration — that pattern produces
merge conflicts and hidden coupling as the catalog grows. At the same time,
the Orchestrator needs a uniform way to go from a YAML entry to a concrete
Python object, and to allow two instances of the same engine (different
models/ports) to coexist.

## Decision

Each integration is a self-contained folder under `services/` (or
`services/inference/` for a native engine, or `harnesses/`) that
self-registers via `@register_service("x")` / `@register_harness("x")`
decorators (`core/registry.py`). `services/__init__.py` and
`harnesses/__init__.py` walk their trees (`pkgutil.walk_packages`) so nested
groupings register too; `core/registry.populate_registries()` is the one
call every lookup path makes first. Instance `name` is the unique identity a
user picks; `base_type` is the factory key that selects the class
(`core/registry.route_entry` for `auto`-routed cases, sweeping each engine's
`claim_route`). Dropping in a new folder with `__init__.py` + `config.py` +
`manager.py` is sufficient — no aggregator edit.

## Consequences

- New integrations are additive, not edits to shared files.
- `name` vs `base_type` lets two `llama_cpp` instances run side by side with
  different models.
- Registration bugs (e.g. missing decorator) fail loudly at
  `populate_registries()` time rather than silently at boot.
- Cost: harness/service modules must import optional third-party deps
  lazily (inside methods), since discovery imports every module
  unconditionally at startup.

## Alternatives considered

- A central `ENGINES = {...}` dict edited per integration — rejected: exactly
  the shared-file contention this decision avoids.
- Entry-points / plugin discovery via `importlib.metadata` — rejected as
  overkill for an in-repo, single-package catalog; revisit if integrations
  ever ship as separate installable packages.

---
Provenance: plan §2 (points 4, 5, 6) and `core/registry.py`.
