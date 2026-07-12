---
name: architecture-review
description: Run make arch and make graph, diff the dependency graph against docs/architecture.md's stated rules, and scan the working diff for contract-surface changes to report which ADR/doc updates are owed. Use before opening a PR that touches layering, a Protocol, the config schema, the telemetry wire protocol, or WorkerConfig.
---

Audit the working diff for architecture drift and report what's owed —
don't fix anything without confirming with the user first, since some
findings (e.g. a genuine new ARCH_RULES exception) need a human decision.

1. Run `make arch` (`scripts/depgraph.py --check` + `scripts/check_docs.py`).
   Report every failure verbatim — rule id, message, file:line.
2. Run `make graph` and check whether `docs/dependency-graph.md` changed.
   If it did, note that the regenerated file needs to be committed
   alongside the rest of the change (this is also what `make arch`'s
   freshness check would have caught).
3. Diff the working tree (`git diff` against the merge-base, or against
   HEAD if unclear) and scan it for contract-surface changes:
   - `ServiceManager` / `Harness` / capability Protocols in
     `core/base_manager.py`, `core/base_harness.py` — any changed method
     signature, new Protocol, or new capability hook.
   - `sovereign.yaml` schema changes in `config.py` / `**/config.py` —
     new fields are usually fine additively; changed *semantics* of an
     existing field (default, validation, meaning) is the trigger.
   - The telemetry wire schema (`workers/protocol.py`'s `EventType` /
     event payload shapes) or `PROTOCOL_VERSION`.
   - `WorkerConfig` fields (`workers/worker_config.py`).
   - Admission/memory math (`core/resources.py`, `core/planning.py`).
4. For each contract-surface change found, check whether:
   - An ADR already covers it (`docs/decisions/` — grep for the relevant
     area) — if not, this is very likely a "write an ADR" finding (point
     at the `/adr` skill).
   - `docs/architecture.md`'s contracts table or layering section
     mentions the changed file — if the change is significant and the doc
     wasn't touched, that's a "docs/architecture.md is stale" finding.
   - `CLAUDE.md`'s architecture map still accurately describes the
     changed file's role.
5. Report a punch list: what's clean, what's missing an ADR, what doc
   needs an update, whether the graph needs regenerating. Don't editorialize
   beyond what the diff and the checks actually show.
