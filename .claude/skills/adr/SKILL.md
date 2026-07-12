---
name: adr
description: Scaffold a new Architecture Decision Record from the template in docs/decisions/README.md. Use when a change touches layering, a core Protocol/contract, sovereign.yaml semantics, the memory/admission model, the process/lifecycle model, the telemetry wire protocol, or testing seams.
---

Scaffold the next-numbered ADR and remind the caller what else needs updating.

1. Read `docs/decisions/README.md` for the convention, the REQUIRED-when
   list, and the template. If the requested change doesn't clearly need an
   ADR, say so and point at the README's list instead of writing one anyway.
2. Find the next ADR number: list `docs/decisions/*.md`, take the highest
   `NNNN` prefix (ignore `README.md`), add one, zero-padded to four digits.
3. Derive a kebab-case title from the user's description of the decision.
   Create `docs/decisions/NNNN-kebab-title.md` using the template's
   structure (`Status: accepted`, `## Context`, `## Decision`,
   `## Consequences`, `## Alternatives considered`, a trailing provenance
   line). Ask the user for Context/Decision/Consequences/Alternatives
   content if it isn't already clear from the conversation — don't invent a
   decision that wasn't actually made.
4. Add a row for the new ADR to the index table at the bottom of
   `docs/decisions/README.md`.
5. Tell the user, explicitly, what else this decision likely obligates them
   to update before merging:
   - `docs/architecture.md` — if the decision changed layering, a contract,
     or an invariant this doc states.
   - `CLAUDE.md`'s architecture map — if a directory's role or a file's
     purpose changed.
   - `scripts/depgraph.py`'s `ARCH_RULES` (and re-run `make arch`) — if the
     decision changed which modules may import which.
   - `make graph` — if the module set changed.
   Do not silently skip this step; a scaffolded ADR with stale surrounding
   docs is worse than no ADR.
6. Run `uv run python scripts/check_docs.py` to confirm the new ADR's
   filename and Status line are well-formed before handing it back.
