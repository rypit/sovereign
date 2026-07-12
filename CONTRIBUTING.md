# Contributing

For humans and agents alike. Sovereign is a declarative control plane for
local LLM infrastructure on macOS/Apple Silicon — see `README.md` for what it
does and `CLAUDE.md` for the codebase map and conventions.

## Commands

```bash
make test        # uv run pytest -q          (~600 tests, <15s, fully hermetic)
make lint        # uv run ruff check .
make typecheck   # uv run mypy               (src/sovereign, must stay clean)
make arch        # dep-graph rules + doc consistency (depgraph --check, check_docs.py)
make check       # everything CI runs: lint, typecheck, arch, test
make graph       # regenerate docs/dependency-graph.md
```

Run `make check` before opening a PR — it's exactly what CI's blocking jobs
run.

## Branch / PR flow

1. Branch off `main`.
2. Make your change; keep `config.py` files Pydantic-only (the golden rule —
   see `docs/architecture.md`), and keep new integrations self-contained
   under `services/` or `harnesses/` (no central registry edit needed).
3. Run `make check`. If `make arch` fails, it names the specific rule id and
   file — fix the import, or (rarely) add a `GRANDFATHERED` entry in
   `scripts/depgraph.py` with a comment explaining why.
4. Open a PR using the template — it asks whether this change needed an ADR
   and which docs it touched.

## Working agreements

See `CLAUDE.md`'s "Working agreements" section for exactly when an ADR is
required and which docs to update alongside a layering or contract change.
The `/adr` and `/architecture-review` Claude Code skills automate both.

## Decisions and architecture

- `docs/decisions/` — ADRs recording load-bearing decisions, with a
  convention and template in `docs/decisions/README.md`.
- `docs/architecture.md` — the living layering/contracts doc; keep it and
  `scripts/depgraph.py`'s `ARCH_RULES` in sync (checked by `make arch`).
- `docs/sovereign-implementation-plan-v1.1.md` — the frozen historical
  design doc (`§N` anchors cited from docstrings); not updated for new work.
