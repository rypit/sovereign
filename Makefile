# Canonical developer verbs — the same commands CI runs.
.PHONY: setup test coverage lint typecheck arch check graph

setup:            ## Bootstrap toolchain + env + integration deps (macOS)
	python3 scripts/setup.py

test:             ## Run the test suite
	uv run pytest -q

coverage:         ## Test suite + line-coverage report (visibility, not a gate)
	uv run pytest -q --cov=sovereign --cov-report=term-missing

lint:             ## Ruff lint
	uv run ruff check .

typecheck:        ## Mypy over src/sovereign + tests
	uv run mypy

arch:             ## Architecture guardrails: dep-graph rules + doc consistency
	uv run python scripts/depgraph.py --check
	uv run python scripts/check_docs.py

check: lint typecheck arch test  ## Everything CI checks, locally

graph:            ## Regenerate the internal dependency graph
	uv run python scripts/depgraph.py
