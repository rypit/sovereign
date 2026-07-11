# Canonical developer verbs — the same commands CI runs.
.PHONY: setup test lint typecheck check graph coverage

setup:            ## Bootstrap toolchain + env + integration deps (macOS)
	python3 scripts/setup.py

test:             ## Run the test suite
	uv run pytest -q

lint:             ## Ruff lint
	uv run ruff check .

typecheck:        ## Mypy over src/sovereign
	uv run mypy

check: lint typecheck test  ## Everything CI checks, locally

graph:            ## Regenerate the internal dependency graph
	uv run python scripts/depgraph.py

coverage:         ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html
