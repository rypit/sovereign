# Canonical developer verbs — the same commands CI runs.
.PHONY: setup test lint typecheck check

setup:            ## Bootstrap toolchain + env + integration deps (macOS)
	python3 scripts/setup.py

test:             ## Run the test suite
	uv run pytest -q

lint:             ## Ruff lint
	uv run ruff check .

typecheck:        ## Mypy over src/sovereign
	uv run mypy

check: lint typecheck test  ## Everything CI checks, locally
