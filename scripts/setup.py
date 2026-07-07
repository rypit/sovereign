#!/usr/bin/env python3
"""Install Sovereign's dependencies: bootstrap, then per-integration provisioning.

Lightweight and idempotent — every step skips work that's already done, so it's
safe to re-run. Stdlib-only so it works on a fresh checkout with just system
``python3`` (it must not import the ``sovereign`` package, which doesn't exist
until after ``uv sync``). The final step delegates to ``sovereign provision``,
the same shared mechanism the Orchestrator runs at boot: it installs each
integration's own dependencies (services/*/Brewfile, harnesses/*/Brewfile,
plus install commands like ``npm install -g cline``).

Usage:
    python3 scripts/setup.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BREWFILE = REPO_ROOT / "Brewfile"

# Ordered: brew bundle first (bootstrap — installs `uv` if missing), then uv
# sync, then uv tool install so `sovereign` is available globally on PATH, then
# `sovereign provision` to install every integration's own dependencies.
COMMANDS: list[list[str]] = [
    ["brew", "bundle", "--file", str(BREWFILE)],
    ["uv", "sync"],
    ["uv", "tool", "install", "--editable", "."],
    ["uv", "run", "sovereign", "provision"],
]


def main() -> int:
    if shutil.which("brew") is None:
        print(
            "Homebrew is required but not found. Install it from https://brew.sh "
            "and re-run.",
            file=sys.stderr,
        )
        return 1

    for cmd in COMMANDS:
        print(f"$ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT)  # noqa: S603 - fixed argv, not shell
        if result.returncode != 0:
            return result.returncode

    print("Setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
