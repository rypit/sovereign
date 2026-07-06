#!/usr/bin/env python3
"""Install Sovereign's dependencies: ``brew bundle`` + ``uv sync``.

Lightweight and idempotent — both steps skip work that's already done, so it's
safe to re-run. Stdlib-only so it works on a fresh checkout with just system
``python3`` (it must not import the ``sovereign`` package, which doesn't exist
until after ``uv sync``).

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

# Ordered: brew bundle first (it installs `uv` if missing), then uv sync,
# then uv tool install so `sovereign` is available globally on PATH.
COMMANDS: list[list[str]] = [
    ["brew", "bundle", "--file", str(BREWFILE)],
    ["uv", "sync"],
    ["uv", "tool", "install", "--editable", "."],
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
