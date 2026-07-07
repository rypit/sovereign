"""Coding-harness integrations.

Each harness lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. Importing this module imports
every harness package, so ``import sovereign.harnesses`` populates the registry.
"""

from sovereign.harnesses import cline_cli, mini_swe_agent  # noqa: F401 - imports register each

__all__ = ["cline_cli", "mini_swe_agent"]
