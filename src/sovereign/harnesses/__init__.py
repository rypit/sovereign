"""Coding-harness integrations.

Each harness lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. Importing this module imports
every harness package, so ``import sovereign.harnesses`` populates the registry.
"""

from sovereign.harnesses import mini_swe_agent  # noqa: F401 - imports register each harness

__all__ = ["mini_swe_agent"]
