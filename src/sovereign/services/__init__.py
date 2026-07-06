"""Service integrations.

Each service lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. Importing this module imports
every service package, so ``import sovereign.services`` populates the registry.
"""

from sovereign.services import (  # noqa: F401 - imports register each service
    docker_engine,
    llama_cpp,
    mlx_lm,
    open_webui,
    searxng,
)

__all__ = ["docker_engine", "llama_cpp", "mlx_lm", "open_webui", "searxng"]
