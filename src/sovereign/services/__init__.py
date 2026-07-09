"""Service integrations.

Each service lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. Importing this module imports
every service module automatically (auto-discovered below), so dropping a new
integration folder in is all it takes — no aggregator edit, no import to remember.
Prefer :func:`sovereign.core.registry.populate_registries` over importing this
module directly.

Discovery walks recursively (``walk_packages``, not ``iter_modules``) so nested
groupings like ``inference/llama_cpp`` are found too, not just direct
children.
"""

import importlib
import pkgutil

for _module_info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    importlib.import_module(_module_info.name)
