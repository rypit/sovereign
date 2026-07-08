"""Coding-harness integrations.

Each harness lives in its own folder with a ``config.py`` + ``manager.py`` and
registers itself via :mod:`sovereign.core.registry`. Importing this module imports
every harness subpackage automatically (auto-discovered below), so dropping a new
integration folder in is all it takes — no aggregator edit, no import to remember.
Prefer :func:`sovereign.core.registry.populate_registries` over importing this
module directly.

Harness managers must stay importable without their optional dependencies
(import heavy/optional packages lazily inside methods) — discovery imports
every subpackage unconditionally.
"""

import importlib
import pkgutil

for _module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module_info.name}")
