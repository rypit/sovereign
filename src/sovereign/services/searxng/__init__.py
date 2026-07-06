"""The ``searxng`` service package.

Importing this package registers :class:`SearxngManager` under the ``searxng``
base_type.
"""

from sovereign.services.searxng.manager import SearxngManager

__all__ = ["SearxngManager"]
