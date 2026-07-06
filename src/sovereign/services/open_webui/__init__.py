"""The ``open_webui`` service package.

Importing this package registers :class:`OpenWebUIManager` under the ``open_webui``
base_type.
"""

from sovereign.services.open_webui.manager import OpenWebUIManager

__all__ = ["OpenWebUIManager"]
