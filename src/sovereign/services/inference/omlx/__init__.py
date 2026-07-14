"""The ``omlx`` service package.

Importing this package registers :class:`OmlxManager` under the ``omlx``
base_type.
"""

from sovereign.services.inference.omlx.manager import OmlxManager

__all__ = ["OmlxManager"]
