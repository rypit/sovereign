"""The ``llama_cpp`` service package.

Importing this package registers :class:`LlamaCppManager` under the ``llama_cpp``
base_type.
"""

from sovereign.services.inference.llama_cpp.manager import LlamaCppManager

__all__ = ["LlamaCppManager"]
