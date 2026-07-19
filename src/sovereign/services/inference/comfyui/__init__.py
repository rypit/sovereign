"""The ``comfyui`` service package.

Importing this package registers :class:`ComfyUIManager` under the ``comfyui``
base_type.
"""

from sovereign.services.inference.comfyui.manager import ComfyUIManager

__all__ = ["ComfyUIManager"]
