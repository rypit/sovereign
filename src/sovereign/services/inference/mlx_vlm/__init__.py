"""The ``mlx_vlm`` service package.

Importing this package registers :class:`MlxVlmManager` under the ``mlx_vlm``
base_type.
"""

from sovereign.services.inference.mlx_vlm.manager import MlxVlmManager

__all__ = ["MlxVlmManager"]
