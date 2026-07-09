"""The ``mlx_lm`` service package.

Importing this package registers :class:`MlxLmManager` under the ``mlx_lm``
base_type.
"""

from sovereign.services.inference_engines.mlx_lm.manager import MlxLmManager

__all__ = ["MlxLmManager"]
