"""The ``docker`` service package.

Importing this package registers :class:`DockerManager` under the
``docker`` base_type.
"""

from sovereign.services.docker.manager import DockerManager

__all__ = ["DockerManager"]
