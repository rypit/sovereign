"""The ``docker_engine`` service package.

Importing this package registers :class:`DockerEngineManager` under the
``docker_engine`` base_type.
"""

from sovereign.services.docker_engine.manager import DockerEngineManager

__all__ = ["DockerEngineManager"]
