"""The ``cline_cli`` harness package.

Importing this package registers :class:`ClineCliHarness` under the
``cline_cli`` base_type.
"""

from sovereign.harnesses.cline_cli.manager import ClineCliHarness

__all__ = ["ClineCliHarness"]
