"""The ``opencode`` harness package.

Importing this package registers :class:`OpencodeHarness` under the
``opencode`` base_type.
"""

from sovereign.harnesses.opencode.manager import OpencodeHarness

__all__ = ["OpencodeHarness"]
