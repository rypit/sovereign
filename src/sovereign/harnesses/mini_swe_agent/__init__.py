"""The ``mini_swe_agent`` harness package.

Importing this package registers :class:`MiniSweAgentHarness` under the
``mini_swe_agent`` base_type.
"""

from sovereign.harnesses.mini_swe_agent.manager import MiniSweAgentHarness

__all__ = ["MiniSweAgentHarness"]
