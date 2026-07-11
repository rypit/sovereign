"""Embedded engine-worker package.

Deliberately empty: no engine imports here (and none should be added) so this
package stays safe to discover/import in any context — including linux CI —
without pulling in platform-only bindings like ``mlx_lm`` or
``llama_cpp``.
"""
