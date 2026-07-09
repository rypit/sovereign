"""Cross-layer model-resolution errors.

These live in ``core`` because they are the contract the orchestration layers
(:mod:`sovereign.orchestrator`, :mod:`sovereign.core.planning`,
:mod:`sovereign.main`) catch around routing and admission — while the code that
*raises* them is the inference-engine HF pipeline
(:mod:`sovereign.services.inference_engines.hf`). Putting them at the boundary
lets the upper layers handle a routing/download failure without importing the
engine package (which would invert the dependency direction).
"""

from __future__ import annotations


class ModelResolutionError(Exception):
    """Base for model resolution problems."""


class ModelAccessError(ModelResolutionError):
    """Gated repo or bad token."""


class ModelNotFoundError(ModelResolutionError):
    """Repo doesn't exist on HuggingFace Hub."""


class ModelDownloadError(ModelResolutionError):
    """Disk space exhausted or mid-download failure."""


class RoutingError(ModelResolutionError):
    """Auto routing is impossible for this model ref."""
