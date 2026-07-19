"""Config schema for the ``comfyui`` native inference engine.

Pydantic-only (§2.3). Parses the ``config:`` block of a ``comfyui`` service
entry. The knobs map onto ``comfy … launch`` / ComfyUI ``main.py`` flags in
the worker adapter (``workers/comfyui_adapter.build_server_argv``):

    workspace_dir -> comfy --workspace (where ComfyUI is installed)
    output_dir    -> --output-directory (after the ``--`` separator)
"""

from __future__ import annotations

from sovereign.core.base_config import NativeEngineConfig


class ComfyUIConfig(NativeEngineConfig):
    """Settings for a single ComfyUI server instance.

    Shared fields (``model``, ``host``, ``served_model_name``, ``log_dir``)
    come from :class:`NativeEngineConfig`. ``model`` is a local checkpoint
    file or a HuggingFace ref to a single-file diffusion checkpoint
    (``org/repo`` when the repo has exactly one top-level ``.safetensors``,
    else ``org/repo/file.safetensors``). ``draft_model`` is inherited but
    rejected pre-flight — ComfyUI has no speculative-decoding surface
    (ADR 0006: surface the gap loudly).
    """

    #: Where comfy-cli installs/finds the ComfyUI checkout (``--workspace``).
    #: User-level rather than per-stack ``.sovereign/`` — an install is
    #: multi-GB of torch, shared across stacks. ``~`` is expanded at use.
    workspace_dir: str = "~/.sovereign/comfyui"

    #: Where generated images land (``--output-directory``). Defaults to
    #: ComfyUI's own ``<workspace>/output`` when unset.
    output_dir: str | None = None
