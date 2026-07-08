"""Config schema for the ``mlx_lm`` native inference engine (Phase 11).

Pydantic-only (§2.3). Parses the ``config:`` block of an ``mlx_lm`` service entry.

Unlike llama.cpp, MLX manages Apple-Silicon unified memory automatically, so there
are no ``-ngl`` / ``-c`` knobs and no API key. The knobs below map onto
``mlx_lm.server`` flags.
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import NativeEngineConfig


class MlxLmConfig(NativeEngineConfig):
    """Settings for a single ``mlx_lm.server`` instance.

    Shared fields (``model``, ``host``, ``draft_model``, ``extra_args``,
    ``log_dir``) come from :class:`NativeEngineConfig`. Here, ``model`` is a
    local MLX model dir or a HuggingFace repo id (e.g.
    ``mlx-community/Llama-3.2-1B-Instruct-4bit``); ``served_model_name`` only
    affects Sovereign's bookkeeping — ``mlx_lm.server`` has no flag for it and
    ignores the request's model field.
    """

    #: ``mlx_lm.server`` console script; a bare name is resolved on ``PATH``.
    binary: str = "mlx_lm.server"

    # Serving knobs — omitted flags let mlx_lm.server pick its own default.
    max_tokens: int | None = Field(default=None, gt=0)  # --max-tokens
    temp: float | None = Field(default=None, ge=0)  # --temp
    top_p: float | None = Field(default=None, gt=0, le=1)  # --top-p
    decode_concurrency: int | None = Field(default=None, gt=0)  # --decode-concurrency
    prompt_cache_size: int | None = Field(default=None, gt=0)  # --prompt-cache-size
    prompt_cache_bytes: int | None = Field(default=None, ge=0)  # --prompt-cache-bytes
    #: Path to trained LoRA adapter weights (--adapter-path).
    adapter_path: str | None = None
    #: Tokens the draft model generates ahead of time (--num-draft-tokens); server default is 3.
    num_draft_tokens: int | None = Field(default=None, ge=0)
    #: Trust remote tokenizer code (--trust-remote-code).
    trust_remote_code: bool = False
