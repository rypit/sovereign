"""Config schema for the ``mlx_lm`` native inference engine (Phase 11).

Pydantic-only (§2.3). Parses the ``config:`` block of an ``mlx_lm`` service entry.

Unlike llama.cpp, MLX manages Apple-Silicon unified memory automatically, so there
are no ``-ngl`` / ``-c`` knobs and no API key. The knobs below map onto
``mlx_lm.server`` flags.
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import SovereignBaseModel


class MlxLmConfig(SovereignBaseModel):
    """Settings for a single ``mlx_lm.server`` instance."""

    #: Model reference — a local MLX model dir/path **or** a HuggingFace repo id
    #: (e.g. ``mlx-community/Llama-3.2-1B-Instruct-4bit``). Required.
    model: str
    #: ``mlx_lm.server`` console script; a bare name is resolved on ``PATH``.
    binary: str = "mlx_lm.server"
    #: Address the server binds to.
    host: str = "127.0.0.1"

    # Serving knobs — omitted flags let mlx_lm.server pick its own default.
    max_tokens: int | None = Field(default=None, gt=0)  # --max-tokens
    temp: float | None = Field(default=None, ge=0)  # --temp
    top_p: float | None = Field(default=None, gt=0, le=1)  # --top-p
    decode_concurrency: int | None = Field(default=None, gt=0)  # --decode-concurrency
    prompt_cache_size: int | None = Field(default=None, gt=0)  # --prompt-cache-size
    prompt_cache_bytes: int | None = Field(default=None, ge=0)  # --prompt-cache-bytes
    #: Path to trained LoRA adapter weights (--adapter-path).
    adapter_path: str | None = None
    #: Draft model for speculative decoding — local path or HuggingFace repo id (--draft-model).
    draft_model: str | None = None
    #: Tokens the draft model generates ahead of time (--num-draft-tokens); server default is 3.
    num_draft_tokens: int | None = Field(default=None, ge=0)
    #: Trust remote tokenizer code (--trust-remote-code).
    trust_remote_code: bool = False
    #: Client-facing model name — the string an OpenAI-compatible client sends as
    #: ``"model"``. mlx_lm.server has no CLI flag for this (it ignores the
    #: request's model field), so this only affects Sovereign's own bookkeeping
    #: (endpoint attribute, manifest, harness wiring). Defaults to ``model``.
    served_model_name: str | None = None

    #: Escape hatch for flags Sovereign doesn't model yet.
    extra_args: list[str] = Field(default_factory=list)
    #: Directory for the captured stdout/stderr log (created on start).
    log_dir: str = ".sovereign/logs"
