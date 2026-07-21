"""Config schema for the ``mlx_vlm`` native inference engine.

Pydantic-only (§2.3). Parses the ``config:`` block of an ``mlx_vlm`` service
entry. The knobs map onto ``mlx_vlm.server`` flags in the worker adapter
(``workers/mlx_vlm_adapter.build_server_argv``):

    max_tokens         -> --max-tokens
    prefill_step_size  -> --prefill-step-size
    vision_cache_size  -> --vision-cache-size
    kv_bits            -> --kv-bits
    kv_quant_scheme    -> --kv-quant-scheme
    kv_group_size      -> --kv-group-size
    max_kv_size        -> --max-kv-size
    quantized_kv_start -> --quantized-kv-start
    draft_kind         -> --draft-kind
    draft_block_size   -> --draft-block-size
    adapter_path       -> --adapter-path
    trust_remote_code  -> --trust-remote-code
    enable_thinking    -> --enable-thinking
    thinking_budget    -> --thinking-budget
    api_key            -> env MLX_VLM_SERVER_API_KEY (never a flag; §"start_env")
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from sovereign.core.base_config import NativeEngineConfig


class MlxVlmConfig(NativeEngineConfig):
    """Settings for a single ``mlx_vlm.server`` instance.

    Shared fields (``model``, ``host``, ``draft_model``, ``served_model_name``,
    ``log_dir``) come from :class:`NativeEngineConfig`. ``model`` is a local
    MLX directory or a HuggingFace repo id of a vision-language MLX snapshot
    (e.g. an ``image-text-to-text`` ``mlx-community/...`` conversion — the
    vision tower ships inside the snapshot; there is no llama.cpp-style
    ``mmproj`` sidecar in MLX land). ``draft_model`` names a drafter snapshot
    for speculative decoding; its family is picked with ``draft_kind`` —
    ``mtp`` uses a repo of split multi-token-prediction head weights.
    ``num_draft_tokens`` is inherited but rejected pre-flight: mlx-vlm sizes
    drafts with ``--draft-block-size`` (kind-dependent semantics), so use
    ``draft_block_size`` instead (ADR 0006: surface the gap loudly).
    """

    # Serving knobs — omitted flags let mlx_vlm.server pick its own default.
    max_tokens: int | None = Field(default=None, gt=0)  # --max-tokens
    prefill_step_size: int | None = Field(default=None, gt=0)  # --prefill-step-size
    #: Encoded-image LRU entries kept hot (--vision-cache-size; server default 20).
    vision_cache_size: int | None = Field(default=None, ge=0)

    # KV-cache quantization.
    kv_bits: int | None = Field(default=None, gt=0)  # --kv-bits
    kv_quant_scheme: Literal["uniform", "turboquant"] | None = None  # --kv-quant-scheme
    kv_group_size: int | None = Field(default=None, gt=0)  # --kv-group-size
    max_kv_size: int | None = Field(default=None, gt=0)  # --max-kv-size
    quantized_kv_start: int | None = Field(default=None, ge=0)  # --quantized-kv-start

    # Speculative decoding (pairs with the inherited ``draft_model``).
    draft_kind: Literal["dflash", "eagle3", "mtp"] | None = None  # --draft-kind
    draft_block_size: int | None = Field(default=None, gt=0)  # --draft-block-size

    #: Path to trained LoRA adapter weights (--adapter-path).
    adapter_path: str | None = None
    #: Trust remote tokenizer/processor code (--trust-remote-code).
    trust_remote_code: bool = False

    # Thinking-mode defaults applied server-side.
    enable_thinking: bool = False  # --enable-thinking
    thinking_budget: int | None = Field(default=None, gt=0)  # --thinking-budget

    #: Optional bearer key mlx-vlm requires on requests (env
    #: MLX_VLM_SERVER_API_KEY — see MlxVlmManager.start_env()).
    api_key: str | None = None
