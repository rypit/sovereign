"""Config schema for the ``llama_cpp`` native inference engine.

Pydantic-only (§2.3). Parses the ``config:`` block of a ``llama_cpp`` service
entry. The memory-management knobs map directly onto ``llama-server`` flags (§7):

    gpu_layers    -> -ngl   (Metal-offloaded layers)
    threads       -> -t     (CPU threads)
    context_size  -> -c     (KV-cache size)
    max_parallel  -> -np     (concurrent request slots)
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import NativeEngineConfig, SovereignBaseModel

# KV-cache quantisation types llama-server accepts for --cache-type-{k,v}.
VALID_KV_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
)


class PromptCachingPolicy(SovereignBaseModel):
    """Prompt/KV cache policy — validated pre-flight, then turned into flags (§7)."""

    enabled: bool = False
    #: Directory where llama-server saves/restores slot KV caches (--slot-save-path).
    cache_path: str | None = None
    #: KV cache dtype (--cache-type-k / --cache-type-v).
    kv_cache_type: str = "f16"


class LlamaPolicy(SovereignBaseModel):
    """The ``policy:`` block for a llama_cpp service."""

    prompt_caching: PromptCachingPolicy | None = None


class LlamaCppConfig(NativeEngineConfig):
    """Settings for a single ``llama-server`` instance.

    Shared fields (``model``, ``host``, ``draft_model``, ``served_model_name``,
    ``extra_args``, ``log_dir``) come from :class:`NativeEngineConfig`. Here,
    ``model`` is a local GGUF path or a HuggingFace repo id
    (``<user>/<model>[:quant]``); ``served_model_name`` maps to ``--alias``.
    """

    #: ``llama-server`` binary; a bare name is resolved on ``PATH``.
    binary: str = "llama-server"

    # Resource knobs — omitted flags let llama-server pick its own default.
    gpu_layers: int | None = Field(default=None, ge=0)  # -ngl
    threads: int | None = Field(default=None, gt=0)  # -t
    context_size: int | None = Field(default=None, gt=0)  # -c
    max_parallel: int | None = Field(default=None, gt=0)  # -np

    #: Max tokens to draft per step (llama-server ``--draft-max``).
    num_draft_tokens: int | None = Field(default=None, gt=0)

    #: Optional bearer key llama-server requires on requests (``--api-key``).
    api_key: str | None = None

    # --- admission-control estimation (§7) ---
    #: Approximate KV-cache bytes per context token (default ~256 KiB, large-model
    #: rough figure). Used for the model-file + KV memory estimate. A top-level
    #: ``memory_gb`` on the service entry overrides the whole estimate.
    kv_bytes_per_token: int = Field(default=262144, gt=0)
