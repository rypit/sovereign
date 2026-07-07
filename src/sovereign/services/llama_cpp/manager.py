"""``llama_cpp`` — the first native inference engine (§12, Phase 4).

Runs ``llama-server`` as a native subprocess (bare metal, for real Metal
acceleration — §2.1) via the shared :class:`NativeEngineManager` lifecycle:
subprocess + HTTP health + ``psutil`` metrics. The model can be a local GGUF
path or a HuggingFace repo id (``<user>/<model>[:quant]``), which llama-server
downloads and caches on first start.
"""

from __future__ import annotations

import os
from pathlib import Path

from sovereign.config import ServiceEntry
from sovereign.core.base_native import (
    NativeEngineManager,
    check_local_artifact,
    local_model_bytes,
    looks_local,
)
from sovereign.core.registry import register_service
from sovereign.services.llama_cpp.config import (
    VALID_KV_CACHE_TYPES,
    LlamaCppConfig,
    LlamaPolicy,
)


@register_service("llama_cpp")
class LlamaCppManager(NativeEngineManager):
    """Supervises one native ``llama-server`` process."""

    base_type = "llama_cpp"
    config_cls = LlamaCppConfig
    #: Provisioned via the package Brewfile (`brew "llama.cpp"`) when missing.
    provisioning_binary = "llama-server"

    def __init__(self, entry: ServiceEntry) -> None:
        super().__init__(entry)
        self.policy = LlamaPolicy.model_validate(entry.policy or {})

    # --- resource estimation (§7) ---
    def estimated_memory_gb(self) -> float:
        """Model (+ draft) weights on disk + KV cache, or a declared override.

        HuggingFace repo ids not yet cached locally contribute 0.0 (admitted).
        """
        if self.memory_override_gb is not None:
            return round(self.memory_override_gb, 2)
        total = local_model_bytes(self.config.model)
        if self.config.draft_model is not None:
            total += local_model_bytes(self.config.draft_model)
        return round(total / (1024**3) + self.estimated_kv_cache_gb(), 2)

    def estimated_kv_cache_gb(self) -> float:
        """KV cache grows with total context (§7 — slots×context is joint)."""
        ctx = self.config.context_size or 0
        return round(ctx * self.config.kv_bytes_per_token / (1024**3), 2)

    def per_slot_context(self) -> int | None:
        """Context available *per agent* — total context divided across -np slots."""
        if self.config.context_size is None:
            return None
        return self.config.context_size // (self.config.max_parallel or 1)

    # --- flag generation ---
    def get_start_args(self) -> list[str]:
        """Translate the validated config into a ``llama-server`` argv (§7)."""
        args = [self.config.binary]
        if looks_local(self.config.model):
            args += ["--model", os.path.expanduser(self.config.model)]
        else:
            args += ["--hf-repo", self.config.model]
        args += ["--host", self.host, "--port", str(self.port)]
        if self.config.gpu_layers is not None:
            args += ["-ngl", str(self.config.gpu_layers)]
        if self.config.threads is not None:
            args += ["-t", str(self.config.threads)]
        if self.config.context_size is not None:
            args += ["-c", str(self.config.context_size)]
        if self.config.max_parallel is not None:
            args += ["-np", str(self.config.max_parallel)]
        if self.config.api_key:
            args += ["--api-key", self.config.api_key]
        if self.config.served_model_name:
            args += ["--alias", self.config.served_model_name]
        if self.config.draft_model is not None:
            if looks_local(self.config.draft_model):
                args += ["--model-draft", os.path.expanduser(self.config.draft_model)]
            else:
                args += ["--hf-repo-draft", self.config.draft_model]
        if self.config.num_draft_tokens is not None:
            args += ["--draft-max", str(self.config.num_draft_tokens)]

        caching = self.policy.prompt_caching
        if caching and caching.enabled and caching.cache_path:
            cache_dir = str(Path(caching.cache_path).expanduser())
            args += [
                "--slot-save-path",
                cache_dir,
                "--cache-type-k",
                caching.kv_cache_type,
                "--cache-type-v",
                caching.kv_cache_type,
            ]

        args += self.config.extra_args
        return args

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if self.config.draft_model is not None:
            check_local_artifact(
                self.config.draft_model, kind="llama_cpp draft_model", service=self.name
            )
        self._validate_prompt_caching()

    def _validate_prompt_caching(self) -> None:
        """Pre-flight the prompt-caching policy (§7): dir writable, dtype valid."""
        caching = self.policy.prompt_caching
        if caching is None or not caching.enabled:
            return
        if not caching.cache_path:
            raise ValueError(
                f"prompt_caching is enabled for '{self.name}' but no cache_path is set."
            )
        if caching.kv_cache_type not in VALID_KV_CACHE_TYPES:
            raise ValueError(
                f"invalid kv_cache_type '{caching.kv_cache_type}' for '{self.name}'; "
                f"expected one of {sorted(VALID_KV_CACHE_TYPES)}."
            )
        cache_dir = Path(caching.cache_path).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"prompt cache dir for '{self.name}' is not writable: {cache_dir} ({exc})"
            ) from exc
