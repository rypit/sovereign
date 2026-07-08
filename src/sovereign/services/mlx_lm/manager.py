"""``mlx_lm`` — a native MLX inference engine (§12, Phase 11).

Runs ``mlx_lm.server`` (Apple MLX, OpenAI-compatible) via the shared
:class:`NativeEngineManager` lifecycle — subprocess + HTTP health + ``psutil``
metrics — adapted to MLX's CLI. The model can be a local MLX directory or a
HuggingFace repo id; Sovereign downloads the snapshot into the shared HF cache
before launch (DOWNLOADING state) and starts the server from the resolved path.
"""

from __future__ import annotations

import os

from sovereign.core.base_native import (
    NativeEngineManager,
    check_local_artifact,
)
from sovereign.core.registry import register_service
from sovereign.services.mlx_lm.config import MlxLmConfig


@register_service("mlx_lm")
class MlxLmManager(NativeEngineManager):
    """Supervises one native ``mlx_lm.server`` process."""

    base_type = "mlx_lm"
    config_cls = MlxLmConfig
    model_artifact_kind = "snapshot"
    binary_hint = "It ships with the `mlx-lm` dependency — run `uv sync`."

    # --- resource estimation (§7) ---
    def estimated_memory_gb(self) -> float:
        """Estimate resident memory from the model weights (or an override).

        Includes the draft model when speculative decoding is configured — both models
        live in unified memory simultaneously. A declared ``prompt_cache_bytes`` is a
        hard KV-cache reservation that also lives in unified memory. For HuggingFace
        repo ids the estimate comes from repo metadata (weight-file sizes); unknown
        (offline + uncached) contributes 0.0.
        """
        if self.memory_override_gb is not None:
            return round(self.memory_override_gb, 2)
        total = self._model_bytes(self.config.model)
        if self.config.draft_model is not None:
            total += self._model_bytes(self.config.draft_model)
        if self.config.prompt_cache_bytes is not None:
            total += self.config.prompt_cache_bytes
        return round(total / (1024**3), 2)

    # --- flag generation ---
    def get_start_args(self) -> list[str]:
        """Translate the validated config into an ``mlx_lm.server`` argv."""
        args = [
            self.config.binary,
            "--model",
            self.resolved_model_path(),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.config.max_tokens is not None:
            args += ["--max-tokens", str(self.config.max_tokens)]
        if self.config.temp is not None:
            args += ["--temp", str(self.config.temp)]
        if self.config.top_p is not None:
            args += ["--top-p", str(self.config.top_p)]
        if self.config.decode_concurrency is not None:
            args += ["--decode-concurrency", str(self.config.decode_concurrency)]
        if self.config.prompt_cache_size is not None:
            args += ["--prompt-cache-size", str(self.config.prompt_cache_size)]
        if self.config.prompt_cache_bytes is not None:
            args += ["--prompt-cache-bytes", str(self.config.prompt_cache_bytes)]
        if self.config.adapter_path is not None:
            args += ["--adapter-path", os.path.expanduser(self.config.adapter_path)]
        if self.config.draft_model is not None:
            args += ["--draft-model", self.resolved_draft_model_path()]
        if self.config.num_draft_tokens is not None:
            args += ["--num-draft-tokens", str(self.config.num_draft_tokens)]
        if self.config.trust_remote_code:
            args += ["--trust-remote-code"]
        args += self.config.extra_args
        return args

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if self.config.adapter_path is not None:
            check_local_artifact(
                self.config.adapter_path, kind="mlx_lm adapter_path", service=self.name
            )
        if self.config.draft_model is not None:
            check_local_artifact(
                self.config.draft_model, kind="mlx_lm draft_model", service=self.name
            )

    def adjust_resources(self, memory_limit_mb: int) -> None:
        """No-op: MLX's Metal cache limit isn't reachable through the server subprocess."""
