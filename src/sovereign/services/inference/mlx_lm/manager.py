"""``mlx_lm`` — a native MLX inference engine (§12, Phase 11).

Runs ``mlx_lm.server`` (Apple MLX, OpenAI-compatible) via the shared
:class:`NativeEngineManager` lifecycle — subprocess + HTTP health + ``psutil``
metrics — adapted to MLX's CLI. The model can be a local MLX directory or a
HuggingFace repo id; Sovereign downloads the snapshot into the shared HF cache
before launch (DOWNLOADING state) and starts the server from the resolved path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from sovereign.core.registry import register_service
from sovereign.services.inference.base import (
    NativeEngineManager,
    check_local_artifact,
)
from sovereign.services.inference.mlx_lm.config import MlxLmConfig

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo


@register_service("mlx_lm")
class MlxLmManager(NativeEngineManager):
    """Supervises one embedded ``mlx_lm.server`` engine worker."""

    base_type = "mlx_lm"
    config_cls = MlxLmConfig
    config: MlxLmConfig
    model_artifact_kind = "snapshot"
    #: Probed (out-of-process) rather than checked via a binary-on-PATH: the
    #: engine now runs in-process inside the Python worker.
    import_probe_modules = ("mlx_lm.server",)
    binary_hint = "It ships with the mlx-lm dependency — run `uv sync`."

    # --- routing (auto base_type) ---
    @classmethod
    def claim_route(cls, ref: ModelRef, info: RepoInfo | None) -> int | None:
        """Claim MLX/safetensors work: a local ``config.json``/``.safetensors`` dir, an
        ``mlx`` tag or ``mlx-community`` org, or ``.safetensors`` siblings on the hub.

        The mlx-tag confidence outranks a ``.gguf`` sibling so an mlx repo that also
        ships GGUFs still routes here (see :class:`LlamaCppManager` for the scale).
        """
        if ref.is_local:
            path = ref.local_path
            assert path is not None
            if path.is_dir() and (
                (path / "config.json").exists() or any(path.glob("*.safetensors"))
            ):
                return 40
            return None
        if info is not None:
            if "mlx" in info.tags or (ref.repo_id or "").startswith("mlx-community/"):
                return 40
            if any(n.endswith(".safetensors") for n, _ in info.siblings):
                return 20
        return None

    # --- resource estimation (§7) ---
    def extra_memory_bytes(self) -> int:
        """mlx_lm's engine-specific term on top of the weights: a declared
        ``prompt_cache_bytes`` is a hard KV-cache reservation that also lives
        in unified memory."""
        return self.config.prompt_cache_bytes or 0

    # --- engine_kwargs mapping (consumed by workers.mlx_lm_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter overlays onto
        ``mlx_lm.server``'s own argparse namespace (see
        ``workers/mlx_lm_adapter.build_server_namespace``)."""
        kwargs: dict[str, Any] = {}
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.temp is not None:
            kwargs["temp"] = self.config.temp
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.decode_concurrency is not None:
            kwargs["decode_concurrency"] = self.config.decode_concurrency
        if self.config.prompt_cache_size is not None:
            kwargs["prompt_cache_size"] = self.config.prompt_cache_size
        if self.config.prompt_cache_bytes is not None:
            kwargs["prompt_cache_bytes"] = self.config.prompt_cache_bytes
        if self.config.adapter_path is not None:
            kwargs["adapter_path"] = os.path.expanduser(self.config.adapter_path)
        if self.config.num_draft_tokens is not None:
            kwargs["num_draft_tokens"] = self.config.num_draft_tokens
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True

        kwargs.update(self.config.engine_kwargs)
        return kwargs

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

    def adjust_resources(self, memory_limit_bytes: int) -> None:
        """No-op: MLX's Metal cache limit isn't reachable through the server subprocess."""
