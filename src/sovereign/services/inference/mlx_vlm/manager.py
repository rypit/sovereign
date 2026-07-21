"""``mlx_vlm`` — vision-language MLX inference as a native engine.

Runs ``mlx_vlm.server`` (Apple MLX, OpenAI-compatible, accepts image content
parts on ``/v1/chat/completions``) in-process inside an engine worker, the
same embedded-binding pattern as ``mlx_lm``. The model is an MLX/safetensors
*snapshot* whose vision tower ships in-repo — MLX VLM conversions have no
llama.cpp-style ``mmproj`` sidecar. Speculative decoding takes a drafter
snapshot (``draft_model`` + ``draft_kind``: dflash/eagle3/**mtp** — for MTP,
a repo of split multi-token-prediction head weights); both snapshots are
downloaded up front and both count toward admission (§7).

mlx-vlm exposes ``/health``/``/metrics`` management endpoints, but their
schema is unverified on hardware, so v1 ships without a telemetry translator:
heartbeat + memory events come from the generic ``engine_worker``, while
prefill/tok-s stats are absent — an ADR 0006 gap surfaced at pre-flight
(same posture as ``omlx``).
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from sovereign.core.registry import register_service
from sovereign.services.inference.base import (
    HTTP_TIMEOUT,
    NativeEngineManager,
    check_local_artifact,
)
from sovereign.services.inference.mlx_vlm.config import MlxVlmConfig

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo

logger = logging.getLogger("sovereign")

#: Config fields forwarded verbatim to the worker adapter whenever set (the
#: adapter kebab-cases each into the matching ``mlx_vlm.server`` flag).
_FORWARDED_FIELDS = (
    "max_tokens",
    "prefill_step_size",
    "vision_cache_size",
    "kv_bits",
    "kv_quant_scheme",
    "kv_group_size",
    "max_kv_size",
    "quantized_kv_start",
    "draft_kind",
    "draft_block_size",
    "thinking_budget",
)


@register_service("mlx_vlm")
class MlxVlmManager(NativeEngineManager):
    """Supervises one embedded ``mlx_vlm.server`` engine worker."""

    base_type = "mlx_vlm"
    config_cls = MlxVlmConfig
    config: MlxVlmConfig
    model_artifact_kind = "snapshot"
    #: Probed (out-of-process) rather than checked via a binary-on-PATH: the
    #: engine runs in-process inside the Python worker, like ``mlx_lm``.
    import_probe_modules = ("mlx_vlm.server",)
    binary_hint = "It ships with the mlx-vlm dependency — run `uv sync`."

    # --- routing (auto base_type) ---
    @classmethod
    def claim_route(cls, ref: ModelRef, info: RepoInfo | None) -> int | None:
        """Deliberately abstain from ``auto`` routing.

        mlx_vlm serves the same MLX/safetensors snapshots as ``mlx_lm``, and
        the router's :class:`RepoInfo` carries no vision signal (tags +
        siblings only), so any claim would contend blindly on the
        cross-engine confidence scale. For now mlx_vlm is opt-in only —
        ``base_type: mlx_vlm`` in ``sovereign.yaml``; a future tag-based
        vision claim outranking ``mlx_lm`` is a deliberate change to the
        precedence contract (with an ADR), not a side effect of adding the
        engine.
        """
        return None

    # --- engine_kwargs mapping (consumed by workers.mlx_vlm_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter maps onto
        ``mlx_vlm.server`` CLI flags (see
        ``workers/mlx_vlm_adapter.build_server_argv``)."""
        kwargs: dict[str, Any] = {}
        for field in _FORWARDED_FIELDS:
            value = getattr(self.config, field)
            if value is not None:
                kwargs[field] = value
        if self.config.adapter_path is not None:
            kwargs["adapter_path"] = os.path.expanduser(self.config.adapter_path)
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.config.enable_thinking:
            kwargs["enable_thinking"] = True

        kwargs.update(self.config.engine_kwargs)
        return kwargs

    def start_env(self) -> dict[str, str]:
        """Pass the API key via ``MLX_VLM_SERVER_API_KEY`` — mlx-vlm reads
        this environment variable natively, so unlike ``llama_cpp``/``omlx``
        (whose external servers only take ``--api-key`` argv) the secret
        never appears on any command line *or* in the dumped worker-config
        JSON; the worker inherits it straight through to the in-process
        server."""
        if self.config.api_key:
            return {"MLX_VLM_SERVER_API_KEY": self.config.api_key}
        return {}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if self.config.adapter_path is not None:
            check_local_artifact(
                self.config.adapter_path, kind="mlx_vlm adapter_path", service=self.name
            )
        if self.config.draft_model is not None:
            check_local_artifact(
                self.config.draft_model, kind="mlx_vlm draft_model", service=self.name
            )
        if self.config.num_draft_tokens is not None:
            # ADR 0006: an engine gap is surfaced loudly, not silently bridged —
            # mlx-vlm sizes drafts with --draft-block-size (semantics depend on
            # draft_kind), not a num-draft-tokens flag.
            raise ValueError(
                f"mlx_vlm service '{self.name}': num_draft_tokens is not supported — "
                "mlx-vlm sizes speculative drafts with draft_block_size "
                "(--draft-block-size). Use draft_block_size instead."
            )
        if self.config.draft_model is None and (
            self.config.draft_kind is not None or self.config.draft_block_size is not None
        ):
            raise ValueError(
                f"mlx_vlm service '{self.name}': draft_kind/draft_block_size require "
                "draft_model — name the drafter snapshot (e.g. a split MTP-head repo) "
                "or remove them."
            )
        logger.info(
            "mlx_vlm service '%s': no telemetry translator for mlx-vlm yet, so the "
            "dashboard shows memory but no TOK/S or prefill progress for this "
            "service (ADR 0006 gap; health and lifecycle are unaffected).",
            self.name,
        )

    # --- Readiness / observability ---
    def is_healthy(self) -> bool:
        """Same probe as the base, plus a Bearer header when ``api_key`` is
        set — mlx-vlm gates its management endpoints (``/health`` included)
        behind the key, and the base's bare ``urlopen`` would 401 forever."""
        if not self.config.api_key:
            return super().is_healthy()
        if self.process is None or self.process.poll() is not None:
            return False
        request = urllib.request.Request(  # noqa: S310 - fixed http scheme
            f"http://{self.host}:{self.port}{self.health_path}",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:  # noqa: S310
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False
