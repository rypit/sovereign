"""``omlx`` — the oMLX inference server as a native engine.

Runs the ``omlx`` CLI (an MLX server with continuous batching and a paged
KV/prefix cache: RAM hot tier + SSD cold tier) as a subprocess of a detached
engine worker, following the ADR 0007 pattern proven by ``llama_cpp``: the
worker (``workers/omlx_adapter.py``) supervises ``omlx serve`` as a *child*
process — tensors live there, not in the tracked worker. Unlike llama_cpp,
omlx exposes no ``/slots``/``/metrics`` scrape surface, so there is no
telemetry-translator loop: heartbeat + memory events come from the generic
``engine_worker`` (and ``get_metrics()``'s recursive child summing), while
prefill/tok-s stats are absent — an ADR 0006 gap surfaced at pre-flight.

The model is an MLX/safetensors snapshot (same artifact kind as ``mlx_lm``):
a local MLX directory or a HuggingFace repo id downloaded into the shared HF
cache before launch. omlx discovers models by *directory layout* rather than
taking a model path, so the adapter symlinks the one resolved snapshot into a
private per-service ``--model-dir`` (one omlx instance = one model,
preserving Sovereign's per-service identity and admission model).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from sovereign.core.registry import register_service
from sovereign.core.units import GB
from sovereign.services.inference.base import (
    NativeEngineManager,
    probe_binary,
)
from sovereign.services.inference.hf import parse_model_ref
from sovereign.services.inference.omlx.config import OmlxConfig

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo

logger = logging.getLogger("sovereign")

#: The binary this engine's worker execs. Provisioned via the sibling
#: Brewfile (`brew "omlx"`); availability is a PATH probe, not an import
#: probe, since the engine runs as an external CLI subprocess (ADR 0007
#: pattern).
_OMLX_BINARY = "omlx"


@register_service("omlx")
class OmlxManager(NativeEngineManager):
    """Supervises one engine worker that runs ``omlx serve`` as a child process."""

    base_type = "omlx"
    config_cls = OmlxConfig
    config: OmlxConfig
    model_artifact_kind = "snapshot"
    binary_hint = (
        "Run `sovereign provision` (installs from the jundot/omlx tap via this "
        "engine's Brewfile), or manually: `brew tap jundot/omlx "
        "https://github.com/jundot/omlx && brew install omlx --HEAD "
        "--with-custom-kernel`."
    )

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Satisfied once ``omlx`` is on ``PATH`` — a binary-on-PATH probe,
        same rationale as ``llama_cpp``: the engine runs as an external CLI
        subprocess, not an in-process binding. The ``Brewfile`` next to this
        module (`brew "omlx"`) is discovered automatically by the shared
        :class:`Provisioner` mixin."""
        return probe_binary(_OMLX_BINARY)

    # --- routing (auto base_type) ---
    @classmethod
    def claim_route(cls, ref: ModelRef, info: RepoInfo | None) -> int | None:
        """Deliberately abstain from ``auto`` routing.

        omlx serves the same MLX/safetensors artifacts as ``mlx_lm``, so any
        claim would contend on the cross-engine confidence scale (see
        :class:`LlamaCppManager` / :class:`MlxLmManager`). For now omlx is
        opt-in only — ``base_type: omlx`` in ``sovereign.yaml``; promoting it
        into the ``auto`` precedence contract is a deliberate later change
        (with an ADR), not a side effect of installing it.
        """
        return None

    # --- resource estimation (§7) ---
    def extra_memory_bytes(self) -> int:
        """omlx's engine-specific term on top of the weights: the in-memory
        hot tier of the paged KV cache. The SSD cold tier is disk, not
        unified memory, so it never counts."""
        if self.config.hot_cache_gb is None:
            return 0
        return self.config.hot_cache_gb * GB

    # --- wiring ---
    def api_model_name(self) -> str:
        """The string clients send as ``"model"``.

        omlx derives model names from ``--model-dir`` subdirectory layout, so
        the name must be a usable relative path: a repo id (``org/name``,
        two-level layout) works as-is; a local model directory contributes
        only its basename. ``served_model_name`` overrides either — the
        adapter names the symlink from this same value, so what omlx serves
        and what :meth:`endpoint` advertises always agree.
        """
        if self.config.served_model_name:
            return self.config.served_model_name
        ref = parse_model_ref(self.config.model)
        if ref.is_local:
            assert ref.local_path is not None
            return ref.local_path.name
        return ref.repo_id or self.config.model

    # --- engine_kwargs mapping (consumed by workers.omlx_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter maps onto ``omlx serve``
        CLI flags (see ``workers/omlx_adapter.build_server_argv``).

        ``model_dir``/``model_name`` drive the adapter's single-model symlink
        layout; the memory guard defaults to Sovereign's own admission
        estimate (rounded GB) so omlx's internal enforcer operates within the
        slice admission control reserved, not the whole machine.
        """
        state_root = self._worker_state_dir() / "omlx" / self.name
        kwargs: dict[str, Any] = {
            "model_dir": str(state_root / "models"),
            "model_name": self.api_model_name(),
        }
        if self.config.max_concurrent_requests is not None:
            kwargs["max_concurrent_requests"] = self.config.max_concurrent_requests

        guard_gb = self.config.memory_guard_gb
        if guard_gb is None:
            estimate = self.estimated_memory_bytes()
            if estimate > 0:
                guard_gb = round(estimate / GB, 2)
        if guard_gb:
            kwargs["memory_guard_gb"] = guard_gb

        if self.config.paged_ssd_cache:
            cache_dir = self.config.paged_ssd_cache_dir or str(state_root / "kv-cache")
            kwargs["paged_ssd_cache_dir"] = os.path.expanduser(cache_dir)
            if self.config.paged_ssd_cache_max_gb is not None:
                kwargs["paged_ssd_cache_max_gb"] = self.config.paged_ssd_cache_max_gb
        if self.config.hot_cache_gb is not None:
            kwargs["hot_cache_gb"] = self.config.hot_cache_gb

        kwargs.update(self.config.engine_kwargs)
        return kwargs

    def start_env(self) -> dict[str, str]:
        """Pass the API key via ``SOVEREIGN_API_KEY`` — the worker adapter reads
        it from the environment and appends ``--api-key`` to ``omlx serve``'s
        argv itself, keeping the secret off the ``ps``-visible parent command
        line and out of the dumped worker-config JSON (same contract as
        ``llama_cpp``)."""
        if self.config.api_key:
            return {"SOVEREIGN_API_KEY": self.config.api_key}
        return {}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if not probe_binary(_OMLX_BINARY):
            raise FileNotFoundError(
                f"omlx binary '{_OMLX_BINARY}' not found on PATH for "
                f"'{self.name}'. {self.binary_hint}"
            )
        if self.config.draft_model is not None:
            # ADR 0006: an engine gap is surfaced loudly, not silently dropped —
            # omlx has no speculative-decoding flags to bridge to.
            raise ValueError(
                f"omlx service '{self.name}': draft_model is not supported — "
                "omlx has no speculative-decoding surface. Remove draft_model "
                "or use base_type: mlx_lm / llama_cpp."
            )
        logger.info(
            "omlx service '%s': omlx exposes no telemetry scrape surface, so the "
            "dashboard shows memory but no TOK/S or prefill progress for this "
            "service (ADR 0006 gap; health and lifecycle are unaffected).",
            self.name,
        )
