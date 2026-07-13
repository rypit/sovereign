"""``llama_cpp`` — the first native inference engine (§12, Phase 4).

Per ADR 0007, runs the native ``llama-server`` binary as a subprocess of a
detached engine-worker process (bare metal, for real Metal acceleration —
§2.1) via the shared :class:`NativeEngineManager` lifecycle: worker
subprocess + HTTP health + telemetry/``psutil`` metrics. The worker
(``workers/llama_cpp_adapter.py``) launches ``llama-server`` as a *child* and
translates its HTTP telemetry surface (``/slots``, ``/metrics``) into
Sovereign's UDS NDJSON events — tensors live in that child, not the tracked
worker process. The model can be a local GGUF path or a HuggingFace repo id
(``org/name[:quant]`` or ``org/name/file.gguf``), downloaded by Sovereign
into the shared HF cache before launch (DOWNLOADING state); the worker
always starts from the resolved local path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.config import ServiceEntry
from sovereign.core.registry import register_service
from sovereign.services.inference.base import (
    NativeEngineManager,
    check_local_artifact,
    probe_binary,
)
from sovereign.services.inference.llama_cpp.config import (
    VALID_KV_CACHE_TYPES,
    LlamaCppConfig,
    LlamaPolicy,
)

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo

logger = logging.getLogger("sovereign")

#: The binary this engine's worker execs. Provisioned via the sibling
#: Brewfile (`brew "llama.cpp"`); availability is a PATH probe, not an
#: import probe, since the engine now runs as an external CLI subprocess.
_LLAMA_SERVER_BINARY = "llama-server"


@register_service("llama_cpp")
class LlamaCppManager(NativeEngineManager):
    """Supervises one engine worker that runs ``llama-server`` as a child process."""

    base_type = "llama_cpp"
    config_cls = LlamaCppConfig
    config: LlamaCppConfig
    model_artifact_kind = "gguf"
    binary_hint = (
        "Install it via Homebrew (`brew install llama.cpp`) or run `sovereign provision`."
    )

    def __init__(self, entry: ServiceEntry) -> None:
        super().__init__(entry)
        self.policy = LlamaPolicy.model_validate(entry.policy or {})

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Satisfied once ``llama-server`` is on ``PATH`` — a binary-on-PATH
        probe (:func:`sovereign.services.inference.base.probe_binary`), not
        an import probe, since ADR 0007 runs this engine as an external CLI
        subprocess rather than an in-process Python binding. The
        ``Brewfile`` next to this module (`brew "llama.cpp"`) is discovered
        automatically by the shared :class:`Provisioner` mixin."""
        return probe_binary(_LLAMA_SERVER_BINARY)

    # --- routing (auto base_type) ---
    @classmethod
    def claim_route(cls, ref: ModelRef, info: RepoInfo | None) -> int | None:
        """Claim GGUF work: a local ``.gguf``, an explicit quant/filename, or ``.gguf``
        siblings on the hub.

        Confidence ordering (higher wins across engines) preserves the original
        rule precedence: an explicit quant/filename or a local GGUF outranks an mlx
        tag (see :class:`MlxLmManager`), which in turn outranks a bare ``.gguf`` sibling.
        """
        if ref.is_local:
            path = ref.local_path
            assert path is not None
            if path.suffix == ".gguf" or (path.is_dir() and any(path.glob("*.gguf"))):
                return 50
            return None
        if ref.quant is not None or ref.filename is not None:
            return 50
        if info is not None and any(n.endswith(".gguf") for n, _ in info.siblings):
            return 30
        return None

    # --- resource estimation (§7) ---
    def extra_memory_bytes(self) -> int:
        """llama_cpp's engine-specific term on top of the weights: the KV cache."""
        return self.estimated_kv_cache_bytes()

    def estimated_kv_cache_bytes(self) -> int:
        """KV cache grows with total context (§7 — slots×context is joint)."""
        ctx = self.config.context_size or 0
        return ctx * self.config.kv_bytes_per_token

    def per_slot_context(self) -> int | None:
        """Context available *per agent* — total context divided across ``-np``
        slots. Per ADR 0007 this division is now **real**: ``llama-server``
        natively multiplexes ``max_parallel`` continuous-batching slots over
        one context window (``-c`` total, ``-np`` slots), so each slot gets
        ``context_size // max_parallel`` tokens of context in practice, not
        just in admission-control math."""
        if self.config.context_size is None:
            return None
        return self.config.context_size // (self.config.max_parallel or 1)

    # --- engine_kwargs mapping (consumed by workers.llama_cpp_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter maps onto
        ``llama-server`` CLI flags (see
        ``workers/llama_cpp_adapter.build_server_argv``)."""
        kwargs: dict[str, Any] = {}
        if self.config.gpu_layers is not None:
            kwargs["gpu_layers"] = self.config.gpu_layers
        if self.config.threads is not None:
            kwargs["threads"] = self.config.threads
        if self.config.context_size is not None:
            kwargs["context_size"] = self.config.context_size
        if self.config.max_parallel is not None:
            kwargs["max_parallel"] = self.config.max_parallel
        if self.config.num_draft_tokens is not None:
            kwargs["num_draft_tokens"] = self.config.num_draft_tokens

        caching = self.policy.prompt_caching
        if caching and caching.enabled:
            kwargs["kv_cache_type"] = caching.kv_cache_type

        kwargs.update(self.config.engine_kwargs)
        return kwargs

    def start_env(self) -> dict[str, str]:
        """Pass the API key via ``SOVEREIGN_API_KEY`` — the worker adapter reads
        it from the environment and appends ``--api-key`` to ``llama-server``'s
        argv itself, keeping the secret off the ``ps``-visible parent command
        line and out of the dumped worker-config JSON."""
        if self.config.api_key:
            return {"SOVEREIGN_API_KEY": self.config.api_key}
        return {}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if not probe_binary(_LLAMA_SERVER_BINARY):
            raise FileNotFoundError(
                f"llama_cpp binary '{_LLAMA_SERVER_BINARY}' not found on PATH for "
                f"'{self.name}'. {self.binary_hint}"
            )
        if self.config.draft_model is not None:
            # Two-model speculative decoding is native to llama-server; the
            # draft GGUF is validated/downloaded like the main model and its
            # weights count toward admission.
            check_local_artifact(
                self.config.draft_model, kind="llama_cpp draft_model", service=self.name
            )
        self._validate_prompt_caching()

    def _validate_prompt_caching(self) -> None:
        """Pre-flight the prompt-caching policy (§7): dir writable, dtype valid.

        ``kv_cache_type`` maps directly onto ``llama-server``'s
        ``--cache-type-k``/``--cache-type-v`` flags; ``cache_path`` still has
        no llama-server ``--slot-save-path`` wiring here (accept-and-warn
        degrade per ADR 0006 Mitigation 2), so it's validated as a writable
        directory but not yet passed through.
        """
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
        logger.warning(
            "llama_cpp service '%s': prompt_caching.cache_path (%s) is inert — "
            "Sovereign doesn't yet wire it to llama-server's --slot-save-path, "
            "so prompt caching degrades to llama-server's own in-memory reuse, "
            "not disk-backed slot-save/restore.",
            self.name,
            cache_dir,
        )
