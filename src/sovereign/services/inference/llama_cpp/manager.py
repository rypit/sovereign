"""``llama_cpp`` — the first native inference engine (§12, Phase 4).

Runs ``llama-server`` as a native subprocess (bare metal, for real Metal
acceleration — §2.1) via the shared :class:`NativeEngineManager` lifecycle:
subprocess + HTTP health + ``psutil`` metrics. The model can be a local GGUF
path or a HuggingFace repo id (``org/name[:quant]`` or ``org/name/file.gguf``),
downloaded by Sovereign into the shared HF cache before launch (DOWNLOADING
state); the server always starts from the resolved local path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.config import ServiceEntry
from sovereign.core.registry import register_service
from sovereign.services.inference.base import (
    NativeEngineManager,
    probe_import,
)
from sovereign.services.inference.llama_cpp.config import (
    VALID_KV_CACHE_TYPES,
    LlamaCppConfig,
    LlamaPolicy,
)

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo

logger = logging.getLogger("sovereign")

#: Metal-wheel index for llama-cpp-python — avoids a from-source cmake build
#: on user machines (see pyproject.toml's platform-marked dependency).
_LLAMA_CPP_PYTHON_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/metal"


@register_service("llama_cpp")
class LlamaCppManager(NativeEngineManager):
    """Supervises one embedded ``llama-cpp-python`` engine worker."""

    base_type = "llama_cpp"
    config_cls = LlamaCppConfig
    config: LlamaCppConfig
    model_artifact_kind = "gguf"
    #: llama-cpp-python has no second-GGUF speculative decoding (§3a hard
    #: gap) — a configured draft_model's weights never actually load, so they
    #: must not count toward this engine's admission-control estimate.
    supports_draft_model = False
    #: Probed (out-of-process) rather than checked via a binary-on-PATH: the
    #: engine now runs in-process inside the Python worker.
    import_probe_modules = ("llama_cpp", "llama_cpp.server.app")
    #: Fallback install: the prebuilt Metal wheel avoids a from-source cmake
    #: build. Only reached if the module import probe still fails after any
    #: Brewfile step (there is none here — see class docstring).
    provisioning_commands = [
        [
            "uv",
            "pip",
            "install",
            "llama-cpp-python[server]>=0.3",
            "--extra-index-url",
            _LLAMA_CPP_PYTHON_WHEEL_INDEX,
        ]
    ]

    def __init__(self, entry: ServiceEntry) -> None:
        super().__init__(entry)
        self.policy = LlamaPolicy.model_validate(entry.policy or {})

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Satisfied once every ``import_probe_modules`` entry imports cleanly —
        there is no Brewfile/binary for this engine any more (it's a Python
        binding, not a CLI); ``provisioning_commands`` installs the wheel."""
        return all(probe_import(m) for m in cls.import_probe_modules)

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
        """Context available *per agent* — nominally total context divided across
        ``-np`` slots. The embedded llama-cpp-python server has no ``-np``
        equivalent: it holds a single ``Llama`` instance and serves requests
        sequentially (§3a hard gap — see ``max_parallel`` warning in
        :meth:`prepare_environment`), so in practice there is exactly one slot
        and this divides across a purely nominal count kept for config
        compatibility and admission-control math."""
        if self.config.context_size is None:
            return None
        return self.config.context_size // (self.config.max_parallel or 1)

    # --- engine_kwargs mapping (consumed by workers.llama_cpp_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter maps onto
        ``llama_cpp.server.settings.ModelSettings`` (see
        ``workers/llama_cpp_adapter.build_model_settings``)."""
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
        """Pass the API key via ``SOVEREIGN_API_KEY`` — the embedded server's
        adapter reads it from the environment, keeping the secret off the
        ``ps``-visible command line/worker-config JSON."""
        if self.config.api_key:
            return {"SOVEREIGN_API_KEY": self.config.api_key}
        return {}

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if self.config.draft_model is not None:
            raise ValueError(
                f"llama_cpp engine no longer supports GGUF draft models for '{self.name}' "
                "(no second-GGUF speculative decoding in llama-cpp-python); remove "
                "draft_model, or use the mlx_lm engine for real speculative decoding."
            )
        if self.config.max_parallel is not None and self.config.max_parallel > 1:
            logger.warning(
                "llama_cpp service '%s': max_parallel=%d has no effect — the embedded "
                "server is single-Llama/sequential; requests queue rather than "
                "running concurrently.",
                self.name,
                self.config.max_parallel,
            )
        self._validate_prompt_caching()

    def _validate_prompt_caching(self) -> None:
        """Pre-flight the prompt-caching policy (§7): dir writable, dtype valid.

        The embedded llama-cpp-python server has no slot-save/restore-to-disk
        equivalent (§3a hard gap): ``enabled`` degrades to an in-process RAM
        cache (``cache=True, cache_type="ram"``, wired in the worker adapter),
        and ``cache_path`` — while still validated as a writable directory for
        config-compatibility — is inert; we warn so this doesn't look silently
        broken.
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
            "llama_cpp service '%s': prompt_caching.cache_path (%s) is inert — the "
            "embedded server degrades to an in-process RAM cache, not disk-backed "
            "slot-save/restore.",
            self.name,
            cache_dir,
        )
