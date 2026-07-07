"""``mlx_lm`` — a native MLX inference engine (§12, Phase 11).

Runs ``mlx_lm.server`` (Apple MLX, OpenAI-compatible) via the shared
:class:`NativeEngineManager` lifecycle — subprocess + HTTP health + ``psutil``
metrics — adapted to MLX's CLI. The model can be a local MLX directory or a
HuggingFace repo id, downloaded on first start with progress surfaced as
activity via the log tailer.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

from sovereign.core.base_native import (
    NativeEngineManager,
    check_local_artifact,
    local_model_bytes,
)
from sovereign.core.registry import register_service
from sovereign.services.mlx_lm.config import MlxLmConfig

# Matches tqdm "Fetching N files: XX%|<bar>| done/total" lines from mlx_lm.server.
_FETCH_RE = re.compile(r"Fetching (\d+) files:\s+(\d+)%[^|]*\|[^|]*\|\s*(\d+)/\1")


@register_service("mlx_lm")
class MlxLmManager(NativeEngineManager):
    """Supervises one native ``mlx_lm.server`` process."""

    base_type = "mlx_lm"
    config_cls = MlxLmConfig
    binary_hint = "It ships with the `mlx-lm` dependency — run `uv sync`."

    # --- resource estimation (§7) ---
    def estimated_memory_gb(self) -> float:
        """Estimate resident memory from the local model weights (or an override).

        Includes the draft model when speculative decoding is configured — both models
        live in unified memory simultaneously. A declared ``prompt_cache_bytes`` is a
        hard KV-cache reservation that also lives in unified memory. For HuggingFace
        repo ids (not yet downloaded) the footprint is unknown and contributes 0.0
        (admitted).
        """
        if self.memory_override_gb is not None:
            return round(self.memory_override_gb, 2)
        total = local_model_bytes(self.config.model)
        if self.config.draft_model is not None:
            total += local_model_bytes(self.config.draft_model)
        if self.config.prompt_cache_bytes is not None:
            total += self.config.prompt_cache_bytes
        return round(total / (1024**3), 2)

    # --- flag generation ---
    def get_start_args(self) -> list[str]:
        """Translate the validated config into an ``mlx_lm.server`` argv."""
        args = [
            self.config.binary,
            "--model",
            os.path.expanduser(self.config.model),
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
            args += ["--draft-model", os.path.expanduser(self.config.draft_model)]
        if self.config.num_draft_tokens is not None:
            args += ["--num-draft-tokens", str(self.config.num_draft_tokens)]
        if self.config.trust_remote_code:
            args += ["--trust-remote-code"]
        args += self.config.extra_args
        return args

    # --- download-progress activity ---
    def _tail_target(self) -> Callable[[Path, threading.Event], None] | None:
        return self._tail_log_for_activity

    def _tail_log_for_activity(self, log_path: Path, stop: threading.Event) -> None:
        """Tail the log file and surface HuggingFace download progress as activity."""
        try:
            with log_path.open() as fh:
                while not stop.is_set():
                    line = fh.readline()
                    if not line:
                        stop.wait(timeout=0.5)
                        continue
                    for m in _FETCH_RE.finditer(line):
                        total, pct, done = m.group(1), m.group(2), m.group(3)
                        self.set_activity(f"downloading model: {done}/{total} files ({pct}%)")
                        if pct == "100":
                            self.clear_activity()
                            return
        except OSError:
            pass

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
