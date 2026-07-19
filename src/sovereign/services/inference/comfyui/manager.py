"""``comfyui`` — the ComfyUI image/video-generation server as a native engine.

Runs ComfyUI (Stable Diffusion / SDXL / Flux workflows over an HTTP API,
natively on Apple Silicon via PyTorch/MPS) as a subprocess of a detached
engine worker, following the ADR 0007 pattern proven by ``llama_cpp`` and
``omlx``: the worker (``workers/comfyui_adapter.py``) supervises
``comfy … launch`` as a *child* process — tensors live there, not in the
tracked worker. Like omlx there is no telemetry-translator loop: ComfyUI has
no ``/slots``/``/metrics``-style scrape surface (generation progress is a
websocket, deferred), so heartbeat + memory events come from the generic
``engine_worker`` while TOK/S/prefill stats are absent — an ADR 0006 gap
surfaced at pre-flight.

The model is a single-file diffusion *checkpoint* (``.safetensors``) — a new
artifact kind next to ``snapshot``/``gguf`` (ADR 0008). ComfyUI discovers
models from a ``models/`` directory tree, so the adapter symlinks the one
resolved checkpoint into a per-service ``models/checkpoints/`` layout wired
in via ``--extra-model-paths-config`` (one comfyui instance = one checkpoint,
preserving Sovereign's per-service identity and admission model).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.core.registry import register_service
from sovereign.services.inference.base import NativeEngineManager, probe_binary
from sovereign.services.inference.comfyui.config import ComfyUIConfig
from sovereign.services.inference.hf import parse_model_ref

if TYPE_CHECKING:
    from sovereign.services.inference.hf import ModelRef, RepoInfo

logger = logging.getLogger("sovereign")

#: The CLI this engine's worker execs. Provisioned via ``uv tool install``
#: (comfy-cli is a pip tool, not a brew formula); availability is a PATH
#: probe, same rationale as ``llama_cpp``/``omlx`` (ADR 0007 pattern).
_COMFY_BINARY = "comfy"

#: Bounded wait for a first-time ``comfy install`` — it clones ComfyUI and
#: installs torch, which can take a while on a cold cache.
_INSTALL_TIMEOUT = 1800.0


def _install_workspace(workspace_dir: str) -> None:
    """Install ComfyUI into ``workspace_dir`` via ``comfy install``.

    A module-level seam (patched in tests) so the pre-flight workspace check
    stays unit-testable without comfy-cli installed. ``--fast-deps`` uses uv
    for the dependency install; ``--skip-manager`` keeps v1 to the bare
    server (custom nodes are out of scope, see ADR 0008).
    """
    result = subprocess.run(  # noqa: S603 - fixed argv from config
        [
            _COMFY_BINARY,
            "--skip-prompt",
            "--workspace",
            workspace_dir,
            "install",
            "--fast-deps",
            "--skip-manager",
        ],
        capture_output=True,
        text=True,
        timeout=_INSTALL_TIMEOUT,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"`comfy install` into '{workspace_dir}' failed: {detail}")


@register_service("comfyui")
class ComfyUIManager(NativeEngineManager):
    """Supervises one engine worker that runs ``comfy … launch`` as a child process."""

    base_type = "comfyui"
    config_cls = ComfyUIConfig
    config: ComfyUIConfig
    model_artifact_kind = "checkpoint"
    binary_hint = (
        "Run `sovereign provision` (installs comfy-cli via `uv tool install "
        "comfy-cli`), or install it manually and ensure `comfy` is on PATH."
    )
    provisioning_commands = [["uv", "tool", "install", "comfy-cli"]]

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Satisfied once ``comfy`` is on ``PATH`` — a binary-on-PATH probe,
        same rationale as ``llama_cpp``/``omlx``: the engine runs as an
        external CLI subprocess, not an in-process binding."""
        return probe_binary(_COMFY_BINARY)

    # --- routing (auto base_type) ---
    @classmethod
    def claim_route(cls, ref: ModelRef, info: RepoInfo | None) -> int | None:
        """Deliberately abstain from ``auto`` routing.

        Diffusion checkpoints are ``.safetensors`` files, so any claim would
        contend on the cross-engine confidence scale with ``mlx_lm``'s
        safetensors fallback (see :class:`MlxLmManager`). comfyui is opt-in
        only — ``base_type: comfyui`` in ``sovereign.yaml``; promoting it
        into the ``auto`` precedence contract is a deliberate later change
        (with an ADR), not a side effect of installing it.
        """
        return None

    # --- wiring ---
    def _workspace_dir(self) -> str:
        return os.path.expanduser(self.config.workspace_dir)

    def api_model_name(self) -> str:
        """The checkpoint name workflows reference (``CheckpointLoaderSimple``).

        ComfyUI identifies checkpoints by their filename within
        ``models/checkpoints/``. The adapter names the symlink from this same
        value, so the name workflows must use and what :meth:`endpoint`
        advertises always agree.
        """
        if self.config.served_model_name:
            return self.config.served_model_name
        ref = parse_model_ref(self.config.model)
        if ref.is_local:
            assert ref.local_path is not None
            return ref.local_path.name
        if ref.filename is not None:
            return Path(ref.filename).name
        return ref.repo_id or self.config.model

    # --- engine_kwargs mapping (consumed by workers.comfyui_adapter) ---
    def engine_kwargs(self) -> dict[str, Any]:
        """Sovereign-side settings the worker's adapter maps onto the
        ``comfy … launch`` argv (see ``workers/comfyui_adapter.build_server_argv``).

        ``models_root``/``checkpoint_name`` drive the adapter's single-checkpoint
        symlink + ``extra_model_paths.yaml`` layout; ``workspace_dir`` selects
        the ComfyUI install to launch.
        """
        state_root = self._worker_state_dir() / "comfyui" / self.name
        kwargs: dict[str, Any] = {
            "workspace_dir": self._workspace_dir(),
            "models_root": str(state_root / "models"),
            "checkpoint_name": self.api_model_name(),
        }
        if self.config.output_dir is not None:
            kwargs["output_dir"] = os.path.expanduser(self.config.output_dir)
        kwargs.update(self.config.engine_kwargs)
        return kwargs

    # --- Resource cooperation ---
    def prepare_environment(self) -> None:
        super().prepare_environment()
        if not probe_binary(_COMFY_BINARY):
            raise FileNotFoundError(
                f"comfy-cli binary '{_COMFY_BINARY}' not found on PATH for "
                f"'{self.name}'. {self.binary_hint}"
            )
        if self.config.draft_model is not None:
            # ADR 0006: an engine gap is surfaced loudly, not silently dropped —
            # ComfyUI has no speculative-decoding surface to bridge to.
            raise ValueError(
                f"comfyui service '{self.name}': draft_model is not supported — "
                "ComfyUI has no speculative-decoding surface. Remove draft_model."
            )
        workspace = self._workspace_dir()
        if not (Path(workspace) / "ComfyUI" / "main.py").exists():
            logger.info(
                "comfyui service '%s': installing ComfyUI into workspace %s "
                "(first run; this clones ComfyUI and installs torch)",
                self.name,
                workspace,
            )
            _install_workspace(workspace)
        logger.info(
            "comfyui service '%s': ComfyUI exposes no telemetry scrape surface, "
            "so the dashboard shows memory but no TOK/S or prefill progress for "
            "this service (ADR 0006 gap; health and lifecycle are unaffected).",
            self.name,
        )
