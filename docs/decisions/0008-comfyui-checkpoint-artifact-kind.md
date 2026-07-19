# 0008. ComfyUI engine + single-file `checkpoint` artifact kind

Status: accepted

## Context

ComfyUI is an image/video-generation workflow server (Stable Diffusion, SDXL,
Flux) that runs natively on Apple Silicon via PyTorch/MPS — a natural fourth
native engine. But it diverges from the existing engines on three axes:

1. **Artifact shape.** Its models are single-file diffusion checkpoints
   (`.safetensors`), not MLX/safetensors repo *snapshots* and not *gguf*
   files — and diffusion repos also ship a nested diffusers/ tree of
   component weights that are not loadable as a checkpoint. The HF pipeline's
   `model_artifact_kind: Literal["snapshot", "gguf"]` (a `RoutesModelRef`
   ClassVar — a core contract) cannot express this, so admission estimation
   would count the whole repo rather than the one checkpoint file.
2. **Model discovery.** ComfyUI discovers models from a `models/` directory
   tree; a running instance serves whatever is in it, and workflows reference
   checkpoints by filename. Sovereign's identity/admission model is one
   service = one model.
3. **Distribution.** ComfyUI is a Python application installed by `comfy-cli`
   (a pip tool), not a brew formula; there is no scrape surface for TOK/S
   telemetry (generation progress is a websocket).

## Decision

- **Widen the artifact-kind contract**: `ModelArtifactKind =
  Literal["snapshot", "gguf", "checkpoint"]` in `core/base_manager.py`.
  `"checkpoint"` means *one* top-level `.safetensors` file:
  `select_checkpoint_file` picks it (explicit `org/repo/file.safetensors`
  ref, or auto only when the repo has exactly one top-level candidate —
  ambiguity fails loudly, same discipline as GGUF quants), and
  `parse_model_ref` accepts `.safetensors` filenames alongside `.gguf`.
  Estimation/caching/download reuse the existing single-file (gguf) paths.
- **One service = one checkpoint**, preserved via layout: the worker adapter
  (`workers/comfyui_adapter.py`) symlinks the resolved checkpoint into a
  per-service `models/checkpoints/` directory and points ComfyUI at it with
  `--extra-model-paths-config` — the omlx symlinked `--model-dir` pattern.
  The checkpoint's filename is the service's `api_model_name()`.
- **ADR 0007 subprocess pattern, supervise-only**: the worker launches
  `comfy … launch` as a child process and supervises it; no telemetry
  translator (nothing to scrape — an ADR 0006 gap the manager logs at
  pre-flight, exactly like omlx). Provisioning installs comfy-cli via
  `uv tool install comfy-cli` (`provisioning_commands`, no Brewfile), and
  pre-flight runs `comfy install` into the configured `workspace_dir`
  (default `~/.sovereign/comfyui` — user-level, since an install is multi-GB
  of torch shared across stacks).
- **Explicit-only routing**: `claim_route` abstains from `auto` (diffusion
  checkpoints are `.safetensors`, so any claim would contend with mlx_lm's
  safetensors fallback). `base_type: comfyui` is required; promotion into
  the `auto` precedence contract is a deliberate later change with its own
  ADR — the omlx precedent.

## Consequences

- Admission control and `sovereign plan` see the true checkpoint size (one
  file), not the whole diffusion repo; the SOURCE column works unchanged.
- A third artifact kind exists in the core contract; future engines with
  single-file weights (upscalers, VAEs as primary models) can reuse it.
- Scope is deliberately v1-narrow: one checkpoint per service. Loras/VAE/
  controlnet lists, custom nodes, and websocket progress telemetry are
  foreclosed *for now*, not forever — each is an additive follow-up.
- Engine asymmetry grows (an engine whose "server" is a Python app managed
  by a pip-installed CLI), but inside the accepted ADR 0007 shape.

## Alternatives considered

- **Treat the repo as a `snapshot`** — rejected: downloads and counts
  gigabytes of diffusers components ComfyUI won't load as a checkpoint,
  inflating admission and disk for nothing.
- **Bring-your-own models directory** (point config at an existing ComfyUI
  install's tree) — rejected as the primary model: no HF pipeline
  integration, no per-model admission, and `plan` goes blind. May return
  later as an *additive* `extra_model_dirs` option.
- **Claim diffusion refs in `auto` routing** — rejected for v1: it touches
  the cross-engine confidence contract; explicit `base_type` costs one YAML
  line and keeps the precedence scale stable.

---
Provenance: follow-up to ADR 0007 (subprocess engines) and ADR 0006 (gap
policy); artifact-kind contract introduced alongside the comfyui integration.
