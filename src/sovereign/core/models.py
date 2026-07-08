"""HuggingFace model metadata, routing, and download library (Phase M1).

Pure library — no typer, no manager imports. Provides:
- ModelRef parsing from raw model strings (local path, repo id, repo:quant, repo/file.gguf)
- Repo metadata fetch, memoised per process (never caches offline/transient failures)
- GGUF file selection with shard grouping and quant disambiguation
- Weight byte estimation from metadata or disk
- Download with byte-level progress via a custom ``tqdm_class`` (works with the Xet
  backend through ``hf_xet``, which reports transfer progress through the same bars)
- Engine routing: mlx_lm vs llama_cpp, with persisted RoutingCache for offline restarts

Helpers ``looks_local`` and ``local_model_bytes`` are defined here and re-exported
from ``base_native`` for backwards compatibility.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)
from tqdm.auto import tqdm as _BaseTqdm

from sovereign.utils.state import read_json, write_json

if TYPE_CHECKING:
    from sovereign.config import ServiceEntry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelResolutionError(Exception):
    """Base for model resolution problems."""


class ModelAccessError(ModelResolutionError):
    """Gated repo or bad token."""


class ModelNotFoundError(ModelResolutionError):
    """Repo doesn't exist on HuggingFace Hub."""


class ModelDownloadError(ModelResolutionError):
    """Disk space exhausted or mid-download failure."""


class RoutingError(ModelResolutionError):
    """Auto routing is impossible for this model ref."""


# ---------------------------------------------------------------------------
# Local-path helpers (moved from base_native; re-exported there for back-compat)
# ---------------------------------------------------------------------------


def looks_local(model: str) -> bool:
    """Whether ``model`` refers to a local path (vs. a HuggingFace repo id)."""
    return model.startswith(("/", "~", ".")) or Path(os.path.expanduser(model)).exists()


def local_model_bytes(model: str) -> int:
    """Bytes on disk for a local model path; 0 for a HuggingFace repo id or missing path."""
    p = Path(os.path.expanduser(model))
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    if p.is_file():
        return p.stat().st_size
    return 0


# ---------------------------------------------------------------------------
# ModelRef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRef:
    raw: str
    is_local: bool
    local_path: Path | None = None  # expanded absolute path, when is_local
    repo_id: str | None = None  # "org/name"
    quant: str | None = None  # from "org/name:Q4_K_M"
    filename: str | None = None  # from "org/name/sub/file.gguf"


def parse_model_ref(value: str) -> ModelRef:
    """Parse a raw model string into a typed ModelRef.

    Parsing order:
    1. Starts with /  ~  .  or path exists on disk → local.
    2. Exactly one colon in "org/name:QUANT" form → repo + quant.
    3. Three or more slash segments ending in .gguf → repo (first two) + filename.
    4. Otherwise "org/name" repo id.
    """
    # 1. Local path
    if looks_local(value):
        return ModelRef(
            raw=value,
            is_local=True,
            local_path=Path(os.path.expanduser(value)),
        )

    # 2. "org/name:QUANT" — single colon
    if value.count(":") == 1 and "/" in value.split(":")[0]:
        repo, quant = value.split(":", 1)
        return ModelRef(raw=value, is_local=False, repo_id=repo, quant=quant)

    # 3. Three or more "/" segments ending in ".gguf"
    parts = value.split("/")
    if len(parts) >= 3 and parts[-1].endswith(".gguf"):
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return ModelRef(raw=value, is_local=False, repo_id=repo_id, filename=filename)

    # 4. Plain "org/name" repo id
    return ModelRef(raw=value, is_local=False, repo_id=value)


# ---------------------------------------------------------------------------
# Repo metadata (memoised; never caches offline/transient None results)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoInfo:
    repo_id: str
    tags: tuple[str, ...]
    siblings: tuple[tuple[str, int | None], ...]  # (rfilename, size_bytes)


# Only successful fetches are stored here — transient/offline None is not pinned.
_repo_info_cache: dict[str, RepoInfo] = {}


def fetch_repo_info(repo_id: str) -> RepoInfo | None:
    """Fetch model metadata from HuggingFace Hub, memoised per process.

    Returns None on transient network or offline errors so callers fall back.
    Raises ModelAccessError for gated repos, ModelNotFoundError when the repo
    doesn't exist. Successful results are cached for the process lifetime.
    """
    if repo_id in _repo_info_cache:
        return _repo_info_cache[repo_id]

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except GatedRepoError as exc:
        raise ModelAccessError(
            f"'{repo_id}' is gated — set HF_TOKEN in .env or run `hf auth login`"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ModelNotFoundError(
            f"Repository '{repo_id}' not found on HuggingFace Hub"
        ) from exc
    except Exception:
        # Transient: connection error, timeout, 5xx, offline — do not cache
        return None

    tags = tuple(info.tags or [])
    siblings = tuple(
        (s.rfilename, getattr(s, "size", None))
        for s in (info.siblings or [])
    )
    result = RepoInfo(repo_id=repo_id, tags=tags, siblings=siblings)
    _repo_info_cache[repo_id] = result
    return result


# ---------------------------------------------------------------------------
# GGUF file selection
# ---------------------------------------------------------------------------

_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}$")


def _quant_key(filename: str) -> str:
    """Stem with shard suffix stripped — identifies the quant variant."""
    stem = filename[:-5]  # remove ".gguf"
    return _SHARD_RE.sub("", stem)


def select_gguf_files(
    info: RepoInfo,
    *,
    quant: str | None,
    filename: str | None,
) -> list[str]:
    """Select GGUF shards for the given quant/filename hint.

    Returns all shards of the chosen quant, sorted by name.  Raises
    ModelResolutionError when selection is ambiguous or impossible.
    """
    # All non-mmproj GGUF files
    candidates = [
        name
        for name, _ in info.siblings
        if name.endswith(".gguf") and not Path(name).name.startswith("mmproj")
    ]
    if not candidates:
        raise ModelResolutionError(f"No GGUF files found in '{info.repo_id}'")

    # Exact filename
    if filename is not None:
        matches = [f for f in candidates if f == filename or Path(f).name == filename]
        if not matches:
            available = ", ".join(sorted(candidates))
            raise ModelResolutionError(
                f"File '{filename}' not found in '{info.repo_id}'. Available: {available}"
            )
        return sorted(matches)

    # Group by quant key (collapses shards)
    quant_groups: dict[str, list[str]] = defaultdict(list)
    for name in candidates:
        quant_groups[_quant_key(name)].append(name)
    distinct = list(quant_groups.keys())

    if quant is not None:
        quant_lower = quant.lower()
        matched = [q for q in distinct if quant_lower in q.lower()]
        if len(matched) == 0:
            available = ", ".join(sorted(distinct))
            raise ModelResolutionError(
                f"Quant '{quant}' not found in '{info.repo_id}'. "
                f"Available quants: {available}"
            )
        if len(matched) > 1:
            options = ", ".join(sorted(matched))
            raise ModelResolutionError(
                f"Quant '{quant}' is ambiguous in '{info.repo_id}': {options}"
            )
        return sorted(quant_groups[matched[0]])

    # Auto-select
    if len(distinct) == 1:
        return sorted(quant_groups[distinct[0]])
    # Prefer Q4_K_M (llama.cpp default)
    q4_matches = [q for q in distinct if "q4_k_m" in q.lower()]
    if len(q4_matches) == 1:
        return sorted(quant_groups[q4_matches[0]])
    available = ", ".join(sorted(distinct))
    raise ModelResolutionError(
        f"Multiple quants available in '{info.repo_id}': {available}. "
        "Specify one with 'repo:quant' syntax."
    )


def weight_bytes(
    info: RepoInfo,
    kind: Literal["snapshot", "gguf"],
    *,
    quant: str | None = None,
    filename: str | None = None,
) -> int | None:
    """Sum of weight file sizes for the given kind. Returns None if any size is unknown."""
    sibling_sizes = dict(info.siblings)

    if kind == "gguf":
        try:
            files = select_gguf_files(info, quant=quant, filename=filename)
        except ModelResolutionError:
            return None
        sizes = [sibling_sizes.get(f) for f in files]
        if any(s is None for s in sizes):
            return None
        return sum(sizes)  # type: ignore[arg-type]

    # snapshot: safetensors weight files
    safetensors = [n for n in sibling_sizes if n.endswith(".safetensors")]
    model_sf = [n for n in safetensors if Path(n).name.startswith("model")]
    consolidated_sf = [n for n in safetensors if Path(n).name.startswith("consolidated")]
    if model_sf and consolidated_sf:
        # Both present: use only model* to avoid double-counting
        safetensors = model_sf
    if not safetensors:
        # Fall back to PyTorch .bin files
        safetensors = [n for n in sibling_sizes if n.endswith(".bin")]
    if not safetensors:
        return None
    sizes = [sibling_sizes.get(f) for f in safetensors]
    if any(s is None for s in sizes):
        return None
    return sum(sizes)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache + estimation
# ---------------------------------------------------------------------------

# Snapshot downloads exclude GGUF files; mlx needs config + tokenizer + safetensors.
_SNAPSHOT_IGNORE = ["*.gguf"]


def cached_model_path(ref: ModelRef, kind: Literal["snapshot", "gguf"]) -> Path | None:
    """Return the local cache path if the model is already downloaded, else None."""
    if ref.is_local:
        return ref.local_path

    if kind == "snapshot":
        try:
            path = snapshot_download(
                ref.repo_id,
                local_files_only=True,
                ignore_patterns=_SNAPSHOT_IGNORE,
            )
            return Path(path)
        except Exception:
            return None

    # gguf: check shards individually
    if ref.repo_id is None:
        return None
    info = _repo_info_cache.get(ref.repo_id)
    if info is not None:
        try:
            filenames = select_gguf_files(info, quant=ref.quant, filename=ref.filename)
        except ModelResolutionError:
            return None
    elif ref.filename is not None:
        filenames = [ref.filename]
    else:
        return None

    first_path: Path | None = None
    for fname in filenames:
        try:
            p = hf_hub_download(ref.repo_id, fname, local_files_only=True)
            if first_path is None:
                first_path = Path(p)
        except Exception:
            return None  # any shard missing → not fully cached
    return first_path


def estimate_model_bytes(ref: ModelRef, kind: Literal["snapshot", "gguf"]) -> int | None:
    """Estimate model weight size in bytes via a fallback chain.

    Order: local disk → cached HF path on disk → HF repo metadata → None.
    """
    # 1. Local
    if ref.is_local:
        n = local_model_bytes(ref.raw)
        return n or None

    # 2. Cached on disk
    cached = cached_model_path(ref, kind)
    if cached is not None:
        if cached.is_dir():
            total = sum(f.stat().st_size for f in cached.rglob("*") if f.is_file())
            return total or None
        if cached.is_file():
            size = cached.stat().st_size
            return size or None

    # 3. Repo metadata
    if ref.repo_id is not None:
        info = fetch_repo_info(ref.repo_id)
        if info is not None:
            return weight_bytes(info, kind, quant=ref.quant, filename=ref.filename)

    # 4. Unknown
    return None


# ---------------------------------------------------------------------------
# Download with byte-level progress
# ---------------------------------------------------------------------------


def _blobs_dir(repo_id: str) -> Path:
    """HF cache blobs directory for a given repo id."""
    from huggingface_hub import constants  # lazy import keeps startup fast

    cache_dir = Path(os.environ.get("HF_HUB_CACHE", str(constants.HF_HUB_CACHE)))
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    return cache_dir / repo_folder / "blobs"


class _ProgressAggregator:
    """Formats a human progress line from cumulative byte counts.

    Testable without tqdm or threads: ``update(current, total)`` is a pure
    function of the counts (plus a monotonic clock, injectable for tests). Keeps
    a small rolling window for a smoothed MB/s and an ETA.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._window: list[tuple[float, int]] = []  # (monotonic_time, bytes)

    def update(self, current: int, total: int, *, now: float | None = None) -> str | None:
        """Return a progress string, or None if nothing useful can be said yet."""
        if total <= 0:
            return None
        now = time.monotonic() if now is None else now
        self._window.append((now, current))
        if len(self._window) > 5:
            self._window.pop(0)

        pct = min(100.0, current / total * 100)
        speed_str = ""
        eta_str = ""
        if len(self._window) >= 2:
            t0, b0 = self._window[0]
            elapsed = now - t0
            if elapsed > 0:
                speed_bps = (current - b0) / elapsed
                speed_str = f" — {speed_bps / (1024 * 1024):.0f} MB/s"
                remaining = total - current
                if speed_bps > 0 and remaining > 0:
                    mins, secs = divmod(int(remaining / speed_bps), 60)
                    eta_str = f", ETA {mins}m{secs:02d}s"

        return (
            f"downloading {self._label}: "
            f"{current / (1024**3):.1f}/{total / (1024**3):.1f} GB "
            f"({pct:.0f}%){speed_str}{eta_str}"
        )


def _make_progress_tqdm(progress: Callable[[str], None], label: str) -> type[_BaseTqdm]:
    """Build a ``tqdm`` subclass that forwards byte-download progress to ``progress``.

    huggingface_hub (and hf_xet for Xet transfers) render download progress through
    ``tqdm_class``; the byte bars carry ``unit="B"``. We aggregate the ``n``/``total``
    of every live byte bar — one shared bar for ``snapshot_download``, one per shard
    for ``hf_hub_download`` — into a single line. Subclassing vanilla ``tqdm`` (not
    huggingface_hub's) keeps the counter live even when stdout isn't a TTY (the daemon
    case); each bar renders into a throwaway buffer so nothing clutters the console.
    """
    live: set[_BaseTqdm] = set()
    lock = threading.Lock()
    agg = _ProgressAggregator(label)

    def _emit() -> None:
        with lock:
            bars = [t for t in live if getattr(t, "unit", "") == "B" and (t.total or 0) > 0]
            current = sum(int(t.n) for t in bars)
            total = sum(int(t.total) for t in bars)
        msg = agg.update(current, total)
        if msg:
            progress(msg)

    class _ProgressTqdm(_BaseTqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.setdefault("disable", False)  # keep the counter live off-TTY
            kwargs.setdefault("file", io.StringIO())  # swallow the rendered bar
            super().__init__(*args, **kwargs)
            with lock:
                live.add(self)

        def update(self, n: float | None = 1) -> bool | None:
            displayed = super().update(n)
            _emit()
            return displayed

        def close(self) -> None:
            with lock:
                live.discard(self)
            super().close()

    return _ProgressTqdm


def download_model(
    ref: ModelRef,
    kind: Literal["snapshot", "gguf"],
    *,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download a model to the HF cache and return its local path.

    Local refs return immediately. Raises ModelDownloadError on disk-space
    issues or download failures; ModelAccessError / ModelNotFoundError propagate.
    """
    if ref.is_local:
        assert ref.local_path is not None
        return ref.local_path

    assert ref.repo_id is not None

    # Pre-flight disk check
    expected: int | None = None
    info = fetch_repo_info(ref.repo_id)  # may raise ModelAccessError / ModelNotFoundError
    if info is not None:
        expected = weight_bytes(info, kind, quant=ref.quant, filename=ref.filename)
    if expected is not None:
        already = _cached_bytes(ref.repo_id)
        needed = max(0, expected - already)
        try:
            from huggingface_hub import constants

            cache_dir = Path(os.environ.get("HF_HUB_CACHE", str(constants.HF_HUB_CACHE)))
            free = shutil.disk_usage(cache_dir).free
        except Exception:
            free = None
        if free is not None and needed * 1.1 > free:
            needed_gb = needed / (1024**3)
            free_gb = free / (1024**3)
            raise ModelDownloadError(
                f"Not enough disk space to download '{ref.repo_id}': "
                f"need {needed_gb:.1f} GB, only {free_gb:.1f} GB free"
            )

    # Byte-level progress via tqdm_class (also carries hf_xet transfer progress).
    tqdm_class = _make_progress_tqdm(progress, ref.repo_id) if progress is not None else None

    try:
        if kind == "snapshot":
            path = snapshot_download(
                ref.repo_id, ignore_patterns=_SNAPSHOT_IGNORE, tqdm_class=tqdm_class
            )
            return Path(path)

        # gguf: download all shards; re-fetch info if needed
        if info is None:
            raise ModelDownloadError(
                f"Cannot fetch metadata for '{ref.repo_id}' (offline?)"
            )
        files = select_gguf_files(info, quant=ref.quant, filename=ref.filename)
        first_path: Path | None = None
        for fname in files:
            p = hf_hub_download(ref.repo_id, fname, tqdm_class=tqdm_class)
            if first_path is None:
                first_path = Path(p)
        assert first_path is not None
        return first_path

    except (ModelAccessError, ModelNotFoundError, ModelDownloadError):
        raise
    except GatedRepoError as exc:
        raise ModelAccessError(
            f"'{ref.repo_id}' is gated — set HF_TOKEN in .env or run `hf auth login`"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ModelNotFoundError(f"Repository '{ref.repo_id}' not found") from exc
    except Exception as exc:
        raise ModelDownloadError(
            f"Download failed for '{ref.repo_id}': {exc}"
        ) from exc


def _cached_bytes(repo_id: str) -> int:
    """Bytes already present in the HF cache blobs dir for a repo."""
    total = 0
    with contextlib.suppress(OSError):
        for p in _blobs_dir(repo_id).iterdir():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Engine routing
# ---------------------------------------------------------------------------


def route_base_type(ref: ModelRef, info: RepoInfo | None) -> str:
    """Determine which engine should serve this model.

    Rules (in order):
    1. Local path suffix / directory contents.
    2. quant or filename set → llama_cpp (no network needed).
    3. Hub metadata: mlx tag / mlx-community org → mlx_lm; .gguf sibling →
       llama_cpp; .safetensors sibling → mlx_lm.
    4. Offline without info → RoutingError.
    """
    if ref.is_local:
        path = ref.local_path
        assert path is not None
        if path.suffix == ".gguf":
            return "llama_cpp"
        if path.is_dir():
            if any(path.glob("*.gguf")):
                return "llama_cpp"
            if (path / "config.json").exists() or any(path.glob("*.safetensors")):
                return "mlx_lm"
        raise RoutingError(
            f"Cannot determine engine for local path '{path}': "
            "no .gguf files and no config.json/safetensors"
        )

    if ref.quant is not None or ref.filename is not None:
        return "llama_cpp"

    if info is not None:
        # mlx tag or mlx-community org → mlx_lm
        if "mlx" in info.tags or (ref.repo_id or "").startswith("mlx-community/"):
            return "mlx_lm"
        siblings_names = [name for name, _ in info.siblings]
        has_gguf = any(n.endswith(".gguf") for n in siblings_names)
        has_safetensors = any(n.endswith(".safetensors") for n in siblings_names)
        if has_gguf:
            return "llama_cpp"
        if has_safetensors:
            return "mlx_lm"
        raise RoutingError(
            f"Cannot route '{ref.raw}': no GGUF or safetensors files found "
            f"(tags={info.tags!r}, siblings={len(info.siblings)})"
        )

    raise RoutingError(
        f"Cannot route '{ref.raw}' offline — "
        "set an explicit base_type or connect once to populate the routing cache"
    )


# ---------------------------------------------------------------------------
# Routing cache — persists routing decisions across restarts
# ---------------------------------------------------------------------------


class RoutingCache:
    """JSON-backed routing cache at ``<state_dir>/models.json``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                self._data = read_json(path)

    def get(self, raw_ref: str) -> dict | None:
        return self._data.get(raw_ref)

    def put(self, raw_ref: str, *, base_type: str, weight_bytes: int | None) -> None:  # noqa: F811
        self._data[raw_ref] = {
            "base_type": base_type,
            "weight_bytes": weight_bytes,
            "resolved_at": datetime.now(UTC).isoformat(),
        }
        with contextlib.suppress(Exception):
            write_json(self._path, self._data)


def resolve_entry_base_type(entry: ServiceEntry, state_dir: Path) -> str:
    """Resolve ``base_type`` for a ServiceEntry, routing ``"auto"`` entries.

    Returns the existing base_type unchanged for non-auto entries.
    """
    if entry.base_type != "auto":
        return entry.base_type

    model: str = entry.config.get("model")  # type: ignore[assignment]
    ref = parse_model_ref(model)
    cache = RoutingCache(state_dir / "models.json")

    if ref.is_local:
        return route_base_type(ref, None)

    info = fetch_repo_info(ref.repo_id)  # type: ignore[arg-type]
    if info is not None:
        base_type = route_base_type(ref, info)
        kind: Literal["snapshot", "gguf"] = "gguf" if base_type == "llama_cpp" else "snapshot"
        wb = weight_bytes(info, kind, quant=ref.quant, filename=ref.filename)
        cache.put(model, base_type=base_type, weight_bytes=wb)
        return base_type

    # Offline: consult routing cache
    cached = cache.get(model)
    if cached:
        return cached["base_type"]

    raise RoutingError(
        f"Cannot route '{model}' offline — "
        "set an explicit base_type or connect once to populate the routing cache"
    )
