"""The HuggingFace model pipeline: metadata, estimation, and download.

Pure library — no typer, no manager imports. Provides:
- ModelRef parsing from raw model strings (local path, repo id, repo:quant, repo/file.gguf)
- Repo metadata fetch, memoised per process (never caches offline/transient failures)
- GGUF file selection with shard grouping and quant disambiguation
- Weight byte estimation from metadata or disk
- Download that forwards huggingface_hub's own tqdm-rendered progress to a callback
- ``RoutingCache``: persisted engine-routing decisions for offline restarts

The engine-routing *decision* lives in :mod:`sovereign.services.inference.routing`
(each engine claims a ref via ``claim_route``); this module supplies the metadata
it reads. Engines call this module through a single seam (``inference.base``
imports it as ``hf_models``), so tests patch
``sovereign.services.inference.hf.<fn>`` and every caller sees it.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx  # a hard dependency of huggingface_hub>=1.0 (its transport layer)
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from tqdm.auto import tqdm as _BaseTqdm

# The model-resolution exceptions live in ``core`` (the cross-layer contract the
# orchestrator/planner/CLI catch); this pipeline raises them.
from sovereign.core.errors import (
    ModelAccessError,
    ModelDownloadError,
    ModelNotFoundError,
    ModelResolutionError,
)
from sovereign.utils.state import read_json, write_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local-path helpers
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
    except (OSError, httpx.HTTPError, HfHubHTTPError) as exc:
        # Transient: connection error, timeout, 5xx, offline — do not cache.
        # Deliberately narrow so genuine bugs surface instead of reading as "offline".
        log.debug("metadata fetch for %s failed (%s); treating as offline", repo_id, exc)
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
    assert ref.repo_id is not None  # non-local refs always carry a repo id

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


EstimateSource = Literal["local", "cached", "hub", "unknown"]


def estimate_model_bytes_with_source(
    ref: ModelRef, kind: Literal["snapshot", "gguf"]
) -> tuple[int | None, EstimateSource]:
    """Estimate model weight size in bytes plus where the number came from.

    Fallback chain: local disk → cached HF path on disk → HF repo metadata → unknown.
    ``sovereign plan`` surfaces the source so a dry-run says whether a model is already
    on disk or would be fetched from the hub.
    """
    # 1. Local
    if ref.is_local:
        n = local_model_bytes(ref.raw)
        return (n or None), "local"

    # 2. Cached on disk
    cached = cached_model_path(ref, kind)
    if cached is not None:
        if cached.is_dir():
            total = sum(f.stat().st_size for f in cached.rglob("*") if f.is_file())
            return (total or None), "cached"
        if cached.is_file():
            return (cached.stat().st_size or None), "cached"

    # 3. Repo metadata
    if ref.repo_id is not None:
        info = fetch_repo_info(ref.repo_id)
        if info is not None:
            return weight_bytes(info, kind, quant=ref.quant, filename=ref.filename), "hub"

    # 4. Unknown
    return None, "unknown"


def estimate_model_bytes(ref: ModelRef, kind: Literal["snapshot", "gguf"]) -> int | None:
    """Estimate model weight size in bytes via a fallback chain.

    Order: local disk → cached HF path on disk → HF repo metadata → None.
    """
    return estimate_model_bytes_with_source(ref, kind)[0]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # cursor moves tqdm emits for multi-bar layout


def _clean_bar(text: str) -> str:
    """Strip tqdm's carriage returns and cursor-movement escapes down to a bare line."""
    return _ANSI_RE.sub("", text).replace("\r", "").strip()


class _ActivityFeed:
    """Forwards huggingface_hub's own tqdm output to a callback, all live bars at once.

    huggingface_hub downloads snapshot files concurrently (``max_workers`` threads) but
    renders progress as a few *aggregate* bars on the main thread — a file counter
    ("Fetching N files"), the summed network transfer ("Downloading bytes") and the summed
    reconstruction — folding the per-file bars into those internally. Each bar renders into
    its own sink; the feed keeps the latest line from every live bar and forwards them as a
    list (one bar per line), so concurrently-active bars all show at once instead of
    overwriting each other. Bar updates arrive from worker threads, hence the lock.
    """

    def __init__(self, progress: Callable[[list[str]], None]) -> None:
        self._progress = progress
        self._bars: dict[int, str] = {}  # bar id -> latest line, kept in first-seen order
        self._lock = threading.Lock()

    def tqdm_class(self) -> type[_BaseTqdm]:
        """A ``tqdm`` subclass whose every bar renders into this feed.

        ``file`` and ``disable`` are forced, not defaulted: the caller explicitly asked
        for progress, so bars must render even off-TTY (the daemon case) and into the feed
        rather than the console.
        """
        feed = self

        class _FeedTqdm(_BaseTqdm):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["file"] = _BarSink(feed, id(self))
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

            def close(self) -> None:
                super().close()  # emits the bar's final line, then it leaves the feed
                feed._drop(id(self))

        return _FeedTqdm

    def _write(self, bar_id: int, text: str) -> None:
        line = _clean_bar(text)
        with self._lock:
            if line:
                self._bars[bar_id] = line
            lines = list(self._bars.values())
        if lines:
            self._progress(lines)

    def _drop(self, bar_id: int) -> None:
        with self._lock:
            self._bars.pop(bar_id, None)


class _BarSink(io.TextIOBase):
    """File-like sink tqdm renders a single bar into; forwards each write to the feed."""

    def __init__(self, feed: _ActivityFeed, bar_id: int) -> None:
        self._feed = feed
        self._bar_id = bar_id

    def write(self, s: str) -> int:
        self._feed._write(self._bar_id, s)
        return len(s)

    def flush(self) -> None:
        pass


def download_model(
    ref: ModelRef,
    kind: Literal["snapshot", "gguf"],
    *,
    progress: Callable[[list[str]], None] | None = None,
) -> Path:
    """Download a model to the HF cache and return its local path.

    Local refs return immediately. When ``progress`` is given, huggingface_hub's
    own tqdm-rendered progress lines are forwarded to it as they render. Raises
    ModelDownloadError on download failures (including disk exhaustion, surfaced
    by huggingface_hub itself); ModelAccessError / ModelNotFoundError propagate.
    """
    if ref.is_local:
        assert ref.local_path is not None
        return ref.local_path

    assert ref.repo_id is not None

    tqdm_class = _ActivityFeed(progress).tqdm_class() if progress is not None else None

    try:
        if kind == "snapshot":
            path = snapshot_download(
                ref.repo_id, ignore_patterns=_SNAPSHOT_IGNORE, tqdm_class=tqdm_class
            )
            return Path(path)

        # gguf: metadata needed to select shards
        info = fetch_repo_info(ref.repo_id)  # may raise ModelAccessError / ModelNotFoundError
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
