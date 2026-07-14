"""Config schema for the ``omlx`` native inference engine.

Pydantic-only (§2.3). Parses the ``config:`` block of an ``omlx`` service
entry. The knobs map onto ``omlx serve`` flags in the worker adapter
(``workers/omlx_adapter.build_server_argv``):

    max_concurrent_requests -> --max-concurrent-requests
    memory_guard_gb         -> --memory-guard-gb
    paged_ssd_cache(_dir)   -> --paged-ssd-cache-dir
    paged_ssd_cache_max_gb  -> --paged-ssd-cache-max-size
    hot_cache_gb            -> --hot-cache-max-size
"""

from __future__ import annotations

from pydantic import Field

from sovereign.core.base_config import NativeEngineConfig


class OmlxConfig(NativeEngineConfig):
    """Settings for a single ``omlx serve`` instance.

    Shared fields (``model``, ``host``, ``served_model_name``, ``log_dir``)
    come from :class:`NativeEngineConfig`. ``model`` is a local MLX directory
    or a HuggingFace repo id (MLX/safetensors snapshot, same artifact kind as
    ``mlx_lm``). ``draft_model`` is inherited but rejected pre-flight — omlx
    has no speculative-decoding surface (ADR 0006: surface the gap loudly).
    """

    #: Concurrent requests omlx's continuous-batching scheduler admits
    #: (``--max-concurrent-requests``; omlx's own default is 8).
    max_concurrent_requests: int | None = Field(default=None, gt=0)

    #: Ceiling for omlx's internal memory enforcer in decimal GB
    #: (``--memory-guard-gb``). When unset, Sovereign pins it from its own
    #: admission estimate so omlx's enforcer and Sovereign's refuse-to-boot
    #: budget (§11.5) agree instead of fighting; omlx's stock default
    #: ("system RAM − 8 GB") would let one service eat the whole budget.
    memory_guard_gb: float | None = Field(default=None, gt=0)

    #: omlx's headline feature — paged KV/prefix cache with an SSD cold tier.
    #: On by default (the reason to pick this engine over ``mlx_lm``); the
    #: cache directory defaults to ``<state_dir>/omlx/<service>/kv-cache``.
    paged_ssd_cache: bool = True
    paged_ssd_cache_dir: str | None = None
    #: Cap for the SSD cold tier in decimal GB (``--paged-ssd-cache-max-size``;
    #: omlx's own default is 100 GB). Disk, not unified memory — no admission term.
    paged_ssd_cache_max_gb: int | None = Field(default=None, gt=0)

    #: In-memory hot tier for KV cache blocks in decimal GB
    #: (``--hot-cache-max-size``; omlx's own default is 0 = disabled). Lives in
    #: unified memory, so it counts toward the admission estimate (§7).
    hot_cache_gb: int | None = Field(default=None, gt=0)

    #: Optional bearer key omlx requires on requests (``--api-key``).
    api_key: str | None = None
