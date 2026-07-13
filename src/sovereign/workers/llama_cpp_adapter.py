"""Engine-worker adapter for ``llama-server`` (the native llama.cpp binary).

Per ADR 0007, this adapter supervises ``llama-server`` as a **subprocess**
rather than loading tensors in-process — the opposite of ``mlx_lm_adapter``.
Only :func:`build_server_argv` is meant to be imported at module scope
elsewhere (pure, unit-testable without ``llama-server`` installed).
:func:`run` is the adapter entrypoint the worker calls: it launches the
child, waits for health, then runs a telemetry-translator poll loop that
reads ``llama-server``'s HTTP surface (``/slots``, ``/metrics``) and
re-emits the same UDS NDJSON events (:mod:`sovereign.workers.protocol`) the
rest of Sovereign already consumes. This module never imports
``llama_cpp``/``fastapi``/``uvicorn`` at all — it stays importable on any
platform.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from sovereign.workers.protocol import EventType

if TYPE_CHECKING:
    from sovereign.workers.telemetry import TelemetryClient
    from sovereign.workers.worker_config import WorkerConfig

logger = logging.getLogger("sovereign")

#: engine_kwargs keys mapped onto a llama-server CLI flag with a straight
#: rename. Verified against `llama-server --help` (b-series builds).
_KWARG_FLAGS: dict[str, str] = {
    "gpu_layers": "--n-gpu-layers",
    "threads": "--threads",
    "context_size": "--ctx-size",
    "max_parallel": "--parallel",  # -np — the gap-closing flag (ADR 0006/0007)
    "num_draft_tokens": "--draft-max",
}

#: engine_kwargs keys consumed here rather than passed through verbatim.
_CONSUMED_KEYS = frozenset({*_KWARG_FLAGS, "kv_cache_type"})

#: How often the telemetry translator polls llama-server's HTTP surface.
_POLL_INTERVAL = 0.5
#: Bounded wait for llama-server to report healthy before giving up.
_HEALTH_TIMEOUT = 60.0
_HEALTH_POLL_INTERVAL = 0.2
_HTTP_TIMEOUT = 2.0
#: How long to wait after SIGTERM before escalating to SIGKILL.
_STOP_TIMEOUT = 10.0


def build_server_argv(
    engine_kwargs: dict[str, Any],
    model_path: str,
    draft_model_path: str | None,
    served_model_name: str | None,
    host: str,
    port: int,
) -> list[str]:
    """Translate Sovereign's engine-agnostic kwargs into a ``llama-server``
    CLI argv (sans program name).

    Recognized keys: ``gpu_layers`` -> ``--n-gpu-layers``, ``threads`` ->
    ``--threads``, ``context_size`` -> ``--ctx-size``, ``max_parallel`` ->
    ``--parallel`` (``-np``, the multi-slot continuous-batching flag this
    adapter closes the gap with — see ADR 0007), ``num_draft_tokens`` ->
    ``--draft-max`` (only emitted when a draft model is set — llama-server
    rejects draft flags without ``--model-draft``), ``kv_cache_type`` ->
    ``--cache-type-k``/``--cache-type-v``. ``draft_model_path`` ->
    ``--model-draft``, ``served_model_name`` -> ``--alias``, ``model_path``
    -> ``-m``, ``host``/``port`` -> ``--host``/``--port``.

    Unknown ``engine_kwargs`` entries pass through the same way as the mlx
    adapter's ``build_server_argv``: kebab-cased ``--flag value`` (bare flag
    for ``True``, dropped for ``False``) — a flag the installed
    ``llama-server`` doesn't know is a loud failure at worker boot, not
    silent misconfiguration.

    The API key is deliberately NOT built here — ``run()`` appends
    ``--api-key`` itself from the ``SOVEREIGN_API_KEY`` environment variable
    so it never appears in a dumped ``WorkerConfig`` JSON, only in argv
    passed directly to the child (still world-readable via ``ps``, same
    trade-off as before; the manager already keeps it off the JSON).
    """
    argv = ["-m", model_path, "--host", host, "--port", str(port)]
    # /metrics and /slots are disabled by default in llama-server; the
    # telemetry translator (poll_once) depends on both, so always enable them.
    # Without these the endpoints 404 and generation_tps/prefill never surface.
    argv += ["--metrics", "--slots"]
    if served_model_name:
        argv += ["--alias", served_model_name]
    if draft_model_path:
        argv += ["--model-draft", draft_model_path]

    for key, flag in _KWARG_FLAGS.items():
        if key not in engine_kwargs:
            continue
        if key == "num_draft_tokens" and not draft_model_path:
            continue
        argv += [flag, str(engine_kwargs[key])]

    kv_cache_type = engine_kwargs.get("kv_cache_type")
    if kv_cache_type is not None:
        argv += ["--cache-type-k", str(kv_cache_type), "--cache-type-v", str(kv_cache_type)]

    for key, value in engine_kwargs.items():
        if key in _CONSUMED_KEYS:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]

    return argv


# --- telemetry translator: llama-server HTTP surface -> UDS NDJSON events ---


class _MetricsDelta:
    """Tracks the previous ``/metrics`` cumulative counters so the poll loop
    can derive instantaneous prompt/generation tps from deltas — llama-server's
    Prometheus surface is cumulative totals, not per-poll rates."""

    def __init__(self) -> None:
        self.prev: dict[str, float] = {}
        self.prev_ts: float | None = None


def _fetch_json(url: str) -> Any | None:
    import json

    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _fetch_text(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse a minimal subset of the Prometheus text exposition format:
    ``name value`` pairs, one per line, ignoring ``#`` comments/blank lines."""
    counters: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, raw_value = parts
        try:
            counters[name] = float(raw_value)
        except ValueError:
            continue
    return counters


def _slot_is_idle(slot: dict[str, Any]) -> bool:
    state = slot.get("state")
    return state in (0, "idle", None, "SLOT_STATE_IDLE")


def _emit_slot_prefill(slot: dict[str, Any], telemetry: TelemetryClient) -> None:
    """Emit ``PREFILL_PROGRESS`` for one active ``/slots`` entry.

    llama-server's per-slot JSON exposes the task id, tokens processed so
    far, and (when known) the total prompt token count; ``total`` is
    ``None`` — an honest indeterminate — when llama-server doesn't report
    one, same parity as the prior in-process adapter.
    """
    request_id = str(slot.get("id_task", slot.get("id", "")))
    processed_raw = slot.get("n_past", slot.get("n_prompt_tokens_processed", 0))
    processed = int(processed_raw) if isinstance(processed_raw, int | float) else 0
    total_raw = slot.get("n_prompt_tokens")
    total = int(total_raw) if isinstance(total_raw, int | float) and total_raw > 0 else None
    telemetry.emit(
        EventType.PREFILL_PROGRESS,
        {"request_id": request_id, "processed": processed, "total": total},
    )


def _emit_generation_stats_from_deltas(
    counters: dict[str, float], delta: _MetricsDelta, dt: float, telemetry: TelemetryClient
) -> None:
    """Derive prompt/generation tps from deltas of llama-server's cumulative
    ``/metrics`` counters and emit one aggregate ``GENERATION_STATS`` event.

    llama-server's own streaming ``timings`` block (per-request tps) isn't
    reachable from a side-channel poller without proxying the request path
    (explicitly out of scope per ADR 0007/plan), so this is a
    worker-aggregate signal keyed ``"aggregate"`` rather than a genuine
    per-``request_id`` figure — the dashboard's tps sparkline only ever
    consumed the latest value, so this preserves that contract.
    """
    predicted_total = counters.get("llamacpp:tokens_predicted_total")
    if predicted_total is None or dt <= 0:
        return
    prev_predicted = delta.prev.get("llamacpp:tokens_predicted_total", predicted_total)
    delta_tokens = predicted_total - prev_predicted
    if delta_tokens <= 0:
        return
    generation_tps = delta_tokens / dt

    prompt_tps: float | None = None
    prompt_total = counters.get("llamacpp:prompt_tokens_total")
    if prompt_total is not None:
        prev_prompt = delta.prev.get("llamacpp:prompt_tokens_total", prompt_total)
        delta_prompt = prompt_total - prev_prompt
        if delta_prompt > 0:
            prompt_tps = delta_prompt / dt

    telemetry.emit(
        EventType.GENERATION_STATS,
        {
            "request_id": "aggregate",
            "prompt_tokens": int(prompt_total) if prompt_total is not None else None,
            "completion_tokens": int(delta_tokens),
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
        },
    )


def poll_once(base_url: str, telemetry: TelemetryClient, delta: _MetricsDelta) -> None:
    """One translation pass: ``/slots`` -> ``PREFILL_PROGRESS``,
    ``/metrics`` deltas -> ``GENERATION_STATS``. Never raises — a malformed
    or unreachable response is a no-op poll, not a crash, since telemetry is
    best-effort observability (mirrors :class:`TelemetryClient`'s own
    contract)."""
    slots = _fetch_json(f"{base_url}/slots")
    if isinstance(slots, list):
        for slot in slots:
            if isinstance(slot, dict) and not _slot_is_idle(slot):
                _emit_slot_prefill(slot, telemetry)

    text = _fetch_text(f"{base_url}/metrics")
    if text is not None:
        counters = _parse_prometheus(text)
        now = time.monotonic()
        if delta.prev_ts is not None:
            dt = now - delta.prev_ts
            _emit_generation_stats_from_deltas(counters, delta, dt, telemetry)
        delta.prev = counters
        delta.prev_ts = now


def telemetry_loop(
    base_url: str,
    telemetry: TelemetryClient,
    stop_event: threading.Event,
    process: subprocess.Popen[Any],
    interval: float = _POLL_INTERVAL,
) -> bool:
    """Poll ``base_url`` until ``stop_event`` is set or ``process`` exits on
    its own. Returns ``True`` if the child exited unexpectedly (a crash the
    caller should surface), ``False`` on a clean/requested shutdown."""
    delta = _MetricsDelta()
    while not stop_event.is_set():
        if process.poll() is not None:
            return True
        try:
            poll_once(base_url, telemetry, delta)
        except Exception:  # noqa: BLE001 - telemetry must never crash the worker
            logger.debug("llama_cpp telemetry poll failed", exc_info=True)
        if stop_event.wait(interval):
            return False
    return False


def _wait_for_health(
    url: str,
    process: subprocess.Popen[Any],
    stop_event: threading.Event,
    timeout: float | None = None,
    interval: float | None = None,
) -> bool:
    # Resolved at call time (not bound as a default) so tests can monkeypatch
    # the module-level constants to shrink the wait.
    if timeout is None:
        timeout = _HEALTH_TIMEOUT
    if interval is None:
        interval = _HEALTH_POLL_INTERVAL
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop_event.is_set() or process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if stop_event.wait(interval):
            return False
    return False


def run(cfg: WorkerConfig, telemetry: TelemetryClient, controller: Any) -> None:
    """Launch ``llama-server`` as a child process and serve until shutdown.

    The worker process (this one) never loads tensors — ``llama-server``
    does, in the child. ``controller.shutdown_callback`` terminates that
    child on SIGTERM (escalating to SIGKILL after a bounded wait); an
    unexpected child exit raises so ``engine_worker`` surfaces a
    ``state_change: crashed`` event, same as any other adapter failure.
    """
    api_key = os.environ.get("SOVEREIGN_API_KEY")
    argv = [
        "llama-server",
        *build_server_argv(
            cfg.engine_kwargs,
            cfg.model_path,
            cfg.draft_model_path,
            cfg.served_model_name,
            cfg.host,
            cfg.port,
        ),
    ]
    if api_key:
        argv += ["--api-key", api_key]

    logger.info("llama_cpp worker %s: launching llama-server", cfg.service)
    process = subprocess.Popen(argv)  # noqa: S603 - argv is constructed, not shell

    stop_event = threading.Event()

    def _shutdown() -> None:
        stop_event.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    controller.shutdown_callback = _shutdown

    base_url = f"http://{cfg.host}:{cfg.port}"
    health_url = f"{base_url}{cfg.health_path}"
    if not _wait_for_health(health_url, process, stop_event):
        if process.poll() is None:
            _shutdown()
        raise RuntimeError(f"llama-server for '{cfg.service}' failed to become healthy")

    telemetry.emit(EventType.STATE_CHANGE, {"state": "serving"})

    crashed = telemetry_loop(base_url, telemetry, stop_event, process)
    if crashed:
        raise RuntimeError(
            f"llama-server for '{cfg.service}' exited unexpectedly "
            f"(code={process.returncode})"
        )
