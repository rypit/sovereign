"""In-house performance prober (§6b) — attach mode.

Streams chat-completion requests at a live, already-running stack's resolved
endpoint (read-only: this module never boots or stops anything) and reports
TTFT, output tok/s, and end-to-end latency with mean + spread over
``spec.trials`` repetitions (§6b measurement discipline: "3+ trials/cell").

``httpx`` is the ``bench`` optional dependency, imported lazily so
``sovereign bench run`` on a suite-only (no perf) spec never needs it
installed.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sovereign.bench.spec import BenchSpec, Thresholds
from sovereign.core.state import read_json
from sovereign.core.units import fmt_size

if TYPE_CHECKING:
    from sovereign.bench.runner import CellExecutor, Job

_INSTALL_HINT = (
    "httpx is not installed. Install the optional bench extra: "
    "`uv sync --extra bench` (or `pip install sovereign[bench]`)."
)

_DEFAULT_PROMPT = "Write a haiku about the sea."


class PerfError(Exception):
    """Raised when attach-mode perf probing can't reach a live, resolved stack."""


def _import_httpx():
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return httpx


async def _stream_once(
    client: Any, url: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """Stream one chat-completion request; return TTFT/duration/token metrics."""
    start = time.perf_counter()
    ttft_s: float | None = None
    output_tokens = 0
    usage: dict[str, Any] | None = None

    async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                output_tokens += 1
            if chunk.get("usage"):
                usage = chunk["usage"]

    total_s = time.perf_counter() - start
    if usage and "completion_tokens" in usage:
        output_tokens = usage["completion_tokens"]
    tok_s = output_tokens / total_s if total_s > 0 and output_tokens else None
    return {
        "ttft_s": ttft_s,
        "total_s": total_s,
        "output_tokens": output_tokens,
        "tok_s": tok_s,
    }


async def probe_endpoint(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    prompt: str = _DEFAULT_PROMPT,
    trials: int = 3,
    max_tokens: int = 64,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Run ``trials`` sequential streaming requests against an OpenAI-compatible
    ``/chat/completions`` endpoint and return one raw sample per trial."""
    httpx = _import_httpx()
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    samples: list[dict[str, Any]] = []
    async with httpx.AsyncClient(headers=headers) as client:
        for _ in range(trials):
            samples.append(await _stream_once(client, url, payload, timeout_s))
    return samples


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean + spread (population stdev) per metric across trials."""

    def _stats(key: str) -> dict[str, float | None]:
        values = [s[key] for s in samples if s.get(key) is not None]
        if not values:
            return {"mean": None, "stdev": None}
        return {
            "mean": statistics.mean(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    ttft = _stats("ttft_s")
    ttft_ms = {
        "mean": ttft["mean"] * 1000 if ttft["mean"] is not None else None,
        "stdev": ttft["stdev"] * 1000 if ttft["stdev"] is not None else None,
    }
    return {
        "trials": len(samples),
        "ttft_ms": ttft_ms,
        "tok_s": _stats("tok_s"),
        "total_s": _stats("total_s"),
        "samples": samples,
    }


def _passes_thresholds(
    result: dict[str, Any], thresholds: Thresholds, available_bytes: int | None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    tok_s_mean = result["tok_s"]["mean"]
    if thresholds.min_tok_s is not None:
        if tok_s_mean is None or tok_s_mean < thresholds.min_tok_s:
            reasons.append(f"tok_s {tok_s_mean} < min_tok_s {thresholds.min_tok_s}")
    ttft_mean = result["ttft_ms"]["mean"]
    if thresholds.max_ttft_ms is not None:
        if ttft_mean is None or ttft_mean > thresholds.max_ttft_ms:
            reasons.append(f"ttft_ms {ttft_mean} > max_ttft_ms {thresholds.max_ttft_ms}")
    if thresholds.min_headroom_bytes is not None:
        if available_bytes is None or available_bytes < thresholds.min_headroom_bytes:
            available_display = (
                fmt_size(available_bytes) if available_bytes is not None else None
            )
            reasons.append(
                f"available {available_display} < "
                f"min_headroom {fmt_size(thresholds.min_headroom_bytes)}"
            )
    return (not reasons, reasons)


def _primary_engine(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The first service exposing a resolved endpoint+model — the engine under
    test for this stack variant (§7b: one primary engine per variant file)."""
    for svc in manifest.get("services", []):
        endpoint = svc.get("endpoint")
        if endpoint and endpoint.get("model"):
            return svc
    return None


async def run_perf_attach_cell(job: Job, spec: BenchSpec, state_dir: str | Path) -> dict[str, Any]:
    """Attach-mode perf measurement for one cell: read the live manifest
    (read-only — never boots/stops anything), probe its primary engine."""
    state_dir = Path(state_dir)
    manifest_path = state_dir / "manifest.json"
    if not manifest_path.exists():
        raise PerfError(
            f"no live stack found at {state_dir} (attach mode needs a running `sovereign up`)"
        )
    manifest = read_json(manifest_path)

    engine = _primary_engine(manifest)
    if engine is None:
        raise PerfError(
            f"no native engine with a resolved endpoint in the live stack for '{job.stack}'"
        )

    endpoint = engine["endpoint"]
    base_url = f"{endpoint['scheme']}://{endpoint['host']}:{endpoint['port']}/v1"
    model = endpoint["model"]

    samples = await probe_endpoint(
        base_url,
        model,
        trials=spec.trials,
        timeout_s=float(spec.budgets.task_timeout_s),
        max_tokens=spec.budgets.max_tokens or 64,
    )
    result = summarize(samples)
    result["engine"] = engine["name"]
    result["co_resident"] = engine.get("co_resident", [])
    result["variant_hash"] = manifest.get("variant_hash")

    available_bytes = manifest.get("memory_budget", {}).get("available_bytes")
    passed, reasons = _passes_thresholds(result, spec.thresholds, available_bytes)
    result["gate_passed"] = passed
    result["gate_reasons"] = reasons
    return result


def make_perf_attach_executor(spec: BenchSpec, state_dir: str | Path) -> CellExecutor:
    """Build a `CellExecutor` (§B1 seam) that runs the attach-mode perf probe."""

    def executor(job: Job) -> dict[str, Any]:
        return asyncio.run(run_perf_attach_cell(job, spec, state_dir))

    return executor
