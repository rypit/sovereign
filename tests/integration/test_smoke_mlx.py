"""Real-boot smoke test for the mlx_lm embedded worker.

Mirrors ``test_smoke_llama.py`` for the second native engine: boots a tiny MLX
model through `sovereign serve`, waits for READY via the persisted state
files, hits the live health endpoint, sends one real chat completion, asserts
the telemetry pipeline surfaced generation stats into status.json (the
``stream_generate`` wrapping runs against the real binding here), then tears
down with `sovereign down` and asserts the exact engine PID is gone.

Opt-in only: marked `integration`, excluded from `make test` (see pytest
addopts), and skipped when `mlx-lm` isn't installed (darwin/arm64-only
dependency, provided by `uv sync` there). CI runs it on a macOS arm64 runner
with the HF cache cached between runs (~/.cache/huggingface).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.integration

# Tiny instruct model: the 4-bit MLX snapshot is ~280 MB.
_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
_PORT = 18435
_BOOT_TIMEOUT = 600.0  # first run downloads the model; cached runs boot in seconds
_STACK_YAML = f"""\
version: "1.1"
resources:
  max_unified_memory_gb: 8
  safety_margin_gb: 1
services:
  - name: engine
    base_type: mlx_lm
    health_check:
      type: http
      endpoint: /health
      port: {_PORT}
      timeout_seconds: 300
    config:
      model: {_MODEL}
      max_tokens: 64
"""


def _sovereign(*args: str, cwd) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "from sovereign.cli import app; app()", *args],
        cwd=cwd,
    )


def _wait_for(predicate, *, timeout: float, interval: float = 1.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")


@pytest.fixture
def stack_dir():
    """A short-lived stack directory under the system temp root.

    Not pytest's tmp_path: the stack dir anchors `.sovereign/telemetry.sock`,
    and macOS caps AF_UNIX socket paths at ~104 bytes — tmp_path's deep
    nesting on CI runners exceeds it, which silently degrades `serve` to
    running without live telemetry (and fails the generation-stats asserts).
    """
    d = Path(tempfile.mkdtemp(prefix="sov-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _dump_worker_log(state_dir: Path, name: str = "engine") -> None:
    """Print the worker's log tail so CI failures are diagnosable."""
    log = state_dir / "logs" / f"{name}.log"
    if log.exists():
        tail = log.read_text(errors="replace").splitlines()[-40:]
        print(f"--- worker log tail ({log}) ---")
        for line in tail:
            print(line)
        print("--- end worker log ---")


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None,
    reason="mlx-lm not installed (darwin/arm64-only dependency)",
)
def test_mlx_stack_boots_serves_and_tears_down(stack_dir) -> None:
    stack = stack_dir / "stack.yaml"
    stack.write_text(_STACK_YAML)
    state_dir = stack_dir / ".sovereign"

    serve = _sovereign("serve", "-f", str(stack), cwd=stack_dir)
    try:
        # READY, observed through the persisted coordination files.
        def _ready():
            if serve.poll() is not None:
                pytest.fail(f"sovereign serve exited early with {serve.returncode}")
            try:
                state = json.loads((state_dir / "state.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return None
            return state if state.get("services", {}).get("engine") == "ready" else None

        state = _wait_for(_ready, timeout=_BOOT_TIMEOUT, what="engine READY in state.json")

        # The runtime handle carries the PID (+ create_time identity) for `down`.
        handle = state["runtime"]["engine"]
        assert handle["kind"] == "native"
        engine_pid = handle["pid"]
        assert psutil.pid_exists(engine_pid)
        assert "create_time" in handle

        # The health endpoint answers for real.
        with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/health", timeout=5) as resp:
            assert resp.status == 200

        # One real completion flows through the embedded worker.
        request = urllib.request.Request(
            f"http://127.0.0.1:{_PORT}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": "Say hi in one word."}],
                    "max_tokens": 16,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["choices"][0]["message"]["content"].strip()

        # Generation stats must reach status.json via the telemetry pipeline —
        # the mlx stream_generate wrapping checked against the real binding.
        def _stats():
            try:
                status = json.loads((state_dir / "status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return None
            telemetry = status.get("services", {}).get("engine", {}).get("telemetry", {})
            return telemetry if telemetry.get("generation_tps") else None

        telemetry = _wait_for(_stats, timeout=30, what="generation_tps in status.json")
        assert telemetry["generation_tps"] > 0

        # Cross-process teardown: `down` from a separate process reaps the PID.
        down = _sovereign("down", "--state-dir", str(state_dir), cwd=stack_dir)
        assert down.wait(timeout=60) == 0
        _wait_for(
            lambda: not psutil.pid_exists(engine_pid),
            timeout=30,
            what="engine PID to disappear",
        )

        state = json.loads((state_dir / "state.json").read_text())
        assert state["services"]["engine"] == "stopped"
        assert state["runtime"] == {}
    finally:
        _dump_worker_log(state_dir)
        if serve.poll() is None:
            serve.send_signal(signal.SIGINT)
            try:
                serve.wait(timeout=30)
            except subprocess.TimeoutExpired:
                serve.kill()
                serve.wait()
