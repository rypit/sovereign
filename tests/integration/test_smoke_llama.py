"""Real-boot smoke test: Sovereign demonstrably boots a model (P2.1).

The hermetic suite never executes anything real; this test does. It runs
`sovereign serve` as a subprocess against a tiny GGUF, waits for READY via the
persisted state files (the same cross-process coordination `sovereign status`
uses), hits the live health endpoint, then stops the stack with
`sovereign down` and asserts the exact engine PID is gone.

Opt-in only: marked `integration`, excluded from `make test` (see pytest
addopts), and skipped when `llama-server` isn't installed. CI runs it on a
macOS runner with llama.cpp installed via brew and the HF cache cached
between runs (~/.cache/huggingface).
"""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import psutil
import pytest

pytestmark = pytest.mark.integration

# Tiny instruct model: the Q2_K single-file GGUF is ~340 MB.
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q2_K"
_PORT = 18434
_BOOT_TIMEOUT = 600.0  # first run downloads the model; cached runs boot in seconds
_STACK_YAML = f"""\
version: "1.1"
resources:
  max_unified_memory_gb: 8
  safety_margin_gb: 1
services:
  - name: engine
    base_type: llama_cpp
    health_check:
      type: http
      endpoint: /health
      port: {_PORT}
      timeout_seconds: 300
    config:
      model: {_MODEL}
      context_size: 2048
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


@pytest.mark.skipif(
    shutil.which("llama-server") is None, reason="llama-server not installed (brew llama.cpp)"
)
def test_llama_stack_boots_serves_and_tears_down(tmp_path) -> None:
    stack = tmp_path / "stack.yaml"
    stack.write_text(_STACK_YAML)
    state_dir = tmp_path / ".sovereign"

    serve = _sovereign("serve", "-f", str(stack), cwd=tmp_path)
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
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_PORT}/health", timeout=5
        ) as resp:
            assert resp.status == 200

        # Cross-process teardown: `down` from a separate process reaps the PID.
        down = _sovereign("down", "--state-dir", str(state_dir), cwd=tmp_path)
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
        if serve.poll() is None:
            serve.send_signal(signal.SIGINT)
            try:
                serve.wait(timeout=30)
            except subprocess.TimeoutExpired:
                serve.kill()
                serve.wait()
