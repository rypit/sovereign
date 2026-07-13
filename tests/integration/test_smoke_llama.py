"""Real-boot smoke test: Sovereign demonstrably boots a model (P2.1).

The hermetic suite never executes anything real; this test does. It runs
`sovereign serve` as a subprocess against a tiny GGUF, waits for READY via the
persisted state files (the same cross-process coordination `sovereign status`
uses), hits the live health endpoint, then stops the stack with
`sovereign down` and asserts the exact engine PID is gone. Along the way it
sends real chat completions through the worker and asserts the telemetry
pipeline surfaced generation stats into status.json.

Per ADR 0007 the llama_cpp worker runs the native `llama-server` binary as a
child and translates its HTTP telemetry surface (`/slots`, `/metrics`) into
Sovereign's events — this test is the one place that whole path (the argv
`build_server_argv` produces AND the translator's real JSON parsing) is
exercised against the real binary, not fakes. It also boots with
`max_parallel: 2` and fires two concurrent completions, so `-np` continuous
batching (the gap ADR 0007 closes) is exercised for real, not just asserted in
mapping tests.

Opt-in only: marked `integration`, excluded from `make test` (see pytest
addopts), and skipped when the `llama-server` binary isn't on PATH. CI's macOS
arm64 smoke job installs it via `brew install llama.cpp` (the Brewfile next to
the manager) and caches the HF model between runs (~/.cache/huggingface).
"""

from __future__ import annotations

import concurrent.futures
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
      max_parallel: 2
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
    shutil.which("llama-server") is None,
    reason="llama-server binary not on PATH (provision via `brew install llama.cpp`)",
)
def test_llama_stack_boots_serves_and_tears_down(stack_dir) -> None:
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
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_PORT}/health", timeout=5
        ) as resp:
            assert resp.status == 200

        # Two concurrent completions flow through the worker: the OpenAI
        # surface works end-to-end (not just the health route), and with
        # `max_parallel: 2` llama-server's `-np` continuous batching serves
        # them over one context/weights — the gap ADR 0007 closes, exercised
        # for real rather than only asserted in build_server_argv tests.
        def _complete(prompt: str) -> dict:
            request = urllib.request.Request(
                f"http://127.0.0.1:{_PORT}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": _MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 16,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as resp:
                assert resp.status == 200
                return json.loads(resp.read())

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            bodies = list(
                pool.map(_complete, ["Say hi in one word.", "Name one color."])
            )
        for body in bodies:
            assert body["choices"][0]["message"]["content"].strip()

        # The completions must surface generation stats through the telemetry
        # pipeline (llama-server /slots + /metrics -> adapter translator ->
        # UDS hub -> cache -> status.json). This is the one place the real
        # translator's HTTP parsing is checked against the live binary — a
        # broken flag argv or a drifted /metrics schema would show up here as
        # a permanently-None generation_tps.
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
