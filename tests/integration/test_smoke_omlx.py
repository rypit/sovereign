"""Real-boot smoke test for the omlx engine.

Mirrors ``test_smoke_llama.py``: runs `sovereign serve` as a subprocess
against a tiny MLX model with `base_type: omlx`, waits for READY via the
persisted state files, hits the live `/v1/models` health endpoint, sends a
real chat completion through omlx's OpenAI surface, then stops the stack
with `sovereign down` and asserts the exact engine PID is gone. This is the
one place the adapter's argv (`build_server_argv`) and single-model symlink
layout (`prepare_model_dir`) are exercised against the real `omlx` binary.

No generation-stats assertion: omlx exposes no telemetry scrape surface
(the ADR 0006 gap the manager announces at pre-flight), so status.json only
ever carries heartbeat/memory telemetry for this engine.

Opt-in only: marked `integration`, excluded from `make test` (see pytest
addopts). The skip guard below covers unprovisioned *local* machines only —
CI's macOS arm64 smoke job installs the binary from the manager's Brewfile
(`brew bundle --file src/sovereign/services/inference/omlx/Brewfile`, which
taps jundot/omlx and builds from HEAD with the custom kernel; see
.github/workflows/ci.yml), so this test always executes there rather than
silently skipping. The HF model is cached between runs
(~/.cache/huggingface).
"""

from __future__ import annotations

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

# Tiny MLX instruct model (~350 MB, 4-bit).
_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
# The name omlx serves it under: api_model_name() flattens the repo id to
# omlx's directory-derived id convention (nested segments join with `--`).
_SERVED = _MODEL.replace("/", "--")
_PORT = 18435
_BOOT_TIMEOUT = 600.0  # first run downloads the model; cached runs boot in seconds
_STACK_YAML = f"""\
version: "1.1"
resources:
  max_unified_memory_gb: 8
  safety_margin_gb: 1
services:
  - name: engine
    base_type: omlx
    health_check:
      type: http
      endpoint: /v1/models
      port: {_PORT}
      timeout_seconds: 300
    config:
      model: {_MODEL}
      max_concurrent_requests: 2
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
    """A short-lived stack directory under the system temp root (not pytest's
    tmp_path — macOS caps AF_UNIX socket paths at ~104 bytes, and the stack
    dir anchors `.sovereign/telemetry.sock`)."""
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
    shutil.which("omlx") is None,
    reason=(
        "omlx binary not on PATH (provision via `sovereign provision` or "
        "`brew bundle --file src/sovereign/services/inference/omlx/Brewfile`)"
    ),
)
def test_omlx_stack_boots_serves_and_tears_down(stack_dir) -> None:
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

        # The single-model symlink layout the adapter prepared points at the
        # resolved snapshot inside the shared HF cache — flat `org--name`, so
        # omlx discovers exactly one model whose id equals _SERVED.
        link = state_dir / "omlx" / "engine" / "models" / _SERVED
        assert link.is_symlink()
        assert (link / "config.json").exists()

        # The health endpoint (omlx's /v1/models) answers for real and lists
        # the model under the name clients must send.
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_PORT}/v1/models", timeout=5
        ) as resp:
            assert resp.status == 200
            listed = json.loads(resp.read())
        assert any(m.get("id") == _SERVED for m in listed.get("data", []))

        # A completion flows through omlx's OpenAI surface end-to-end.
        request = urllib.request.Request(
            f"http://127.0.0.1:{_PORT}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": _SERVED,
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
