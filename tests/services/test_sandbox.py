"""Per-invoke Docker sandbox helpers (ADR 0009)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sovereign.core.base_config import SandboxConfig
from sovereign.services.docker import sandbox as sbx


class FakeDocker:
    """Records every docker argv; scripted per-subcommand results."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.results: dict[str, subprocess.CompletedProcess[str]] = {}
        self.raises: dict[str, BaseException] = {}

    def script(self, prefix: str, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.results[prefix] = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, args, *, binary="docker", timeout=None, check=False):
        self.calls.append(list(args))
        key = " ".join(args)
        for prefix, exc in self.raises.items():
            if key.startswith(prefix):
                raise exc
        for prefix, result in self.results.items():
            if key.startswith(prefix):
                return result
        return subprocess.CompletedProcess(args, 0, "", "")

    def argv(self, prefix: str) -> list[list[str]]:
        return [c for c in self.calls if " ".join(c).startswith(prefix)]


@pytest.fixture
def docker(monkeypatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(sbx, "run_docker", fake)
    return fake


# --- SandboxConfig schema ---
def test_sandbox_config_defaults() -> None:
    cfg = SandboxConfig()
    assert cfg.enabled and cfg.image is None and cfg.network == "engine_only"
    assert cfg.memory_bytes is None and cfg.mounts == []
    assert cfg.proxy_image == "alpine/socat" and cfg.binary == "docker"


def test_sandbox_config_memory_gb_alias() -> None:
    assert SandboxConfig.model_validate({"memory_gb": 2}).memory_bytes == 2 * 10**9


def test_sandbox_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        SandboxConfig.model_validate({"imgae": "typo:latest"})


def test_sandbox_config_rejects_unknown_network() -> None:
    with pytest.raises(ValueError):
        SandboxConfig.model_validate({"network": "bridge"})


# --- rewrite_base_url ---
def test_rewrite_base_url_preserves_path() -> None:
    port, url = sbx.rewrite_base_url("http://127.0.0.1:8080/v1")
    assert port == 8080
    assert url == f"http://{sbx.PROXY_ALIAS}:{sbx.PROXY_PORT}/v1"


def test_rewrite_base_url_rejects_remote_host() -> None:
    with pytest.raises(sbx.SandboxError):
        sbx.rewrite_base_url("http://example.com:8080/v1")


def test_rewrite_base_url_requires_port() -> None:
    with pytest.raises(sbx.SandboxError):
        sbx.rewrite_base_url("http://localhost/v1")


# --- network + proxy lifecycle ---
def test_create_internal_network_argv(docker: FakeDocker) -> None:
    sbx.create_internal_network("sov-sbx-h-1", harness_name="h")
    (argv,) = docker.argv("network create")
    assert "--internal" in argv
    assert f"{sbx.SANDBOX_LABEL}=1" in argv
    assert f"{sbx.HARNESS_LABEL}=h" in argv
    assert argv[-1] == "sov-sbx-h-1"


def test_create_internal_network_failure_raises(docker: FakeDocker) -> None:
    docker.script("network create", returncode=1, stderr="boom")
    with pytest.raises(sbx.SandboxError, match="boom"):
        sbx.create_internal_network("sid", harness_name="h")


def test_start_egress_proxies_argv_and_alias(docker: FakeDocker) -> None:
    docker.script("inspect", stdout="true\n")
    sbx.start_egress_proxies("sid", {sbx.PROXY_ALIAS: 11434}, harness_name="h")
    (run_argv,) = docker.argv("run")
    assert "host.docker.internal:host-gateway" in run_argv
    assert f"TCP-LISTEN:{sbx.PROXY_PORT},fork,reuseaddr" in run_argv
    assert "TCP:host.docker.internal:11434" in run_argv
    (connect_argv,) = docker.argv("network connect")
    assert connect_argv == [
        "network", "connect", "--alias", sbx.PROXY_ALIAS, "sid", f"sid-{sbx.PROXY_ALIAS}",
    ]


def test_start_egress_proxies_one_sidecar_per_route(docker: FakeDocker) -> None:
    docker.script("inspect", stdout="true\n")
    sbx.start_egress_proxies(
        "sid", {sbx.PROXY_ALIAS: 11434, "sovereign-searxng": 8888}, harness_name="h"
    )
    run_argvs = docker.argv("run")
    assert len(run_argvs) == 2
    assert any("TCP:host.docker.internal:11434" in a for a in run_argvs)
    assert any("TCP:host.docker.internal:8888" in a for a in run_argvs)
    connects = docker.argv("network connect")
    aliases = [a[a.index("--alias") + 1] for a in connects]
    assert aliases == [sbx.PROXY_ALIAS, "sovereign-searxng"]


def test_start_egress_proxies_never_ready_rolls_back(docker: FakeDocker, monkeypatch) -> None:
    monkeypatch.setattr(sbx.time, "sleep", lambda _s: None)
    docker.script("inspect", stdout="false\n")
    with pytest.raises(sbx.SandboxError, match="never reached running"):
        sbx.start_egress_proxies("sid", {sbx.PROXY_ALIAS: 11434}, harness_name="h")
    assert docker.argv("ps -aq") and docker.argv("network rm sid")


def test_start_egress_proxies_run_failure_rolls_back(docker: FakeDocker) -> None:
    docker.script("run", returncode=125, stderr="no such image")
    with pytest.raises(sbx.SandboxError, match="no such image"):
        sbx.start_egress_proxies("sid", {sbx.PROXY_ALIAS: 11434}, harness_name="h")
    assert docker.argv("network rm sid")


def test_service_alias_and_env_var() -> None:
    assert sbx.service_alias("searxng") == "sovereign-searxng"
    assert sbx.service_env_var("searxng") == "SOVEREIGN_SEARXNG_URL"
    assert sbx.service_env_var("my-mcp.v2") == "SOVEREIGN_MY_MCP_V2_URL"


# --- run_sandboxed ---
def test_run_sandboxed_argv(docker: FakeDocker) -> None:
    sbx.run_sandboxed(
        "sid",
        harness_name="h",
        image="img:1",
        argv=["cline", "--yolo"],
        mounts=["/tmp/w:/workspace"],
        env={"CLINE_DIR": "/cline"},
        network="sid",
        memory_bytes=2 * 10**9,
        workdir="/workspace",
    )
    (argv,) = docker.argv("run")
    joined = " ".join(argv)
    assert "--network sid" in joined
    assert "-e CLINE_DIR=/cline" in joined
    assert "-v /tmp/w:/workspace" in joined
    assert f"--memory {2 * 10**9}" in joined
    assert "-w /workspace" in joined
    assert argv[-3:] == ["img:1", "cline", "--yolo"]


def test_run_sandboxed_env_is_explicit_allowlist(docker: FakeDocker, monkeypatch) -> None:
    monkeypatch.setenv("HOST_SECRET", "leak-me")
    sbx.run_sandboxed(
        "sid", harness_name="h", image="i", argv=["x"], mounts=[], env={"A": "1"}, network="none"
    )
    (argv,) = docker.argv("run")
    env_pairs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert env_pairs == ["A=1"]


def test_run_sandboxed_timeout_kills_and_carries_stdout(docker: FakeDocker) -> None:
    docker.raises["run"] = subprocess.TimeoutExpired(cmd="run", timeout=5, output=b"partial")
    with pytest.raises(sbx.SandboxTimeout) as excinfo:
        sbx.run_sandboxed(
            "sid", harness_name="h", image="i", argv=["x"], mounts=[], env={}, network="none",
            timeout=5,
        )
    assert excinfo.value.partial_stdout == "partial"
    assert docker.argv("rm -f sid-agent")


# --- teardown / sweep ---
def test_teardown_removes_by_sandbox_id_label(docker: FakeDocker) -> None:
    docker.script("ps -aq", stdout="c1\nc2\n")
    sbx.teardown_sandbox("sid")
    (ps_argv,) = docker.argv("ps -aq")
    assert f"label={sbx.SANDBOX_ID_LABEL}=sid" in ps_argv
    assert docker.argv("rm -f c1") and docker.argv("rm -f c2")
    assert docker.argv("network rm sid")


def test_teardown_never_raises(docker: FakeDocker) -> None:
    docker.raises["ps -aq"] = OSError("gone")
    sbx.teardown_sandbox("sid")
    assert docker.argv("network rm sid")


def test_sweep_filters_by_label_and_scopes_by_harness(docker: FakeDocker) -> None:
    docker.script("ps -aq", stdout="c1\nc2\n")
    docker.script("network ls -q", stdout="n1\n")
    removed = sbx.sweep_leftovers(harness="h")
    assert removed == 3
    (ps_argv,) = docker.argv("ps -aq")
    assert f"label={sbx.SANDBOX_LABEL}" in ps_argv
    assert f"label={sbx.HARNESS_LABEL}=h" in ps_argv
    assert docker.argv("rm -f c1") and docker.argv("rm -f c2") and docker.argv("network rm n1")


def test_sweep_global_has_no_harness_filter(docker: FakeDocker) -> None:
    docker.script("ps -aq", stdout="")
    docker.script("network ls -q", stdout="")
    sbx.sweep_leftovers()
    (ps_argv,) = docker.argv("ps -aq")
    assert not any(a.startswith(f"label={sbx.HARNESS_LABEL}") for a in ps_argv)


def test_sweep_survives_missing_docker(docker: FakeDocker) -> None:
    docker.raises["ps -aq"] = FileNotFoundError("docker")
    assert sbx.sweep_leftovers() == 0


# --- daemon probe / ensure_image ---
def test_daemon_reachable_true_and_false(docker: FakeDocker) -> None:
    docker.script("version", stdout="27.0.1\n")
    assert sbx.daemon_reachable()
    docker.script("version", returncode=1, stdout="")
    assert not sbx.daemon_reachable()
    docker.raises["version"] = FileNotFoundError("docker")
    assert not sbx.daemon_reachable()


def test_ensure_image_present_is_noop(docker: FakeDocker) -> None:
    sbx.ensure_image("img:1")
    assert docker.argv("image inspect img:1") and not docker.argv("build")


def test_ensure_image_builds_from_context(docker: FakeDocker, tmp_path: Path) -> None:
    docker.script("image inspect", returncode=1)
    sbx.ensure_image("img:1", context_dir=tmp_path)
    (argv,) = docker.argv("build")
    assert argv == ["build", "-t", "img:1", str(tmp_path)]


def test_ensure_image_build_failure_raises(docker: FakeDocker, tmp_path: Path) -> None:
    docker.script("image inspect", returncode=1)
    docker.script("build", returncode=1, stderr="bad Dockerfile")
    with pytest.raises(sbx.SandboxError, match="bad Dockerfile"):
        sbx.ensure_image("img:1", context_dir=tmp_path)


def test_ensure_image_pulls_without_context(docker: FakeDocker, monkeypatch) -> None:
    docker.script("image inspect", returncode=1)
    pulled: list[str] = []
    monkeypatch.setattr(sbx, "stream_pull", lambda tag, *, binary, on_progress: pulled.append(tag))
    sbx.ensure_image("user/img:2")
    assert pulled == ["user/img:2"]


def test_sandbox_id_is_unique_and_prefixed() -> None:
    a, b = sbx.sandbox_id("h"), sbx.sandbox_id("h")
    assert a.startswith("sov-sbx-h-") and a != b
