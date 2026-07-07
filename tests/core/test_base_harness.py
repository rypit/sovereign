"""Harness track (H1): `BaseHarness` — template resolution + fingerprint."""

from __future__ import annotations

from sovereign.config import HarnessEntry
from sovereign.core.base_harness import BaseHarness
from sovereign.core.resolver import ConsumerKind, ResolvedEndpoint, Resolver, ServiceRegistry


def _entry(config: dict | None = None) -> HarnessEntry:
    return HarnessEntry(
        name="cline_local",
        base_type="cline_cli",
        dependencies=["llama_heavy_v1"],
        config=config or {"base_url": "{{ llama_heavy_v1.endpoint }}/v1"},
    )


def _resolver() -> Resolver:
    reg = ServiceRegistry()
    reg.register(
        "llama_heavy_v1", ResolvedEndpoint("http", "127.0.0.1", 11435, model="llama3-70b")
    )
    return Resolver(reg, env={})


def test_resolve_populates_resolved_config() -> None:
    harness = BaseHarness(_entry())
    harness.resolve(_resolver())
    assert harness.resolved_config == {"base_url": "http://127.0.0.1:11435/v1"}
    assert harness.resolver is not None


def test_default_consumer_kind_is_native() -> None:
    assert BaseHarness.consumer_kind is ConsumerKind.NATIVE


def test_docker_consumer_kind_rewrites_loopback() -> None:
    class DockerHarness(BaseHarness):
        consumer_kind = ConsumerKind.DOCKER

    harness = DockerHarness(_entry())
    harness.resolve(_resolver())
    assert harness.resolved_config == {"base_url": "http://host.docker.internal:11435/v1"}


def test_fingerprint_is_stable_for_same_config() -> None:
    h1 = BaseHarness(_entry())
    h1.resolve(_resolver())
    h2 = BaseHarness(_entry())
    h2.resolve(_resolver())
    assert h1.fingerprint() == h2.fingerprint()
    assert h1.fingerprint()["base_type"] == "cline_cli"


def test_fingerprint_changes_with_config() -> None:
    h1 = BaseHarness(_entry({"model": "a"}))
    h1.resolve(_resolver())
    h2 = BaseHarness(_entry({"model": "b"}))
    h2.resolve(_resolver())
    assert h1.fingerprint()["config_hash"] != h2.fingerprint()["config_hash"]


def test_name_and_dependencies_from_entry() -> None:
    harness = BaseHarness(_entry())
    assert harness.name == "cline_local"
    assert harness.dependencies == ["llama_heavy_v1"]
