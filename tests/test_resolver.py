"""Phase 5: consumer-aware resolver — templates, secrets, host rewriting."""

from __future__ import annotations

import pytest

from sovereign.core.resolver import (
    ConsumerKind,
    ResolutionError,
    ResolvedEndpoint,
    Resolver,
    ServiceRegistry,
)


def _registry() -> ServiceRegistry:
    reg = ServiceRegistry()
    reg.register("llama_heavy_v1", ResolvedEndpoint("http", "127.0.0.1", 11435))
    return reg


def _resolver(env: dict | None = None) -> Resolver:
    return Resolver(_registry(), env=env or {})


def test_endpoint_native_uses_loopback() -> None:
    got = _resolver().resolve("{{ llama_heavy_v1.endpoint }}", ConsumerKind.NATIVE)
    assert got == "http://127.0.0.1:11435"


def test_endpoint_docker_uses_host_gateway() -> None:
    r = _resolver()
    got = r.resolve("{{ llama_heavy_v1.endpoint }}", ConsumerKind.DOCKER)
    assert got == "http://host.docker.internal:11435"


def test_host_and_port_attributes() -> None:
    r = _resolver()
    assert r.resolve("{{ llama_heavy_v1.host }}", ConsumerKind.DOCKER) == "host.docker.internal"
    assert r.resolve("{{ llama_heavy_v1.host }}", ConsumerKind.NATIVE) == "127.0.0.1"
    assert r.resolve("{{ llama_heavy_v1.port }}", ConsumerKind.DOCKER) == "11435"
    assert r.resolve("{{ llama_heavy_v1.scheme }}", ConsumerKind.NATIVE) == "http"


def test_non_loopback_host_not_rewritten_for_docker() -> None:
    reg = ServiceRegistry()
    reg.register("remote", ResolvedEndpoint("http", "10.0.0.5", 8000))
    got = Resolver(reg, env={}).resolve("{{ remote.endpoint }}", ConsumerKind.DOCKER)
    assert got == "http://10.0.0.5:8000"


def test_composite_string_multiple_templates() -> None:
    template = (
        "{{ llama_heavy_v1.scheme }}://"
        "{{ llama_heavy_v1.host }}:{{ llama_heavy_v1.port }}/v1"
    )
    got = _resolver().resolve(template, ConsumerKind.NATIVE)
    assert got == "http://127.0.0.1:11435/v1"


def test_env_secret_resolution() -> None:
    r = _resolver(env={"LLAMA_API_KEY": "s3cret"})
    assert r.resolve("${ENV:LLAMA_API_KEY}", ConsumerKind.NATIVE) == "s3cret"


def test_mixed_template_and_secret() -> None:
    r = _resolver(env={"TOKEN": "abc"})
    got = r.resolve("{{ llama_heavy_v1.endpoint }}?key=${ENV:TOKEN}", ConsumerKind.DOCKER)
    assert got == "http://host.docker.internal:11435?key=abc"


def test_unknown_service_raises() -> None:
    with pytest.raises(ResolutionError, match="unknown service 'ghost'"):
        _resolver().resolve("{{ ghost.endpoint }}", ConsumerKind.NATIVE)


def test_unknown_attribute_raises() -> None:
    with pytest.raises(ResolutionError, match="unknown endpoint attribute 'bogus'"):
        _resolver().resolve("{{ llama_heavy_v1.bogus }}", ConsumerKind.NATIVE)


def test_missing_secret_raises() -> None:
    with pytest.raises(ResolutionError, match="MISSING.*not set"):
        _resolver(env={}).resolve("${ENV:MISSING}", ConsumerKind.NATIVE)


def test_resolve_mapping_passthrough_non_strings() -> None:
    r = _resolver(env={"K": "v"})
    mapping = {
        "URL": "{{ llama_heavy_v1.endpoint }}",
        "KEY": "${ENV:K}",
        "PORT": 3000,
        "FLAG": True,
    }
    got = r.resolve_mapping(mapping, ConsumerKind.DOCKER)
    assert got == {
        "URL": "http://host.docker.internal:11435",
        "KEY": "v",
        "PORT": 3000,
        "FLAG": True,
    }


def test_registry_contains_and_names() -> None:
    reg = _registry()
    assert "llama_heavy_v1" in reg
    assert reg.names() == ["llama_heavy_v1"]
