"""``sovereign.yaml`` schema and loader (§5).

Parses a declarative stack description into validated Pydantic models. This is
pure desired-state: ``{{ ... }}`` templates and ``${ENV:...}`` secrets are carried
through **unresolved** — resolution against the runtime registry is the resolver's
job (Phase 5), not the schema's.

Benchmarks are deliberately absent: they live in a separate bench spec run
imperatively, never at boot.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from sovereign.core.base_config import GbBytes, SovereignBaseModel, validate_identifier


class ConfigError(Exception):
    """Raised when a ``sovereign.yaml`` file cannot be read or validated."""


class Priority(StrEnum):
    """Boot priority / QoS class, mapped later to ``os.nice()`` / ``taskpolicy``."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResourcesConfig(SovereignBaseModel):
    """Global unified-memory budget (§7)."""

    max_unified_memory_bytes: GbBytes = Field(gt=0, validation_alias="max_unified_memory_gb")
    safety_margin_bytes: GbBytes = Field(ge=0, validation_alias="safety_margin_gb")
    default_priority: Priority = Priority.MEDIUM


class HealthCheck(SovereignBaseModel):
    """Declarative readiness probe; executed by the manager's ``is_healthy()`` (§2.7)."""

    type: Literal["http"]
    endpoint: str
    port: int = Field(gt=0, lt=65536)
    timeout_seconds: int = Field(default=60, gt=0)


class ServiceEntry(SovereignBaseModel):
    """One supervised service in the stack."""

    name: str
    #: Which engine serves this service. ``"auto"`` (the default, also when omitted)
    #: routes to ``llama_cpp``/``mlx_lm`` from the model's HuggingFace metadata.
    base_type: str = "auto"
    priority: Priority | None = None
    dependencies: list[str] = Field(default_factory=list)

    #: Admission-control memory estimate/override. YAML declares GB (``memory_gb``);
    #: this field holds int bytes. Used by the ResourceBudgeter when the manager
    #: can't estimate its own footprint (§7).
    memory_bytes: GbBytes | None = Field(default=None, gt=0, validation_alias="memory_gb")

    health_check: HealthCheck | None = None

    # Flexible passthrough blocks — shape is owned by each service's own config.py,
    # validated when the concrete manager consumes them (later phases).
    affinity: dict[str, Any] | None = None
    policy: dict[str, Any] | None = None
    env_overrides: dict[str, Any] | None = None
    secrets: dict[str, Any] | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name", "base_type")(validate_identifier)

    @model_validator(mode="after")
    def _check_auto_routing(self) -> ServiceEntry:
        if self.base_type == "auto" and not self.config.get("model"):
            raise ValueError(
                f"service '{self.name}': base_type 'auto' (or omitted) requires config.model"
            )
        return self


class HarnessEntry(SovereignBaseModel):
    """One coding harness — a leaf consumer of the registry (§4b)."""

    name: str
    base_type: str
    dependencies: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    _check_name = field_validator("name", "base_type")(validate_identifier)


class SovereignConfig(SovereignBaseModel):
    """Top-level parsed ``sovereign.yaml`` (§5)."""

    version: str
    resources: ResourcesConfig
    services: list[ServiceEntry] = Field(default_factory=list)
    harnesses: list[HarnessEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_names_and_dependencies(self) -> SovereignConfig:
        entries: list[ServiceEntry | HarnessEntry] = [*self.services, *self.harnesses]
        names = [e.name for e in entries]

        # 1. Names unique across services *and* harnesses combined.
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            dupes = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate entry name(s): {dupes}")

        # 2 & 3. Dependencies must reference an existing entry, and nothing may
        # depend on itself.
        known = set(names)
        for entry in entries:
            for dep in entry.dependencies:
                if dep == entry.name:
                    raise ValueError(f"{entry.name!r} cannot depend on itself")
                if dep not in known:
                    raise ValueError(
                        f"{entry.name!r} depends on unknown entry {dep!r}"
                    )
        return self


def load_config(path: str | Path) -> SovereignConfig:
    """Read and validate a ``sovereign.yaml`` file.

    Wraps I/O and validation failures in :class:`ConfigError` with a readable
    message; Pydantic's own error already pinpoints the offending field.
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    try:
        return SovereignConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{exc}") from exc
