"""The core contract for coding harnesses: ``Harness`` (§4b).

Harnesses are **leaf consumers** of the service registry: they reuse the resolver
and dependency edges, but nothing depends on them. Three capabilities, mirroring
the service lifecycle:

* ``prepare_environment()`` — install/validate everything the tool needs
  (toolchain, binary, package), so a harness declared in ``sovereign.yaml``
  is usable without manual setup. The harness analog of a service's
  ``PROVISIONING`` phase; must be idempotent.
* ``materialize()`` — project resolved endpoints/secrets into the tool's own
  config format. Runs only after dependencies are ``READY``; re-runs when an
  endpoint changes.
* ``invoke(task)`` — run one headless, non-interactive session to completion.
  Not all harnesses support this.

``Task`` and ``RunResult`` are intentionally minimal here; fields grow when the
harness/bench tracks land.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sovereign.core.provisioning import Provisioner
from sovereign.core.resolver import ConsumerKind, Resolver

if TYPE_CHECKING:
    from sovereign.config import HarnessEntry


@dataclass
class Task:
    """A single unit of work handed to a harness for a headless run."""

    #: Human-readable identifier for the task (used in bench cell keys / reports).
    id: str
    #: The instruction / prompt given to the harness.
    prompt: str
    #: Working directory the harness operates in (e.g. a throwaway sandbox).
    workdir: str | None = None
    #: Arbitrary per-task metadata (suite name, seed, budgets, ...).
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RunResult:
    """The outcome of a single ``invoke()`` session."""

    task_id: str
    #: Whether the harness reports the run as complete. Ground truth is graded
    #: separately (diff + tests) — a self-report of success is metadata, not proof.
    success: bool
    #: Process exit code, when the harness exposes one.
    exit_code: int | None = None
    #: Captured output / transcript path / free-form notes.
    output: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Harness(Protocol):
    """Contract for a configure-then-run-on-demand coding harness."""

    #: Unique instance ID (e.g. ``"cline_local"``).
    name: str
    #: Names of services that must be ``READY`` before this harness is usable.
    dependencies: list[str]

    def prepare_environment(self) -> None:
        """Pre-flight/provisioning hook run before ``materialize()``.

        Installs or validates everything the tool needs (toolchain, binary,
        package) so failures surface as a clean error rather than a failed
        invoke. Idempotent — re-materialization re-runs it.
        """
        ...

    def materialize(self) -> None:
        """Write resolved endpoints/secrets into the tool's own config format."""
        ...

    def invoke(self, task: Task) -> RunResult:
        """Run one headless, non-interactive session and return its result."""
        ...


class BaseHarness(Provisioner):
    """Shared scaffolding for concrete harnesses: provisioning, template
    resolution, and fingerprinting.

    Concrete harnesses subclass this and implement ``materialize()``/``invoke()``
    from the :class:`Harness` Protocol above; ``prepare_environment()`` installs
    the class's declared dependencies (Brewfile next to the module +
    ``provisioning_commands``) via the shared :class:`Provisioner` mixin.
    ``consumer_kind`` picks which host a ``{{ }}`` template resolves to (NATIVE
    for a harness running on the host, DOCKER for one running inside a sandbox
    container).
    """

    #: How this harness reaches service endpoints — see :class:`ConsumerKind`.
    consumer_kind: ConsumerKind = ConsumerKind.NATIVE

    def __init__(self, entry: HarnessEntry) -> None:
        self.entry = entry
        self.name = entry.name
        self.dependencies = entry.dependencies
        self.resolver: Resolver | None = None
        self.resolved_config: dict[str, object] = {}

    def prepare_environment(self) -> None:
        """Install this harness's declared dependencies (idempotent)."""
        self.provision()

    def resolve(self, resolver: Resolver) -> None:
        """Resolve ``{{ }}``/``${ENV:}`` templates in this harness's config block.

        Called by the Orchestrator once all ``dependencies`` are ``READY``, and
        again whenever one of those endpoints changes.
        """
        self.resolver = resolver
        self.resolved_config = resolver.resolve_mapping(self.entry.config, self.consumer_kind)

    def fingerprint(self) -> dict[str, object]:
        """Stable identity for the manifest and bench cell keys."""
        config_hash = hashlib.sha256(
            json.dumps(self.resolved_config, sort_keys=True, default=str).encode()
        ).hexdigest()
        return {"base_type": self.entry.base_type, "config_hash": config_hash}
