"""Per-integration environment provisioning — one contract for services and harnesses.

Declaring an integration in ``sovereign.yaml`` is sufficient: Sovereign installs
whatever it needs (toolchain included). Each integration folder owns its setup
artifacts (§2.4, self-contained folders) — a ``Brewfile`` sitting next to the
class's module is discovered by convention, and extra install commands are
declared on the class. Both the Orchestrator's ``prepare_environment()`` phase
and ``sovereign provision`` (used by ``scripts/setup.py``) run the same code.

The API is class-level so the setup path can provision integrations without
instantiating a manager/harness (which would require a full ``ServiceEntry``).
"""

from __future__ import annotations

import inspect
import logging
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

log = logging.getLogger(__name__)

# Generous timeouts: `brew bundle` may build/download large formulae (Node),
# and install commands may hit slow registries on first run.
BREW_TIMEOUT = 600.0
COMMAND_TIMEOUT = 300.0

# Classes whose provision() already ran (successfully or not) this process —
# re-materialization and sweeps must not hammer brew/npm after a failure.
_ATTEMPTED: set[type] = set()


class ProvisioningError(Exception):
    """Raised when an integration's dependencies cannot be installed."""


def reset_attempts() -> None:
    """Forget which classes already attempted provisioning (for tests)."""
    _ATTEMPTED.clear()


def _run(cmd: list[str], *, timeout: float) -> tuple[int, str]:
    """Run one install command, returning (returncode, stderr-ish detail)."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv from class declarations
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {timeout:.0f}s"
    except OSError as exc:
        return 1, str(exc)
    return result.returncode, (result.stderr or result.stdout or "").strip()


class Provisioner:
    """Mixin giving an integration class idempotent dependency installation.

    Defaults describe the common case declaratively:

    * ``provisioning_binary`` — a binary whose presence on PATH means the
      integration is already provisioned (``None`` = always satisfied).
    * a ``Brewfile`` next to the class's module — toolchain dependencies,
      installed via ``brew bundle`` (picked up automatically when present).
    * ``provisioning_commands`` — install commands run after the Brewfile
      (e.g. ``npm install -g ...``, ``uv pip install ...``).

    Integrations with non-binary checks (an importable package, say) override
    :meth:`provisioning_satisfied` instead.
    """

    #: Binary whose presence means provisioning is satisfied (None = nothing to do).
    provisioning_binary: ClassVar[str | None] = None
    #: Install commands run after the Brewfile, in order.
    provisioning_commands: ClassVar[list[list[str]]] = []

    @classmethod
    def provisioning_brewfile(cls) -> Path | None:
        """A ``Brewfile`` in the class's own folder, if it ships one."""
        try:
            module_file = inspect.getfile(cls)
        except (TypeError, OSError):
            return None
        brewfile = Path(module_file).parent / "Brewfile"
        return brewfile if brewfile.is_file() else None

    @classmethod
    def provisioning_satisfied(cls) -> bool:
        """Whether the environment already has what this integration needs."""
        if cls.provisioning_binary is None:
            return True
        return shutil.which(cls.provisioning_binary) is not None

    @classmethod
    def provision(cls) -> None:
        """Install this integration's dependencies; no-op once satisfied.

        One attempt per class per process: after a failure, subsequent calls
        re-raise immediately instead of re-running installers.
        """
        if cls.provisioning_satisfied():
            return
        if cls in _ATTEMPTED:
            raise ProvisioningError(
                f"{cls.__name__}: dependencies still missing after an earlier "
                "provisioning attempt this session."
            )
        _ATTEMPTED.add(cls)

        brewfile = cls.provisioning_brewfile()
        if brewfile is not None:
            if shutil.which("brew") is None:
                raise ProvisioningError(
                    f"{cls.__name__} needs Homebrew to install its toolchain "
                    f"({brewfile}), but `brew` was not found. Install it from "
                    "https://brew.sh and retry."
                )
            log.info("provisioning %s: brew bundle --file %s", cls.__name__, brewfile)
            code, detail = _run(
                ["brew", "bundle", "--file", str(brewfile)], timeout=BREW_TIMEOUT
            )
            if code != 0:
                raise ProvisioningError(
                    f"{cls.__name__}: `brew bundle --file {brewfile}` failed: {detail}"
                )

        for cmd in cls.provisioning_commands:
            log.info("provisioning %s: %s", cls.__name__, " ".join(cmd))
            code, detail = _run(list(cmd), timeout=COMMAND_TIMEOUT)
            if code != 0:
                raise ProvisioningError(f"{cls.__name__}: `{' '.join(cmd)}` failed: {detail}")

        if not cls.provisioning_satisfied():
            raise ProvisioningError(
                f"{cls.__name__}: dependencies still missing after provisioning "
                "(Brewfile + install commands ran without error)."
            )
