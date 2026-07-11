"""``WorkerConfig``: the JSON handoff from a manager to its engine-worker process.

The manager writes this file (mode 0600 — it may be read alongside a
process-wide ``SOVEREIGN_API_KEY`` env var, so it stays off argv and off group/
other read) before ``Popen``-ing the worker; the worker's only argv is
``--config <path>``. Stdlib + dataclasses only — importable from both the
parent (managers) and the worker entrypoint without pulling in engine
bindings.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Config schema version. Bump on incompatible field changes.
WORKER_CONFIG_VERSION = 1


@dataclass
class WorkerConfig:
    """Everything an engine-worker process needs to boot, minus secrets.

    ``engine_kwargs`` is the escape hatch for engine-specific settings (e.g.
    ``gpu_layers``, ``context_size``) that the adapter maps onto the real
    binding's API — kept as a plain dict here so this module never needs to
    know about any particular engine.
    """

    service: str
    engine: str
    host: str
    port: int
    health_path: str
    telemetry_socket: str
    model_path: str
    v: int = WORKER_CONFIG_VERSION
    draft_model_path: str | None = None
    served_model_name: str | None = None
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


def dump_worker_config(cfg: WorkerConfig, path: str | Path) -> None:
    """Atomically write ``cfg`` as JSON at ``path`` with mode 0600.

    Writes to a temp file in the same directory then ``os.replace()`` (atomic
    same-filesystem rename), and applies the restrictive mode before the
    rename lands the file at its final name — a reader never observes a
    world-readable version of the file, even momentarily.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=False) + "\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_worker_config(path: str | Path) -> WorkerConfig:
    """Read a ``WorkerConfig`` written by ``dump_worker_config``."""
    data = json.loads(Path(path).read_text())
    return WorkerConfig(**data)
