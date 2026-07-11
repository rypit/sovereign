"""Byte-unit conventions (bytes-everywhere internal standard).

Internal representation is always int bytes. YAML declares decimal gigabytes
(1 GB = 10**9 bytes); display uses humanize's decimal units (binary=False),
so YAML input and displayed output agree.
"""
from __future__ import annotations

import humanize

#: Decimal gigabyte — the YAML input unit and the display unit base.
GB = 10**9


def gb_to_bytes(gb: float | int) -> int:
    """Convert a human-declared GB value (possibly fractional) to int bytes."""
    return round(gb * GB)


def fmt_size(n: int | float) -> str:
    """Human-readable decimal size string, e.g. 27_000_000_000 -> '27.0 GB'."""
    return humanize.naturalsize(n, binary=False)
