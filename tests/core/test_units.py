"""core/units.py — the bytes-everywhere GB<->bytes conversion and display seam.

Pins the actual humanize.naturalsize(binary=False) strings the rest of the
codebase's golden assertions depend on.
"""

from __future__ import annotations

from sovereign.core.units import GB, fmt_size, gb_to_bytes


def test_gb_constant_is_decimal_billion() -> None:
    assert GB == 10**9


def test_gb_to_bytes_integer() -> None:
    assert gb_to_bytes(128) == 128 * 10**9


def test_gb_to_bytes_fractional() -> None:
    assert gb_to_bytes(1.5) == 1_500_000_000


def test_fmt_size_gb() -> None:
    assert fmt_size(27 * 10**9) == "27.0 GB"
    assert fmt_size(93 * 10**9) == "93.0 GB"


def test_fmt_size_zero() -> None:
    assert fmt_size(0) == "0 Bytes"


def test_fmt_size_mb() -> None:
    assert fmt_size(15_500_000) == "15.5 MB"
