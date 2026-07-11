"""macOS process physical-footprint helper, shared by native engine managers and workers.

``psutil``'s RSS misses Metal/GPU-resident buffers for unified-memory workloads;
``proc_pid_rusage``'s ``ri_phys_footprint`` is what Activity Monitor's "Memory"
column and ``top``'s MEM actually show. Split out of
``services/inference/base.py`` into a leaf module so both the parent-side
manager and the (future) in-process worker heartbeat thread can use it without
either depending on the other.
"""

from __future__ import annotations

import ctypes
import struct
import sys

_RUSAGE_INFO_V4 = 4
# Byte offset of ri_phys_footprint within struct rusage_info_v4 (sys/resource.h):
# 16-byte ri_uuid + 7 preceding uint64 fields (ri_user_time, ri_system_time,
# ri_pkg_idle_wkups, ri_interrupt_wkups, ri_pageins, ri_wired_size,
# ri_resident_size) = 16 + 7*8 = 72. Stable across macOS versions (Apple only
# appends fields to this struct, never reorders existing ones).
_PHYS_FOOTPRINT_OFFSET = 72
# Deliberately larger than the real struct (~280 bytes) — the kernel writes up
# to sizeof(rusage_info_v4) into whatever buffer we hand it; an undersized
# buffer would be a real memory-safety bug, not just a wrong read.
_RUSAGE_BUFFER_SIZE = 512


def _parse_phys_footprint(raw: bytes) -> int:
    """Extract ri_phys_footprint (a little-endian uint64) from a rusage_info_v4 buffer."""
    return struct.unpack_from("<Q", raw, _PHYS_FOOTPRINT_OFFSET)[0]


def macos_phys_footprint(pid: int) -> int | None:
    """The kernel's per-process physical-memory ledger (bytes) — what Activity
    Monitor's "Memory" column and ``top``'s MEM show, unlike psutil's RSS, which
    misses Metal/GPU-resident buffers for unified-memory workloads. Returns
    None on any failure (non-macOS, missing symbol, syscall error, unexpected
    layout) so callers fall back to RSS.
    """
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.create_string_buffer(_RUSAGE_BUFFER_SIZE)
        libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        libc.proc_pid_rusage.restype = ctypes.c_int
        rc = libc.proc_pid_rusage(pid, _RUSAGE_INFO_V4, buf)
        if rc != 0:
            return None
        return _parse_phys_footprint(buf.raw)
    except (OSError, AttributeError, struct.error):
        return None
