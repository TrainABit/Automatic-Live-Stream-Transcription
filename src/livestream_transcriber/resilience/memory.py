"""Process memory observability for cgroup-style limits."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["current_rss_mb", "memory_pressure_level"]

_WARN_RATIO = 0.85


def current_rss_mb() -> float | None:
    """Best-effort resident set size in megabytes, or ``None`` if unavailable."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024.0 * 1024.0)
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource

        # ru_maxrss is the *peak*, not the current value: fine as a fallback.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    # macOS reports bytes, Linux kilobytes.
    return float(usage) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(usage) / 1024.0


def memory_pressure_level(rss_mb: float, *, limit_mb: float) -> int:
    """Return 0 (ok), 1 (warn at 85 %) or 2 (critical at 100 %) relative to *limit_mb*.

    A non-positive limit disables the check.
    """
    if limit_mb <= 0:
        return 0
    ratio = rss_mb / limit_mb
    if ratio >= 1.0:
        return 2
    if ratio >= _WARN_RATIO:
        return 1
    return 0
