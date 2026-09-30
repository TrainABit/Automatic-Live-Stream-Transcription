"""Optional POSIX niceness for CPU-heavy background work such as local inference."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["apply_thread_nice", "run_with_nice"]

T = TypeVar("T")

# Per thread: the target this thread has already been brought to.
_applied = threading.local()


def apply_thread_nice(nice: int) -> None:
    """Bring the calling thread to niceness ``nice`` (absolute, idempotent).

    ``os.nice`` is *relative*: calling ``os.nice(5)`` before every inference
    call would walk the thread to 10, 15 and then the ceiling of 19. The delta
    to the target is computed once per thread instead. A thread that is already
    at least that nice is left alone, since lowering niceness needs privileges.

    On Linux niceness is per thread, so only the calling worker thread is
    affected; on macOS it applies to the whole process.
    """
    if nice <= 0 or getattr(_applied, "target", None) == nice:
        return
    try:
        current = os.nice(0)
        if current < nice:
            os.nice(nice - current)
    except OSError:
        pass
    _applied.target = nice


def run_with_nice(nice: int, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Call ``fn(*args, **kwargs)`` after :func:`apply_thread_nice`."""
    apply_thread_nice(nice)
    return fn(*args, **kwargs)
