"""Wait for a condition instead of sleeping a guessed amount of time.

A fixed ``asyncio.sleep`` is a bet on how fast a child process starts or a pipe
fills. On a loaded machine (CI, or a laptop running parallel suites) the bet is
lost now and then, and the test fails for a reason that has nothing to do with
what it checks. Poll the condition itself, with a timeout generous enough to
cover a machine under load; a condition that holds returns at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

# Long enough for an interpreter to start on a machine with every core busy;
# a hang still fails the test instead of the CI job.
DEFAULT_TIMEOUT_S = 30.0


async def wait_until(
    condition: Callable[[], object],
    *,
    what: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    interval: float = 0.01,
) -> None:
    """Return once ``condition()`` is truthy; fail naming ``what`` after ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout:g} s waiting for {what}")
        await asyncio.sleep(interval)
