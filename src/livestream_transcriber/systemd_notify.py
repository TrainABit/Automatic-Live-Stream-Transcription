"""Optional systemd ``Type=notify`` integration with a progress-driven watchdog.

Under systemd, ``WatchdogSec=`` restarts a service that stops petting the
watchdog. Petting from a plain timer is not enough for a transcriber: the event
loop and the heartbeat keep running after the audio source or the STT lane has
wedged, so the timer would never stop and the watchdog would never fire.

The rule implemented by :class:`SystemdNotifier`:

* **idle** (probing for a stream, between sessions): pet on a timer;
* **active** (a session is transcribing): pet only when the pipeline reports
  *progress*, i.e. a chunk was processed, not merely received.

Choose ``WatchdogSec=`` well above the in-process stall watchdog so that the
supervisor gets a chance to restart the source before systemd kills the
process. Everything here is a no-op when ``NOTIFY_SOCKET`` is not set, so the
class can be used unconditionally.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Callable

from .logging_setup import get_logger

__all__ = ["IDLE_PET_INTERVAL_S", "MIN_PET_INTERVAL_S", "SystemdNotifier", "sd_notify"]

log = get_logger(__name__)

MIN_PET_INTERVAL_S = 15.0
IDLE_PET_INTERVAL_S = 30.0


def sd_notify(message: str) -> bool:
    """Send one datagram to ``NOTIFY_SOCKET``. Returns ``False`` outside systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract socket namespace
        addr = "\0" + addr[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.settimeout(1.0)
        sock.connect(addr)
        sock.sendall(message.encode("utf-8"))
        return True
    except OSError:
        log.debug("sd_notify failed", exc_info=True)
        return False
    finally:
        sock.close()


class SystemdNotifier:
    """Readiness and watchdog notifications with per-instance state.

    Instance state (rather than module globals) keeps tests independent and
    lets a process run more than one notifier if it ever needs to.
    """

    def __init__(
        self,
        *,
        min_pet_interval: float = MIN_PET_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_pet_interval = min_pet_interval
        self._clock = clock
        self._active = False
        self._active_since: float | None = None
        self._last_pet: float | None = None
        self._last_progress: float | None = None

    @property
    def enabled(self) -> bool:
        """True when running under a supervisor that listens for notifications."""
        return bool(os.environ.get("NOTIFY_SOCKET"))

    @property
    def active(self) -> bool:
        return self._active

    def ready(self, status: str | None = None) -> None:
        """Tell systemd startup is finished (``READY=1``), with an optional status."""
        lines = ["READY=1"]
        if status:
            lines.append("STATUS=" + status.replace("\n", " ")[:200])
        lines.append("WATCHDOG=1")
        self._send_pet("\n".join(lines))

    def stopping(self) -> None:
        sd_notify("STOPPING=1")

    def set_active(self, active: bool) -> None:
        """Mark a session as running (progress-driven) or over (timer-driven).

        One pet on entry covers source resolution and process start-up; after
        that only :meth:`note_progress` keeps the watchdog fed.
        """
        self._active = active
        self._last_progress = None
        if active:
            self._active_since = self._clock()
            self.pet(force=True)
        else:
            self._active_since = None

    def pet(self, *, force: bool = False) -> None:
        """Send ``WATCHDOG=1``, at most once per ``min_pet_interval`` unless forced."""
        now = self._clock()
        if (
            not force
            and self._last_pet is not None
            and now - self._last_pet < self._min_pet_interval
        ):
            return
        self._send_pet("WATCHDOG=1")

    def note_progress(self) -> None:
        """Record that a chunk was fully processed; pet if a session is active."""
        self._last_progress = self._clock()
        if self._active:
            self.pet()

    def stall_seconds(self) -> float | None:
        """Seconds since the last progress while active, else ``None``.

        Before the first processed chunk this counts from the start of the
        session, so a pipeline that never produces anything is visible too.
        """
        if not self._active:
            return None
        reference = self._last_progress if self._last_progress is not None else self._active_since
        return None if reference is None else self._clock() - reference

    async def watchdog_loop(
        self, stop: asyncio.Event, interval: float = IDLE_PET_INTERVAL_S
    ) -> None:
        """Pet on a timer while idle; stay quiet while a session is active."""
        while not stop.is_set():
            if not self._active:
                self.pet(force=True)
            try:
                await asyncio.wait_for(stop.wait(), interval)
                return
            except TimeoutError:
                continue

    def _send_pet(self, message: str) -> None:
        if sd_notify(message):
            self._last_pet = self._clock()
