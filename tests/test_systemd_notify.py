from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from livestream_transcriber.systemd_notify import SystemdNotifier, sd_notify


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def notify_socket(
    monkeypatch: pytest.MonkeyPatch, short_socket_dir: Path
) -> Iterator[socket.socket]:
    """A datagram socket standing in for systemd's ``NOTIFY_SOCKET``."""
    path = short_socket_dir / "n.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(path))
    sock.settimeout(1.0)
    monkeypatch.setenv("NOTIFY_SOCKET", str(path))
    try:
        yield sock
    finally:
        sock.close()


def _recv(sock: socket.socket) -> str:
    return sock.recv(4096).decode()


def _nothing_pending(sock: socket.socket) -> bool:
    sock.settimeout(0.05)
    try:
        sock.recv(4096)
    except TimeoutError:
        return True
    finally:
        sock.settimeout(1.0)
    return False


def test_sd_notify_is_a_noop_without_socket():
    assert sd_notify("READY=1") is False
    assert SystemdNotifier().enabled is False


def test_short_socket_dir_fits_af_unix_even_from_a_long_checkout(short_socket_dir: Path):
    path = short_socket_dir / "n.sock"
    assert len(os.fsencode(str(path))) < 104
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind(str(path))
    finally:
        sock.close()


def test_ready_reports_status_and_arms_the_watchdog(notify_socket: socket.socket):
    notifier = SystemdNotifier()
    assert notifier.enabled
    notifier.ready("listening\nfor streams")
    message = _recv(notify_socket)
    assert "READY=1" in message
    assert "STATUS=listening for streams" in message
    assert "WATCHDOG=1" in message


def test_stopping_is_reported(notify_socket: socket.socket):
    SystemdNotifier().stopping()
    assert _recv(notify_socket) == "STOPPING=1"


def test_progress_pets_only_while_active_and_rate_limited(notify_socket: socket.socket):
    clock = FakeClock()
    notifier = SystemdNotifier(min_pet_interval=15.0, clock=clock)

    notifier.note_progress()  # idle: recorded but silent
    assert _nothing_pending(notify_socket)

    notifier.set_active(True)  # one forced pet on entry
    assert _recv(notify_socket) == "WATCHDOG=1"

    clock.now += 5
    notifier.note_progress()  # inside the rate limit window
    assert _nothing_pending(notify_socket)

    clock.now += 11
    notifier.note_progress()
    assert _recv(notify_socket) == "WATCHDOG=1"


def test_stall_is_measured_only_while_active():
    clock = FakeClock()
    notifier = SystemdNotifier(clock=clock)
    assert notifier.stall_seconds() is None

    notifier.set_active(True)
    clock.now += 20
    assert notifier.stall_seconds() == 20  # counted from activation

    notifier.note_progress()
    clock.now += 3
    assert notifier.stall_seconds() == 3

    notifier.set_active(False)
    assert notifier.stall_seconds() is None
    assert not notifier.active


async def test_watchdog_loop_pets_while_idle_and_stops(notify_socket: socket.socket):
    notifier = SystemdNotifier()
    stop = asyncio.Event()
    task = asyncio.create_task(notifier.watchdog_loop(stop, interval=0.05))
    await asyncio.sleep(0.02)
    assert _recv(notify_socket) == "WATCHDOG=1"
    stop.set()
    await asyncio.wait_for(task, 1.0)


async def test_watchdog_loop_is_quiet_while_active(notify_socket: socket.socket):
    notifier = SystemdNotifier()
    notifier.set_active(True)
    assert _recv(notify_socket) == "WATCHDOG=1"  # entry pet
    stop = asyncio.Event()
    task = asyncio.create_task(notifier.watchdog_loop(stop, interval=0.02))
    await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(task, 1.0)
    assert _nothing_pending(notify_socket)
