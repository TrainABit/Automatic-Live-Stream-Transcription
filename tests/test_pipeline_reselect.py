"""In-session reselection: keep, fail over, end the session, or hold while blocked."""

from __future__ import annotations

import asyncio
from typing import Any

from livestream_transcriber.pipeline.reselect import attach_reselect
from livestream_transcriber.stream.auto_resume import ProbeBackoff, SourceStatusMonitor

from .support.fake_source import FakeSource
from .support.scripted_probes import (
    BACKUP,
    BOT,
    ERR,
    LIVE,
    OFF,
    PRIMARY,
    AlertRecorder,
    ScriptedProbe,
    scripted_selector,
)


class Harness:
    """A source with reselect attached to a scripted selector."""

    def __init__(
        self,
        *rounds: tuple[str, str],
        hold_seconds: float = 60.0,
        reuse_seconds: float = 30.0,
        backoff: ProbeBackoff | None = None,
    ) -> None:
        self.selector, self.probe = scripted_selector(*rounds)
        self.source: Any = FakeSource(hang=True)
        self.session_end = asyncio.Event()
        self.alerts = AlertRecorder()
        self.idle = 0.0
        self.ended = 0
        self.backoff = backoff or ProbeBackoff(0.05, 0.4)
        self.status = SourceStatusMonitor(self.alerts)
        self.watch_hold = attach_reselect(
            self.source,
            self.selector,
            session_end=self.session_end,
            status=self.status,
            backoff=self.backoff,
            media_idle_seconds=lambda: self.idle,
            hold_seconds=hold_seconds,
            reuse_seconds=reuse_seconds,
            alert=self.alerts,
            on_end=self._on_end,
        )

    def _on_end(self) -> None:
        self.ended += 1

    def assume_live(self) -> None:
        """Start from an already reported ``live`` state, as a running session does."""
        self.status.assume("live")

    async def ask(self, current: str, reason: str = "stream_ended", attempt: int = 1) -> Any:
        return await self.source.reselect(current, reason, attempt)


class TestLiveCandidates:
    async def test_the_captured_source_still_live_is_kept(self) -> None:
        h = Harness((LIVE, OFF))
        h.assume_live()
        assert await h.ask(PRIMARY.url) is None
        assert not h.session_end.is_set()
        assert h.alerts.sent == []

    async def test_a_dead_source_fails_over_to_the_live_backup(self) -> None:
        h = Harness((OFF, LIVE))
        h.assume_live()
        assert await h.ask(PRIMARY.url) == BACKUP.url
        assert h.alerts.titles == ["Source failover"]
        assert not h.session_end.is_set()

    async def test_a_live_peer_does_not_pull_a_healthy_capture_away(self) -> None:
        h = Harness((LIVE, LIVE))
        h.assume_live()
        assert await h.ask(BACKUP.url) is None
        assert h.alerts.sent == []


class TestEndingTheSession:
    async def test_every_candidate_offline_ends_the_session_not_the_process(self) -> None:
        h = Harness((OFF, OFF))
        assert await h.ask(PRIMARY.url) is None
        assert h.session_end.is_set()
        assert h.ended == 1
        assert h.source.close_calls == 1

    async def test_the_end_is_reported_once(self) -> None:
        h = Harness((OFF, OFF))
        await h.ask(PRIMARY.url)
        await h.ask(PRIMARY.url, attempt=2)
        assert h.ended == 1
        assert h.source.close_calls == 1

    async def test_the_captured_source_offline_while_a_peer_probe_failed_still_ends(self) -> None:
        h = Harness((OFF, ERR))
        await h.ask(PRIMARY.url)
        assert h.session_end.is_set()
        assert any("session ended" in body for body in h.alerts.bodies)


class TestUnknownStatus:
    async def test_a_bot_check_keeps_the_session_alive(self) -> None:
        h = Harness((BOT, BOT))
        assert await h.ask(PRIMARY.url) is None
        assert not h.session_end.is_set()
        assert h.source.close_calls == 0

    async def test_a_failing_probe_of_the_captured_source_is_not_proof_it_is_gone(self) -> None:
        h = Harness((ERR, OFF))
        await h.ask(PRIMARY.url)
        assert not h.session_end.is_set()

    async def test_starved_of_media_while_unknown_ends_the_session(self) -> None:
        h = Harness((BOT, BOT), hold_seconds=30)
        h.idle = 45.0
        assert await h.ask(PRIMARY.url) is None
        assert h.session_end.is_set()
        assert "Capture session ended" in h.alerts.titles

    async def test_the_hold_watch_ends_a_starved_session_between_reconnects(self) -> None:
        h = Harness((BOT, BOT), hold_seconds=0.2)
        await h.ask(PRIMARY.url)
        watch = asyncio.create_task(h.watch_hold())
        try:
            h.idle = 5.0
            await asyncio.wait_for(h.session_end.wait(), 3)
        finally:
            await asyncio.wait_for(watch, 3)
        assert h.source.close_calls == 1

    async def test_the_hold_watch_leaves_a_session_with_flowing_media_alone(self) -> None:
        h = Harness((BOT, BOT), hold_seconds=0.2)
        await h.ask(PRIMARY.url)
        watch = asyncio.create_task(h.watch_hold())
        await asyncio.sleep(0.3)
        assert not h.session_end.is_set()
        h.session_end.set()
        await asyncio.wait_for(watch, 3)

    async def test_a_conclusive_answer_clears_the_unknown_state(self) -> None:
        h = Harness((BOT, BOT), (LIVE, OFF), hold_seconds=0.2)
        await h.ask(PRIMARY.url, attempt=1)
        await asyncio.sleep(0.06)  # the back-off after a failed round is over
        await h.ask(PRIMARY.url, reason="stall", attempt=2)
        watch = asyncio.create_task(h.watch_hold())
        h.idle = 5.0
        await asyncio.sleep(0.3)
        assert not h.session_end.is_set()
        h.session_end.set()
        await asyncio.wait_for(watch, 3)


class TestProbeEconomy:
    async def test_the_same_reconnect_reuses_one_probe(self) -> None:
        h = Harness((LIVE, OFF))
        await h.ask(PRIMARY.url, "stream_ended", 1)
        await h.ask(PRIMARY.url, "stream_ended", 1)
        assert isinstance(h.probe, ScriptedProbe)
        assert h.probe.rounds_used == 1

    async def test_another_reconnect_probes_again_when_probes_are_conclusive(self) -> None:
        h = Harness((LIVE, OFF))
        await h.ask(PRIMARY.url, "stream_ended", 1)
        await h.ask(PRIMARY.url, "stream_ended", 2)
        assert h.probe.rounds_used == 2

    async def test_while_backing_off_the_last_answer_is_held(self) -> None:
        h = Harness((BOT, BOT), backoff=ProbeBackoff(60.0, 600.0))
        await h.ask(PRIMARY.url, "stall", 1)
        # A second failed round is needed before the back-off doubles; either way the
        # probe is not due again within the interval.
        await h.ask(PRIMARY.url, "stall", 2)
        await h.ask(PRIMARY.url, "stall", 3)
        assert h.probe.rounds_used <= 2
