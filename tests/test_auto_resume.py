"""Idle loop, probe back-off and availability alerts."""

from __future__ import annotations

import asyncio
import logging

import pytest

from livestream_transcriber.config import Settings
from livestream_transcriber.logging_setup import ConsoleFormatter, JsonlFormatter
from livestream_transcriber.stream import auto_resume
from livestream_transcriber.stream.auto_resume import (
    ProbeBackoff,
    SourceStatusMonitor,
    interruptible_sleep,
    notify_stream_transition,
    until_stopped,
    wait_until_live,
)
from livestream_transcriber.stream.fallback import (
    LiveSourceSelector,
    SourceSelection,
    decide_source,
)
from tests.support.scripted_probes import (
    BACKUP,
    BOT,
    ERR,
    LIVE,
    OFF,
    PRIMARY,
    AlertRecorder,
    probe_row,
    scripted_selector,
)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every probe-loop sleep instead of waiting it out."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float, stop: asyncio.Event) -> bool:
        recorded.append(seconds)
        await asyncio.sleep(0)
        return stop.is_set()

    monkeypatch.setattr(auto_resume, "interruptible_sleep", fake_sleep)
    return recorded


def _whole(seconds: list[float]) -> list[int]:
    # The loop sleeps "until the last round is <delay> old" on the real clock; the
    # test itself takes milliseconds.
    return [round(s) for s in seconds]


def _rounds(*pairs: tuple[str, str]) -> list[SourceSelection]:
    return [decide_source((probe_row(PRIMARY, a), probe_row(BACKUP, b))) for a, b in pairs]


async def _feed(*pairs: tuple[str, str], in_session: bool = False) -> AlertRecorder:
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    for selection in _rounds(*pairs):
        await monitor.observe(selection, in_session=in_session)
    return alerts


# --------------------------------------------------------------------- primitives


async def test_interruptible_sleep_honours_stop() -> None:
    stop = asyncio.Event()
    stop.set()
    assert await interruptible_sleep(10.0, stop) is True


async def test_interruptible_sleep_runs_out_without_stop() -> None:
    assert await interruptible_sleep(0.01, asyncio.Event()) is False
    assert await interruptible_sleep(0, asyncio.Event()) is False


async def test_until_stopped_returns_the_result_or_none() -> None:
    async def quick() -> int:
        return 7

    assert await until_stopped(quick(), asyncio.Event()) == 7

    stop = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 1

    task = asyncio.create_task(until_stopped(slow(), stop))
    await asyncio.sleep(0.01)
    stop.set()
    assert await asyncio.wait_for(task, 5) is None
    assert cancelled.is_set(), "the abandoned probe must be cancelled, not leaked"


# --------------------------------------------------------------------- wait_until_live


async def test_wait_until_live_probes_until_online(sleeps: list[float]) -> None:
    selector, probe = scripted_selector((OFF, OFF), (LIVE, OFF))
    got = await wait_until_live(selector, asyncio.Event())
    assert got is not None and got.capture_allowed
    assert probe.rounds_used == 2


async def test_wait_until_live_returns_none_when_stopped() -> None:
    selector, probe = scripted_selector((OFF, OFF))
    stop = asyncio.Event()
    stop.set()
    assert await wait_until_live(selector, stop) is None
    assert probe.calls == 0


async def test_an_initial_selection_is_acted_on_without_probing(sleeps: list[float]) -> None:
    selector, probe = scripted_selector((OFF, OFF))
    (live,) = _rounds((LIVE, OFF))
    got = await wait_until_live(selector, asyncio.Event(), initial=live)
    assert got is live
    assert probe.calls == 0


async def test_the_stable_window_must_pass_before_a_session_opens(
    sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(auto_resume.time, "monotonic", lambda: clock[0])

    async def tick_sleep(seconds: float, stop: asyncio.Event) -> bool:
        clock[0] += 10.0
        return stop.is_set()

    monkeypatch.setattr(auto_resume, "interruptible_sleep", tick_sleep)
    # Live, offline (a flap resets the window), then live for good.
    selector, probe = scripted_selector(
        (LIVE, OFF), (OFF, OFF), (LIVE, OFF), (LIVE, OFF), (LIVE, OFF)
    )
    got = await wait_until_live(selector, asyncio.Event(), stable_seconds=15.0)
    assert got is not None
    assert probe.rounds_used == 5


async def test_settings_drive_the_backoff_and_the_stable_window(sleeps: list[float]) -> None:
    settings = Settings(
        resume_probe_interval_seconds=45.0,
        resume_backoff_max_seconds=50.0,
        resume_online_stable_seconds=0.0,
    )
    selector, _ = scripted_selector((BOT, BOT), (BOT, BOT), (LIVE, OFF))
    await wait_until_live(selector, asyncio.Event(), settings=settings)
    assert _whole(sleeps) == [45, 50]


# --------------------------------------------------------------------- alerts (idle loop)


async def test_bot_check_that_starts_while_idle_is_alerted(sleeps: list[float]) -> None:
    selector, _ = scripted_selector(
        (OFF, OFF), (OFF, OFF), (BOT, OFF), (BOT, OFF), (OFF, OFF), (OFF, OFF), (LIVE, OFF)
    )
    alerts = AlertRecorder()

    got = await wait_until_live(selector, asyncio.Event(), alert=alerts)

    assert got is not None and got.capture_allowed
    # offline at start, the block begins, the block clears, the stream starts
    # (a change between two not-live states is alerted on its second round)
    assert alerts.titles == [
        "Stream offline",
        "Bot check in force",
        "Stream offline",
        "Stream online",
    ]
    assert "blocked: the platform is asking for a bot check" in alerts.bodies[1]
    assert "primary: BOT_BLOCKED" in alerts.bodies[1]


async def test_probe_errors_are_their_own_alert(sleeps: list[float]) -> None:
    selector, _ = scripted_selector(
        (OFF, OFF), (ERR, OFF), (ERR, OFF), (BOT, OFF), (BOT, OFF), (LIVE, OFF)
    )
    alerts = AlertRecorder()
    await wait_until_live(selector, asyncio.Event(), alert=alerts)
    assert alerts.titles == [
        "Stream offline",
        "Stream status uncertain",
        "Bot check in force",
        "Stream online",
    ]


async def test_idle_bot_check_reaches_the_log_without_an_alert_callable(
    sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """With no callable configured, the WARNING line is the alert."""
    caplog.set_level(logging.INFO)
    selector, _ = scripted_selector((OFF, OFF), (BOT, OFF), (BOT, OFF), (LIVE, OFF))

    await wait_until_live(selector, asyncio.Event())

    alerts = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage() == "stream alert"
    ]
    assert [getattr(r, "state", None) for r in alerts] == ["offline", "bot_check", "live"]
    assert "not a bot" not in alerts[1].alert  # selection lines only
    assert "bot check" in alerts[1].alert


async def test_a_failing_alert_callable_never_breaks_the_loop(
    sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(title: str, body: str) -> None:
        raise RuntimeError("webhook down")

    selector, _ = scripted_selector((OFF, OFF), (LIVE, OFF))
    got = await wait_until_live(selector, asyncio.Event(), alert=broken)
    assert got is not None
    assert any(r.getMessage() == "alert delivery failed" for r in caplog.records)


async def test_idle_loop_logs_info_only_when_something_changed(
    sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    selector, _ = scripted_selector(*([(OFF, OFF)] * 6), (LIVE, OFF))

    await wait_until_live(selector, asyncio.Event())

    info = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and r.name.startswith("livestream_transcriber.stream")
    ]
    assert info == [
        "source probe",
        "source probe",
        "stream availability transition",
        "source probe",
        "stream availability transition",
    ]
    repeats = [r for r in caplog.records if "probing again" in r.getMessage()]
    assert len(repeats) == 6 and all(r.levelno == logging.DEBUG for r in repeats)


# --------------------------------------------------------------------- back-off


async def test_probe_interval_backs_off_while_the_bot_check_persists(sleeps: list[float]) -> None:
    selector, probe = scripted_selector(*([(BOT, OFF)] * 8), (OFF, OFF), (LIVE, OFF))

    await wait_until_live(selector, asyncio.Event())

    # doubling per failed round from the 30 s base, capped at 900 s, back to the base
    # on the first conclusive probe
    assert _whole(sleeps) == [30, 60, 120, 240, 480, 900, 900, 900, 30]
    assert probe.calls == 20


async def test_probe_errors_back_off_like_a_bot_check(sleeps: list[float]) -> None:
    selector, _ = scripted_selector((ERR, OFF), (ERR, BOT), (ERR, OFF), (LIVE, ERR))

    got = await wait_until_live(selector, asyncio.Event())

    assert got is not None and got.selection_reason == "live_despite_probe_failure"
    assert _whole(sleeps) == [30, 60, 120]


async def test_an_intermittent_block_keeps_backing_off_and_pages_once(
    sleeps: list[float],
) -> None:
    """A partial block answers bot check and offline in turn. Each offline round must
    not reset the back-off and flip the alert state, or the flagged address would still
    be probed every 30 s and the alert channel paged every round."""
    flapping = [(BOT, OFF), (OFF, OFF)] * 4
    selector, _ = scripted_selector(*flapping, (OFF, OFF), (OFF, OFF), (BOT, OFF), (LIVE, OFF))
    alerts = AlertRecorder()

    await wait_until_live(selector, asyncio.Event(), alert=alerts)

    # each offline round drops to the base interval (a stream start is seen at once),
    # each relapse doubles on from the level reached; two clean rounds in a row forget
    # the level
    assert _whole(sleeps) == [30, 30, 60, 30, 120, 30, 240, 30, 30, 30, 30]
    assert alerts.titles == ["Bot check in force", "Stream offline", "Stream online"]


async def test_back_off_cap_is_configurable(sleeps: list[float]) -> None:
    selector, _ = scripted_selector(*([(BOT, BOT)] * 4), (LIVE, OFF))
    await wait_until_live(selector, asyncio.Event(), backoff=ProbeBackoff(30.0, 45.0))
    assert _whole(sleeps) == [30, 45, 45, 45]


async def test_a_held_back_off_is_waited_out_before_the_first_probe(sleeps: list[float]) -> None:
    """A back-off already running (a session held through a bot check) is honoured by
    the idle loop: no immediate probe from the flagged address."""
    backoff = ProbeBackoff(30.0, 900.0)
    backoff.observe(decide_source((probe_row(PRIMARY, BOT), probe_row(BACKUP, OFF))))
    backoff.observe(decide_source((probe_row(PRIMARY, BOT), probe_row(BACKUP, OFF))))
    selector, probe = scripted_selector((LIVE, OFF))

    await wait_until_live(selector, asyncio.Event(), backoff=backoff)

    assert _whole(sleeps) == [60]
    assert probe.calls == 2


def test_backoff_resets_on_a_conclusive_round_and_counts_a_round_once() -> None:
    clock = [0.0]
    backoff = ProbeBackoff(30.0, 900.0, clock=lambda: clock[0])
    assert backoff.due() and backoff.remaining() == 0.0
    (blocked,) = _rounds((BOT, OFF))
    assert backoff.observe(blocked) == 30.0
    # the same round handed from a session to the idle loop is one round
    assert backoff.observe(blocked) == 30.0
    assert backoff.failures == 1
    (again,) = _rounds((BOT, OFF))
    assert backoff.observe(again) == 60.0
    assert not backoff.due()
    clock[0] = 59.0
    assert not backoff.due() and backoff.remaining() == pytest.approx(1.0)
    clock[0] = 60.0
    assert backoff.due()
    (live_other,) = _rounds((BOT, LIVE))
    assert backoff.observe(live_other) == 30.0
    assert backoff.failures == 0
    # conclusive: an in-session reselect may always ask
    assert backoff.due()


def test_backoff_never_overflows_during_a_week_long_block() -> None:
    backoff = ProbeBackoff(30.0, 900.0)
    for _ in range(5000):
        backoff.record_failure()
    assert backoff.delay == 900.0


def test_a_refused_connect_is_conclusive_but_still_waits_an_interval() -> None:
    clock = [0.0]
    backoff = ProbeBackoff(30.0, 900.0, clock=lambda: clock[0])
    assert backoff.record_not_live() == 30.0
    assert backoff.failures == 0
    assert backoff.remaining() == 30.0, "no hot loop while the listing lags the stream end"


def test_backoff_from_settings() -> None:
    backoff = ProbeBackoff.from_settings(
        Settings(resume_probe_interval_seconds=20.0, resume_backoff_max_seconds=100.0)
    )
    assert (backoff.base_s, backoff.max_s) == (20.0, 100.0)


# --------------------------------------------------------------------- monitor


async def test_intermittent_block_after_offline_was_reported_is_alerted_once() -> None:
    alerts = await _feed((OFF, OFF), (OFF, OFF), *([(BOT, OFF), (OFF, OFF)] * 5))
    assert alerts.titles == ["Stream offline", "Bot check in force"]


async def test_bot_check_next_to_a_probe_error_is_a_bot_check() -> None:
    (selection,) = _rounds((BOT, ERR))
    assert selection.state == "bot_check"
    alerts = await _feed((OFF, OFF), (BOT, ERR), (BOT, BOT), (ERR, BOT), (BOT, OFF))
    assert alerts.titles == ["Stream offline", "Bot check in force"]


async def test_a_new_bot_check_episode_after_it_cleared_is_alerted_again() -> None:
    alerts = await _feed(
        (OFF, OFF),
        (BOT, OFF), (BOT, OFF),  # episode 1
        (OFF, OFF), (OFF, OFF),  # cleared, offline reported
        (BOT, OFF), (OFF, OFF), (BOT, OFF),  # episode 2, intermittent
    )  # fmt: skip
    assert alerts.titles == [
        "Stream offline",
        "Bot check in force",
        "Stream offline",
        "Bot check in force",
    ]


async def test_a_single_bot_check_round_is_not_an_episode() -> None:
    alerts = await _feed((OFF, OFF), (BOT, OFF), (OFF, OFF), (OFF, OFF), (OFF, OFF))
    assert alerts.titles == ["Stream offline"]


async def test_reason_change_from_bot_check_to_probe_errors_is_alerted() -> None:
    alerts = await _feed((OFF, OFF), (BOT, BOT), (BOT, BOT), (ERR, ERR), (ERR, ERR))
    assert alerts.titles == ["Stream offline", "Bot check in force", "Stream status uncertain"]


async def test_live_peer_bot_check_flapping_pages_once_per_episode() -> None:
    alerts = await _feed(
        (LIVE, OFF), (LIVE, BOT), (LIVE, OFF), (LIVE, BOT), (LIVE, OFF), (LIVE, BOT),
        in_session=True,
    )  # fmt: skip
    assert alerts.titles == ["Stream online", "Bot check in force"]


async def test_a_bot_checked_peer_of_a_live_candidate_is_alerted_once() -> None:
    """One candidate live, the other bot-checked: capture goes ahead (state live), but
    the flagged address is news once per episode."""
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    # the episode ends after two rounds without a bot check; the next one pages again
    for selection in _rounds(
        (LIVE, OFF), (LIVE, BOT), (LIVE, BOT), (LIVE, OFF), (LIVE, OFF), (LIVE, BOT)
    ):
        await monitor.observe(selection, in_session=True)

    assert alerts.titles == ["Stream online", "Bot check in force", "Bot check in force"]
    assert "backup: BOT_BLOCKED" in alerts.bodies[1]
    assert "capture continues on primary" in alerts.bodies[1]
    assert monitor.state == "live"


async def test_the_monitor_alerts_once_and_says_when_it_probes_next(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    (blocked,) = _rounds((BOT, OFF))

    assert await monitor.observe(blocked, next_probe_s=60.0)
    assert not await monitor.observe(blocked)

    assert alerts.titles == ["Bot check in force"]
    assert "next probe in 60 s" in alerts.bodies[0]
    warnings = [r for r in caplog.records if r.getMessage() == "stream alert"]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert warnings[0].kind == "stream_status"


async def test_in_session_bot_check_says_capture_continues() -> None:
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    (live,) = _rounds((LIVE, OFF))
    (blocked,) = _rounds((BOT, OFF))
    await monitor.observe(live)
    await monitor.observe(blocked, in_session=True, next_probe_s=30.0)
    assert alerts.titles[-1] == "Bot check in force"
    assert "capture continues on the current media URLs" in alerts.bodies[-1]
    assert "was: live" in alerts.bodies[-1]


async def test_immediate_skips_the_confirmation_rounds() -> None:
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    await monitor.observe(_rounds((OFF, OFF))[0])
    assert not await monitor.observe(_rounds((ERR, OFF))[0])  # not confirmed yet
    assert await monitor.observe(_rounds((BOT, OFF))[0], immediate=True)
    assert alerts.titles == ["Stream offline", "Bot check in force"]


async def test_an_assumed_state_is_not_reported_again() -> None:
    alerts = AlertRecorder()
    monitor = SourceStatusMonitor(alerts)
    monitor.assume("live")
    assert not await monitor.observe(_rounds((LIVE, OFF))[0])
    assert alerts.sent == []


async def test_notify_stream_transition_is_a_one_off_alert() -> None:
    alerts = AlertRecorder()
    (selection,) = _rounds((OFF, LIVE))
    await notify_stream_transition(alerts, online=True, selection=selection, detail="started")
    await notify_stream_transition(alerts, online=False)
    assert alerts.titles == ["Stream online", "Stream offline"]
    assert alerts.bodies[0].startswith("started\nprimary: OFFLINE")


# --------------------------------------------------------------------- transition log line


def _reading(record: logging.LogRecord) -> list[str]:
    """What an operator's log parser sees in the console and JSON sinks."""
    readings = []
    for line in (ConsoleFormatter(color=False).format(record), JsonlFormatter().format(record)):
        readings.append(
            "online"
            if "true" in line.lower()
            else "offline"
            if "false" in line.lower()
            else "undecided"
        )
    return readings


async def test_the_transition_line_decides_online_only_for_live_and_offline(
    sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """A bot check or an uncertain probe says neither "online" nor "offline": inside a
    session the capture goes on on its media URLs, so a parser must not read it as the
    stream going away."""
    caplog.set_level(logging.INFO)
    selector, _ = scripted_selector(
        (OFF, OFF), (BOT, OFF), (BOT, OFF), (ERR, OFF), (ERR, OFF), (LIVE, OFF)
    )
    await wait_until_live(selector, asyncio.Event())

    moves = [r for r in caplog.records if r.getMessage() == "stream availability transition"]
    assert [(r.state, getattr(r, "online", "<missing>")) for r in moves] == [
        ("offline", False),
        ("bot_check", None),
        ("uncertain", None),
        ("live", True),
    ]


async def test_an_in_session_bot_check_keeps_the_state_machine_honest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    selector, _ = scripted_selector((LIVE, OFF), (BOT, OFF), (LIVE, OFF), (ERR, OFF))
    monitor = SourceStatusMonitor(None)
    for _ in range(4):
        assert await monitor.observe(await selector.select(), in_session=True)

    moves = [r for r in caplog.records if r.getMessage() == "stream availability transition"]
    assert [(r.previous, r.state) for r in moves] == [
        (None, "live"),
        ("live", "bot_check"),
        ("bot_check", "live"),
        ("live", "uncertain"),
    ]


def test_the_selector_type_is_what_the_idle_loop_expects() -> None:
    selector, _ = scripted_selector((OFF, OFF))
    assert isinstance(selector, LiveSourceSelector)
