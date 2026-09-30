"""Ordered fallback: decisions, failover, probe classification and log hygiene."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from livestream_transcriber.models import StreamInfo
from livestream_transcriber.stream import fallback, resolver
from livestream_transcriber.stream.base import StreamResolutionError
from livestream_transcriber.stream.fallback import (
    BLOCKED_BOT,
    BLOCKED_NONE_LIVE,
    BLOCKED_UNCERTAIN,
    STATE_BOT_CHECK,
    STATE_LIVE,
    STATE_OFFLINE,
    STATE_UNCERTAIN,
    LiveSourceSelector,
    ProbeResult,
    SourceCandidate,
    classify_probe_error,
    decide_source,
    maybe_failover,
    probe_candidate,
    short_error,
)
from tests.support.scripted_probes import (
    BACKUP,
    BOT,
    BOT_ERROR,
    ERR,
    LIVE,
    OFF,
    PRIMARY,
    probe_row,
    scripted_selector,
)


def _decide(primary: str, backup: str) -> fallback.SourceSelection:
    return decide_source((probe_row(PRIMARY, primary), probe_row(BACKUP, backup)))


# --------------------------------------------------------------------- decisions


def test_exactly_one_live_is_selected() -> None:
    selection = _decide(OFF, LIVE)
    assert selection.capture_allowed and selection.state == STATE_LIVE
    assert selection.selected_name == "backup"
    assert selection.selected_url == BACKUP.url
    assert selection.selected_stream_id == "stream-1"
    assert selection.selection_reason == "exactly_one_live"


def test_a_live_candidate_is_captured_despite_a_failed_peer_probe() -> None:
    for peer in (ERR, BOT):
        selection = _decide(peer, LIVE)
        assert selection.selected_name == "backup"
        assert selection.selection_reason == "live_despite_probe_failure"


def test_several_live_candidates_prefer_the_listed_order() -> None:
    selection = _decide(LIVE, LIVE)
    assert selection.selected_name == "primary"
    assert selection.selection_reason == "multiple_live_prefer_listed_order"


def test_none_live_blocks_capture_with_a_reason() -> None:
    offline = _decide(OFF, OFF)
    assert not offline.capture_allowed
    assert (offline.state, offline.blocked) == (STATE_OFFLINE, BLOCKED_NONE_LIVE)
    assert not offline.probe_failed

    uncertain = _decide(OFF, ERR)
    assert (uncertain.state, uncertain.blocked) == (STATE_UNCERTAIN, BLOCKED_UNCERTAIN)
    assert uncertain.probe_failed

    bot = _decide(BOT, OFF)
    assert (bot.state, bot.blocked) == (STATE_BOT_CHECK, BLOCKED_BOT)
    assert bot.probe_failed


def test_a_bot_check_next_to_a_probe_error_is_a_bot_check() -> None:
    """The address is flagged; filing it as "uncertain" would let a block that
    answers bot-check and error in turn flip between two states unalerted."""
    assert _decide(BOT, ERR).state == STATE_BOT_CHECK
    assert _decide(ERR, BOT).state == STATE_BOT_CHECK


def test_a_live_row_without_a_url_is_not_capturable() -> None:
    row = ProbeResult("primary", 0.0, 0.0, LIVE, True, None, None)
    assert not decide_source([row]).capture_allowed


# --------------------------------------------------------------------- failover


def test_failover_only_when_the_current_url_is_no_longer_live() -> None:
    both = _decide(LIVE, LIVE)  # would select primary
    assert maybe_failover(BACKUP.url, both) == (None, None)  # still live: do not flap

    only_backup = _decide(OFF, LIVE)
    new_url, alert = maybe_failover(PRIMARY.url, only_backup)
    assert new_url == BACKUP.url
    assert alert is not None
    assert alert.startswith("Source failover")
    assert "primary: OFFLINE" in alert and "selected: backup" in alert
    assert "reason=exactly_one_live" in alert


def test_failover_is_not_needed_on_the_selected_url() -> None:
    assert maybe_failover(BACKUP.url, _decide(OFF, LIVE)) == (None, None)
    # a trailing slash and the case of scheme and host do not make another source
    shouted = "HTTPS://STREAMS.EXAMPLE.TEST/backup/live/"
    assert maybe_failover(shouted, _decide(OFF, LIVE)) == (None, None)
    assert maybe_failover(BACKUP.url + "/", _decide(OFF, LIVE)) == (None, None)


def test_no_failover_while_nothing_is_capturable() -> None:
    assert maybe_failover(PRIMARY.url, _decide(OFF, OFF)) == (None, None)
    assert maybe_failover(PRIMARY.url, _decide(BOT, OFF)) == (None, None)


# --------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ERROR: [site] abc: This channel is not currently live", OFF),
        ("The live event will begin in 3 hours", OFF),
        ("Premiere will begin shortly", OFF),
        ("stream is offline (HTTP 404)", OFF),
        ("ERROR: Sign in to confirm you're not a bot", BOT),
        ("Sign in to confirm you’re not a bot", BOT),  # noqa: RUF001 - typographic apostrophe
        ("unable to download webpage: connection timed out", ERR),
        ("HTTP Error 429: Too Many Requests", ERR),
        ("something nobody has seen before", ERR),
        ("", ERR),
    ],
)
def test_classify_probe_error(message: str, expected: str) -> None:
    assert classify_probe_error(message) == expected


def test_short_error_folds_whitespace_and_keeps_short_text() -> None:
    assert short_error(None) is None
    assert short_error("timed   out\n  again") == "timed out again"
    cut = short_error("x" * 500, limit=20)
    assert cut is not None and len(cut) == 20 and cut.endswith("…")
    assert short_error("failed on https://cdn.example/x?sig=abc now") == "failed on <url> now"


# --------------------------------------------------------------------- probing


class _FakeResolver:
    def __init__(self, answers: dict[str, StreamInfo | Exception]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    async def __call__(self, url: str, **_kw: Any) -> StreamInfo:
        self.calls.append(url)
        answer = self.answers[url]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_probe_candidate_maps_resolver_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeResolver(
        {
            "https://x.test/live": StreamInfo(
                url="https://x.test/live", is_live=True, stream_id="abc123"
            ),
            "https://x.test/vod": StreamInfo(url="https://x.test/vod", is_live=False),
            "https://x.test/off": StreamResolutionError("This channel is not currently live"),
            "https://x.test/bot": StreamResolutionError("Sign in to confirm you're not a bot"),
            "https://x.test/err": StreamResolutionError("connection timed out"),
        }
    )
    monkeypatch.setattr(fallback, "resolve_stream", fake)

    async def probe(name: str) -> ProbeResult:
        return await probe_candidate(SourceCandidate(name, f"https://x.test/{name}"), "best")

    live = await probe("live")
    assert (live.status, live.is_live, live.stream_id, live.url) == (
        LIVE,
        True,
        "abc123",
        "https://x.test/live",
    )
    vod = await probe("vod")
    assert (vod.status, vod.url, vod.stream_id) == (OFF, None, None)
    assert (await probe("off")).status == OFF
    bot = await probe("bot")
    assert bot.status == BOT and "not a bot" in (bot.error or "")
    assert (await probe("err")).status == ERR


async def test_candidates_are_probed_concurrently() -> None:
    """One after another, N candidates cost N times the slowest extraction."""
    started: list[str] = []
    both_started = asyncio.Event()

    async def probe(candidate: SourceCandidate, _fmt: str, **_kw: Any) -> ProbeResult:
        started.append(candidate.name)
        if len(started) == 2:
            both_started.set()
        # Only returns once the other candidate's probe has begun: a sequential
        # selector would deadlock here and hit the timeout.
        await asyncio.wait_for(both_started.wait(), 5)
        return probe_row(candidate, OFF)

    selector = LiveSourceSelector([PRIMARY, BACKUP], probe=probe)
    selection = await asyncio.wait_for(selector.select(), 10)
    assert sorted(started) == ["backup", "primary"]
    assert [row.name for row in selection.probe_results] == ["primary", "backup"]


async def test_a_crashing_probe_does_not_lose_the_other_answers() -> None:
    async def probe(candidate: SourceCandidate, _fmt: str, **_kw: Any) -> ProbeResult:
        if candidate is PRIMARY:
            raise RuntimeError("extractor bug")
        return probe_row(candidate, LIVE)

    selection = await LiveSourceSelector([PRIMARY, BACKUP], probe=probe).select()
    assert [row.status for row in selection.probe_results] == [ERR, LIVE]
    assert selection.selected_name == "backup"
    assert "RuntimeError" in (selection.probe_results[0].error or "")


async def test_a_cancelled_select_propagates_cancellation() -> None:
    async def probe(candidate: SourceCandidate, _fmt: str, **_kw: Any) -> ProbeResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(LiveSourceSelector([PRIMARY], probe=probe).select())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_a_selector_needs_a_candidate() -> None:
    with pytest.raises(ValueError, match="candidate"):
        LiveSourceSelector([])


def test_candidate_lookup_helpers() -> None:
    selector, _ = scripted_selector((OFF, LIVE))
    assert selector.candidate_for(BACKUP.url) is BACKUP
    assert selector.candidate_for(BACKUP.url + "/") is BACKUP
    assert selector.candidate_for("https://elsewhere.test/live") is None
    assert selector.candidate_for(None) is None
    assert selector.language_for(PRIMARY.url) == "en"
    assert selector.language_for(BACKUP.url) == "de"
    assert selector.language_for("https://elsewhere.test/live") is None


async def test_status_of_reports_the_row_for_a_url() -> None:
    selector, _ = scripted_selector((OFF, LIVE))
    selection = await selector.select()
    assert selector.status_of(selection, PRIMARY.url) == OFF
    assert selector.status_of(selection, BACKUP.url) == LIVE
    assert selector.status_of(selection, "https://elsewhere.test/live") is None


# --------------------------------------------------------------------- log hygiene


def _probe_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "source probe"]


async def test_repeated_probe_results_log_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """An idle process probes every candidate every 30 s for hours. Only a change of
    a candidate's status is INFO; repeats are DEBUG."""
    caplog.set_level(logging.DEBUG)
    selector, _ = scripted_selector((OFF, OFF), (OFF, OFF), (BOT, OFF), (BOT, OFF))
    for _ in range(4):
        await selector.select()

    levels = [(r.candidate, r.status, r.levelname) for r in _probe_records(caplog)]
    assert levels == [
        ("primary", OFF, "INFO"),
        ("backup", OFF, "INFO"),
        ("primary", OFF, "DEBUG"),
        ("backup", OFF, "DEBUG"),
        ("primary", BOT, "INFO"),
        ("backup", OFF, "DEBUG"),
        ("primary", BOT, "DEBUG"),
        ("backup", OFF, "DEBUG"),
    ]


async def test_probe_error_text_is_truncated_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    selector, _ = scripted_selector((BOT, OFF))
    selection = await selector.select()

    bot = next(r for r in _probe_records(caplog) if r.status == BOT)
    assert len(BOT_ERROR) > 600
    assert len(bot.error) <= 160
    assert "https://" not in bot.error
    assert "not a bot" in bot.error
    # the row itself keeps the whole message for classification and tests
    assert selection.probe_results[0].error == BOT_ERROR


# --------------------------------------------------------------------- resolve budget


class FakeExtractor:
    """Stands in for ``resolve_stream_sync``; one call is one extraction."""

    def __init__(self, live: dict[str, bool]) -> None:
        self.live = dict(live)
        self.calls: list[str] = []

    def __call__(self, url: str, *, format_selector: str, **_kw: Any) -> StreamInfo:
        self.calls.append(url)
        if not self.live.get(url):
            raise StreamResolutionError(f"ERROR: [site] {url}: The channel is not currently live")
        return StreamInfo(url=url, is_live=True, media_url=f"{url}/audio.m3u8")


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> FakeExtractor:
    resolver.forget_resolved()
    fake = FakeExtractor({PRIMARY.url: True, BACKUP.url: False})
    monkeypatch.setattr(resolver, "resolve_stream_sync", fake)
    yield fake
    resolver.forget_resolved()


async def test_a_probe_feeds_the_resolve_cache(extractor: FakeExtractor) -> None:
    """The probe that finds a candidate live runs the same extraction the capture
    repeats seconds later; the resolver remembers it, so that costs nothing."""
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector="best")
    selection = await selector.select()
    assert selection.selected_url == PRIMARY.url

    info = await resolver.resolve_stream(PRIMARY.url, format_selector="best", max_age=20.0)
    assert info.media_url == f"{PRIMARY.url}/audio.m3u8"
    assert sorted(extractor.calls) == sorted([PRIMARY.url, BACKUP.url])
