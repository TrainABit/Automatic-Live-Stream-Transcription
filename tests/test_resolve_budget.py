"""Extractions per stream start and per reconnect.

Every resolve of a page URL is an extraction from an address the platform may already be
bot-checking. The source selector probes every candidate; the capture then resolved the
chosen URL again, and a reconnect could add a second reselect and a second resolve on top.
These tests count the calls through the real resolver, selector and capture source, with
only yt-dlp itself faked.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import pytest

from livestream_transcriber.models import SegmentInfo, StreamInfo
from livestream_transcriber.stream import resolver
from livestream_transcriber.stream.base import StreamNotLiveError, StreamResolutionError
from livestream_transcriber.stream.fallback import (
    LiveSourceSelector,
    SourceCandidate,
    maybe_failover,
)
from livestream_transcriber.stream.resolver import resolve_stream
from livestream_transcriber.stream.source import CaptureOptions, LiveStreamSource

PRIMARY = SourceCandidate("primary", "https://streams.example.test/primary/live")
BACKUP = SourceCandidate("backup", "https://streams.example.test/backup/live")
P, B = PRIMARY.url, BACKUP.url


class FakeYtDlp:
    """Stands in for ``resolve_stream_sync``; one call is one extraction."""

    def __init__(self, live: dict[str, bool]) -> None:
        self.live = dict(live)
        self.calls: list[str] = []
        # URLs whose next answer is "resolves, but not live" (then normal).
        self.not_live_once: set[str] = set()

    def __call__(self, url: str, *, format_selector: str, **_kw: Any) -> StreamInfo:
        self.calls.append(url)
        if url in self.not_live_once:
            self.not_live_once.discard(url)
            return StreamInfo(url=url, is_live=False, media_url=f"{url}/recording.m3u8")
        if not self.live.get(url):
            raise StreamResolutionError(f"ERROR: [site] {url}: The channel is not currently live")
        return StreamInfo(url=url, is_live=True, media_url=f"{url}/audio.m3u8")


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    resolver.forget_resolved()
    yield
    resolver.forget_resolved()


@pytest.fixture
def ytdlp(monkeypatch: pytest.MonkeyPatch) -> FakeYtDlp:
    fake = FakeYtDlp({P: True, B: False})
    monkeypatch.setattr(resolver, "resolve_stream_sync", fake)
    return fake


def _options(**kw: Any) -> CaptureOptions:
    return CaptureOptions(reconnect_initial_delay=0.01, reconnect_max_delay=0.01, **kw)


def _source(
    url: str, options: CaptureOptions, selector: LiveSourceSelector | None = None
) -> tuple[LiveStreamSource, list[str]]:
    reselects: list[str] = []

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        # What a session's reselect hook does: probe, then maybe fail over.
        reselects.append(reason)
        assert selector is not None
        selection = await selector.select()
        new_url, _alert = maybe_failover(current, selection)
        return new_url

    return LiveStreamSource(url, options, reselect=reselect if selector else None), reselects


async def test_stream_start_reuses_the_selector_probe(
    ytdlp: FakeYtDlp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection probes both candidates, then connect() would resolve the chosen one a
    third time within the same second."""
    options = _options()
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    selection = await selector.select()
    assert selection.selected_url == P
    source, _ = _source(P, options, selector)

    async def no_capture(_info: StreamInfo) -> None:
        return None

    monkeypatch.setattr(source, "_supervise", no_capture)
    info = await source.connect()
    await source.close()
    assert info.media_url == f"{P}/audio.m3u8"
    assert sorted(ytdlp.calls) == sorted([P, B])


async def test_reconnect_after_a_live_end_is_one_probe_round(ytdlp: FakeYtDlp) -> None:
    """A live ``stream_ended`` asks the selector (two probes); the resolve of the
    still-live candidate that follows is the probe's answer, not a third call."""
    options = _options()
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    source, reselects = _source(P, options, selector)
    info, switched = await source._next_stream_info(
        reason="stream_ended", is_live=True, empty_streak=0, attempt=1
    )
    assert info is not None and info.media_url == f"{P}/audio.m3u8"
    assert switched is False
    assert reselects == ["stream_ended"]
    assert sorted(ytdlp.calls) == sorted([P, B])


async def test_failover_after_a_failed_resolve_takes_the_probe_answer(ytdlp: FakeYtDlp) -> None:
    """The current candidate went offline during a transient drop: one failed resolve,
    one reselect round, and the new candidate's probe result is used as is."""
    ytdlp.live = {P: False, B: True}
    options = _options()
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    source, reselects = _source(P, options, selector)
    info, switched = await source._next_stream_info(
        reason="ffmpeg_exit_1", is_live=True, empty_streak=0, attempt=1
    )
    assert switched is True and source.url == B
    assert info is not None and info.media_url == f"{B}/audio.m3u8"
    assert reselects == ["ffmpeg_exit_1"]
    assert ytdlp.calls.count(B) == 1 and ytdlp.calls.count(P) == 2


async def test_one_reselect_per_iteration_at_most(ytdlp: FakeYtDlp) -> None:
    """A reselect that kept the candidate, then a failed resolve, must not run a second
    reselect (two more probes) in the same iteration."""
    options = _options()
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    source, reselects = _source(P, options, selector)
    original = selector._probe

    async def probe_then_drop(*args: Any, **kwargs: Any) -> Any:
        # The probe says primary is live; by the time the capture resolves, it is not.
        result = await original(*args, **kwargs)
        ytdlp.live[P] = False
        resolver.forget_resolved()
        return result

    selector._probe = probe_then_drop
    info, switched = await source._next_stream_info(
        reason="stream_ended", is_live=True, empty_streak=0, attempt=1
    )
    assert info is None and switched is False
    assert reselects == ["stream_ended"]
    # Two probes + one own resolve; not two probe rounds + two resolves.
    assert len(ytdlp.calls) == 3


async def test_one_odd_not_live_answer_does_not_end_a_live_capture(ytdlp: FakeYtDlp) -> None:
    """The capture's re-resolve says "not live" once; the selector's probe of the same
    URL a moment later says live. The probe is the newer answer, so the capture goes on
    instead of ending on one odd reply."""
    options = _options(max_reconnect_attempts=5)
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    source, reselects = _source(P, options, selector)
    played: list[bool] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        played.append(info.is_live)
        segment.audio_chunks = 5
        if len(played) == 2:
            source._closing.set()
            return "duration_reached"
        return "ffmpeg_exit_1"

    source._run_segment = run_segment
    ytdlp.not_live_once.add(P)
    await source._supervise(StreamInfo(url=P, is_live=True, media_url="m"))
    assert played == [True, True]
    assert reselects == ["live_ended"]
    assert sorted(ytdlp.calls) == sorted([P, P, B])


async def test_a_failed_segment_is_not_retried_on_the_resolve_that_fed_it(
    ytdlp: FakeYtDlp,
) -> None:
    """A segment that dies seconds in (a 403 on a signed manifest) would be retried on
    exactly the URL that had just failed, if its resolve were younger than the cache
    lifetime. Every retry resolves."""
    options = _options(max_reconnect_attempts=5)
    source, _ = _source(P, options)
    fed: list[StreamInfo] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        fed.append(info)
        if len(fed) == 3:
            source._closing.set()
            return "duration_reached"
        return "ffmpeg_exit_1"

    source._run_segment = run_segment
    await source._supervise(await resolve_stream(P, format_selector=options.stream_format))
    assert ytdlp.calls == [P, P, P]
    assert len({id(info) for info in fed}) == 3


async def test_after_a_live_end_the_probe_feeds_the_next_segment(ytdlp: FakeYtDlp) -> None:
    """What the cache is for: the selector's probe, made after the segment ended, is the
    next segment's resolve. No third call."""
    options = _options(max_reconnect_attempts=5)
    selector = LiveSourceSelector([PRIMARY, BACKUP], format_selector=options.stream_format)
    source, reselects = _source(P, options, selector)
    fed: list[StreamInfo] = []

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        fed.append(info)
        segment.audio_chunks = 5
        if len(fed) == 2:
            source._closing.set()
            return "duration_reached"
        return "stream_ended"

    source._run_segment = run_segment
    await source._supervise(await resolve_stream(P, format_selector=options.stream_format))
    assert reselects == ["stream_ended"]
    assert sorted(ytdlp.calls) == sorted([P, P, B])
    assert fed[1] is not fed[0]


async def test_a_failed_resolve_forgets_the_older_success(ytdlp: FakeYtDlp) -> None:
    """Otherwise a newer failure would be answered by an older success for up to the
    cache lifetime."""
    await resolve_stream(P, format_selector="best")
    ytdlp.live[P] = False
    with pytest.raises(StreamResolutionError):
        await resolve_stream(P, format_selector="best")
    assert resolver.recently_resolved(P, format_selector="best", max_age=30) is None
    # The next caller willing to reuse hears the failure, not the older success.
    with pytest.raises(StreamResolutionError):
        await resolve_stream(P, format_selector="best", max_age=30)
    assert ytdlp.calls == [P, P, P]


async def test_a_reuse_can_demand_an_answer_made_since_a_moment(ytdlp: FakeYtDlp) -> None:
    await resolve_stream(P, format_selector="best")
    assert resolver.recently_resolved(P, format_selector="best", max_age=30, since=time.time() - 60)
    assert (
        resolver.recently_resolved(P, format_selector="best", max_age=30, since=time.time() + 1)
        is None
    )
    await resolve_stream(P, format_selector="best", max_age=30, since=time.time() + 1)
    assert ytdlp.calls == [P, P]


async def test_the_cache_can_be_turned_off(ytdlp: FakeYtDlp) -> None:
    options = _options(resolve_cache_seconds=0)
    source, _ = _source(P, options)
    await source._try_resolve(max_age=options.resolve_cache_seconds)
    await source._next_stream_info(reason="ffmpeg_exit_1", is_live=True, empty_streak=0, attempt=1)
    assert ytdlp.calls == [P, P]


async def test_the_cache_is_keyed_by_everything_that_shapes_the_answer(
    ytdlp: FakeYtDlp, monkeypatch: pytest.MonkeyPatch
) -> None:
    await resolve_stream(P, format_selector="best")
    await resolve_stream(P, format_selector="best", max_age=30)
    assert ytdlp.calls == [P]
    await resolve_stream(P, format_selector="worst", max_age=30)
    await resolve_stream(P, format_selector="best", proxy="http://p.example.test:1", max_age=30)
    await resolve_stream(P, format_selector="best", cookiefile="/some/cookies.txt", max_age=30)
    assert ytdlp.calls == [P, P, P, P]
    # Too old: resolved "an hour ago".
    monkeypatch.setattr(resolver.time, "time", lambda: 1e12)
    await resolve_stream(P, format_selector="best", max_age=30)
    assert len(ytdlp.calls) == 5


async def test_the_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def extract(url: str, **_kw: Any) -> StreamInfo:
        return StreamInfo(url=url, is_live=True, media_url=url)

    monkeypatch.setattr(resolver, "resolve_stream_sync", extract)
    for i in range(resolver._RESOLVED_LIMIT + 5):
        await resolve_stream(f"https://x.test/{i}", format_selector="best")
    assert len(resolver._RESOLVED) == resolver._RESOLVED_LIMIT
    assert (
        resolver.recently_resolved("https://x.test/0", format_selector="best", max_age=30) is None
    )


async def test_local_files_are_never_cached(tmp_path: Any) -> None:
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x")
    await resolve_stream(str(clip), format_selector="best")
    assert not resolver._RESOLVED


async def test_failures_are_not_remembered(ytdlp: FakeYtDlp) -> None:
    ytdlp.live[P] = False
    for _ in range(2):
        with pytest.raises(StreamResolutionError):
            await resolve_stream(P, format_selector="best", max_age=30)
    assert ytdlp.calls == [P, P]


async def test_failed_re_resolves_log_one_short_warning_per_streak(
    ytdlp: FakeYtDlp, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reconnect through a bot check must not log "re-resolve failed" at ERROR with
    the whole extractor text."""
    caplog.set_level(logging.DEBUG)
    options = _options(resolve_cache_seconds=0)
    source, _ = _source(P, options)
    long_error = (
        "ERROR: [site] x: Sign in to confirm you're not a bot. " + "Use the cookies option. " * 200
    )

    def bot(url: str, **_kw: Any) -> StreamInfo:
        ytdlp.calls.append(url)
        raise StreamResolutionError(long_error)

    monkeypatch.setattr(resolver, "resolve_stream_sync", bot)
    for _ in range(4):
        assert await source._try_resolve(max_age=0) is None
    failed = [r for r in caplog.records if r.getMessage() == "re-resolve failed"]
    assert [r.levelno for r in failed] == [logging.WARNING] + [logging.DEBUG] * 3
    assert all(len(r.error) <= 160 for r in failed)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    monkeypatch.setattr(resolver, "resolve_stream_sync", ytdlp)
    ytdlp.live[P] = True
    assert await source._try_resolve(max_age=0) is not None
    ok = [r for r in caplog.records if r.getMessage() == "re-resolved stream"]
    assert ok[-1].after_failures == 4
    caplog.clear()
    ytdlp.live[P] = False
    await source._try_resolve(max_age=0)
    assert [r.levelno for r in caplog.records if r.getMessage() == "re-resolve failed"] == [
        logging.WARNING
    ]


# --------------------------------------------------------------------- connect()


async def test_connect_refuses_a_listed_live_source_that_resolves_as_a_recording(
    ytdlp: FakeYtDlp,
) -> None:
    """A probe said live, this resolve says the broadcast is over. Capturing it as a
    finite recording would push the whole post-live DVR through STT at decode speed."""
    ytdlp.not_live_once.add(P)
    source = LiveStreamSource(P, _options())
    source.require_live = True
    with pytest.raises(StreamNotLiveError, match="broadcast is over"):
        await source.connect()
    await source.close()


async def test_connect_picks_the_queue_policy_from_the_resolve(
    ytdlp: FakeYtDlp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_capture(_info: StreamInfo) -> None:
        return None

    options = _options(queue_size=5, file_queue_size=3)
    live = LiveStreamSource(P, options)
    monkeypatch.setattr(live, "_supervise", no_capture)
    await live.connect()
    assert live.lossless is False and live._audio_q.maxsize == 5
    await live.close()

    forced = LiveStreamSource(P, options, lossless=True)
    monkeypatch.setattr(forced, "_supervise", no_capture)
    await forced.connect()
    assert forced.lossless is True and forced._audio_q.maxsize == 3
    await forced.close()


async def test_a_finite_source_gets_the_lossless_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    async def no_capture(_info: StreamInfo) -> None:
        return None

    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x")
    finite = LiveStreamSource(str(clip), _options(queue_size=5, file_queue_size=3))
    monkeypatch.setattr(finite, "_supervise", no_capture)
    await finite.connect()
    assert finite.lossless is True and finite._audio_q.maxsize == 3
    await finite.close()

    dropping = LiveStreamSource(str(clip), _options(), lossless=False)
    monkeypatch.setattr(dropping, "_supervise", no_capture)
    await dropping.connect()
    assert dropping.lossless is False
    await dropping.close()
