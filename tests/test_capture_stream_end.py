"""When a live capture must stop reconnecting.

* A playlist the platform still lists but no longer extends: every reconnect would replay
  its last seconds, stall, "produce data", reset the back-off and go round again,
  forever, re-transcribing the same few seconds. Ending the capture instead would only
  move that loop up a level (a process supervisor restarts it into the same source), so
  a frozen playlist is *held*: no ffmpeg, one playlist GET per poll, until it moves or
  the stream is over.
* A re-resolve that says the stream is no longer live: the platform then serves the
  recording, and the capture would play hours of it at full decode speed through STT.

Offline: stub ffmpeg, patched playlist fetch, patched resolve. The frozen-playlist logic
needs the opt-in HLS window, which is what supplies the playlist's live edge.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from livestream_transcriber.config import Settings
from livestream_transcriber.models import SegmentInfo, StreamInfo
from livestream_transcriber.stream import hls_window
from livestream_transcriber.stream import source as source_mod
from livestream_transcriber.stream.hls_window import HlsInput, HlsLiveWindow
from livestream_transcriber.stream.source import (
    CaptureOptions,
    LiveStreamSource,
    playlist_stalled,
    should_attempt_reselect,
)
from tests.support.ffmpeg_stubs import stub_stalled

PRIMARY = "https://streams.example.test/primary/live"
BACKUP = "https://streams.example.test/backup/live"


def _playlist(seq: int, n: int = 8) -> bytes:
    lines = ["#EXTM3U", "#EXT-X-TARGETDURATION:1", f"#EXT-X-MEDIA-SEQUENCE:{seq}"]
    for i in range(n):
        lines += ["#EXTINF:1.000,", f"https://cdn.example.test/sq/{seq + i}/seg.ts"]
    return ("\n".join(lines) + "\n").encode()


def _live(url: str = PRIMARY, *, is_live: bool = True) -> StreamInfo:
    return StreamInfo(
        url=url,
        is_live=is_live,
        title="post-live recording" if not is_live else "live",
        media_url=f"{url}/audio/index.m3u8",
    )


def _options(tmp_path: Path, **kw: object) -> CaptureOptions:
    base: dict[str, object] = {
        "ffmpeg_binary": stub_stalled(tmp_path),
        "audio_stall_seconds": 0.3,
        "live_chunk_seconds": 0.1,  # the stub writes 0.1 s blocks: one chunk each
        "reconnect_initial_delay": 0.01,
        "reconnect_max_delay": 0.01,
        "hls_window": True,
    }
    base.update(kw)
    return CaptureOptions(**base)


class _Playlists:
    """The platform's remote playlists: each stays frozen at its media sequence (500
    unless set) until a test moves it."""

    def __init__(self) -> None:
        self.seq: dict[str, int] = {}
        self.fetches = 0

    def __call__(self, url: str, timeout: float = 0) -> bytes:
        self.fetches += 1
        return _playlist(self.seq.get(url, 500))


async def _until(condition: Callable[[], object], timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
        assert loop.time() < end, "condition not reached"
        await asyncio.sleep(0.01)


def _start(source: LiveStreamSource, initial: StreamInfo) -> asyncio.Task[None]:
    # What connect() does after its resolve; close() then cancels this task.
    source._supervisor = asyncio.create_task(source._supervise(initial))
    return source._supervisor


class _Harness:
    """Counts resolves; after ``guard`` of them, closes the source (so an endless loop
    ends and the test can assert on how far it got)."""

    def __init__(
        self, source: LiveStreamSource, infos: Callable[[str], StreamInfo], guard: int = 6
    ) -> None:
        self.source = source
        self.infos = infos
        self.guard = guard
        self.resolves = 0
        self.reselects: list[str] = []

    async def resolve(self, url: str, *_a: object, **_k: object) -> StreamInfo:
        self.resolves += 1
        if self.resolves >= self.guard:
            self.source._closing.set()
        return self.infos(url)

    async def reselect(self, current: str, reason: str, attempt: int) -> str | None:
        self.reselects.append(reason)
        return None


def _held(source: LiveStreamSource, harness: _Harness, rechecks: int) -> Callable[[], bool]:
    """Held with ``rechecks`` selector asks behind it; or, if it did not hold, still
    reconnecting into ever more segments."""
    return lambda: harness.reselects.count("stale_playlist") >= rechecks or len(source.segments) > 4


async def test_frozen_playlist_is_held_without_ffmpeg_until_it_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    playlists = _Playlists()
    monkeypatch.setattr(hls_window, "_default_fetch", playlists)
    holds: list[bool] = []
    source = LiveStreamSource(PRIMARY, _options(tmp_path), on_hold=holds.append)
    harness = _Harness(source, lambda url: _live(url), guard=10_000)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    try:
        task = _start(source, _live())
        # Several selector rechecks while held: no new segment, nothing decoded or
        # transcribed again, and the process stays up.
        await _until(_held(source, harness, rechecks=3))
        assert [s.reason for s in source.segments] == ["audio_stalled", "audio_stalled"]
        assert all(s.audio_chunks == 3 for s in source.segments)
        polls = playlists.fetches
        await _until(lambda: playlists.fetches >= polls + 5)
        assert len(source.segments) == 2
        assert not task.done()
        assert harness.resolves >= 3
        # A hold has no audio to prove liveness with: the hook lets a watchdog elsewhere
        # know that silence is expected.
        assert holds == [True]

        playlists.seq[_live().media_url or ""] = 530  # the encoder is back
        await _until(lambda: len(source.segments) >= 3)
        assert holds[:2] == [True, False]
    finally:
        await source.close()


async def test_a_failing_hold_hook_does_not_break_the_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    playlists = _Playlists()
    monkeypatch.setattr(hls_window, "_default_fetch", playlists)

    def broken(_holding: bool) -> None:
        raise RuntimeError("hook bug")

    source = LiveStreamSource(PRIMARY, _options(tmp_path), on_hold=broken)
    harness = _Harness(source, lambda url: _live(url), guard=10_000)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    task = _start(source, _live())
    try:
        await _until(_held(source, harness, rechecks=2))
        assert not task.done()
    finally:
        await source.close()


async def test_a_held_stream_that_stops_being_live_ends_the_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hls_window, "_default_fetch", _Playlists())
    live = {"now": True}
    source = LiveStreamSource(PRIMARY, _options(tmp_path))
    harness = _Harness(source, lambda url: _live(url, is_live=live["now"]), guard=10_000)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    task = _start(source, _live())
    try:
        await _until(_held(source, harness, rechecks=2))
        assert len(source.segments) == 2
        live["now"] = False  # the broadcast ended; the URL now serves the recording
        await asyncio.wait_for(asyncio.shield(task), 20)
        assert len(source.segments) == 2  # not one second of the recording
        assert not source._closing.is_set()  # it ended by itself
    finally:
        await source.close()


async def test_a_held_capture_follows_the_selector_to_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hls_window, "_default_fetch", _Playlists())
    source = LiveStreamSource(PRIMARY, _options(tmp_path))
    harness = _Harness(source, lambda url: _live(url), guard=10_000)
    target: dict[str, str | None] = {"url": None}

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        harness.reselects.append(reason)
        return target["url"]

    source.reselect = reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    _start(source, _live())
    try:
        await _until(_held(source, harness, rechecks=2))
        assert len(source.segments) == 2
        target["url"] = BACKUP
        await _until(lambda: len(source.segments) >= 3)
        assert source.url == BACKUP and source.info is not None and source.info.url == BACKUP
    finally:
        await source.close()


async def test_a_new_broadcast_behind_the_same_url_resumes_the_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen broadcast stays frozen; the URL now resolves to a new one whose
    playlist starts over at a lower media sequence."""
    playlists = _Playlists()
    monkeypatch.setattr(hls_window, "_default_fetch", playlists)
    broadcast = {"n": 1}

    def infos(url: str) -> StreamInfo:
        return StreamInfo(
            url=url, is_live=True, media_url=f"{url}/b{broadcast['n']}/audio/index.m3u8"
        )

    source = LiveStreamSource(PRIMARY, _options(tmp_path))
    harness = _Harness(source, infos, guard=10_000)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    _start(source, infos(PRIMARY))
    try:
        await _until(_held(source, harness, rechecks=2))
        assert len(source.segments) == 2
        playlists.seq[f"{PRIMARY}/b2/audio/index.m3u8"] = 0
        broadcast["n"] = 2
        await _until(lambda: len(source.segments) >= 3)
        assert source.info is not None and (source.info.media_url or "").endswith(
            "/b2/audio/index.m3u8"
        )
    finally:
        await source.close()


async def test_stalls_on_a_moving_playlist_keep_reconnecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def advancing(url: str, timeout: float = 0) -> bytes:
        calls["n"] += 1
        return _playlist(500 + 10 * calls["n"])

    monkeypatch.setattr(hls_window, "_default_fetch", advancing)
    source = LiveStreamSource(PRIMARY, _options(tmp_path))
    harness = _Harness(source, lambda url: _live(url), guard=4)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)

    await asyncio.wait_for(source._supervise(_live()), timeout=60)

    assert len(source.segments) == 4  # stopped by the guard, not by us
    assert "stale_playlist" not in harness.reselects
    source._close_queue()


async def test_the_stall_limit_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``stale_playlist_stalls=0`` a frozen playlist is reconnected forever, as it
    would be without the hold logic."""
    monkeypatch.setattr(hls_window, "_default_fetch", _Playlists())
    source = LiveStreamSource(PRIMARY, _options(tmp_path, stale_playlist_stalls=0))
    harness = _Harness(source, lambda url: _live(url), guard=4)
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)
    await asyncio.wait_for(source._supervise(_live()), timeout=60)
    assert len(source.segments) == 4
    assert "stale_playlist" not in harness.reselects
    source._close_queue()


async def test_no_longer_live_ends_capture_instead_of_replaying_the_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    played: list[bool] = []
    source = LiveStreamSource(
        PRIMARY, CaptureOptions(reconnect_initial_delay=0.01, reconnect_max_delay=0.01)
    )
    harness = _Harness(source, lambda url: _live(url, is_live=False))
    source.reselect = harness.reselect
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        played.append(info.is_live)
        segment.audio_chunks = 12
        return "audio_stalled"

    source._run_segment = run_segment
    await asyncio.wait_for(source._supervise(_live()), timeout=60)

    assert played == [True]
    assert harness.reselects == ["live_ended"]
    assert harness.resolves == 1
    source._close_queue()


async def test_no_longer_live_fails_over_when_the_backup_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    played: list[tuple[str, bool]] = []
    source = LiveStreamSource(
        PRIMARY, CaptureOptions(reconnect_initial_delay=0.01, reconnect_max_delay=0.01)
    )
    harness = _Harness(source, lambda url: _live(url, is_live=(url == BACKUP)), guard=4)
    monkeypatch.setattr(source_mod, "resolve_stream", harness.resolve)

    async def reselect(current: str, reason: str, attempt: int) -> str | None:
        harness.reselects.append(reason)
        return BACKUP

    source.reselect = reselect

    async def run_segment(info: StreamInfo, segment: SegmentInfo, _remaining: float | None) -> str:
        played.append((info.url, info.is_live))
        segment.audio_chunks = 12
        if len(played) > 1:
            source._closing.set()
            return "duration_reached"
        return "audio_stalled"

    source._run_segment = run_segment
    await asyncio.wait_for(source._supervise(_live()), timeout=60)

    assert played == [(PRIMARY, True), (BACKUP, True)]
    assert harness.reselects == ["live_ended"]
    source._close_queue()


def test_a_frozen_edge_is_only_claimed_for_an_audio_stall() -> None:
    assert playlist_stalled("audio_stalled", 100, 100, None)
    assert playlist_stalled("audio_stalled", 100, 100, 100)
    assert not playlist_stalled("audio_stalled", 100, 104, None)  # moved during
    assert not playlist_stalled("audio_stalled", 105, 105, 100)  # moved since the last
    assert not playlist_stalled("ffmpeg_exit_1", 100, 100, 100)
    assert not playlist_stalled("audio_stalled", None, None, 100)  # no window


def test_a_stale_playlist_asks_the_selector() -> None:
    assert should_attempt_reselect(reason="stale_playlist", is_live=True)
    assert not should_attempt_reselect(reason="stale_playlist", is_live=False)


async def test_window_reports_its_live_edge() -> None:
    playlists = iter([_playlist(100), _playlist(90), _playlist(104)])
    window = HlsLiveWindow(
        "https://cdn.example.test/audio/index.m3u8",
        keep=4,
        fetcher=lambda _url: next(playlists),
        name="audio",
    )
    await window.start()
    try:
        assert (window.first_edge, window.edge) == (107, 107)
        window._refresh_blocking()  # an older snapshot: ignored
        assert window.edge == 107
        window._refresh_blocking()
        assert (window.first_edge, window.edge) == (107, 111)
        assert HlsInput(window.path, window).live_edges() == (107, 111)
    finally:
        await window.aclose()
    assert HlsInput("u").live_edges() == (None, None)
    # The held capture's poll reads the same edge from one GET.
    assert hls_window.fetch_live_edge(window.url, lambda _url: _playlist(104)) == 111
    with pytest.raises(ValueError):
        hls_window.fetch_live_edge(window.url, lambda _url: b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:9\n")


def test_stall_limit_is_configurable_and_validated() -> None:
    assert CaptureOptions().stale_playlist_stalls == 2
    assert Settings(capture_stale_playlist_stalls=0).capture_stale_playlist_stalls == 0
    assert (
        CaptureOptions.from_settings(
            Settings(capture_stale_playlist_stalls=5)
        ).stale_playlist_stalls
        == 5
    )
    with pytest.raises(ValueError):
        Settings(capture_stale_playlist_stalls=-1)
