"""Unit tests for the HLS live window. Offline, no ffmpeg; only loopback HTTP."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import urllib.request

import pytest

from livestream_transcriber.stream import ffmpeg, hls_window
from livestream_transcriber.stream.hls_window import (
    DEFAULT_KEEP,
    HlsLiveWindow,
    fetch_live_edge,
    make_playlist_fetcher,
    parse_m3u8,
    prepare_hls_input,
    render_window,
)


def _playlist(*, n: int = 50, seq: int = 100, prefix: str = "https://media.example/seg") -> str:
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-TARGETDURATION:1",
        f"#EXT-X-MEDIA-SEQUENCE:{seq}",
        "#EXT-X-PLAYLIST-TYPE:EVENT",
    ]
    for i in range(n):
        lines.append("#EXTINF:1.000,")
        lines.append(f"{prefix}/sq/{seq + i}/dur/1.000/file/seg.ts")
    return "\n".join(lines) + "\n"


def _slim(text: str, url: str, *, keep: int) -> str:
    body, _ = render_window(parse_m3u8(text, url), keep=keep)
    return body


def test_looks_like_hls() -> None:
    looks_like_hls = ffmpeg.looks_like_hls
    assert looks_like_hls("https://x/playlist/index.m3u8")
    assert looks_like_hls("/tmp/window.m3u8")
    assert looks_like_hls("https://cdn.example/api/manifest/hls_playlist/id/x")
    assert not looks_like_hls("/tmp/clip.mp4")
    assert not looks_like_hls(None)


def test_window_and_ffmpeg_command_share_one_hls_heuristic() -> None:
    """Both decide "is this HLS?" the same way, and no segment URL is synthesised:
    guessing the next ``/sq/N+1/`` URL keeps a stale signature and hangs TLS."""
    assert hls_window.looks_like_hls is ffmpeg.looks_like_hls
    for name in ("increment_sq_url", "slim_playlist", "_sq_of", "_SQ_RE"):
        assert not hasattr(hls_window, name), name
    assert "extra_uris" not in render_window.__code__.co_varnames


def test_render_window_refresh_stamp_changes_body() -> None:
    parsed = parse_m3u8(_playlist(n=6, seq=1), "https://x/index.m3u8")
    a, _ = render_window(parsed, keep=4, refresh_stamp=1)
    b, _ = render_window(parsed, keep=4, refresh_stamp=2)
    assert a != b
    assert "#EXT-X-LST-REFRESH:1" in a
    assert "#EXT-X-LST-REFRESH:2" in b


def test_slim_playlist_keeps_only_the_live_edge() -> None:
    slim = _slim(_playlist(n=50, seq=200), "https://media.example/index.m3u8", keep=8)
    assert slim.startswith("#EXTM3U")
    assert "#EXT-X-ENDLIST" not in slim
    assert "#EXT-X-PLAYLIST-TYPE" not in slim
    assert "#EXT-X-MEDIA-SEQUENCE:242" in slim  # 200 + (50 - 8)
    assert slim.count("#EXTINF") == 8
    assert "/sq/249/" in slim
    assert "/sq/200/" not in slim


def test_relative_segment_uris_become_absolute() -> None:
    raw = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:1\n#EXTINF:2.0,\nseg1.ts\n"
    slim = _slim(raw, "https://cdn.example/live/index.m3u8", keep=8)
    assert "https://cdn.example/live/seg1.ts" in slim


def test_parse_strips_endlist() -> None:
    parsed = parse_m3u8(_playlist(n=4, seq=1) + "#EXT-X-ENDLIST\n", "https://x/index.m3u8")
    assert len(parsed.segments) == 4
    assert parsed.media_sequence == 1
    assert parsed.target_duration == 1.0


def _tagged_playlist(n: int, seq: int, tags_before: dict[int, list[str]]) -> str:
    """``n`` segments with a PROGRAM-DATE-TIME each, plus extra tags in front of the
    segments named in ``tags_before`` (index -> lines)."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-TARGETDURATION:1",
        f"#EXT-X-MEDIA-SEQUENCE:{seq}",
    ]
    for i in range(n):
        lines.extend(tags_before.get(i, []))
        lines.append(f"#EXT-X-PROGRAM-DATE-TIME:2026-01-02T12:{i // 60:02d}:{i % 60:02d}.000Z")
        lines.append("#EXTINF:1.000,")
        lines.append(f"https://media.example/seg{seq + i}.ts")
    return "\n".join(lines) + "\n"


def _header_and_segments(body: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split a rendered window into its header (up to the sequence lines, which
    render_window writes last) and each segment URI with its tags."""
    lines = [ln for ln in body.splitlines() if ln.strip("#")]  # drop the size pad
    end = max(
        i
        for i, ln in enumerate(lines)
        if ln.startswith(("#EXT-X-MEDIA-SEQUENCE", "#EXT-X-DISCONTINUITY-SEQUENCE"))
    )
    header, segments, pending = lines[: end + 1], {}, []
    for ln in lines[end + 1 :]:
        if ln.startswith("#"):
            pending.append(ln)
        else:
            segments[ln], pending = pending, []
    return header, segments


def test_mid_playlist_discontinuity_stays_with_its_segment() -> None:
    """A DISCONTINUITY marks the segment after it. A parser that moves any tag found
    between segments into the header announces the discontinuity before the first
    segment and drops it from the right one."""
    raw = _tagged_playlist(10, 100, {7: ["#EXT-X-DISCONTINUITY"]})
    body = _slim(raw, "https://media.example/index.m3u8", keep=5)
    header, segments = _header_and_segments(body)
    assert list(segments) == [f"https://media.example/seg{n}.ts" for n in range(105, 110)]
    assert not any(ln.startswith(("#EXT-X-DISCONTINUITY", "#EXT-X-PROGRAM-DATE")) for ln in header)
    assert segments["https://media.example/seg107.ts"] == [
        "#EXT-X-DISCONTINUITY",
        "#EXT-X-PROGRAM-DATE-TIME:2026-01-02T12:00:07.000Z",
        "#EXTINF:1.000,",
    ]
    for n in (105, 106, 108, 109):
        assert segments[f"https://media.example/seg{n}.ts"] == [
            f"#EXT-X-PROGRAM-DATE-TIME:2026-01-02T12:00:{n - 100:02d}.000Z",
            "#EXTINF:1.000,",
        ]


def test_window_stays_slim_with_per_segment_tags() -> None:
    """A one-hour DVR playlist with a date-time per segment still renders a window
    of ``keep`` segments, not a 3600-line header."""
    body = _slim(
        _tagged_playlist(3600, 1, {}), "https://media.example/index.m3u8", keep=DEFAULT_KEEP
    )
    assert body.count("#EXT-X-PROGRAM-DATE-TIME") == DEFAULT_KEEP
    assert len(body.splitlines()) <= 3 * DEFAULT_KEEP + 8


def test_dropped_discontinuities_advance_the_discontinuity_sequence() -> None:
    raw = _tagged_playlist(
        10, 100, {2: ["#EXT-X-DISCONTINUITY"], 7: ["#EXT-X-DISCONTINUITY"]}
    ).replace(
        "#EXT-X-MEDIA-SEQUENCE:100", "#EXT-X-MEDIA-SEQUENCE:100\n#EXT-X-DISCONTINUITY-SEQUENCE:4"
    )
    body = _slim(raw, "https://media.example/index.m3u8", keep=5)
    header, segments = _header_and_segments(body)
    # Segment 102's discontinuity left the window: the first kept segment is in
    # discontinuity sequence 5; 107's is still announced in place.
    assert [ln for ln in header if "DISCONTINUITY" in ln] == ["#EXT-X-DISCONTINUITY-SEQUENCE:5"]
    assert "#EXT-X-MEDIA-SEQUENCE:105" in header
    assert segments["https://media.example/seg107.ts"][0] == "#EXT-X-DISCONTINUITY"
    assert sum(tags.count("#EXT-X-DISCONTINUITY") for tags in segments.values()) == 1


def test_key_changes_keep_their_segments() -> None:
    """A KEY applies from its segment on. One inside the window stays there; one on
    a dropped segment moves onto the first kept segment."""
    k1 = '#EXT-X-KEY:METHOD=AES-128,URI="https://keys.example/1"'
    k2 = '#EXT-X-KEY:METHOD=AES-128,URI="https://keys.example/2"'
    k3 = '#EXT-X-KEY:METHOD=AES-128,URI="https://keys.example/3"'
    raw = _tagged_playlist(10, 100, {0: [k1], 3: [k2], 7: [k3]})
    header, segments = _header_and_segments(_slim(raw, "https://media.example/index.m3u8", keep=5))
    assert [ln for ln in header if ln.startswith("#EXT-X-KEY")] == [k1]
    keys = {uri: [t for t in tags if t.startswith("#EXT-X-KEY")] for uri, tags in segments.items()}
    assert keys == {
        "https://media.example/seg105.ts": [k2],
        "https://media.example/seg106.ts": [],
        "https://media.example/seg107.ts": [k3],
        "https://media.example/seg108.ts": [],
        "https://media.example/seg109.ts": [],
    }


def test_live_edge_of_a_fetched_playlist() -> None:
    raw = _playlist(n=10, seq=40).encode()
    assert fetch_live_edge("https://x/a.m3u8", lambda _url: raw) == 49
    with pytest.raises(ValueError, match="no segments"):
        fetch_live_edge("https://x/a.m3u8", lambda _url: b"#EXTM3U\n")


# Refresh interval for armed windows in these tests: fast enough that a few hundred
# milliseconds cover many refresh ticks (production: 2 s).
_TICK_S = 0.01


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.002)


def _uris(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("http")]


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


async def test_every_served_refresh_is_a_whole_window() -> None:
    """A GET that races a refresh gets the old window or the new one, whole.

    A window rewritten in place (seek, write, truncate) lets a GET land in between
    and read the new head glued to the old tail."""
    sig = "/sig/" + "A" * 700 + "/file/seg.ts"  # signed CDN URLs run about 1 KB

    def playlist(seq: int, n: int) -> bytes:
        text = _playlist(n=n, seq=seq, prefix="https://cdn.example/vp")
        return text.replace("/file/seg.ts", sig).encode()

    window = HlsLiveWindow(
        "https://x/audio.m3u8", fetcher=lambda _url: playlist(1000, 50), name="audio"
    )
    await window.start()
    server = hls_window._LocalPlaylistServer({"/audio/index.m3u8": window.path})
    url = server.url_for("/audio/index.m3u8")
    stop = threading.Event()
    reads: list[str] = []

    def reader() -> None:
        while not stop.is_set() and len(reads) < 150:
            with urllib.request.urlopen(url, timeout=2) as resp:
                reads.append(resp.read().decode())

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        n = 0
        deadline = time.monotonic() + 10.0
        while thread.is_alive() and time.monotonic() < deadline:
            n += 1
            # Alternate a full window with a short one: sizes differ a lot.
            window._ingest(playlist(1000 + 10 * n, 50 if n % 2 else 5), force=True)
            window._write()
    finally:
        stop.set()
        thread.join()
        server.close()
        await window.aclose()

    assert len(reads) >= 100
    torn = []
    for body in reads:
        lines = body.split("\n")
        whole = (
            lines[0] == "#EXTM3U"
            and lines[-1] == ""
            and set(lines[-2]) == {"#"}  # the size pad closes every window
            and len(_uris(body)) == body.count("#EXTINF") in (DEFAULT_KEEP, 5)
        )
        if not whole:
            torn.append(len(body))
    assert torn == []


async def test_close_waits_for_a_refresh_write_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling the refresh task does not stop its worker thread. Closing must not
    remove the directory under a write that is still running (the rename-based write
    creates a temp file there)."""
    state = {"seq": 10}

    def fetch(_url: str) -> bytes:
        state["seq"] += 1
        return _playlist(n=12, seq=state["seq"]).encode()

    window = HlsLiveWindow(
        "https://media.example/index.m3u8",
        keep=4,
        fetcher=fetch,
        name="test",
        refresh_s=_TICK_S,
    )
    path = await window.start()
    entered, release = threading.Event(), threading.Event()
    real_mkstemp = tempfile.mkstemp

    def slow_mkstemp(*args, **kwargs):
        entered.set()
        release.wait(5)
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(hls_window.tempfile, "mkstemp", slow_mkstemp)
    window.arm()
    try:
        await _until(entered.is_set)  # a refresh is inside its write now
        closing = asyncio.create_task(window.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done()
    finally:
        release.set()
    await asyncio.wait_for(closing, 5)
    assert not os.path.exists(os.path.dirname(path))
    before = state["seq"]
    await asyncio.sleep(5 * _TICK_S)
    assert state["seq"] == before  # nothing refreshes after close


async def test_window_does_not_advance_until_armed() -> None:
    raw = _playlist(n=12, seq=10)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return raw.encode()

    window = HlsLiveWindow(
        "https://media.example/index.m3u8",
        keep=4,
        fetcher=fetch,
        name="test",
        refresh_s=_TICK_S,
    )
    path = await window.start()
    try:
        before = _read(path)
        await asyncio.sleep(15 * _TICK_S)
        assert calls == ["https://media.example/index.m3u8"]
        assert _read(path) == before
        window.arm()
        await _until(lambda: len(calls) >= 3)
    finally:
        await window.aclose()


async def test_older_refresh_is_ignored() -> None:
    playlists = [
        _playlist(n=8, seq=100).encode(),
        _playlist(n=8, seq=90).encode(),  # older edge
        _playlist(n=8, seq=108).encode(),
    ]
    state = {"i": 0}
    newer_may_arrive = threading.Event()

    def fetch(_url: str) -> bytes:
        i = state["i"]
        state["i"] += 1
        if i >= 2:
            newer_may_arrive.wait(5)
        return playlists[min(i, len(playlists) - 1)]

    window = HlsLiveWindow(
        "https://media.example/index.m3u8",
        keep=4,
        fetcher=fetch,
        name="test",
        refresh_s=_TICK_S,
    )
    path = await window.start()
    window.arm()
    try:
        assert "/sq/107/" in _read(path)
        # The third fetch has begun, so the older snapshot was fully handled.
        await _until(lambda: state["i"] >= 3)
        mid = _read(path)
        assert "/sq/107/" in mid  # older snapshot ignored
        assert "/sq/97/" not in mid
        assert window.edge == 107
        newer_may_arrive.set()
        await _until(lambda: "/sq/115/" in _read(path))
        assert window.edge == 115
    finally:
        newer_may_arrive.set()
        await window.aclose()


async def test_window_refresh_rewrites_from_fetcher() -> None:
    state = {"n": 0}

    def fetch(_url: str) -> bytes:
        seq = 10 + state["n"] * 4
        state["n"] += 1
        return _playlist(n=12, seq=seq).encode()

    window = HlsLiveWindow(
        "https://media.example/index.m3u8",
        keep=4,
        fetcher=fetch,
        name="test",
        refresh_s=_TICK_S,
    )
    path = await window.start()
    window.arm()
    try:
        text = _read(path)
        assert text.count("#EXTINF") == 4
        assert window.remote_segments == 12
        last_before = _uris(text)[-1]
        await _until(lambda: _uris(_read(path))[-1] != last_before)
        assert _read(path).count("#EXTINF") == 4
    finally:
        await window.aclose()
    assert window.path == ""


async def test_window_directory_uses_a_neutral_prefix() -> None:
    window = HlsLiveWindow(
        "https://x/a.m3u8", fetcher=lambda _url: _playlist(n=6).encode(), name="audio"
    )
    path = await window.start()
    try:
        assert os.path.basename(os.path.dirname(path)).startswith("lst-hls-audio-")
    finally:
        await window.aclose()


async def test_prepare_hls_input_is_a_noop_when_not_live() -> None:
    prepared = await prepare_hls_input("https://x/a.m3u8", is_live=False)
    assert prepared.url == "https://x/a.m3u8"
    assert not prepared.windowed
    assert prepared.live_edges() == (None, None)
    await prepared.aclose()


async def test_prepare_hls_input_skips_non_hls_urls() -> None:
    prepared = await prepare_hls_input("https://x/stream.mp3", is_live=True)
    assert not prepared.windowed
    assert prepared.url == "https://x/stream.mp3"


async def test_prepare_hls_input_serves_a_local_window() -> None:
    def fetch(_url: str) -> bytes:
        return _playlist(n=80, seq=5, prefix="https://a.example/seg").encode()

    prepared = await prepare_hls_input("https://x/audio.m3u8", is_live=True, fetcher=fetch)
    try:
        assert prepared.windowed
        assert prepared.url.startswith("http://127.0.0.1:")
        assert prepared.url.endswith("/audio/index.m3u8")
        first, latest = prepared.live_edges()
        assert first == latest == 84
        with urllib.request.urlopen(prepared.url, timeout=2) as resp:
            body = resp.read().decode()
        assert body.count("#EXTINF") == DEFAULT_KEEP
    finally:
        await prepared.aclose()
    assert not prepared.windowed


async def test_prepare_hls_input_falls_back_when_the_fetch_fails() -> None:
    def boom(_url: str) -> bytes:
        raise TimeoutError("playlist fetch failed")

    prepared = await prepare_hls_input("https://x/audio.m3u8", is_live=True, fetcher=boom)
    try:
        assert not prepared.windowed
        assert prepared.url == "https://x/audio.m3u8"
    finally:
        await prepared.aclose()


def test_window_keep_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        HlsLiveWindow("https://x/a.m3u8", keep=1)


def test_refresh_interval_must_be_positive() -> None:
    with pytest.raises(ValueError):
        HlsLiveWindow("https://x/a.m3u8", refresh_s=0)


def test_playlist_fetcher_uses_the_http_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Resp:
        headers: dict[str, str] = {}

        def read(self) -> bytes:
            return b"#EXTM3U\n"

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    class Opener:
        def open(self, req, timeout=None):
            captured["url"] = req.full_url
            return Resp()

    def build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr("livestream_transcriber.stream.hls_window.build_opener", build_opener)
    proxy = "http://user:your-proxy-password@proxy.example.test:8877"
    fetch = make_playlist_fetcher(proxy)
    assert fetch("https://example.test/a.m3u8") == b"#EXTM3U\n"
    handler = captured["handlers"][0]
    assert handler.proxies["https"] == proxy
    assert captured["url"] == "https://example.test/a.m3u8"


def test_default_fetcher_goes_through_the_guarded_urlopen() -> None:
    """A remote fetch is refused by the suite's network guard, which proves the
    fetcher looks ``urlopen`` up at call time instead of binding it on import."""
    with pytest.raises(OSError, match="network blocked"):
        make_playlist_fetcher(None)("https://media.example/a.m3u8")
