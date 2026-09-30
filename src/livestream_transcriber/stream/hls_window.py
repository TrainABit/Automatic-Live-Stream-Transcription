"""Keep ffmpeg on the live edge of a huge HLS DVR playlist (opt-in).

Some live platforms serve a DVR playlist listing thousands of one-second segments,
several MB of text. ffmpeg's HLS demuxer re-downloads the whole playlist on every
reload, which can take longer than the segments it lists are worth.

This module fetches the remote playlist (gzip, about a second), writes a local
window of the last N *real* segment URLs, and refreshes that window in the
background. ffmpeg still does the decode; it just reloads a file of a few dozen
kilobytes from a loopback HTTP server. Every URL in the window is one the platform
listed and signed: synthesising the next segment URL was tried and dropped, since
those keep a stale signature. Older DVR snapshots are ignored so MEDIA-SEQUENCE
never rewinds.

It is off by default (``capture_hls_window``): a generic HLS stream has a short
playlist and gains nothing from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import urljoin
from urllib.request import ProxyHandler, Request, build_opener

from ..logging_setup import get_logger
from .ffmpeg import looks_like_hls

log = get_logger(__name__)

__all__ = [
    "DEFAULT_KEEP",
    "USER_AGENT",
    "HlsInput",
    "HlsLiveWindow",
    "ParsedPlaylist",
    "fetch_live_edge",
    "live_edge",
    "make_playlist_fetcher",
    "parse_m3u8",
    "prepare_hls_input",
    "render_window",
]

# A window this short is still tiny next to a multi-MB DVR playlist, yet long enough
# that ffmpeg's demuxer cannot fall behind the oldest kept segment between reloads.
DEFAULT_KEEP = 45
_FETCH_TIMEOUT_S = 45.0
_REFRESH_S = 2.0
# Some CDNs throttle ffmpeg's default Lavf user agent. Playlist fetches and segment
# requests use a browser-like one instead (see FFmpegSpec.user_agent).
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _read_playlist_response(resp: Any) -> bytes:
    data: bytes = resp.read()
    encoding = (resp.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def _playlist_request(url: str) -> Request:
    return Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        },
    )


def _default_fetch(url: str, timeout: float = _FETCH_TIMEOUT_S) -> bytes:
    # Looked up at call time so tests can guard urllib.request.urlopen.
    with urllib.request.urlopen(_playlist_request(url), timeout=timeout) as resp:
        return _read_playlist_response(resp)


def make_playlist_fetcher(proxy: str | None) -> Callable[[str], bytes]:
    """Fetch playlists directly, or through ``proxy`` when one is set.

    The opener is built per call site so the proxy is not installed as the process
    default: STT and webhook traffic must stay on the direct path. Only HTTP
    proxies are supported, since ffmpeg's http client and urllib speak nothing else.
    """
    if not proxy:
        return _default_fetch
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))

    def fetch(url: str, timeout: float = _FETCH_TIMEOUT_S) -> bytes:
        with opener.open(_playlist_request(url), timeout=timeout) as resp:
            return _read_playlist_response(resp)

    return fetch


# Tags about the playlist as a whole; they stay in the header of every window.
_PLAYLIST_TAGS = frozenset(
    {
        "#EXT-X-VERSION",
        "#EXT-X-TARGETDURATION",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        "#EXT-X-START",
        "#EXT-X-SERVER-CONTROL",
        "#EXT-X-PART-INF",
        "#EXT-X-ALLOW-CACHE",
        "#EXT-X-I-FRAMES-ONLY",
    }
)
# Tags that belong to the segment after them, wherever they appear. Hoisted into
# the header, a mid-playlist DISCONTINUITY would lose its place.
_SEGMENT_TAGS = frozenset(
    {
        "#EXTINF",
        "#EXT-X-BYTERANGE",
        "#EXT-X-DISCONTINUITY",
        "#EXT-X-PROGRAM-DATE-TIME",
        "#EXT-X-DATERANGE",
        "#EXT-X-GAP",
        "#EXT-X-BITRATE",
        "#EXT-X-PART",
        "#EXT-X-CUE-OUT",
        "#EXT-X-CUE-OUT-CONT",
        "#EXT-X-CUE-IN",
    }
)
# Segment tags that stay in force for every later segment until repeated. Before the
# first segment they go in the header; a later one moves onto the first kept segment
# when the window drops the segment that carried it.
_STICKY_TAGS = ("#EXT-X-KEY", "#EXT-X-MAP")

# A private tag that changes on every rewrite. It is ignored by players, and it
# forces a visible difference between two windows of the same length.
_REFRESH_TAG = "#EXT-X-LST-REFRESH"


def _tag_name(line: str) -> str:
    return line.split(":", 1)[0]


@dataclass(frozen=True, slots=True)
class ParsedPlaylist:
    header: tuple[str, ...]
    media_sequence: int
    target_duration: float
    segments: tuple[tuple[tuple[str, ...], str], ...]
    discontinuity_sequence: int | None = None


def parse_m3u8(text: str, playlist_url: str) -> ParsedPlaylist:
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    header: list[str] = []
    segments: list[tuple[tuple[str, ...], str]] = []
    pending: list[str] = []
    media_sequence = 0
    discontinuity_sequence: int | None = None
    target_duration = 1.0
    for line in lines:
        if not line:
            continue
        if not line.startswith("#"):
            segments.append((tuple(pending), urljoin(playlist_url, line)))
            pending = []
            continue
        name = _tag_name(line)
        if name in ("#EXTM3U", "#EXT-X-ENDLIST"):
            continue
        if name == "#EXT-X-PLAYLIST-TYPE":
            # A VOD/EVENT type tag would make ffmpeg treat the slim file as finite.
            continue
        if name in ("#EXT-X-MEDIA-SEQUENCE", "#EXT-X-DISCONTINUITY-SEQUENCE"):
            try:
                value = int(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                continue
            if name == "#EXT-X-MEDIA-SEQUENCE":
                media_sequence = value
            else:
                discontinuity_sequence = value
            continue
        if name == "#EXT-X-TARGETDURATION":
            with contextlib.suppress(IndexError, ValueError):
                target_duration = float(line.split(":", 1)[1].strip())
            header.append(line)
            continue
        if name in _PLAYLIST_TAGS:
            header.append(line)
            continue
        if name in _SEGMENT_TAGS or segments or pending:
            pending.append(line)
            continue
        # Anything else ahead of the first segment (an initial KEY/MAP, an unknown
        # tag, a comment) describes the whole playlist.
        header.append(line)
    if target_duration <= 0:
        target_duration = 1.0
    return ParsedPlaylist(
        header=tuple(header),
        media_sequence=media_sequence,
        target_duration=target_duration,
        segments=tuple(segments),
        discontinuity_sequence=discontinuity_sequence,
    )


def live_edge(parsed: ParsedPlaylist) -> int:
    """Media sequence of the newest segment the playlist lists."""
    return parsed.media_sequence + len(parsed.segments) - 1


def fetch_live_edge(url: str, fetcher: Callable[[str], bytes]) -> int:
    """One GET of a live playlist: the media sequence of its newest segment.

    Blocking; call it through ``asyncio.to_thread``. Raises when the playlist
    cannot be fetched or lists no segment.
    """
    parsed = parse_m3u8(fetcher(url).decode("utf-8", "replace"), url)
    if not parsed.segments:
        raise ValueError("remote playlist contained no segments")
    return live_edge(parsed)


def _last_segments(parsed: ParsedPlaylist, keep: int) -> ParsedPlaylist:
    """The playlist cut to its last ``keep`` segments, still meaning the same.

    MEDIA-SEQUENCE and DISCONTINUITY-SEQUENCE advance past what was dropped, and a
    KEY/MAP carried by a dropped segment moves onto the first kept one.
    """
    if keep <= 0 or len(parsed.segments) <= keep:
        return parsed
    dropped, kept = parsed.segments[:-keep], list(parsed.segments[-keep:])
    sticky: dict[str, str] = {}
    discontinuities = 0
    for tags, _uri in dropped:
        for tag in tags:
            name = _tag_name(tag)
            if name in _STICKY_TAGS:
                sticky[name] = tag
            elif name == "#EXT-X-DISCONTINUITY":
                discontinuities += 1
    first_tags, first_uri = kept[0]
    own = {_tag_name(tag) for tag in first_tags}
    carried = tuple(tag for name, tag in sticky.items() if name not in own)
    if carried:
        kept[0] = (carried + first_tags, first_uri)
    sequence = parsed.discontinuity_sequence
    if discontinuities:
        sequence = (sequence or 0) + discontinuities
    return ParsedPlaylist(
        header=parsed.header,
        media_sequence=parsed.media_sequence + len(dropped),
        target_duration=parsed.target_duration,
        segments=tuple(kept),
        discontinuity_sequence=sequence,
    )


def render_window(
    parsed: ParsedPlaylist,
    *,
    keep: int,
    refresh_stamp: int | None = None,
) -> tuple[str, int]:
    """Return ``(m3u8_text, media_sequence)`` for the last ``keep`` segments."""
    window = _last_segments(parsed, keep)
    lines = ["#EXTM3U"]
    if refresh_stamp is not None:
        # ffmpeg's HLS demuxer skips reloads when the playlist's byte size is
        # unchanged. A slim window is always about ``keep`` segments, so size alone
        # plateaus; a monotonic tag forces a visible change on each refresh.
        lines.append(f"{_REFRESH_TAG}:{refresh_stamp}")
    lines.extend([*window.header, f"#EXT-X-MEDIA-SEQUENCE:{window.media_sequence}"])
    if window.discontinuity_sequence is not None:
        lines.append(f"#EXT-X-DISCONTINUITY-SEQUENCE:{window.discontinuity_sequence}")
    for tags, uri in window.segments:
        lines.extend(tags or ("#EXTINF:1.000,",))
        lines.append(uri)
    return "\n".join(lines) + "\n", window.media_sequence


class _LocalPlaylistServer:
    """Serve an on-disk sliding playlist over loopback HTTP.

    ffmpeg's HLS demuxer reloads remote manifests reliably; a local ``file://``
    playlist is easy to desync via inode swaps or stale size heuristics.
    """

    def __init__(self, routes: dict[str, str]) -> None:
        routes = dict(routes)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                target = routes.get(self.path.split("?", 1)[0])
                if not target:
                    self.send_error(404)
                    return
                try:
                    data = Path(target).read_bytes()
                except OSError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.apple.mpegurl")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._http.server_address[1]
        self._thread = Thread(
            target=self._http.serve_forever, name="hls-playlist-http", daemon=True
        )
        self._thread.start()

    def url_for(self, route: str) -> str:
        return f"http://127.0.0.1:{self.port}{route}"

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._thread.join(timeout=2.0)


class HlsLiveWindow:
    """Local sliding playlist that ffmpeg can reload without touching the DVR."""

    def __init__(
        self,
        url: str,
        *,
        keep: int = DEFAULT_KEEP,
        fetcher: Callable[[str], bytes] | None = None,
        name: str = "audio",
        refresh_s: float = _REFRESH_S,
    ) -> None:
        if keep < 2:
            raise ValueError("keep must be >= 2")
        if refresh_s <= 0:
            raise ValueError("refresh_s must be > 0")
        self.url = url
        self.keep = keep
        self.name = name
        self.refresh_s = refresh_s
        """Seconds between playlist refreshes once armed."""
        self._fetcher = fetcher or _default_fetch
        self._dir: tempfile.TemporaryDirectory[str] | None = None
        self.path: str = ""
        self.remote_segments = 0
        self.fetch_s = 0.0
        self.target_duration = 1.0
        self.first_edge: int | None = None
        """Media sequence of the newest remote segment at start."""
        self.edge: int | None = None
        """Media sequence of the newest remote segment seen so far."""
        self._parsed: ParsedPlaylist | None = None
        self._task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()
        # A refresh runs in a worker thread, which cancelling its task does not
        # stop. aclose() sets _retired and then waits for this lock, so no write is
        # in flight (or starts) once the directory is removed.
        self._write_lock = Lock()
        self._retired = False

    def _ingest(self, raw: bytes, *, force: bool = False) -> bool:
        parsed = parse_m3u8(raw.decode("utf-8", "replace"), self.url)
        if not parsed.segments:
            raise ValueError("remote playlist contained no segments")
        edge = live_edge(parsed)
        if not force and self.edge is not None and edge <= self.edge:
            # A slower refresh can return an older DVR snapshot. Rewinding
            # MEDIA-SEQUENCE makes ffmpeg's demuxer stall.
            return False
        self.remote_segments = len(parsed.segments)
        self.edge = edge
        if self.first_edge is None:
            self.first_edge = self.edge
        self.target_duration = min(2.0, max(0.5, parsed.target_duration))
        self._parsed = _last_segments(parsed, self.keep)
        return True

    def _install_playlist(self, raw: bytes) -> None:
        self._ingest(raw, force=True)
        self._write()

    def _refresh_blocking(self) -> None:
        raw = self._fetcher(self.url)
        if self._ingest(raw):
            self._write()

    async def start(self) -> str:
        if self.path:
            raise RuntimeError("already started")
        t0 = time.monotonic()
        raw = await asyncio.to_thread(self._fetcher, self.url)
        self.fetch_s = time.monotonic() - t0
        self._dir = tempfile.TemporaryDirectory(
            prefix=f"lst-hls-{self.name}-", ignore_cleanup_errors=True
        )
        self.path = str(Path(self._dir.name) / "index.m3u8")
        # A multi-MB DVR playlist parses in tens of milliseconds, but it must not
        # run on the event loop, which also reads ffmpeg.
        await asyncio.to_thread(self._install_playlist, raw)
        log.info(
            "hls live window ready",
            extra={
                "input": self.name,
                "remote_segments": self.remote_segments,
                "kept": self.keep,
                "fetch_s": round(self.fetch_s, 2),
                "target_duration": self.target_duration,
            },
        )
        return self.path

    def arm(self) -> None:
        """Start refreshing in the background."""
        if self._task is not None or not self.path:
            return
        self._task = asyncio.create_task(self._advance_loop(), name=f"hls-window-{self.name}")

    def _write(self) -> None:
        with self._write_lock:
            if not self._retired:
                self._write_locked()

    def _write_locked(self) -> None:
        assert self._parsed is not None and self.path
        stamp = time.time_ns()
        body, _ = render_window(self._parsed, keep=self.keep, refresh_stamp=stamp)
        # Some ffmpeg builds treat an unchanged byte length as "playlist unchanged"
        # even when MEDIA-SEQUENCE moved. Pad so every refresh differs in size as
        # well as content.
        body = body.rstrip("\n") + "\n" + ("#" * ((stamp % 24) + 1)) + "\n"
        # The loopback server opens the path afresh per request, so a rename is
        # atomic for it: every GET sees one whole window, never a new head glued to
        # an old tail.
        fd, tmp = tempfile.mkstemp(prefix="idx-", suffix=".tmp", dir=os.path.dirname(self.path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    async def _refresh(self) -> None:
        try:
            await asyncio.to_thread(self._refresh_blocking)
        except Exception:
            log.debug("hls playlist refresh failed", extra={"input": self.name}, exc_info=True)

    async def _advance_loop(self) -> None:
        """Re-fetch the remote playlist and rewrite the slim window."""
        try:
            while not self._closed.is_set():
                try:
                    await asyncio.wait_for(self._closed.wait(), self.refresh_s)
                    return
                except TimeoutError:
                    pass
                if self._refresh_task is not None and not self._refresh_task.done():
                    continue
                self._refresh_task = asyncio.create_task(
                    self._refresh(), name=f"hls-refresh-{self.name}"
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("hls live window stopped", extra={"input": self.name})

    async def _retire(self, timeout: float = 2.0) -> None:
        """Stop writing; wait (without blocking the loop) for a write in flight."""
        self._retired = True
        deadline = time.monotonic() + timeout
        while not self._write_lock.acquire(blocking=False):
            if time.monotonic() > deadline:
                log.warning("hls window write still running at close", extra={"input": self.name})
                return
            await asyncio.sleep(0.005)
        self._write_lock.release()

    async def aclose(self) -> None:
        self._closed.set()
        for task in (self._task, self._refresh_task):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._task = None
        self._refresh_task = None
        await self._retire()
        if self._dir is not None:
            self._dir.cleanup()
            self._dir = None
        self.path = ""


@dataclass
class HlsInput:
    """What ffmpeg should open for one HLS source, and what backs it."""

    url: str
    window: HlsLiveWindow | None = None
    _server: _LocalPlaylistServer | None = None

    @property
    def windowed(self) -> bool:
        return self.window is not None

    def live_edges(self) -> tuple[int | None, int | None]:
        """``(at start, latest)`` live-edge media sequence of the remote playlist.

        ``(None, None)`` without a window (a VOD, a local file, or a window that
        failed so ffmpeg reads the remote playlist).
        """
        if self.window is None:
            return None, None
        return self.window.first_edge, self.window.edge

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        if self.window is not None:
            await self.window.aclose()
            self.window = None


async def prepare_hls_input(
    url: str,
    *,
    is_live: bool,
    fetcher: Callable[[str], bytes] | None = None,
    proxy: str | None = None,
    keep: int = DEFAULT_KEEP,
) -> HlsInput:
    """Point a live HLS input at a local sliding playlist. A no-op otherwise.

    Any failure to build the window falls back to the remote playlist: the window
    is an optimisation, never a reason for a capture to fail.
    """
    if not is_live or not looks_like_hls(url):
        return HlsInput(url)
    window = HlsLiveWindow(url, keep=keep, fetcher=fetcher or make_playlist_fetcher(proxy))
    try:
        await window.start()
    except Exception as exc:
        log.warning(
            "hls live window failed; using remote playlist",
            extra={"error": type(exc).__name__},
        )
        await window.aclose()
        return HlsInput(url)
    route = f"/{window.name}/index.m3u8"
    server = _LocalPlaylistServer({route: window.path})
    window.arm()
    return HlsInput(server.url_for(route), window, server)
