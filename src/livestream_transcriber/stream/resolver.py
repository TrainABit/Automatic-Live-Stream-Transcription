"""Turning a user-supplied source into something ffmpeg can open.

Three kinds of input are told apart:

* an existing **local file** is used as it is;
* a **direct media URL** (an ``.m3u8`` manifest or a plain media file over http)
  bypasses yt-dlp: ffmpeg opens it as given, so no extractor round trip is paid;
* anything else (a video page, a channel's ``/live`` URL) goes through the
  yt-dlp Python API, which knows how to find the audio rendition.

Every successful resolve is remembered for a short while. A source probe resolves
a candidate, and the capture that starts right after asks for the same URL again;
the cache turns those two extractions into one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..logging_setup import get_logger
from ..models import StreamInfo
from ..netutil import HttpError, get_json
from ..redact import redact_url
from .base import StreamResolutionError
from .ffmpeg import is_http_url, looks_like_hls

log = get_logger(__name__)

__all__ = [
    "forget_resolved",
    "js_runtime_status",
    "recently_resolved",
    "resolve_stream",
    "resolve_stream_sync",
    "ydl_opts",
]

# File extensions ffmpeg can read straight from an http URL. A URL that ends in
# one of these is opened directly; a finite file is not a live stream.
_MEDIA_EXTENSIONS = frozenset(
    {
        ".m3u8", ".mp3", ".mp4", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus",
        ".webm", ".mkv", ".mov", ".ts", ".mpd",
    }
)  # fmt: skip
_PLAYLIST_TIMEOUT_S = 10.0

# yt-dlp's warnings explain a resolve that then fails or comes back thin: no
# JavaScript runtime for the site's challenges, a signature solve that failed,
# formats it had to skip. They are logged, each text at most once per
# _WARNING_EVERY_S, because a periodic probe would otherwise repeat the same
# warning on every call.
_WARNING_EVERY_S = 600.0
_WARNING_LIMIT = 64
# text -> (monotonic time it was last logged, repeats suppressed since)
_warnings_logged: dict[str, tuple[float, int]] = {}
_warnings_lock = threading.Lock()  # probes resolve in parallel worker threads
_clock = time.monotonic


def _has_codec(fmt: dict[str, Any], key: str) -> bool:
    """True when ``fmt`` positively declares a usable codec for ``key``.

    Codec metadata is unreliable: live HLS audio renditions arrive with the
    ``acodec`` key *absent altogether*, so a ``get(key, "none")`` test would
    reject the audio track. Treat "absent" as unknown, never as "no codec".
    """
    value = fmt.get(key)
    return bool(value) and value != "none"


def _pick_media(info: dict[str, Any]) -> tuple[str | None, str, dict[str, str]]:
    """Return ``(media_url, format_id, http_headers)`` from a yt-dlp info dict.

    yt-dlp resolves a selector to either one format or a ``requested_formats``
    list (when the selector merges renditions). For a transcriber only the audio
    matters, so from such a list take the entry that carries audio, preferring one
    that carries no video.
    """
    requested = [f for f in (info.get("requested_formats") or []) if f]
    if requested:
        ids = "+".join(str(f.get("format_id")) for f in requested)
        audio_only = [f for f in requested if not _has_codec(f, "vcodec")]
        with_audio = [f for f in requested if _has_codec(f, "acodec")]
        chosen = (audio_only or with_audio or requested)[0]
        return chosen.get("url"), ids, dict(chosen.get("http_headers") or {})
    return (
        info.get("url"),
        str(info.get("format_id") or ""),
        dict(info.get("http_headers") or {}),
    )


def _local_file_info(url: str) -> StreamInfo | None:
    """Treat an existing local path as a source.

    A downloaded clip then flows through exactly the pipeline a live stream does.
    """
    if url.startswith(("http://", "https://", "ytsearch")):
        return None
    path = Path(url).expanduser()
    if not path.is_file():
        return None
    return StreamInfo(
        url=url,
        title=path.name,
        is_live=False,
        media_url=str(path.resolve()),
        format_id="local-file",
    )


def _is_direct_media(url: str) -> bool:
    """A http(s) URL ffmpeg can open as it is: an HLS manifest or a media file."""
    if not is_http_url(url):
        return False
    if looks_like_hls(url):
        return True
    return Path(urlsplit(url).path).suffix.lower() in _MEDIA_EXTENSIONS


def _playlist_is_live(url: str) -> bool:
    """Whether the HLS playlist at ``url`` is a live one.

    A playlist that lists ``#EXT-X-ENDLIST`` (or declares itself VOD) is a finished
    recording. A master playlist says nothing either way, so it counts as live: the
    caller then treats it with the drop-oldest policy that suits a live feed.
    """
    try:
        body = get_json(url, timeout=_PLAYLIST_TIMEOUT_S)
    except HttpError as exc:
        if exc.status in (404, 410):
            raise StreamResolutionError(
                f"{redact_url(url)} is offline (HTTP {exc.status})"
            ) from exc
        raise StreamResolutionError(f"could not fetch playlist: HTTP {exc.status}") from exc
    except OSError as exc:
        raise StreamResolutionError(f"could not fetch playlist: {type(exc).__name__}") from exc
    text = str(body.get("text") or "")
    if not text.lstrip().startswith("#EXTM3U"):
        raise StreamResolutionError(f"{redact_url(url)} is not an HLS playlist")
    if "#EXT-X-STREAM-INF" in text:
        return True
    return "#EXT-X-ENDLIST" not in text and "#EXT-X-PLAYLIST-TYPE:VOD" not in text


def _direct_media_info(url: str, *, proxy: str | None) -> StreamInfo:
    name = Path(urlsplit(url).path).name or None
    if looks_like_hls(url):
        # Behind a proxy the peek would bypass it, so assume live instead.
        is_live = True if proxy else _playlist_is_live(url)
        format_id = "direct-hls"
    else:
        is_live, format_id = False, "direct-media"
    return StreamInfo(url=url, title=name, is_live=is_live, media_url=url, format_id=format_id)


def _note_ytdlp_warning(message: str) -> None:
    text = _public_error(message)
    # "[site] <video id>: ..." is the same warning for every video.
    key = re.sub(r"^\[[\w:]+\] [\w-]{6,}: ", "", text)
    now = _clock()
    with _warnings_lock:
        last = _warnings_logged.get(key)
        if last is not None and now - last[0] < _WARNING_EVERY_S:
            _warnings_logged[key] = (last[0], last[1] + 1)
            return
        _warnings_logged.pop(key, None)
        _warnings_logged[key] = (now, 0)
        while len(_warnings_logged) > _WARNING_LIMIT:
            _warnings_logged.pop(next(iter(_warnings_logged)))
    log.warning(
        "yt-dlp warning",
        extra={"warning": text, "repeats_suppressed": last[1] if last else 0},
    )


class _YtdlpLogger:
    """yt-dlp's ``logger``: warnings reach our log, rate-limited.

    Progress chatter (``debug``/``info``) is dropped. Errors are too: yt-dlp
    raises them as ``DownloadError``, which becomes the StreamResolutionError the
    caller logs and classifies.
    """

    def debug(self, _msg: str) -> None:
        return

    def info(self, _msg: str) -> None:
        return

    def warning(self, msg: str) -> None:
        _note_ytdlp_warning(msg)

    def error(self, _msg: str) -> None:
        return


def _probe_js_runtime(name: str) -> tuple[str, bool] | None:
    """``(version, new enough for yt-dlp)`` of JS runtime ``name``, or None.

    Asks yt-dlp's own runtime classes (they know its minimum versions); if this
    yt-dlp has none, a runtime on PATH counts as usable.
    """
    try:
        from yt_dlp.globals import supported_js_runtimes

        runtime_cls = supported_js_runtimes.value.get(name)
    except Exception:
        runtime_cls = None
    if runtime_cls is None:
        return ("unknown version", True) if shutil.which(name) else None
    try:
        info = runtime_cls().info
    except Exception:
        return None
    return (str(info.version), bool(info.supported)) if info else None


def js_runtime_status(
    probe: Callable[[str], tuple[str, bool] | None] | None = None,
) -> tuple[bool, str]:
    """Can yt-dlp solve a site's JavaScript challenges here? ``(ok, detail)``.

    yt-dlp enables only deno by default (``js_runtimes``); without it, extraction
    for some sites falls back to JS-less player clients and formats may be
    missing. node is reported so the operator knows what is there.
    """
    probe = probe or _probe_js_runtime
    deno, node = probe("deno"), probe("node")
    if deno and deno[1]:
        return True, f"deno {deno[0]}"
    missing = f"deno {deno[0]} is too old (yt-dlp needs a newer one)" if deno else "no deno on PATH"
    if node:
        missing += f"; node {node[0]} found, but yt-dlp enables only deno by default"
    return False, (
        f"{missing}; extraction falls back to JS-less clients and some "
        "formats may be missing. Install deno."
    )


def ydl_opts(
    format_selector: str,
    *,
    cookiefile: str | Path | None = None,
    proxy: str | None = None,
) -> dict[str, Any]:
    """yt-dlp options. ``cookiefile`` is a path only, never cookie bytes."""
    opts: dict[str, Any] = {
        "quiet": True,
        "logger": _YtdlpLogger(),
        "noplaylist": True,
        "skip_download": True,
        "format": format_selector,
        # Take the live edge, not the start of the DVR window.
        "live_from_start": False,
        "socket_timeout": 20,
    }
    if cookiefile:
        path = Path(cookiefile).expanduser()
        if not path.is_file():
            raise StreamResolutionError(f"cookie file is set but missing: {path}")
        opts["cookiefile"] = str(path)
    if proxy:
        opts["proxy"] = proxy
    return opts


@contextlib.contextmanager
def _private_cookie_copy(cookiefile: str | Path | None) -> Iterator[str | None]:
    """A private, writable copy of the operator's cookie file for one call.

    yt-dlp writes its cookie jar back to ``cookiefile`` when the ``YoutubeDL``
    closes. When the file sits on a read-only path (a mounted secret, a sandboxed
    service), that write is a ``PermissionError`` and would fail the whole resolve.
    The copy takes the write; the original is only ever read. The copy is 0600 in
    the temp dir and removed afterwards.
    """
    if not cookiefile:
        yield None
        return
    source = Path(cookiefile).expanduser()
    if not source.is_file():
        raise StreamResolutionError(f"cookie file is set but missing: {source}")
    fd, copy = tempfile.mkstemp(prefix="lst-cookies-", suffix=".txt")
    try:
        try:
            with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst)
        except OSError as exc:
            raise StreamResolutionError(
                f"cookie file is not readable: {source} ({type(exc).__name__})"
            ) from exc
        yield copy
    finally:
        with contextlib.suppress(OSError):
            os.unlink(copy)


def _public_error(text: str) -> str:
    """Drop signed media URLs before an error string can reach a log."""
    cleaned = re.sub(r"https?://\S+", "<url>", text or "")
    return " ".join(cleaned.split())[:400]


def resolve_stream_sync(
    url: str,
    *,
    format_selector: str,
    cookiefile: str | Path | None = None,
    proxy: str | None = None,
) -> StreamInfo:
    """Blocking resolve. Call via :func:`resolve_stream` from async code.

    ``cookiefile`` is a Netscape cookie *path* kept outside the repository. Some
    sites require a logged-in session ("confirm you are not a bot"); the file is
    a live account credential and must never be committed. yt-dlp only ever sees
    a private copy of it (see :func:`_private_cookie_copy`).
    """
    local = _local_file_info(url)
    if local is not None:
        log.info("using local file as stream source", extra={"path": local.media_url})
        return local
    if _is_direct_media(url):
        info = _direct_media_info(url, proxy=proxy)
        log.info("using direct media url", extra={"url": redact_url(url), "is_live": info.is_live})
        return info

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover
        raise StreamResolutionError("yt-dlp is not installed") from exc

    with _private_cookie_copy(cookiefile) as cookie_copy:
        opts = ydl_opts(format_selector, cookiefile=cookie_copy, proxy=proxy)
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise StreamResolutionError(
                f"yt-dlp could not resolve {redact_url(url)}: {_public_error(str(exc))}"
            ) from exc
        except Exception as exc:  # network stack, extractor bugs, ...
            raise StreamResolutionError(
                f"yt-dlp failed on {redact_url(url)}: {_public_error(str(exc))}"
            ) from exc

    if not info:
        raise StreamResolutionError(f"yt-dlp returned no information for {redact_url(url)}")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise StreamResolutionError(f"{redact_url(url)} resolved to an empty playlist")
        info = entries[0]

    media_url, format_id, headers = _pick_media(info)
    if not media_url:
        raise StreamResolutionError(
            f"no playable media URL for {redact_url(url)} "
            f"(format selector {format_selector!r} matched nothing)"
        )
    return StreamInfo(
        url=url,
        title=info.get("title"),
        channel=info.get("channel") or info.get("uploader"),
        is_live=bool(info.get("is_live")),
        media_url=media_url,
        format_id=format_id,
        stream_id=info.get("id"),
        headers=headers,
    )


# Recent successful resolves, keyed by everything that shapes the answer. Only the
# event loop thread touches this (the yt-dlp work itself runs in a worker thread).
_RESOLVED: dict[tuple[str | None, ...], StreamInfo] = {}
_RESOLVED_LIMIT = 8


def _cache_key(
    url: str,
    format_selector: str,
    cookiefile: str | Path | None,
    proxy: str | None,
) -> tuple[str | None, ...]:
    return (url, format_selector, str(cookiefile) if cookiefile else None, proxy)


def recently_resolved(
    url: str,
    *,
    format_selector: str,
    max_age: float,
    cookiefile: str | Path | None = None,
    proxy: str | None = None,
    since: float | None = None,
) -> StreamInfo | None:
    """The last successful resolve of ``url`` if it is at most ``max_age`` s old.

    ``since``: and was made at or after that time (``time.time()`` scale).
    No network. ``max_age <= 0`` never matches.
    """
    if max_age <= 0:
        return None
    info = _RESOLVED.get(_cache_key(url, format_selector, cookiefile, proxy))
    if info is None or time.time() - info.resolved_at > max_age:
        return None
    if since is not None and info.resolved_at < since:
        return None
    return info


def forget_resolved() -> None:
    """Drop every remembered resolve (tests, operator tooling)."""
    _RESOLVED.clear()


async def resolve_stream(
    url: str,
    *,
    format_selector: str,
    cookiefile: str | Path | None = None,
    proxy: str | None = None,
    max_age: float = 0.0,
    since: float | None = None,
) -> StreamInfo:
    """Resolve without blocking the event loop.

    ``max_age > 0`` accepts a remembered answer that recent (and, with ``since``,
    made at or after that time) instead of a new extraction. Every successful
    network resolve is remembered, whatever ``max_age`` the caller passed, so a
    probe feeds the capture. A failed one forgets the remembered answer: it is
    older than the failure.
    """
    key = _cache_key(url, format_selector, cookiefile, proxy)
    cached = recently_resolved(
        url,
        format_selector=format_selector,
        max_age=max_age,
        cookiefile=cookiefile,
        proxy=proxy,
        since=since,
    )
    if cached is not None:
        log.debug(
            "stream resolve reused",
            extra={"age_s": round(time.time() - cached.resolved_at, 1), "is_live": cached.is_live},
        )
        return cached
    try:
        info = await asyncio.to_thread(
            resolve_stream_sync,
            url,
            format_selector=format_selector,
            cookiefile=cookiefile,
            proxy=proxy,
        )
    except Exception:
        _RESOLVED.pop(key, None)
        raise
    if info.format_id != "local-file":
        _RESOLVED.pop(key, None)
        _RESOLVED[key] = info
        while len(_RESOLVED) > _RESOLVED_LIMIT:
            _RESOLVED.pop(next(iter(_RESOLVED)))
    return info
