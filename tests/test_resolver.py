from __future__ import annotations

import http.cookiejar
import http.server
import logging
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from livestream_transcriber.stream import resolver
from livestream_transcriber.stream.base import StreamResolutionError
from livestream_transcriber.stream.fallback import STATUS_OFFLINE, classify_probe_error
from livestream_transcriber.stream.resolver import (
    _has_codec,
    _pick_media,
    js_runtime_status,
    resolve_stream_sync,
    ydl_opts,
)

# --------------------------------------------------------------------- picking a rendition


def test_a_single_format_uses_its_own_url_and_headers() -> None:
    info = {"url": "https://x.test/all.m3u8", "format_id": "95", "http_headers": {"Referer": "r"}}
    assert _pick_media(info) == ("https://x.test/all.m3u8", "95", {"Referer": "r"})


def test_the_audio_rendition_is_taken_from_a_merged_selection() -> None:
    info = {
        "requested_formats": [
            {"format_id": "299", "vcodec": "avc1", "acodec": "none", "url": "V"},
            {"format_id": "140", "vcodec": "none", "acodec": "mp4a", "url": "A"},
        ]
    }
    assert _pick_media(info) == ("A", "299+140", {})


def test_audio_without_an_acodec_key_is_still_found() -> None:
    """Live HLS audio renditions often omit ``acodec``. A ``get("acodec", "none")`` test
    would reject the track and hand ffmpeg an empty URL."""
    info = {
        "requested_formats": [
            {"format_id": "270", "vcodec": "avc1.4D4028", "acodec": "none", "url": "V"},
            {"format_id": "234", "vcodec": "none", "url": "A"},  # no acodec key at all
        ]
    }
    assert _pick_media(info)[:2] == ("A", "270+234")


def test_missing_codec_metadata_on_both_takes_the_one_that_is_not_video() -> None:
    info = {
        "requested_formats": [
            {"format_id": "1", "vcodec": "avc1", "url": "V"},
            {"format_id": "2", "url": "A"},
        ]
    }
    assert _pick_media(info)[0] == "A"


def test_a_muxed_pair_falls_back_to_a_format_that_has_audio() -> None:
    info = {
        "requested_formats": [{"format_id": "18", "vcodec": "avc1", "acodec": "mp4a", "url": "M"}]
    }
    assert _pick_media(info)[0] == "M"


@pytest.mark.parametrize(
    ("fmt", "key", "expected"),
    [
        ({"acodec": "mp4a"}, "acodec", True),
        ({"acodec": "none"}, "acodec", False),
        ({"acodec": None}, "acodec", False),
        ({}, "acodec", False),
        ({"vcodec": "avc1"}, "vcodec", True),
    ],
)
def test_has_codec(fmt: dict[str, Any], key: str, expected: bool) -> None:
    assert _has_codec(fmt, key) is expected


# --------------------------------------------------------------------- local files and direct URLs


def test_local_file_is_accepted_as_a_source(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"not really audio, but it exists")
    info = resolve_stream_sync(str(clip), format_selector="best")
    assert info.is_live is False
    assert info.format_id == "local-file"
    assert info.media_url == str(clip.resolve())
    assert info.title == "clip.wav"


def test_missing_local_file_falls_through_to_ytdlp(tmp_path: Path) -> None:
    with pytest.raises(StreamResolutionError):
        resolve_stream_sync(str(tmp_path / "nope.wav"), format_selector="best")


class _Playlist(http.server.BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, str]] = {}

    def do_GET(self) -> None:
        status, body = self.routes.get(self.path, (404, "not found"))
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def playlist_server() -> Iterator[str]:
    _Playlist.routes = {
        "/live.m3u8": (200, "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nseg1.ts\n"),
        "/vod.m3u8": (
            200,
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nseg1.ts\n#EXT-X-ENDLIST\n",
        ),
        "/vod-typed.m3u8": (200, "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:VOD\n#EXTINF:2.0,\nseg1.ts\n"),
        "/master.m3u8": (200, "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlow/index.m3u8\n"),
        "/html.m3u8": (200, "<html>an error page served with 200</html>"),
        "/gone.m3u8": (410, "gone"),
        "/broken.m3u8": (500, "oops"),
    }
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Playlist)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_a_direct_hls_url_bypasses_ytdlp(
    playlist_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ytdlp(*_a: object, **_k: object) -> None:
        raise AssertionError("yt-dlp must not be used for a direct media URL")

    monkeypatch.setattr("yt_dlp.YoutubeDL", no_ytdlp)
    url = f"{playlist_server}/live.m3u8"
    info = resolve_stream_sync(url, format_selector="best")
    assert info.media_url == url
    assert info.format_id == "direct-hls"
    assert info.is_live is True
    assert info.title == "live.m3u8"


def test_a_finished_playlist_is_not_live(playlist_server: str) -> None:
    for name in ("vod", "vod-typed"):
        info = resolve_stream_sync(f"{playlist_server}/{name}.m3u8", format_selector="best")
        assert info.is_live is False, name


def test_a_master_playlist_counts_as_live(playlist_server: str) -> None:
    assert resolve_stream_sync(f"{playlist_server}/master.m3u8", format_selector="best").is_live


def test_a_missing_playlist_is_offline_not_an_error_class_of_its_own(playlist_server: str) -> None:
    with pytest.raises(StreamResolutionError) as exc:
        resolve_stream_sync(f"{playlist_server}/absent.m3u8", format_selector="best")
    assert classify_probe_error(str(exc.value)) == STATUS_OFFLINE
    with pytest.raises(StreamResolutionError, match="offline"):
        resolve_stream_sync(f"{playlist_server}/gone.m3u8", format_selector="best")


def test_a_broken_or_fake_playlist_is_a_resolution_error(playlist_server: str) -> None:
    with pytest.raises(StreamResolutionError, match="HTTP 500"):
        resolve_stream_sync(f"{playlist_server}/broken.m3u8", format_selector="best")
    with pytest.raises(StreamResolutionError, match="not an HLS playlist"):
        resolve_stream_sync(f"{playlist_server}/html.m3u8", format_selector="best")


def test_an_unreachable_playlist_is_a_resolution_error() -> None:
    # The suite's network guard refuses non-loopback hosts, like an unreachable one.
    with pytest.raises(StreamResolutionError, match="could not fetch playlist"):
        resolve_stream_sync("https://media.example.invalid/live.m3u8", format_selector="best")


def test_direct_hls_behind_a_proxy_is_assumed_live_without_a_fetch() -> None:
    info = resolve_stream_sync(
        "https://media.example.invalid/live.m3u8",
        format_selector="best",
        proxy="http://proxy.example.test:8080",
    )
    assert info.is_live is True


@pytest.mark.parametrize(
    "url",
    ["https://media.example.invalid/a.mp3", "https://media.example.invalid/x/clip.MP4?token=1"],
)
def test_a_direct_media_file_is_finite(url: str) -> None:
    info = resolve_stream_sync(url, format_selector="best")
    assert info.media_url == url
    assert info.is_live is False
    assert info.format_id == "direct-media"


def test_a_signed_url_is_not_logged_by_the_bypass(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    resolve_stream_sync(
        "https://media.example.invalid/a.mp3?token=your-secret-token&sig=abc", format_selector="x"
    )
    assert "your-secret-token" not in caplog.text


# --------------------------------------------------------------------- yt-dlp options and cookies


def test_ydl_opts_include_cookiefile_path_not_bytes(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    opts = ydl_opts("bestaudio/best", cookiefile=cookies)
    assert opts["cookiefile"] == str(cookies)
    assert opts["format"] == "bestaudio/best"
    assert "cookie" not in "".join(str(v) for k, v in opts.items() if k != "cookiefile").lower()


def test_ydl_opts_missing_cookiefile_raises(tmp_path: Path) -> None:
    with pytest.raises(StreamResolutionError, match="cookie file"):
        ydl_opts("best", cookiefile=tmp_path / "absent.txt")


def test_ydl_opts_omit_cookiefile_and_proxy_when_unset() -> None:
    opts = ydl_opts("best")
    assert "cookiefile" not in opts and "proxy" not in opts
    assert opts["live_from_start"] is False and opts["skip_download"] is True
    assert ydl_opts("best", proxy="http://p.example.test:1")["proxy"] == "http://p.example.test:1"


_COOKIES = (
    "# Netscape HTTP Cookie File\n"
    ".example.test\tTRUE\t/\tTRUE\t2147483647\tSID\toriginal-session-value\n"
)


def _fake_extract(seen: dict[str, Any]) -> Any:
    def extract_info(self: Any, url: str, download: bool = False, **_kw: Any) -> dict[str, Any]:
        # yt-dlp loads the jar lazily; reading it proves which file it used.
        seen["cookies"] = {c.name: c.value for c in self.cookiejar}
        seen["cookiefile"] = self.params.get("cookiefile")
        # A rotated session cookie, as a site hands out: yt-dlp saves it on close.
        self.cookiejar.set_cookie(
            http.cookiejar.Cookie(
                0,
                "NEW",
                "rotated",
                None,
                False,
                ".example.test",
                True,
                True,
                "/",
                True,
                True,
                2147483647,
                False,
                None,
                None,
                {},
            )
        )
        return {"url": "https://media.example.test/all.m3u8", "format_id": "95", "is_live": True}

    return extract_info


def test_cookie_file_on_a_read_only_path_does_not_fail_the_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """yt-dlp saves its cookie jar back to ``cookiefile`` when the YoutubeDL closes. On a
    read-only mount that would raise PermissionError and fail every resolve."""
    yt_dlp = pytest.importorskip("yt_dlp")
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    cookies = secrets_dir / "cookies.txt"
    cookies.write_text(_COOKIES, encoding="utf-8")
    before = cookies.read_bytes()
    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_tmp))
    seen: dict[str, Any] = {}
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", _fake_extract(seen))
    cookies.chmod(0o400)
    secrets_dir.chmod(0o500)
    try:
        info = resolve_stream_sync(
            "https://www.example.test/watch?v=demoVideo01",
            format_selector="best",
            cookiefile=str(cookies),
        )
    finally:
        secrets_dir.chmod(0o700)
        cookies.chmod(0o600)
    assert info.media_url == "https://media.example.test/all.m3u8"
    assert seen["cookies"] == {"SID": "original-session-value"}
    assert seen["cookiefile"] != str(cookies)
    assert cookies.read_bytes() == before
    # The private copy took yt-dlp's write-back and is gone again.
    assert list(private_tmp.iterdir()) == []


def test_cookie_copy_is_private_neutrally_named_and_removed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yt_dlp = pytest.importorskip("yt_dlp")
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(_COOKIES, encoding="utf-8")
    private_tmp = tmp_path / "private-tmp"
    private_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_tmp))
    seen: list[tuple[int, str]] = []

    def boom(self: Any, url: str, download: bool = False, **_kw: Any) -> None:
        path = Path(self.params["cookiefile"])
        seen.append((path.stat().st_mode & 0o777, path.name))
        raise yt_dlp.utils.DownloadError("ERROR: Sign in to confirm you're not a bot")

    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", boom)
    with pytest.raises(StreamResolutionError, match="not a bot"):
        resolve_stream_sync(
            "https://www.example.test/watch?v=demoVideo01",
            format_selector="best",
            cookiefile=str(cookies),
        )
    assert [mode for mode, _ in seen] == [0o600]
    assert seen[0][1].startswith("lst-cookies-")
    assert list(private_tmp.iterdir()) == []


def test_missing_cookie_file_is_still_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(StreamResolutionError, match="cookie file is set but missing"):
        resolve_stream_sync(
            "https://www.example.test/watch?v=demoVideo01",
            format_selector="best",
            cookiefile=str(tmp_path / "absent.txt"),
        )


def test_resolver_errors_carry_no_signed_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    yt_dlp = pytest.importorskip("yt_dlp")

    def boom(self: Any, url: str, download: bool = False, **_kw: Any) -> None:
        raise yt_dlp.utils.DownloadError(
            "ERROR https://rr1.example.test/videoplayback?sig=abc&ip=203.0.113.9 HTTP Error 403"
        )

    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", boom)
    with pytest.raises(StreamResolutionError) as exc:
        resolve_stream_sync("https://www.example.test/watch?v=demoVideo01", format_selector="best")
    assert "sig=abc" not in str(exc.value) and "203.0.113.9" not in str(exc.value)
    assert "<url>" in str(exc.value) and "403" in str(exc.value)


def test_a_playlist_result_resolves_to_its_first_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    yt_dlp = pytest.importorskip("yt_dlp")
    answer: dict[str, Any] = {"_type": "playlist", "entries": [None, {"url": "M", "id": "e1"}]}
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", lambda self, url, download=False: answer)
    info = resolve_stream_sync("https://www.example.test/watch?v=demoVideo01", format_selector="b")
    assert (info.media_url, info.stream_id) == ("M", "e1")

    answer["entries"] = []
    with pytest.raises(StreamResolutionError, match="empty playlist"):
        resolve_stream_sync("https://www.example.test/watch?v=demoVideo01", format_selector="b")


def test_no_media_url_in_the_answer_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    yt_dlp = pytest.importorskip("yt_dlp")
    monkeypatch.setattr(
        yt_dlp.YoutubeDL, "extract_info", lambda self, url, download=False: {"id": "x"}
    )
    with pytest.raises(StreamResolutionError, match="no playable media URL"):
        resolve_stream_sync("https://www.example.test/watch?v=demoVideo01", format_selector="b")
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", lambda self, url, download=False: None)
    with pytest.raises(StreamResolutionError, match="no information"):
        resolve_stream_sync("https://www.example.test/watch?v=demoVideo01", format_selector="b")


# --------------------------------------------------------------------- yt-dlp warnings


_JS_WARNING = (
    "No supported JavaScript runtime could be found. Only deno is enabled by "
    "default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your "
    "command/config. Extraction without a JS runtime has been deprecated, "
    "and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  "
    "for details on installing one"
)
_URL = "https://www.example.test/watch?v=demoVideo01"
_INFO = {"url": "https://media.example.test/all.m3u8", "format_id": "95", "is_live": True}


def _ytdlp_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == "yt-dlp warning"]


@pytest.fixture
def warning_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A fresh rate limiter on a hand-driven clock."""
    now = [1000.0]
    monkeypatch.setattr(resolver, "_warnings_logged", {})
    monkeypatch.setattr(resolver, "_clock", lambda: now[0])
    return now


def _extract_warning(*warnings: str) -> Any:
    def extract_info(self: Any, url: str, download: bool = False, **_kw: Any) -> dict[str, Any]:
        for text in warnings:
            self.report_warning(text)
        return dict(_INFO)

    return extract_info


def test_ytdlp_warnings_are_logged_once_per_interval(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, warning_clock: list[float]
) -> None:
    """yt-dlp's warnings must reach the log, or a missing JavaScript runtime leaves no
    trace at all. Each distinct warning is logged at most once per interval, since a
    periodic probe would repeat it on every call."""
    yt_dlp = pytest.importorskip("yt_dlp")
    monkeypatch.setattr(
        yt_dlp.YoutubeDL,
        "extract_info",
        _extract_warning(
            f"[site] {_JS_WARNING}",
            "[site] demoVideo01: Signature solving failed: Some formats may be missing.",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="livestream_transcriber.stream.resolver"):
        for _ in range(3):
            resolve_stream_sync(_URL, format_selector="best")
            warning_clock[0] += 30.0
        first = _ytdlp_warnings(caplog)
        assert [r.levelno for r in first] == [logging.WARNING, logging.WARNING]
        assert "No supported JavaScript runtime" in first[0].warning
        assert "github.com" not in first[0].warning  # URLs scrubbed
        assert first[1].warning.startswith("[site] demoVideo01: Signature solving")
        # Ten minutes on both are logged again, with the count held back; the signature
        # warning for another video counts as the same one.
        warning_clock[0] += 600.0
        monkeypatch.setattr(
            yt_dlp.YoutubeDL,
            "extract_info",
            _extract_warning(
                f"[site] {_JS_WARNING}",
                "[site] zyxwVUTSrqp: Signature solving failed: Some formats may be missing.",
            ),
        )
        resolve_stream_sync(_URL, format_selector="best")
    again = _ytdlp_warnings(caplog)[2:]
    assert [r.repeats_suppressed for r in again] == [2, 2]


def test_a_warning_before_a_failed_resolve_is_visible(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, warning_clock: list[float]
) -> None:
    """No JS runtime, so the formats the selector wants are missing and the resolve
    fails. The error is raised (and classified) as usual, and the warning that explains
    it is in the log next to it."""
    yt_dlp = pytest.importorskip("yt_dlp")

    def extract_info(self: Any, url: str, download: bool = False, **_kw: Any) -> None:
        self.report_warning(f"[site] {_JS_WARNING}")
        raise yt_dlp.utils.DownloadError("ERROR: [site] abc: Requested format is not available")

    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", extract_info)
    with caplog.at_level(logging.WARNING, logger="livestream_transcriber.stream.resolver"):
        with pytest.raises(StreamResolutionError, match="Requested format is not available") as exc:
            resolve_stream_sync(_URL, format_selector="best")
    assert "JavaScript" not in str(exc.value)  # classification input unchanged
    warnings = _ytdlp_warnings(caplog)
    assert ["No supported JavaScript runtime" in r.warning for r in warnings] == [True]


def test_ytdlp_progress_lines_do_not_reach_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, warning_clock: list[float]
) -> None:
    yt_dlp = pytest.importorskip("yt_dlp")

    def extract_info(self: Any, url: str, download: bool = False, **_kw: Any) -> dict[str, Any]:
        self.to_screen("[site] Extracting URL: " + url)
        self.write_debug("player client list")
        return dict(_INFO)

    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", extract_info)
    with caplog.at_level(logging.DEBUG):
        resolve_stream_sync(_URL, format_selector="best")
    assert not [
        r
        for r in caplog.records
        if "Extracting URL" in r.getMessage() or "player client" in r.getMessage()
    ]


# --------------------------------------------------------------------- JS runtime


def test_js_runtime_status() -> None:
    def probe_of(**found: tuple[str, bool]) -> Any:
        return lambda name: found.get(name)

    ok, detail = js_runtime_status(probe_of(deno=("2.9.4", True), node=("24.11.1", True)))
    assert (ok, detail) == (True, "deno 2.9.4")
    ok, detail = js_runtime_status(probe_of(node=("24.11.1", True)))
    assert not ok
    assert "no deno on PATH" in detail and "node 24.11.1 found" in detail
    assert "only deno by default" in detail and "Install deno" in detail
    ok, detail = js_runtime_status(probe_of(deno=("1.40.0", False)))
    assert not ok and "deno 1.40.0 is too old" in detail
    ok, detail = js_runtime_status(probe_of())
    assert not ok and "no deno on PATH" in detail and "node" not in detail


def test_js_runtime_probe_uses_ytdlp_minimum_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deno older than yt-dlp's minimum is found but not usable."""
    pytest.importorskip("yt_dlp")
    fake = tmp_path / "deno"
    fake.write_text("#!/bin/sh\necho 'deno 1.46.3 (stable, release, x86_64-unknown-linux-gnu)'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolver._probe_js_runtime("deno") == ("1.46.3", False)
    assert resolver._probe_js_runtime("node") is None
