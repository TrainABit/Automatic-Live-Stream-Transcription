"""Which rendition the default format selector picks.

Runs yt-dlp's real format selection over synthetic format lists shaped like a live HLS
ladder (several video-only renditions plus one audio-only one) and like a VOD with
separate audio formats. No network: the info dict is handed to yt-dlp directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from livestream_transcriber.config import Settings
from livestream_transcriber.stream.resolver import _pick_media
from livestream_transcriber.stream.source import CaptureOptions

yt_dlp = pytest.importorskip("yt_dlp")

DEFAULT_SELECTOR = "bestaudio/best"


def _video(fid: int, height: int) -> dict[str, Any]:
    return {
        "format_id": str(fid),
        "url": f"https://media.example.test/{fid}/index.m3u8",
        "ext": "mp4",
        "protocol": "m3u8_native",
        "height": height,
        "width": height * 16 // 9,
        "vcodec": "avc1.4d4028",
        "acodec": "none",
        "tbr": height / 4,
    }


_AUDIO_HLS = {
    "format_id": "234",
    "url": "https://media.example.test/234/index.m3u8",
    "ext": "mp4",
    "protocol": "m3u8_native",
    "vcodec": "none",  # live HLS audio carries no acodec key at all
    "tbr": 128,
}

_LIVE_LADDER = [_video(269, 144), _video(230, 360), _video(232, 720), _video(270, 1080), _AUDIO_HLS]


def _process(formats: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    info = {
        "id": "abcdefghijk",
        "title": "live",
        "extractor": "generic",
        "extractor_key": "Generic",
        "webpage_url": "https://www.example.test/watch?v=abcdefghijk",
        "is_live": True,
        "formats": [dict(f) for f in formats],
    }
    opts = {"quiet": True, "no_warnings": True, "format": selector}
    with yt_dlp.YoutubeDL(opts) as ydl:
        result: dict[str, Any] = ydl.process_ie_result(info, download=False)
    return result


def _pick(formats: list[dict[str, Any]], selector: str = DEFAULT_SELECTOR) -> str:
    return str(_process(formats, selector)["format_id"])


def test_default_takes_the_audio_only_rendition_of_a_live_ladder() -> None:
    """No video is decoded at all: the transcriber never asks for a video rendition."""
    assert _pick(_LIVE_LADDER) == "234"


def test_the_picked_media_url_is_the_audio_rendition() -> None:
    url, format_id, _headers = _pick_media(_process(_LIVE_LADDER, DEFAULT_SELECTOR))
    assert (url, format_id) == ("https://media.example.test/234/index.m3u8", "234")


def test_a_muxed_only_source_still_resolves() -> None:
    muxed = dict(_video(96, 1080), acodec="mp4a.40.2", format_id="96")
    assert _pick([muxed]) == "96"


def test_a_vod_prefers_the_best_audio_only_format() -> None:
    formats = [
        {"format_id": "139", "url": "https://media.example.test/139", "ext": "m4a",
         "vcodec": "none", "acodec": "mp4a.40.5", "abr": 48},
        {"format_id": "140", "url": "https://media.example.test/140", "ext": "m4a",
         "vcodec": "none", "acodec": "mp4a.40.2", "abr": 128},
        dict(_video(18, 360), acodec="mp4a.40.2"),
    ]  # fmt: skip
    assert _pick(formats) == "140"


def test_the_selector_reaches_settings_and_capture_options() -> None:
    assert Settings().capture_stream_format == DEFAULT_SELECTOR
    assert CaptureOptions().stream_format == DEFAULT_SELECTOR
    custom = Settings(capture_stream_format="worstaudio")
    assert CaptureOptions.from_settings(custom).stream_format == "worstaudio"
