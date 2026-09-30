"""CaptureOptions, the source constructor helpers and the package's public surface."""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path

import pytest
from pydantic import SecretStr

import livestream_transcriber.stream as stream
from livestream_transcriber.config import Settings
from livestream_transcriber.models import AudioChunk
from livestream_transcriber.stream.source import CaptureOptions, LiveStreamSource

SRC = Path(stream.__file__).resolve().parents[1]


def test_defaults_mirror_the_settings_defaults() -> None:
    assert CaptureOptions.from_settings(Settings()) == CaptureOptions()


def test_options_follow_the_settings_fields() -> None:
    settings = Settings(
        capture_sample_rate=8000,
        capture_live_chunk_seconds=1.5,
        capture_file_chunk_seconds=4.0,
        capture_queue_size=7,
        capture_file_queue_size=3,
        capture_reconnect_initial_delay=2.0,
        capture_reconnect_max_delay=9.0,
        capture_max_reconnect_attempts=4,
        capture_resolve_cache_seconds=5.0,
        capture_audio_stall_seconds=12.0,
        capture_stream_format="worstaudio",
        capture_ffmpeg_binary="/opt/ffmpeg",
        capture_ffmpeg_loglevel="error",
        capture_hls_window=True,
        capture_hls_live_start_index=-5,
    )
    options = CaptureOptions.from_settings(settings)
    assert (options.sample_rate, options.live_chunk_seconds, options.file_chunk_seconds) == (
        8000,
        1.5,
        4.0,
    )
    assert (options.queue_size, options.file_queue_size) == (7, 3)
    assert (options.reconnect_initial_delay, options.reconnect_max_delay) == (2.0, 9.0)
    assert options.max_reconnect_attempts == 4 and options.resolve_cache_seconds == 5.0
    assert options.audio_stall_seconds == 12.0 and options.stream_format == "worstaudio"
    assert options.ffmpeg_binary == "/opt/ffmpeg" and options.ffmpeg_loglevel == "error"
    assert options.hls_window is True and options.hls_live_start_index == -5


def test_chunk_sizes_are_whole_samples() -> None:
    options = CaptureOptions(sample_rate=16000, live_chunk_seconds=2.5, file_chunk_seconds=5.0)
    assert options.chunk_bytes_for(live=True) == 80000
    assert options.chunk_bytes_for(live=False) == 160000
    assert options.chunk_seconds_for(live=True) == 2.5
    odd = CaptureOptions(sample_rate=44100, live_chunk_seconds=0.3333)
    assert odd.chunk_bytes_for(live=True) % 2 == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"sample_rate": 100},
        {"sample_rate": 96000},
        {"live_chunk_seconds": 0},
        {"file_chunk_seconds": -1},
        {"queue_size": 0},
        {"file_queue_size": 0},
        {"reconnect_initial_delay": -1},
        {"max_reconnect_attempts": -1},
    ],
)
def test_invalid_options_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CaptureOptions(**bad)


def test_the_options_are_immutable() -> None:
    with pytest.raises(AttributeError):
        CaptureOptions().sample_rate = 8000


def test_a_source_built_from_settings_carries_cookies_and_proxy(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    settings = Settings(
        capture_cookies_file=cookies,
        capture_proxy=SecretStr("http://proxy.example.test:8080"),
        capture_queue_size=9,
    )
    source = LiveStreamSource.from_settings("https://example.test/live", settings, duration=5.0)
    assert source._cookiefile == cookies
    assert source._http_proxy == "http://proxy.example.test:8080"
    assert source.options.queue_size == 9
    assert source._duration == 5.0
    bare = LiveStreamSource.from_settings("https://example.test/live", Settings())
    assert bare._cookiefile is None and bare._http_proxy is None


def test_the_proxy_secret_is_not_in_the_source_repr_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    settings = Settings(
        capture_proxy=SecretStr("http://user:your-proxy-password@proxy.example.test")
    )
    source = LiveStreamSource.from_settings("https://example.test/live", settings)
    assert "your-proxy-password" not in repr(source.options)
    assert "your-proxy-password" not in caplog.text


async def test_overflow_logging_thins_out_but_every_drop_is_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A persistently slow consumer must not log an ERROR per chunk."""
    import asyncio
    import time

    from livestream_transcriber.models import SegmentInfo
    from livestream_transcriber.stream.ffmpeg import FFmpegAudioPipe

    class _Pipe(FFmpegAudioPipe):
        def __init__(self, blocks: int) -> None:
            self.blocks = blocks

        async def read_audio(self):
            for _ in range(self.blocks):
                yield b"\x01\x00" * 1600  # 0.1 s
                await asyncio.sleep(0)

    caplog.set_level(logging.ERROR)
    source = LiveStreamSource(
        "https://example.test/live", CaptureOptions(live_chunk_seconds=0.1, queue_size=1)
    )
    segment = SegmentInfo(index=0, started_at=time.time(), offset=0.0)
    await source._pump_audio(_Pipe(250), segment, is_live=True)
    assert source.stats.audio_chunks_dropped == 249
    overflow = [r for r in caplog.records if "queue overflow" in r.getMessage()]
    assert 1 <= len(overflow) <= 6
    assert overflow[0].dropped_total == 1
    source._close_queue()


def test_every_exported_stream_error_is_raised_somewhere() -> None:
    """A public error type nothing raises would mislead a caller that catches it."""
    errors = [
        name
        for name in stream.__all__
        if inspect.isclass(getattr(stream, name))
        and issubclass(getattr(stream, name), BaseException)
    ]
    assert errors
    code = "\n".join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
    never_made = [name for name in errors if not re.search(rf"(?<!class )\b{name}\(", code)]
    assert never_made == []


def test_the_stream_package_never_kills_the_process() -> None:
    """A capture library reports a stuck capture through a callback; whether the process
    should exit is the application's decision, not a hidden ``os._exit`` in here."""
    for path in (SRC / "stream").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "os._exit" not in text and "sys.exit" not in text, path.name


def test_the_public_names_import() -> None:
    for name in stream.__all__:
        assert hasattr(stream, name), name
    assert AudioChunk  # the chunk type the whole package produces
