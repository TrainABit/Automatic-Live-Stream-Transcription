from __future__ import annotations

import asyncio
import sys

import pytest

from livestream_transcriber.stream.ffmpeg import (
    FFmpegAudioPipe,
    FFmpegSpec,
    build_ffmpeg_command,
    ffmpeg_available,
    is_http_url,
    looks_like_hls,
)


def _pairs(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag]


def test_a_single_input_and_a_single_audio_output_on_stdout() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="https://x.test/y.m3u8"))
    assert _pairs(cmd, "-i") == ["https://x.test/y.m3u8"]
    assert _pairs(cmd, "-map") == ["0:a:0"]
    assert cmd[-1] == "pipe:1"
    assert "-vn" in cmd
    assert cmd.count("ffmpeg") == 1, "never a second binary"


def test_there_is_no_video_output() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="https://x.test/y.m3u8"))
    assert not any(tok in cmd for tok in ("-filter:v", "rawvideo", "0:v:0", "-pix_fmt"))


def test_audio_output_is_mono_pcm_at_the_configured_rate() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="u", sample_rate=22050))
    assert _pairs(cmd, "-ar") == ["22050"]
    assert _pairs(cmd, "-ac") == ["1"]
    assert _pairs(cmd, "-f") == ["s16le"]
    assert _pairs(cmd, "-acodec") == ["pcm_s16le"]


def test_audio_output_fills_timestamp_gaps() -> None:
    """A gap in the input becomes silence, so the sample counter stays a true clock."""
    cmd = build_ffmpeg_command(FFmpegSpec(url="u"))
    assert _pairs(cmd, "-filter:a") == ["aresample=async=1"]


def test_duration_limits_the_output() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="u", duration=12.0))
    assert _pairs(cmd, "-t") == ["12.000"]
    assert cmd.index("-t") > cmd.index("-i"), "-t is an output option"
    assert "-t" not in build_ffmpeg_command(FFmpegSpec(url="u"))


def test_seek_is_an_input_option() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="/tmp/clip.wav", input_seek=3.5))
    assert _pairs(cmd, "-ss") == ["3.500"]
    assert cmd.index("-ss") < cmd.index("-i")


def test_realtime_reads_a_file_at_its_native_rate() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="/tmp/clip.wav", realtime=True))
    assert "-re" in cmd and cmd.index("-re") < cmd.index("-i")
    assert "-re" not in build_ffmpeg_command(FFmpegSpec(url="/tmp/clip.wav"))


def test_http_options_only_for_http_inputs() -> None:
    """ffmpeg errors out when protocol options are passed to other demuxers."""
    local = build_ffmpeg_command(FFmpegSpec(url="/tmp/a.mp4"))
    assert "-reconnect" not in local and "-live_start_index" not in local

    http = build_ffmpeg_command(FFmpegSpec(url="https://x.test/y.m3u8"))
    assert "-reconnect" in http and "-live_start_index" in http
    assert _pairs(http, "-rw_timeout") == ["15000000"]


def test_live_start_index_only_for_hls() -> None:
    plain = build_ffmpeg_command(FFmpegSpec(url="https://x.test/y.mp3"))
    assert "-reconnect" in plain  # still http
    assert "-live_start_index" not in plain  # but not an HLS manifest


def test_live_start_index_can_be_disabled_or_set() -> None:
    off = FFmpegSpec(url="https://x.test/y.m3u8", hls_live_start_index=None)
    assert "-live_start_index" not in build_ffmpeg_command(off)
    on = FFmpegSpec(url="https://x.test/y.m3u8", hls_live_start_index=-5)
    assert _pairs(build_ffmpeg_command(on), "-live_start_index") == ["-5"]


def test_local_inputs_get_no_protocol_options() -> None:
    for url in ("/tmp/window.m3u8", "/tmp/clip.mp4"):
        cmd = build_ffmpeg_command(FFmpegSpec(url=url))
        assert "-protocol_whitelist" not in cmd
        assert "-reconnect" not in cmd


def test_progress_stats_are_off() -> None:
    """The periodic "size= time= speed=" line would push real errors out of the
    40-line stderr tail kept for diagnostics."""
    cmd = build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8"))
    assert "-nostats" in cmd
    assert cmd.index("-nostats") < cmd.index("-i")
    assert "-nostdin" in cmd


def test_thread_queue_size_is_opt_in() -> None:
    assert "-thread_queue_size" not in build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8"))
    sized = build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8", thread_queue_size=1024))
    assert _pairs(sized, "-thread_queue_size") == ["1024"]


def test_http_proxy_never_reaches_the_argv() -> None:
    """A proxy URL carries credentials and the argv is world-readable
    (``/proc/<pid>/cmdline``); the child's environment carries the proxy instead."""
    proxy = "http://proxy-user:your-proxy-password@proxy.example.test:8877"
    for url in ("https://x.test/a.m3u8", "http://127.0.0.1:9/a.m3u8", "/tmp/window.m3u8"):
        cmd = build_ffmpeg_command(FFmpegSpec(url=url, http_proxy=proxy))
        assert "-http_proxy" not in cmd
        assert not any("your-proxy-password" in tok or "proxy.example.test" in tok for tok in cmd)


def test_remote_hls_requests_gzip() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8"))
    assert _pairs(cmd, "-headers") == ["Accept-Encoding: gzip\r\n"]
    assert "-headers" not in build_ffmpeg_command(FFmpegSpec(url="/tmp/window.m3u8"))
    assert "-headers" not in build_ffmpeg_command(FFmpegSpec(url="http://127.0.0.1:9/a.m3u8"))
    assert "-headers" not in build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.mp3"))


def test_resolver_headers_are_passed_but_credentials_are_not() -> None:
    spec = FFmpegSpec(
        url="https://x.test/a.m3u8",
        headers={
            "Referer": "https://x.test/",
            "Cookie": "session=your-cookie-value",
            "Authorization": "Bearer your-token",
            "User-Agent": "ignored-here",
            "Accept-Encoding": "identity",
            "X-Bad": "line\r\nInjected: yes",
        },
    )
    (block,) = _pairs(build_ffmpeg_command(spec), "-headers")
    assert block == "Referer: https://x.test/\r\nAccept-Encoding: gzip\r\n"
    cmd = build_ffmpeg_command(spec)
    assert not any("your-cookie-value" in tok or "your-token" in tok for tok in cmd)


def test_user_agent_is_http_only() -> None:
    http = build_ffmpeg_command(FFmpegSpec(url="https://x.test/a.m3u8", user_agent="TestUA/1"))
    assert _pairs(http, "-user_agent") == ["TestUA/1"]
    local = build_ffmpeg_command(FFmpegSpec(url="/tmp/window.m3u8", user_agent="TestUA/1"))
    assert "-user_agent" not in local


def test_extra_input_args_come_before_the_input() -> None:
    cmd = build_ffmpeg_command(FFmpegSpec(url="u", extra_input_args=("-f", "wav")))
    assert cmd[cmd.index("-i") - 2 : cmd.index("-i") + 2] == ["-f", "wav", "-i", "u"]


def test_url_shape_helpers() -> None:
    assert is_http_url("https://x") and is_http_url("http://x") and not is_http_url("/tmp/x")
    assert looks_like_hls("https://x.test/a/index.m3u8?token=1")
    assert not looks_like_hls("https://x.test/a.mp3")
    assert ffmpeg_available("definitely-not-a-binary-name") is None


async def test_a_missing_binary_is_a_clear_error() -> None:
    pipe = FFmpegAudioPipe(FFmpegSpec(url="u", binary="definitely-not-a-binary-name"))
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        await pipe.start()


async def test_start_twice_is_refused() -> None:
    pipe = FFmpegAudioPipe(FFmpegSpec(url="u", binary=sys.executable))
    pipe._proc = object()
    with pytest.raises(RuntimeError, match="already started"):
        await pipe.start()
    pipe._proc = None


async def test_spawn_failure_leaves_nothing_to_clean_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def spawn_fails(*_a: object, **_k: object) -> None:
        raise OSError("exec failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_fails)
    pipe = FFmpegAudioPipe(FFmpegSpec(url="https://x.test/v.mp3", binary=sys.executable))
    with pytest.raises(OSError, match="exec failed"):
        await pipe.start()
    await pipe.stop()  # idempotent, and safe after a failed start
