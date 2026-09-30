"""The benchmark runner: clip discovery, references, chunking, providers, reports."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from livestream_transcriber.bench.runner import (
    BenchError,
    ProviderSpec,
    decode_audio,
    discover_clips,
    parse_provider_specs,
    read_reference,
    run_bench,
    split_pcm,
    write_report,
)
from livestream_transcriber.config import ConfigError, Settings
from livestream_transcriber.stt.base import MockTranscriber, Transcriber, unavailable

from .support.audio import RATE, silence, tone

SRT = """\
1
00:00:00,000 --> 00:00:02,000
Hello <i>there</i>,

2
00:00:02,000 --> 00:00:04,000
{\\an8}general kenobi
"""

VTT = """\
WEBVTT
Kind: captions
Language: en

NOTE this is a comment

00:00:00.000 --> 00:00:02.000 align:start
Hello there,

00:00:02.000 --> 00:00:04.000
general kenobi
"""


def write_wav(path: Path, pcm: bytes) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    return path


def make_clips(directory: Path) -> Path:
    """Two 6 s clips: one with a reference, one without."""
    directory.mkdir()
    write_wav(directory / "talk.wav", tone(6.0))
    (directory / "talk.txt").write_text("hello world\n", encoding="utf-8")
    write_wav(directory / "noise.wav", tone(6.0))
    return directory


def stt_factory(text: str = "hello world"):  # type: ignore[no-untyped-def]
    def factory(_settings: Settings, _spec: ProviderSpec) -> Transcriber:
        return MockTranscriber(text=text)

    return factory


def decode_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


class TestProviderSpecs:
    def test_providers_and_models(self) -> None:
        specs = parse_provider_specs("local:tiny, OpenAI ,local:tiny,,onnx")
        assert [s.label for s in specs] == ["local:tiny", "openai", "onnx"]
        assert specs[0].model == "tiny"
        assert specs[1].model is None

    def test_a_model_may_contain_slashes_and_colons(self) -> None:
        (spec,) = parse_provider_specs("openrouter:openai/whisper-large-v3")
        assert (spec.provider, spec.model) == ("openrouter", "openai/whisper-large-v3")

    def test_nothing_is_an_error(self) -> None:
        with pytest.raises(BenchError):
            parse_provider_specs(" , ")


class TestClips:
    def test_audio_is_paired_with_its_reference(self, tmp_path: Path) -> None:
        clips = discover_clips(make_clips(tmp_path / "clips"))
        assert [c.name for c in clips] == ["noise", "talk"]
        assert clips[0].reference is None
        assert clips[1].reference == tmp_path / "clips" / "talk.txt"

    def test_srt_is_preferred_over_nothing_and_txt_over_srt(self, tmp_path: Path) -> None:
        base = tmp_path / "clips"
        base.mkdir()
        write_wav(base / "a.wav", tone(1.0))
        (base / "a.srt").write_text(SRT, encoding="utf-8")
        (base / "a.txt").write_text("plain", encoding="utf-8")
        (clip,) = discover_clips(base)
        assert clip.reference == base / "a.txt"

    def test_other_files_are_ignored(self, tmp_path: Path) -> None:
        base = make_clips(tmp_path / "clips")
        (base / "README.md").write_text("x")
        (base / "orphan.txt").write_text("no audio")
        assert [c.name for c in discover_clips(base)] == ["noise", "talk"]

    def test_a_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(BenchError, match="not found"):
            discover_clips(tmp_path / "nope")

    def test_an_empty_directory_says_what_to_put_there(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(BenchError, match=r"no audio clips"):
            discover_clips(tmp_path / "empty")


class TestReferences:
    def test_plain_text_is_whitespace_normalised(self, tmp_path: Path) -> None:
        path = tmp_path / "r.txt"
        path.write_text("  one\n two\t three  \n", encoding="utf-8")
        assert read_reference(path) == "one two three"

    def test_srt_keeps_only_the_words(self, tmp_path: Path) -> None:
        path = tmp_path / "r.srt"
        path.write_text(SRT, encoding="utf-8")
        assert read_reference(path) == "Hello there, general kenobi"

    def test_vtt_drops_headers_notes_and_timing(self, tmp_path: Path) -> None:
        path = tmp_path / "r.vtt"
        path.write_text(VTT, encoding="utf-8")
        assert read_reference(path) == "Hello there, general kenobi"

    def test_a_byte_order_mark_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "r.txt"
        path.write_bytes(b"\xef\xbb\xbfhello")
        assert read_reference(path) == "hello"


class TestChunking:
    def test_even_split(self) -> None:
        pieces = split_pcm(tone(10.0), RATE, 5.0)
        assert [(s, e) for s, e, _ in pieces] == [(0.0, 5.0), (5.0, 10.0)]

    def test_a_short_tail_is_merged_into_the_previous_chunk(self) -> None:
        pieces = split_pcm(tone(5.2), RATE, 5.0)
        assert len(pieces) == 1
        assert pieces[0][1] == pytest.approx(5.2)

    def test_a_long_enough_tail_stays_a_chunk(self) -> None:
        pieces = split_pcm(tone(6.0), RATE, 5.0)
        assert [round(e - s, 1) for s, e, _ in pieces] == [5.0, 1.0]

    def test_chunks_cover_the_audio_exactly(self) -> None:
        pcm = tone(7.3)
        assert b"".join(p for _, _, p in split_pcm(pcm, RATE, 2.0)) == pcm

    def test_a_non_positive_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_seconds"):
            split_pcm(tone(1.0), RATE, 0)

    def test_no_audio_no_chunks(self) -> None:
        assert split_pcm(b"", RATE, 5.0) == []


class TestDecoding:
    def test_a_wav_is_decoded_to_the_same_samples(self, ffmpeg_bin: str, tmp_path: Path) -> None:
        pcm = tone(1.0)
        path = write_wav(tmp_path / "a.wav", pcm)
        assert decode_audio(path, ffmpeg=ffmpeg_bin) == pcm

    def test_other_rates_are_resampled(self, ffmpeg_bin: str, tmp_path: Path) -> None:
        path = tmp_path / "a.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(b"\x00\x10" * 8000)
        decoded = decode_audio(path, ffmpeg=ffmpeg_bin)
        assert len(decoded) / 2 == pytest.approx(RATE, rel=0.02)

    def test_garbage_is_a_bench_error(self, ffmpeg_bin: str, tmp_path: Path) -> None:
        path = tmp_path / "broken.wav"
        path.write_bytes(b"this is not audio")
        with pytest.raises(BenchError, match="could not decode"):
            decode_audio(path, ffmpeg=ffmpeg_bin)

    def test_a_missing_ffmpeg_says_how_to_get_it(self, tmp_path: Path) -> None:
        with pytest.raises(BenchError, match="not found on PATH"):
            decode_audio(tmp_path / "a.wav", ffmpeg="definitely-not-ffmpeg")


class TestRunBench:
    def run(self, tmp_path: Path, **kwargs):  # type: ignore[no-untyped-def]
        clips = make_clips(tmp_path / "clips")
        kwargs.setdefault("transcriber_factory", stt_factory())
        return run_bench(
            clips,
            kwargs.pop("specs", [ProviderSpec("mock")]),
            Settings(),
            decoder=decode_wav,
            **kwargs,
        )

    def test_a_perfect_provider_scores_zero_errors(self, tmp_path: Path) -> None:
        report = self.run(tmp_path, chunk_seconds=10.0)
        (provider,) = report.providers
        talk = next(c for c in provider.clips if c.name == "talk")
        assert talk.hypothesis == "hello world"
        assert talk.wer is not None and talk.wer.rate == 0.0
        assert provider.wer is not None and provider.wer.rate == 0.0
        # The clip without a reference reports latency only.
        noise = next(c for c in provider.clips if c.name == "noise")
        assert noise.wer is None
        assert len(noise.latencies) == 1

    def test_a_wrong_provider_is_scored(self, tmp_path: Path) -> None:
        report = self.run(
            tmp_path, chunk_seconds=10.0, transcriber_factory=stt_factory("goodbye world")
        )
        (provider,) = report.providers
        assert provider.wer is not None
        assert provider.wer.rate == pytest.approx(0.5)

    def test_one_request_per_chunk_and_a_cold_pass_before(self, tmp_path: Path) -> None:
        made: list[MockTranscriber] = []

        def factory(_s: Settings, _spec: ProviderSpec) -> Transcriber:
            made.append(MockTranscriber(text="x"))
            return made[-1]

        self.run(tmp_path, chunk_seconds=3.0, transcriber_factory=factory)
        # 2 clips x 2 chunks (3 s + 3 s), and one warm-up call on the first audible chunk.
        assert len(made[0].calls) == 5
        assert made[0].calls[0] == made[0].calls[1]

    def test_the_cold_pass_can_be_switched_off(self, tmp_path: Path) -> None:
        report = self.run(tmp_path, chunk_seconds=10.0, cold_pass=False)
        assert report.providers[0].cold_start_seconds is None

    def test_silent_chunks_are_skipped_not_sent(self, tmp_path: Path) -> None:
        clips = tmp_path / "clips"
        clips.mkdir()
        write_wav(clips / "quiet.wav", silence(6.0))
        stt = MockTranscriber(text="never")
        report = run_bench(
            clips,
            [ProviderSpec("mock")],
            Settings(),
            transcriber_factory=lambda _s, _p: stt,
            decoder=decode_wav,
            chunk_seconds=10.0,
        )
        assert stt.calls == []
        clip = report.providers[0].clips[0]
        assert (clip.chunks, clip.skipped_silent) == (1, 1)

    def test_unavailable_answers_count_as_failures(self, tmp_path: Path) -> None:
        class Failing:
            def transcribe(self, pcm: bytes, sample_rate: int, *, start: float, end: float):  # type: ignore[no-untyped-def]
                return unavailable(start, end, model="x")

        report = self.run(
            tmp_path, chunk_seconds=10.0, transcriber_factory=lambda _s, _p: Failing()
        )
        assert report.providers[0].failures == 2
        assert report.providers[0].latencies == []

    def test_boilerplate_output_is_counted_as_a_hallucination(self, tmp_path: Path) -> None:
        report = self.run(
            tmp_path, chunk_seconds=10.0, transcriber_factory=stt_factory("Thanks for watching!")
        )
        assert report.providers[0].hallucinations == 2

    def test_a_provider_that_cannot_start_is_reported_and_skipped(self, tmp_path: Path) -> None:
        def factory(_s: Settings, spec: ProviderSpec) -> Transcriber:
            if spec.provider == "openai":
                raise ConfigError("an API key is required: set LST_OPENAI_API_KEY")
            return MockTranscriber(text="hello world")

        report = self.run(
            tmp_path,
            specs=[ProviderSpec("openai"), ProviderSpec("mock")],
            transcriber_factory=factory,
            chunk_seconds=10.0,
        )
        broken, working = report.providers
        assert broken.error is not None and "LST_OPENAI_API_KEY" in broken.error
        assert broken.clips == []
        assert working.error is None and len(working.clips) == 2
        assert "unavailable" in report.to_markdown()

    def test_progress_is_reported(self, tmp_path: Path) -> None:
        seen: list[str] = []
        self.run(tmp_path, chunk_seconds=10.0, progress=seen.append)
        assert any("cold pass" in line for line in seen)
        assert any("talk" in line for line in seen)

    def test_the_default_model_of_the_provider_is_listed(self, tmp_path: Path) -> None:
        report = self.run(tmp_path, chunk_seconds=10.0)
        assert report.providers[0].model == "mock"

    def test_the_default_factory_builds_the_real_provider(self, tmp_path: Path) -> None:
        clips = make_clips(tmp_path / "clips")
        report = run_bench(
            clips, [ProviderSpec("mock")], Settings(), decoder=decode_wav, chunk_seconds=10.0
        )
        assert report.providers[0].error is None
        assert report.providers[0].clips[0].hypothesis.startswith("mock transcript")


class TestReports:
    def report(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        clips = make_clips(tmp_path / "clips")
        return run_bench(
            clips,
            [ProviderSpec("mock")],
            Settings(),
            transcriber_factory=stt_factory(),
            decoder=decode_wav,
            chunk_seconds=10.0,
        )

    def test_json_has_the_environment_and_per_clip_results(self, tmp_path: Path) -> None:
        data = json.loads(self.report(tmp_path).to_json())
        assert data["clips"] == ["noise", "talk"]
        (provider,) = data["providers"]
        assert provider["provider"] == "mock"
        assert provider["wer"]["rate"] == 0.0
        assert {c["name"] for c in provider["per_clip"]} == {"noise", "talk"}
        env = data["environment"]
        assert env["chunk_seconds"] == 10.0
        assert "python" in env and "platform" in env

    def test_the_environment_does_not_name_the_host_or_user(self, tmp_path: Path) -> None:
        text = self.report(tmp_path).to_json()
        assert "/Users/" not in text
        assert "/home/" not in text

    def test_markdown_has_a_row_per_provider_and_per_clip(self, tmp_path: Path) -> None:
        text = self.report(tmp_path).to_markdown()
        assert "| Provider | Model | WER |" in text
        assert "| mock | mock | 0.0% |" in text
        assert "| mock:mock | talk |" in text
        assert "## Per clip" in text

    def test_write_report_creates_parent_directories(self, tmp_path: Path) -> None:
        report = self.report(tmp_path)
        write_report(report, tmp_path / "a" / "bench.json", tmp_path / "b" / "bench.md")
        assert json.loads((tmp_path / "a" / "bench.json").read_text())["clips"]
        assert (tmp_path / "b" / "bench.md").read_text().startswith("# Benchmark")

    def test_write_report_with_no_paths_writes_nothing(self, tmp_path: Path) -> None:
        write_report(self.report(tmp_path), None, None)
        assert not (tmp_path / "bench.json").exists()


class TestCli:
    def test_bench_prints_a_table_and_writes_the_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.cli import main

        clips = make_clips(tmp_path / "clips")
        code = main(
            [
                "bench", "--clips", str(clips), "--stt", "mock", "--out", str(tmp_path / "b.json"),
                "--markdown", str(tmp_path / "b.md"), "--chunk-seconds", "10",
            ]
        )  # fmt: skip
        assert code == 0
        assert "| mock | mock |" in capsys.readouterr().out
        assert json.loads((tmp_path / "b.json").read_text())["providers"]
        assert (tmp_path / "b.md").is_file()

    def test_bench_without_clips_is_a_config_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.cli import main

        assert main(["bench", "--clips", str(tmp_path / "none"), "--stt", "mock"]) == 2
        assert "not found" in capsys.readouterr().err

    def test_bench_where_no_provider_can_start_is_a_config_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from livestream_transcriber.cli import main

        clips = make_clips(tmp_path / "clips")
        assert main(["bench", "--clips", str(clips), "--stt", "openai"]) == 2
        assert "unavailable" in capsys.readouterr().out
