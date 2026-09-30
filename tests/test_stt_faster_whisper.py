"""FasterWhisperTranscriber against a fake ``faster_whisper`` module.

The fake stands in for the package, so loading, transcription, resampling and
every failure branch run offline in milliseconds with no model files.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from typing import Any

import pytest

from livestream_transcriber.stt.base import LoadBackoff
from livestream_transcriber.stt.providers.faster_whisper import FasterWhisperTranscriber
from tests.support.audio import RATE, tone

PACKAGE = "livestream_transcriber"


class Word:
    def __init__(self, word: str, start: float, end: float, probability: float = 0.9) -> None:
        self.word, self.start, self.end, self.probability = word, start, end, probability


class Segment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0, words=None) -> None:
        self.text, self.start, self.end = text, start, end
        self.words = words
        self.avg_logprob = -0.25
        self.no_speech_prob = 0.01


class FakeFasterWhisper:
    """Records every model construction and every transcribe call."""

    def __init__(
        self,
        *,
        segments: tuple[Segment, ...] = (
            Segment(" Welcome to the stream.", 0.2, 1.8),
            Segment(" Today we talk about tea ", 2.0, 3.5),
        ),
        load_delay: float = 0.0,
        load_error: Exception | None = None,
        transcribe_error: Exception | None = None,
        language: str = "en",
    ) -> None:
        self.segments = segments
        self.load_delay = load_delay
        self.load_error = load_error
        self.transcribe_error = transcribe_error
        self.language = language
        self.constructed: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        fake = self

        class WhisperModel:
            def __init__(self, name: str, **kwargs: Any) -> None:
                fake.constructed.append({"name": name, **kwargs})
                if fake.load_delay:
                    time.sleep(fake.load_delay)
                if fake.load_error is not None:
                    raise fake.load_error

            def transcribe(self, audio, **kwargs: Any):
                fake.calls.append({"samples": len(audio), "dtype": str(audio.dtype), **kwargs})
                if fake.transcribe_error is not None:
                    raise fake.transcribe_error
                info = types.SimpleNamespace(language=fake.language)
                return iter(fake.segments), info

        self.module = types.ModuleType("faster_whisper")
        self.module.WhisperModel = WhisperModel  # type: ignore[attr-defined]


@pytest.fixture
def fake_fw(monkeypatch):
    def install(**kwargs: Any) -> FakeFasterWhisper:
        fake = FakeFasterWhisper(**kwargs)
        monkeypatch.setitem(sys.modules, "faster_whisper", fake.module)
        return fake

    return install


def test_transcribes_and_returns_chunk_relative_segments(fake_fw):
    fake = fake_fw()
    stt = FasterWhisperTranscriber(model="small", threads=1, language="en")
    got = stt.transcribe(tone(), RATE, start=10.0, end=11.0)
    assert got is not None
    assert got.text == "Welcome to the stream. Today we talk about tea"
    assert (got.start, got.end) == (10.0, 11.0)
    assert got.segments == [
        {"start": 0.2, "end": 1.8, "text": "Welcome to the stream.",
         "avg_logprob": -0.25, "no_speech_prob": 0.01},
        {"start": 2.0, "end": 3.5, "text": "Today we talk about tea",
         "avg_logprob": -0.25, "no_speech_prob": 0.01},
    ]  # fmt: skip
    assert got.words is None
    assert (got.provider, got.model, got.degraded) == ("local", "faster-whisper/small", False)
    assert got.language == "en" and got.provider_latency is not None
    assert fake.constructed == [
        {"name": "small", "device": "cpu", "compute_type": "int8", "cpu_threads": 1,
         "num_workers": 1}
    ]  # fmt: skip
    (call,) = fake.calls
    assert call["language"] == "en" and call["samples"] == RATE and call["dtype"] == "float32"


def test_decoding_options_are_passed_through(fake_fw):
    fake = fake_fw()
    stt = FasterWhisperTranscriber(beam_size=5, vad_filter=False, word_timestamps=True)
    stt.transcribe(tone(), RATE, start=0.0, end=1.0)
    (call,) = fake.calls
    assert (call["beam_size"], call["vad_filter"], call["word_timestamps"]) == (5, False, True)
    assert call["language"] is None  # auto-detect is the default


def test_defaults_favour_live_use(fake_fw):
    fake = fake_fw()
    FasterWhisperTranscriber().transcribe(tone(), RATE, start=0.0, end=1.0)
    (call,) = fake.calls
    assert (call["beam_size"], call["vad_filter"], call["word_timestamps"]) == (1, True, False)


def test_word_records_carry_chunk_relative_times(fake_fw):
    fake_fw(
        segments=(
            Segment(
                " hello world",
                0.1,
                1.0,
                words=[Word(" hello", 0.1, 0.5, 0.95), Word(" world", 0.55, 1.0, 0.8)],
            ),
        )
    )
    got = FasterWhisperTranscriber(word_timestamps=True).transcribe(
        tone(), RATE, start=40.0, end=41.0
    )
    assert got is not None
    assert got.words == [
        {"start": 0.1, "end": 0.5, "text": "hello", "confidence": 0.95},
        {"start": 0.55, "end": 1.0, "text": "world", "confidence": 0.8},
    ]


def test_the_detected_language_is_reported_when_none_was_forced(fake_fw):
    fake_fw(language="de")
    got = FasterWhisperTranscriber().transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.language == "de"


def test_the_models_dir_becomes_the_download_root(fake_fw, tmp_path):
    fake = fake_fw()
    FasterWhisperTranscriber(models_dir=tmp_path).transcribe(tone(), RATE, start=0.0, end=1.0)
    assert fake.constructed[0]["download_root"] == str(tmp_path)


def test_non_16k_audio_is_resampled(fake_fw):
    fake = fake_fw()
    FasterWhisperTranscriber().transcribe(tone(1.0, 8000), 8000, start=0.0, end=1.0)
    assert fake.calls[0]["samples"] == 16000


def test_a_short_chunk_never_loads_the_model(fake_fw):
    fake = fake_fw()
    assert FasterWhisperTranscriber().transcribe(tone(0.25), RATE, start=0.0, end=0.25) is None
    assert fake.constructed == []


def test_no_text_is_no_speech(fake_fw):
    fake_fw(segments=(Segment("  "), Segment("")))
    assert FasterWhisperTranscriber().transcribe(tone(), RATE, start=0.0, end=1.0) is None


def test_concurrent_first_calls_load_exactly_one_model(fake_fw):
    fake = fake_fw(load_delay=0.05)
    stt = FasterWhisperTranscriber()
    barrier = threading.Barrier(3)
    results: list[Any] = []

    def worker(start: float) -> None:
        barrier.wait()
        results.append(stt.transcribe(tone(), RATE, start=start, end=start + 1.0))

    threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(fake.constructed) == 1
    assert len(results) == 3 and all(r is not None and r.text for r in results)


def test_a_missing_package_is_unavailable_and_logged_once(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    stt = FasterWhisperTranscriber()
    with caplog.at_level(logging.WARNING, logger=PACKAGE):
        first = stt.transcribe(tone(), RATE, start=0.0, end=1.0)
        second = stt.transcribe(tone(), RATE, start=1.0, end=2.0)
    assert first is not None and first.unavailable
    assert second is not None and second.unavailable
    assert caplog.text.count("faster-whisper is not installed") == 1
    assert "livestream-transcriber[local]" in caplog.text


def test_a_load_failure_is_unavailable_and_not_retried_per_chunk(fake_fw, caplog):
    fake = fake_fw(load_error=RuntimeError("no model files"))
    stt = FasterWhisperTranscriber()
    with caplog.at_level(logging.WARNING, logger=PACKAGE):
        assert stt.transcribe(tone(), RATE, start=0.0, end=1.0).unavailable
        assert stt.transcribe(tone(), RATE, start=1.0, end=2.0).unavailable
    assert len(fake.constructed) == 1
    assert caplog.text.count("failed to load") == 1
    assert stt.describe()["load_failed"] is True


def test_a_failed_load_is_retried_after_the_backoff(fake_fw):
    now = [0.0]
    fake = fake_fw(load_error=RuntimeError("download interrupted"))
    stt = FasterWhisperTranscriber(load_backoff=LoadBackoff(30.0, 120.0, clock=lambda: now[0]))
    assert stt.transcribe(tone(), RATE, start=0.0, end=1.0).unavailable
    now[0] = 29.0
    assert stt.transcribe(tone(), RATE, start=1.0, end=2.0).unavailable
    assert len(fake.constructed) == 1  # still inside the pause

    fake.load_error = None  # the cause went away
    now[0] = 31.0
    got = stt.transcribe(tone(), RATE, start=2.0, end=3.0)
    assert got is not None and not got.unavailable
    assert len(fake.constructed) == 2
    assert stt.describe()["load_failed"] is False


def test_the_load_pause_doubles_up_to_its_maximum():
    now = [0.0]
    backoff = LoadBackoff(10.0, 25.0, clock=lambda: now[0])
    waits = []
    for _ in range(4):
        backoff.failed()
        start = now[0]
        while backoff.blocked:
            now[0] += 1.0
        waits.append(now[0] - start)
    assert waits == [10.0, 20.0, 25.0, 25.0]
    backoff.succeeded()
    assert not backoff.failing


def test_a_decoding_error_is_unavailable_not_a_crash(fake_fw):
    fake_fw(transcribe_error=RuntimeError("ctranslate2 failed"))
    got = FasterWhisperTranscriber().transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.unavailable


def test_a_lazy_decoding_error_is_caught_too(monkeypatch, fake_fw):
    """faster-whisper decodes while the segment generator is consumed."""
    fake = fake_fw()

    def exploding():
        yield Segment("first")
        raise RuntimeError("decoder died mid-stream")

    original = fake.module.WhisperModel.transcribe

    def transcribe(self, audio, **kwargs):
        original(self, audio, **kwargs)
        return exploding(), types.SimpleNamespace(language="en")

    monkeypatch.setattr(fake.module.WhisperModel, "transcribe", transcribe)
    got = FasterWhisperTranscriber().transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.unavailable


def test_repeated_decoding_errors_log_once_not_per_chunk(fake_fw, caplog):
    fake_fw(transcribe_error=RuntimeError("ctranslate2 failed"))
    stt = FasterWhisperTranscriber()
    with caplog.at_level(logging.WARNING, logger=PACKAGE):
        for i in range(50):
            assert stt.transcribe(tone(), RATE, start=float(i), end=i + 1.0).unavailable
    lines = [
        r for r in caplog.records if r.name.startswith(PACKAGE) and r.levelno >= logging.WARNING
    ]
    assert len(lines) == 1
