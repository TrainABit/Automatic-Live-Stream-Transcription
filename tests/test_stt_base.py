"""The transcript type, the offline stand-ins and the audio helpers of the STT core."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from livestream_transcriber.netutil import NonJsonBody
from livestream_transcriber.stt.base import (
    STT_UNAVAILABLE,
    FixtureTranscriber,
    MockTranscriber,
    NullTranscriber,
    Transcript,
    bad_body,
    normalise_timed_items,
    pcm_is_silent,
    pcm_to_wav,
    too_short,
    transcribe_chunk,
    unavailable,
)
from tests.support.audio import RATE, chunk, silence, tone


def test_pcm_silence_gate():
    assert pcm_is_silent(silence())
    assert pcm_is_silent(b"")
    assert not pcm_is_silent(tone())


def test_pcm_silence_gate_threshold_is_a_peak_level():
    quiet = b"\x0a\x00" * RATE  # amplitude 10, about -70 dBFS
    assert pcm_is_silent(quiet)
    assert not pcm_is_silent(quiet, threshold_dbfs=-80.0)


def test_too_short_is_half_a_second():
    assert too_short(tone(0.4), RATE)
    assert not too_short(tone(0.5), RATE)
    assert too_short(tone(0.4, 8000), 8000)


def test_pcm_to_wav_is_a_readable_mono_wav():
    wav = pcm_to_wav(tone(0.5), RATE)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        assert w.getnframes() == RATE // 2


def test_transcribe_chunk_skips_silence_and_keeps_the_session_span():
    mock = MockTranscriber("hello world")
    assert transcribe_chunk(mock, chunk(10.0, silence())) is None
    assert mock.calls == []

    got = transcribe_chunk(mock, chunk(1234.2, tone(5.0)))
    assert got is not None
    assert (got.text, got.start, got.end) == ("hello world", 1234.2, 1239.2)
    assert mock.calls == [(1234.2, 1239.2, len(tone(5.0)))]


def test_transcribe_chunk_can_skip_the_gate():
    mock = MockTranscriber("hello")
    assert transcribe_chunk(mock, chunk(0.0, silence()), skip_silence=False) is not None


def test_mock_script_and_null_never_invent():
    scripted = MockTranscriber(script=["one", None, "three"])
    assert scripted.transcribe(b"x", RATE, start=0, end=1).text == "one"
    assert scripted.transcribe(b"x", RATE, start=1, end=2) is None
    assert scripted.transcribe(b"x", RATE, start=2, end=3).text == "three"
    assert scripted.transcribe(b"x", RATE, start=3, end=4) is None  # script used up
    assert NullTranscriber().transcribe(b"x", RATE, start=0, end=1) is None


def test_mock_numbers_its_calls_on_request():
    mock = MockTranscriber("part {n}")
    texts = [mock.transcribe(b"x", RATE, start=i, end=i + 1).text for i in range(3)]
    assert texts == ["part 1", "part 2", "part 3"]


def test_mock_script_keeps_extras_but_takes_the_chunk_span():
    item = Transcript(
        start=0, end=0, text="scripted", status=STT_UNAVAILABLE, words=[{"text": "scripted"}]
    )
    got = MockTranscriber(script=[item]).transcribe(b"x", RATE, start=7.0, end=9.0)
    assert got is not None
    assert (got.start, got.end, got.unavailable, got.words) == (7.0, 9.0, True, item.words)


def test_fixture_matches_by_time_overlap(tmp_path: Path):
    path = tmp_path / "speech.jsonl"
    path.write_text(
        "# comment header must not break the loader\n"
        '{"start": 100.0, "end": 104.0, "text": "welcome to the stream"}\n'
        '{"start_ts": 200.0, "end_ts": 203.0, "text": "see you soon"}\n'
        '{"start": 300.0, "end": 301.0, "text": "   "}\n',
        encoding="utf-8",
    )
    stt = FixtureTranscriber.from_jsonl(path)
    assert len(stt.utterances) == 2
    hit = stt.transcribe(tone(), RATE, start=101.0, end=106.0)
    assert hit is not None
    assert (hit.text, hit.start, hit.end) == ("welcome to the stream", 100.0, 104.0)
    assert stt.transcribe(tone(), RATE, start=10.0, end=12.0) is None
    assert stt.transcribe(tone(), RATE, start=104.0, end=110.0) is None  # touching is not overlap


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ('{"text": "no times"}', "missing"),
        ('{"start": 1.0, "text": "no end"}', "missing"),
        ('{"start": 5.0, "end": 5.0, "text": "empty span"}', "must be after"),
        ("not json", "invalid fixture row"),
        ("[1, 2]", "invalid fixture row"),
    ],
)
def test_a_fixture_row_without_usable_times_is_rejected(tmp_path: Path, row: str, problem: str):
    path = tmp_path / "speech.jsonl"
    path.write_text(f'{{"start": 0, "end": 1, "text": "fine"}}\n{row}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=problem) as info:
        FixtureTranscriber.from_jsonl(path)
    assert f"{path}:2" in str(info.value)


def test_transcript_describe_and_flags():
    t = Transcript(start=1.234, end=2.5, text="hi", provider="local", degraded=True)
    assert t.describe() == {
        "start": 1.23,
        "end": 2.5,
        "text": "hi",
        "confidence": None,
        "provider": "local",
        "degraded": True,
    }
    assert not t.unavailable
    assert t.duration == pytest.approx(1.266)
    gone = unavailable(0, 1, model="m", quota_exceeded=True)
    assert gone.unavailable and gone.quota_exceeded and gone.text == ""
    assert gone.describe()["status"] == STT_UNAVAILABLE


def test_bad_body_classification():
    assert bad_body(NonJsonBody(text="<html>")) == "non_json_body"
    assert bad_body({}) == "empty_body"
    assert bad_body({"text": ""}) is None


def test_normalise_timed_items_keeps_native_times_and_never_invents_them():
    items = normalise_timed_items(
        [
            {"word": "hello", "start": 0.5, "end": 0.9, "confidence": 0.8},
            {"text": "no times", "start": None, "end": True},
            "not a record",
        ],
    )
    assert items == [
        {"start": 0.5, "end": 0.9, "text": "hello", "confidence": 0.8},
        {"start": None, "end": None, "text": "no times"},
    ]
    assert normalise_timed_items([]) is None
    assert normalise_timed_items(None) is None
