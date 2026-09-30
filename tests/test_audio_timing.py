"""Session-clock timestamps and narrowing a rule match to the words that matched."""

from __future__ import annotations

import pytest

from livestream_transcriber.audio.speech import SpeechStream
from livestream_transcriber.audio.timing import (
    align_match_timing,
    sessionize_transcript_timestamps,
)
from livestream_transcriber.stt.base import Transcript


def words(*items: tuple[str, float, float]) -> list[dict]:
    return [{"text": text, "start": start, "end": end} for text, start, end in items]


# ------------------------------------------------------------- sessionizing --


def test_sessionize_shifts_relative_word_times():
    raw = Transcript(start=10.0, end=15.0, text="hello", words=words(("hello", 0.5, 1.0)))
    shifted = sessionize_transcript_timestamps(raw)
    assert shifted.words[0]["start"] == pytest.approx(10.5)
    assert shifted.words[0]["end"] == pytest.approx(11.0)
    assert raw.words[0]["start"] == 0.5  # the provider's own records are never mutated


def test_sessionize_normalises_unicode_to_nfc():
    decomposed = "überlegen"
    got = sessionize_transcript_timestamps(Transcript(start=0, end=1, text=decomposed))
    assert got.text == "überlegen"


def test_a_transcript_without_times_is_returned_as_is():
    raw = Transcript(start=3.0, end=5.0, text="plain")
    assert sessionize_transcript_timestamps(raw) is raw


def test_sessionize_is_idempotent_early_in_the_session():
    """A word at 2.6 s inside a chunk starting at 2.5 s must not be shifted twice."""
    raw = Transcript(
        start=2.5, end=5.0, text="goodbye",
        words=words(("goodbye", 0.35, 0.60)),
        segments=[{"text": "goodbye", "start": 0.0, "end": 0.9}],
    )  # fmt: skip
    once = sessionize_transcript_timestamps(raw)
    twice = sessionize_transcript_timestamps(once)
    assert (once.words[0]["start"], once.words[0]["end"]) == pytest.approx((2.85, 3.10))
    assert (twice.words[0]["start"], twice.words[0]["end"]) == pytest.approx((2.85, 3.10))
    assert (twice.segments[0]["start"], twice.segments[0]["end"]) == pytest.approx((2.5, 3.4))


def test_a_record_lands_on_one_clock():
    """A last word that ends past the chunk is still chunk-relative: both values shift."""
    raw = Transcript(start=10.0, end=12.5, text="goodbye", words=words(("goodbye", 2.3, 3.1)))
    word = sessionize_transcript_timestamps(raw).words[0]
    assert (word["start"], word["end"]) == pytest.approx((12.3, 13.1))


def test_records_without_numeric_times_are_kept_but_marked():
    raw = Transcript(
        start=5.0, end=8.0, text="a b",
        words=[{"text": "a", "start": None, "end": None}, {"text": "b", "start": 0.5, "end": 0.9}],
    )  # fmt: skip
    got = sessionize_transcript_timestamps(raw).words
    assert got[0]["start"] is None and got[1]["start"] == pytest.approx(5.5)


def test_a_merged_rolling_transcript_keeps_session_word_times():
    stream = SpeechStream()
    first = sessionize_transcript_timestamps(
        Transcript(
            start=2.5, end=5.0, text="I am now",
            words=words(("I", 1.6, 1.8), ("am", 1.9, 2.1), ("now", 2.2, 2.45)),
        )
    )  # fmt: skip
    second = sessionize_transcript_timestamps(
        Transcript(
            start=5.0, end=7.5, text="now leaving",
            words=words(("now", 0.05, 0.30), ("leaving", 0.35, 0.60)),
        )
    )  # fmt: skip
    stream.ingest(first)
    merged = stream.ingest(second)
    assert merged.revision and merged.transcript is not None
    spoken = merged.transcript
    assert (spoken.start, spoken.end) == (2.5, 7.5)
    assert align_match_timing(2.5, 7.5, spoken, matched_text="leaving") == pytest.approx(
        (5.35, 5.60)
    )


# ------------------------------------------------------------ narrowing a match --


def test_word_timestamps_narrow_the_interval_to_the_matched_words():
    chunk_start, chunk_end = 4984.0, 4989.0
    transcript = Transcript(
        start=chunk_start, end=chunk_end, text="so welcome back everyone",
        words=words(("so", 2.10, 2.30), ("welcome", 2.72, 3.20), ("back", 3.25, 3.50),
                    ("everyone", 3.60, 4.10)),
    )  # fmt: skip
    start, end = align_match_timing(chunk_start, chunk_end, transcript, matched_text="welcome back")
    assert start == pytest.approx(4986.72) and end == pytest.approx(4987.50)
    assert chunk_end - end >= 1.5  # much earlier than the chunk end


def test_a_single_word_match():
    transcript = Transcript(
        start=100.0, end=105.0, text="please subscribe now",
        words=words(("please", 0.5, 0.9), ("subscribe", 1.0, 1.7), ("now", 1.8, 2.0)),
    )  # fmt: skip
    assert align_match_timing(100.0, 105.0, transcript, matched_text="Subscribe!") == pytest.approx(
        (101.0, 101.7)
    )


def test_a_repeated_phrase_picks_the_occurrence_nearest_the_given_interval():
    transcript = Transcript(
        start=0.0, end=10.0, text="go go stop go",
        words=words(("go", 0.5, 0.8), ("go", 1.0, 1.3), ("stop", 4.0, 4.5), ("go", 8.0, 8.4)),
    )  # fmt: skip
    assert align_match_timing(7.5, 9.0, transcript, matched_text="go") == pytest.approx((8.0, 8.4))
    assert align_match_timing(0.0, 1.5, transcript, matched_text="go") == pytest.approx((0.5, 0.8))


def test_an_exact_word_beats_a_word_that_merely_contains_the_text():
    transcript = Transcript(
        start=0.0, end=6.0, text="drink in the ink",
        words=words(("drink", 0.5, 0.9), ("in", 1.0, 1.2), ("the", 1.3, 1.4), ("ink", 3.0, 3.4)),
    )  # fmt: skip
    assert align_match_timing(0.0, 6.0, transcript, matched_text="in") == pytest.approx((1.0, 1.2))


def test_a_match_survives_different_tokenisation():
    """The provider split a hyphenated word that the matcher saw as one."""
    transcript = Transcript(
        start=0.0, end=5.0, text="send an e mail today",
        words=words(("send", 0.2, 0.5), ("an", 0.6, 0.7), ("e", 1.0, 1.1), ("mail", 1.15, 1.6),
                    ("today", 2.0, 2.5)),
    )  # fmt: skip
    assert align_match_timing(0.0, 5.0, transcript, matched_text="e-mail") == pytest.approx(
        (1.0, 1.6)
    )


def test_a_match_inside_a_longer_word_maps_to_that_word():
    transcript = Transcript(
        start=0.0, end=5.0, text="very profitable",
        words=words(("very", 0.2, 0.5), ("profitable", 0.6, 1.4)),
    )  # fmt: skip
    assert align_match_timing(0.0, 5.0, transcript, matched_text="profit") == pytest.approx(
        (0.6, 1.4)
    )


def test_without_word_times_the_containing_segment_is_used():
    transcript = Transcript(
        start=20.0, end=30.0, text="first part. second part.",
        segments=[
            {"start": 0.0, "end": 4.0, "text": "first part."},
            {"start": 4.5, "end": 9.0, "text": "second part."},
        ],
    )  # fmt: skip
    assert align_match_timing(20.0, 30.0, transcript, matched_text="second") == pytest.approx(
        (24.5, 29.0)
    )


def test_without_any_timestamps_the_whole_chunk_is_kept():
    transcript = Transcript(start=100.0, end=102.5, text="see you later")
    assert align_match_timing(100.0, 102.5, transcript, matched_text="later") == (100.0, 102.5)


def test_records_without_times_place_nothing():
    transcript = Transcript(
        start=200.0, end=205.0, text="see you later",
        words=[{"text": "see", "start": None, "end": None},
               {"text": "later", "start": None, "end": None}],
    )  # fmt: skip
    assert align_match_timing(200.0, 205.0, transcript, matched_text="later") == (200.0, 205.0)


def test_words_that_do_not_hold_the_match_keep_the_whole_span():
    """A stitched utterance can carry another chunk's word records."""
    transcript = Transcript(
        start=400.0, end=405.0, text="see you later",
        words=words(("look", 0.2, 0.5), ("over", 0.55, 0.7), ("there", 0.75, 1.0)),
    )  # fmt: skip
    assert align_match_timing(400.0, 405.0, transcript, matched_text="later") == (400.0, 405.0)


def test_no_matched_text_only_clamps_the_interval():
    transcript = Transcript(start=10.0, end=15.0, text="x", words=words(("x", 0.5, 1.0)))
    assert align_match_timing(9.0, 20.0, transcript) == (10.0, 15.0)
    assert align_match_timing(11.0, 12.0, transcript, matched_text="   ") == (11.0, 12.0)


def test_early_session_words_are_shifted_once_when_aligning():
    raw = Transcript(
        start=2.5, end=5.0, text="all done", words=words(("all", 0.10, 0.30), ("done", 0.35, 0.60))
    )  # fmt: skip
    spoken = sessionize_transcript_timestamps(raw)  # what the pipeline does on arrival
    assert align_match_timing(2.5, 5.0, spoken, matched_text="all done") == pytest.approx(
        (2.60, 3.10)
    )
    # Aligning the raw transcript directly gives the same answer.
    assert align_match_timing(2.5, 5.0, raw, matched_text="all done") == pytest.approx((2.60, 3.10))


def test_a_zero_length_word_gets_a_minimum_span_inside_the_chunk():
    transcript = Transcript(start=0.0, end=5.0, text="hi", words=words(("hi", 2.0, 2.0)))
    assert align_match_timing(0.0, 5.0, transcript, matched_text="hi") == pytest.approx((2.0, 2.35))
    edge = Transcript(start=0.0, end=5.0, text="hi", words=words(("hi", 5.0, 5.0)))
    start, end = align_match_timing(0.0, 5.0, edge, matched_text="hi")
    assert 0.0 <= start <= end <= 5.0


def test_a_word_far_past_the_chunk_end_is_still_chunk_relative():
    """Times are never guessed to be absolute: a provider that overruns is shifted like the rest."""
    raw = Transcript(start=10.0, end=12.0, text="late", words=words(("late", 3.0, 3.4)))
    word = sessionize_transcript_timestamps(raw).words[0]
    assert (word["start"], word["end"]) == pytest.approx((13.0, 13.4))
