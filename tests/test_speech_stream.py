"""Stitching and de-duplicating overlapping chunk transcripts (neutral phrases)."""

from __future__ import annotations

from livestream_transcriber.audio.speech import MIXED, SpeechStream, stitch
from livestream_transcriber.stt.base import Transcript


def t(text: str, start: float, end: float, **kwargs) -> Transcript:
    return Transcript(start=start, end=end, text=text, **kwargs)


def test_stitch_joins_overlapping_windows():
    assert stitch("we are going live in", "live in five minutes") == (
        "we are going live in five minutes"
    )
    assert stitch("we are going", "are going live now") == "we are going live now"


def test_stitch_keeps_the_original_spelling_and_punctuation():
    assert stitch("Willkommen zum Stream,", "zum stream, heute") == "Willkommen zum Stream, heute"
    assert stitch("Hello, world.", "world. Again") == "Hello, world. Again"


def test_stitch_needs_a_real_overlap():
    assert stitch("hello there", "general kenobi") is None
    assert stitch("", "anything") is None
    assert stitch("...", "anything") is None


def test_an_exact_duplicate_is_suppressed():
    stream = SpeechStream()
    first = stream.ingest(t("welcome to the stream", 10.0, 15.0))
    second = stream.ingest(t("Welcome to the stream!", 12.0, 17.0))
    assert first.transcript is not None and first.revision is False
    assert second.transcript is None
    assert (len(stream.raw), len(stream.normalized)) == (2, 1)


def test_a_subset_of_the_previous_utterance_is_suppressed():
    stream = SpeechStream()
    stream.ingest(t("welcome to the stream", 10.0, 15.0))
    assert stream.ingest(t("to the stream", 12.0, 16.0)).transcript is None


def test_an_extension_is_a_revision_not_a_new_utterance():
    stream = SpeechStream()
    stream.ingest(t("welcome to the", 10.0, 14.0))
    got = stream.ingest(t("welcome to the stream", 12.0, 16.0))
    assert got.revision is True and got.transcript is not None
    assert got.transcript.text == "welcome to the stream"
    assert (got.transcript.start, got.transcript.end) == (10.0, 16.0)
    assert len(stream.normalized) == 1


def test_a_suffix_prefix_overlap_merges_and_keeps_the_casing():
    stream = SpeechStream()
    stream.ingest(t("We are going live", 10.0, 14.0))
    got = stream.ingest(t("going live in five minutes", 13.5, 16.0))
    assert got.revision is True and got.transcript is not None
    assert got.transcript.text == "We are going live in five minutes"
    assert len(stream.normalized) == 1


def test_the_same_words_in_another_order_is_a_repeat():
    stream = SpeechStream(jaccard_dup=0.75)
    stream.ingest(t("one two three four", 10.0, 14.0))
    assert stream.ingest(t("three one two four", 13.0, 17.0)).transcript is None
    assert len(stream.normalized) == 1


def test_a_lower_similarity_threshold_is_respected():
    strict = SpeechStream(jaccard_dup=0.95)
    strict.ingest(t("one two three four", 10.0, 14.0))
    assert strict.ingest(t("one two three five", 13.0, 17.0)).transcript is not None
    loose = SpeechStream(jaccard_dup=0.5)
    loose.ingest(t("one two three four", 10.0, 14.0))
    assert loose.ingest(t("one two three five", 13.0, 17.0)).transcript is None


def test_distinct_speech_is_a_new_utterance():
    stream = SpeechStream()
    stream.ingest(t("good morning everyone", 10.0, 14.0))
    got = stream.ingest(t("let us begin with the news", 14.0, 18.0))
    assert got.transcript is not None and got.revision is False
    assert len(stream.normalized) == 2


def test_far_apart_utterances_are_not_fused():
    stream = SpeechStream(max_gap=2.5)
    stream.ingest(t("see you tomorrow", 10.0, 14.0))
    got = stream.ingest(t("see you tomorrow", 30.0, 34.0))
    assert got.revision is False and got.transcript is not None
    assert len(stream.normalized) == 2


def test_empty_text_adds_nothing():
    stream = SpeechStream()
    assert stream.ingest(t("  ...  ", 0.0, 1.0)).transcript is None
    assert len(stream.raw) == 1 and not stream.normalized


def test_provenance_survives_a_merge():
    stream = SpeechStream()
    stream.ingest(t("we are going", 0.0, 3.0, provider="openai", model="whisper-1"))
    got = stream.ingest(
        t("are going live", 3.0, 6.0, provider="local", model="faster-whisper/small", degraded=True)
    )
    merged = got.transcript
    assert merged is not None and got.revision
    assert (merged.provider, merged.model) == (MIXED, MIXED)
    assert merged.degraded is True


def test_a_single_provider_label_is_kept():
    stream = SpeechStream()
    stream.ingest(t("we are going", 0.0, 3.0, provider="openai", model="whisper-1"))
    got = stream.ingest(t("are going live", 3.0, 6.0, provider="openai", model="whisper-1"))
    assert got.transcript is not None
    assert (got.transcript.provider, got.transcript.model) == ("openai", "whisper-1")
    assert got.transcript.degraded is False


def test_a_stitch_merges_word_and_segment_records_without_duplicates():
    stream = SpeechStream()
    stream.ingest(
        t(
            "we are going",
            0.0,
            3.0,
            words=[
                {"start": 0.1, "end": 0.4, "text": "we"},
                {"start": 1.0, "end": 1.4, "text": "are"},
                {"start": 2.0, "end": 2.6, "text": "going"},
            ],
        )
    )
    got = stream.ingest(
        t(
            "are going live",
            3.0,
            6.0,
            words=[
                {"start": 1.0, "end": 1.4, "text": "are"},
                {"start": 2.0, "end": 2.6, "text": "going"},
                {"start": 3.2, "end": 3.6, "text": "live"},
            ],
        )
    )
    assert got.transcript is not None
    assert [w["text"] for w in got.transcript.words] == ["we", "are", "going", "live"]


def test_history_is_bounded_however_long_the_stream_runs():
    stream = SpeechStream(max_history=10)
    for i in range(500):
        stream.ingest(t(f"utterance number {i} spoken here", i * 20.0, i * 20.0 + 4.0))
    assert len(stream.raw) == 10 and len(stream.normalized) == 10
    assert stream.normalized[-1].text == "utterance number 499 spoken here"


def test_a_word_inside_a_longer_word_is_not_a_repeat():
    stream = SpeechStream()
    stream.ingest(t("we can not know what happens", 10.0, 14.0))
    got = stream.ingest(t("no", 14.5, 15.0))
    assert got.transcript is not None
