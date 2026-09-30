"""WER/CER counts, the hallucination checks and the latency statistics."""

from __future__ import annotations

import pytest

from livestream_transcriber.bench.metrics import (
    ErrorCounts,
    aggregate,
    cer,
    edit_counts,
    has_repetition,
    is_boilerplate,
    is_hallucination,
    latency_stats,
    percentile,
    real_time_factor,
    wer,
)


class TestEditCounts:
    def test_identical_sequences_have_no_errors(self) -> None:
        assert edit_counts(["a", "b"], ["a", "b"]) == ErrorCounts(0, 0, 0, 2)

    def test_one_substitution(self) -> None:
        assert edit_counts(["a", "b", "c"], ["a", "x", "c"]) == ErrorCounts(1, 0, 0, 3)

    def test_one_deletion(self) -> None:
        assert edit_counts(["a", "b", "c"], ["a", "c"]) == ErrorCounts(0, 1, 0, 3)

    def test_one_insertion(self) -> None:
        assert edit_counts(["a", "c"], ["a", "b", "c"]) == ErrorCounts(0, 0, 1, 2)

    def test_the_cheapest_alignment_wins(self) -> None:
        # Two ways to explain it: 1 substitution + 1 deletion, or 2 deletions + 2 insertions.
        counts = edit_counts(list("abcd"), list("axc"))
        assert counts.errors == 2
        assert counts.reference_length == 4

    def test_everything_missing_is_all_deletions(self) -> None:
        assert edit_counts(["a", "b"], []) == ErrorCounts(0, 2, 0, 2)

    def test_everything_invented_is_all_insertions(self) -> None:
        assert edit_counts([], ["a", "b"]) == ErrorCounts(0, 0, 2, 0)


class TestWer:
    def test_a_known_value(self) -> None:
        # reference has 6 words; one substituted, one dropped: 2 / 6.
        counts = wer("the quick brown fox jumps over", "the quick red fox over")
        assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 1, 0)
        assert counts.rate == pytest.approx(2 / 6)

    def test_case_and_punctuation_are_not_errors(self) -> None:
        assert wer("Hello, World!", "hello world").rate == 0.0

    def test_unicode_composition_is_not_an_error(self) -> None:
        assert wer("café", "café").rate == 0.0

    def test_an_empty_reference_and_an_empty_hypothesis_agree(self) -> None:
        counts = wer("", "")
        assert counts.rate == 0.0
        assert counts.reference_length == 0

    def test_an_empty_reference_with_text_has_no_defined_rate(self) -> None:
        counts = wer("", "something")
        assert counts.insertions == 1
        assert counts.rate is None

    def test_an_empty_hypothesis_is_a_wer_of_one(self) -> None:
        assert wer("one two three", "").rate == 1.0

    def test_the_rate_can_exceed_one(self) -> None:
        assert wer("hi", "hi there all of you").rate == pytest.approx(4.0)


class TestCer:
    def test_a_known_value(self) -> None:
        # "kitten" -> "sitting": the classic distance of 3 over 6 characters.
        counts = cer("kitten", "sitting")
        assert counts.errors == 3
        assert counts.rate == pytest.approx(3 / 6)

    def test_spaces_are_characters(self) -> None:
        assert cer("a b", "ab").deletions == 1


class TestAggregate:
    def test_counts_are_summed_not_rates_averaged(self) -> None:
        short = wer("yes", "no")  # 100 % on one word
        long = wer(" ".join(["word"] * 99), " ".join(["word"] * 99))  # 0 % on 99 words
        total = aggregate([short, long])
        assert total.rate == pytest.approx(1 / 100)
        assert total.reference_length == 100

    def test_nothing_aggregates_to_zero(self) -> None:
        assert aggregate([]) == ErrorCounts()

    def test_describe_rounds_the_rate(self) -> None:
        described = wer("a b c", "a b d").describe()
        assert described["rate"] == pytest.approx(0.3333, abs=1e-4)
        assert described["substitutions"] == 1


class TestHallucinations:
    @pytest.mark.parametrize(
        "text",
        [
            "Thanks for watching!",
            "Thank you for watching.",
            "Please like and subscribe",
            "Subtitles by the community",
            "Untertitel im Auftrag des ZDF",
            "Bis zum nächsten Mal",
        ],
    )
    def test_stock_phrases_are_flagged(self, text: str) -> None:
        assert is_boilerplate(text)
        assert is_hallucination(text)

    def test_the_same_words_inside_a_long_sentence_are_speech(self) -> None:
        text = (
            "and that is the end of the quarterly report so thanks for watching "
            "everyone and see you soon at the next meeting"
        )
        assert not is_boilerplate(text)

    def test_ordinary_speech_is_not_flagged(self) -> None:
        assert not is_hallucination("The release ships on Tuesday.")

    def test_a_loop_is_a_repetition(self) -> None:
        assert has_repetition("well well well well well")
        assert has_repetition("go home go home go home go home")

    def test_a_short_echo_is_not(self) -> None:
        assert not has_repetition("no no no")

    def test_empty_text_is_fine(self) -> None:
        assert not is_hallucination("")


class TestLatency:
    def test_percentile_is_nearest_rank_of_a_measured_value(self) -> None:
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        assert percentile(values, 0.5) == 3.0
        assert percentile(values, 0.95) == 5.0
        assert percentile(values, 0.0) == 1.0

    def test_percentile_of_nothing_is_none(self) -> None:
        assert percentile([], 0.5) is None

    def test_stats(self) -> None:
        stats = latency_stats([1.0, 2.0, 3.0, 4.0])
        assert stats == {"n": 4, "mean": 2.5, "p50": 2.0, "p95": 4.0, "max": 4.0}

    def test_stats_of_nothing(self) -> None:
        assert latency_stats([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def test_real_time_factor(self) -> None:
        assert real_time_factor(2.0, 10.0) == pytest.approx(0.2)
        assert real_time_factor(1.0, 0.0) is None
