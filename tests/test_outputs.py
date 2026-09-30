"""Transcript sinks: JSONL, SRT, VTT, console, SQLite and the composite."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from livestream_transcriber.outputs import (
    CompositeSink,
    ConsoleSink,
    JsonlSink,
    SqliteSink,
    SrtSink,
    TimedText,
    TranscriptSegment,
    VttSink,
    format_timestamp,
)
from livestream_transcriber.rules import RuleEngine, RuleSet, TextSegment
from livestream_transcriber.store import Database


def segment(
    text: str = "hello world", start: float = 1.0, end: float = 3.5, **kw: object
) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, **kw)  # type: ignore[arg-type]


class TestTimestamps:
    @pytest.mark.parametrize(
        ("seconds", "srt", "vtt"),
        [
            (0, "00:00:00,000", "00:00:00.000"),
            (1.5, "00:00:01,500", "00:00:01.500"),
            (61.007, "00:01:01,007", "00:01:01.007"),
            (3600, "01:00:00,000", "01:00:00.000"),
            (3725.25, "01:02:05,250", "01:02:05.250"),
            (359999.999, "99:59:59,999", "99:59:59.999"),
            (59.9996, "00:01:00,000", "00:01:00.000"),  # rounds up across the minute
            (-4, "00:00:00,000", "00:00:00.000"),
            (0.0004, "00:00:00,000", "00:00:00.000"),
            (0.0005, "00:00:00,001", "00:00:00.001"),
        ],
    )
    def test_format(self, seconds: float, srt: str, vtt: str) -> None:
        assert format_timestamp(seconds) == srt
        assert format_timestamp(seconds, decimal=".") == vtt


class TestProviderSegmentsBecomeCues:
    def test_a_segment_with_provider_times_is_split_into_cues(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt")
        sink.write(
            segment(
                "one two",
                10.0,
                15.0,
                parts=(TimedText(10.2, 11.5, "one"), TimedText(12.0, 14.0, "two")),
            )
        )
        sink.close()
        assert (tmp_path / "a.srt").read_text(encoding="utf-8") == (
            "1\n00:00:10,200 --> 00:00:11,500\none\n\n2\n00:00:12,000 --> 00:00:14,000\ntwo\n\n"
        )

    def test_from_transcript_reads_and_clamps_the_segment_times(self) -> None:
        class Result:
            start, end, text = 10.0, 12.0, "one two"
            segments = [
                {"start": 9.5, "end": 11.0, "text": "one"},
                {"start": 10.5, "end": 13.0, "text": "two"},
            ]

        parts = TranscriptSegment.from_transcript(Result()).parts
        assert [(p.start, p.end, p.text) for p in parts] == [
            (10.0, 11.0, "one"),
            (11.0, 12.0, "two"),
        ]

    def test_unusable_provider_times_fall_back_to_one_cue(self) -> None:
        class Result:
            start, end, text = 10.0, 12.0, "one two"
            segments = [
                {"start": 10.0, "end": 11.0, "text": "one"},
                {"start": None, "end": None, "text": "two"},
            ]

        assert TranscriptSegment.from_transcript(Result()).parts == ()


class TestSrt:
    def test_writes_numbered_cues(self, tmp_path: Path) -> None:
        path = tmp_path / "out" / "a.srt"
        sink = SrtSink(path)
        sink.write(segment("First line", 0.0, 2.0))
        sink.write(segment("Second line", 2.5, 4.125))
        sink.close()
        assert path.read_text(encoding="utf-8") == (
            "1\n00:00:00,000 --> 00:00:02,000\nFirst line\n\n"
            "2\n00:00:02,500 --> 00:00:04,125\nSecond line\n\n"
        )

    def test_is_readable_while_still_open(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt")
        sink.write(segment("live"))
        assert "live" in (tmp_path / "a.srt").read_text(encoding="utf-8")
        sink.close()

    def test_long_lines_wrap_and_whitespace_collapses(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt", max_line_chars=20)
        sink.write(segment("this  is a\nfairly long sentence that needs wrapping"))
        sink.close()
        body = (tmp_path / "a.srt").read_text(encoding="utf-8").split("\n")[2:-2]
        assert len(body) >= 3 and all(len(line) <= 20 for line in body)
        assert " ".join(body) == "this is a fairly long sentence that needs wrapping"

    def test_blank_text_is_skipped_and_zero_length_cues_are_stretched(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt")
        sink.write(segment("   "))
        sink.write(segment("instant", 5.0, 5.0))
        sink.close()
        text = (tmp_path / "a.srt").read_text(encoding="utf-8")
        assert text.startswith("1\n00:00:05,000 --> 00:00:05,500\ninstant")
        assert "2\n" not in text

    def test_arrow_in_text_cannot_break_the_format(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt")
        sink.write(segment("a --> b"))
        sink.close()
        assert (tmp_path / "a.srt").read_text(encoding="utf-8").count("-->") == 1

    def test_append_continues_numbering(self, tmp_path: Path) -> None:
        path = tmp_path / "a.srt"
        first = SrtSink(path)
        first.write(segment("one"))
        first.close()
        second = SrtSink(path, append=True)
        second.write(segment("two", 5, 6))
        second.close()
        text = path.read_text(encoding="utf-8")
        assert "\n2\n00:00:05,000 --> 00:00:06,000\ntwo" in text

    def test_default_mode_overwrites(self, tmp_path: Path) -> None:
        path = tmp_path / "a.srt"
        for word in ("old", "new"):
            sink = SrtSink(path)
            sink.write(segment(word))
            sink.close()
        assert "old" not in path.read_text(encoding="utf-8")

    def test_writing_after_close_is_an_error(self, tmp_path: Path) -> None:
        sink = SrtSink(tmp_path / "a.srt")
        sink.close()
        sink.close()  # idempotent
        with pytest.raises(ValueError, match="closed"):
            sink.write(segment())


class TestVtt:
    def test_header_and_dot_separated_milliseconds(self, tmp_path: Path) -> None:
        path = tmp_path / "a.vtt"
        sink = VttSink(path)
        sink.write(segment("Hello", 0.0, 1.5))
        sink.write(segment("Wörld", 3725.25, 3727))
        sink.close()
        assert path.read_text(encoding="utf-8") == (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.500\nHello\n\n"
            "01:02:05.250 --> 01:02:07.000\nWörld\n\n"
        )

    def test_header_is_written_once_when_appending(self, tmp_path: Path) -> None:
        path = tmp_path / "a.vtt"
        for word in ("one", "two"):
            sink = VttSink(path, append=True)
            sink.write(segment(word))
            sink.close()
        assert path.read_text(encoding="utf-8").count("WEBVTT") == 1

    def test_markup_is_escaped(self, tmp_path: Path) -> None:
        path = tmp_path / "a.vtt"
        sink = VttSink(path)
        sink.write(segment("<b>fish</b> & chips"))
        sink.close()
        assert "&lt;b>fish&lt;/b> &amp; chips" in path.read_text(encoding="utf-8")


class TestJsonl:
    def test_one_object_per_segment_with_the_documented_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "t" / "a.jsonl"
        sink = JsonlSink(path)
        sink.write(
            segment(
                "Willkommen zum Stream",
                1.23456,
                4.0,
                language="de",
                provider="mock",
                model="m",
                latency=0.98765,
                wallclock=1_700_000_000.5,
            )
        )
        sink.write(segment("second"))
        sink.close()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first == {
            "start": 1.235,
            "end": 4.0,
            "text": "Willkommen zum Stream",
            "language": "de",
            "provider": "mock",
            "model": "m",
            "confidence": None,
            "latency": 0.988,
            "wallclock": 1_700_000_000.5,
        }
        assert json.loads(lines[1])["language"] is None
        assert "Willkommen" in lines[0]  # not ASCII-escaped

    def test_each_line_is_flushed_immediately(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path / "a.jsonl")
        sink.write(segment("visible"))
        assert "visible" in (tmp_path / "a.jsonl").read_text(encoding="utf-8")
        sink.close()

    def test_appends_by_default_and_can_truncate(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        for _ in range(2):
            sink = JsonlSink(path)
            sink.write(segment())
            sink.close()
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        fresh = JsonlSink(path, append=False)
        fresh.close()
        assert path.read_text(encoding="utf-8") == ""

    def test_text_with_newlines_stays_on_one_line(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path / "a.jsonl")
        sink.write(segment("line one\nline two"))
        sink.close()
        (line,) = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(line)["text"] == "line one\nline two"


class TestConsole:
    def test_pretty_line_with_stream_time(self) -> None:
        out = io.StringIO()
        sink = ConsoleSink(out, color=False)
        sink.write(segment(" hello there ", 3725.9, 3728))
        assert out.getvalue() == "[1:02:05] hello there\n"

    def test_language_is_optional(self) -> None:
        out = io.StringIO()
        ConsoleSink(out, color=False, show_language=True).write(segment("hi", language="de"))
        assert out.getvalue() == "[0:00:01] (de) hi\n"

    def test_colour_only_when_enabled(self) -> None:
        plain, coloured = io.StringIO(), io.StringIO()
        ConsoleSink(plain, color=False).write(segment())
        ConsoleSink(coloured, color=True).write(segment())
        assert "\033[" not in plain.getvalue()
        assert "\033[" in coloured.getvalue()

    def test_no_colour_on_a_non_terminal_by_default(self) -> None:
        out = io.StringIO()
        ConsoleSink(out).write(segment())
        assert "\033[" not in out.getvalue()

    def test_no_color_environment_variable_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")

        class Tty(io.StringIO):
            def isatty(self) -> bool:
                return True

        out = Tty()
        ConsoleSink(out).write(segment())
        assert "\033[" not in out.getvalue()

    def test_rule_hits_are_highlighted_under_their_line(self) -> None:
        rules = RuleSet.from_yaml(
            "rules: [{id: giveaway, type: keyword, keywords: [giveaway], severity: warning}]"
        )
        (hit,) = RuleEngine(rules).evaluate(TextSegment("a giveaway", start=0, end=1))
        out = io.StringIO()
        sink = ConsoleSink(out, color=False)
        sink.write(segment("a giveaway"))
        sink.write_hit(hit)
        assert out.getvalue().splitlines() == [
            "[0:00:01] a giveaway",
            "    >> WARNING giveaway: giveaway",
        ]
        coloured = io.StringIO()
        ConsoleSink(coloured, color=True).write_hit(hit)
        assert "\033[1;33m" in coloured.getvalue()


class TestSqliteSink:
    def test_segments_land_in_the_transcripts_table(self, tmp_path: Path) -> None:
        with Database(tmp_path / "lst.db") as db:
            sid = db.start_session(url="u")
            sink = SqliteSink(db, sid)
            sink.write(segment("stored", 1, 2, language="en", provider="mock", latency=0.5))
            sink.close()
            (row,) = db.recent_transcripts(5, session_id=sid)
            assert (row["text"], row["language"], row["provider"], row["latency"]) == (
                "stored", "en", "mock", 0.5,
            )  # fmt: skip
            assert (row["start_s"], row["end_s"]) == (1.0, 2.0)
            db.counts()  # still open: the sink does not close the database


class TestTranscriptSegment:
    def test_from_transcript_adapts_stt_results(self) -> None:
        class Result:
            start, end, text = 2.0, 4.0, "hi"
            provider, model = "mock", "m"
            provider_latency, confidence = 0.3, 0.8

        seg = TranscriptSegment.from_transcript(Result(), language="fr", wallclock=9.0)
        assert (seg.start, seg.end, seg.text, seg.language) == (2.0, 4.0, "hi", "fr")
        assert (seg.provider, seg.model, seg.latency, seg.confidence) == ("mock", "m", 0.3, 0.8)
        assert seg.wallclock == 9.0

    def test_from_transcript_tolerates_missing_optional_attributes(self) -> None:
        class Bare:
            start, end, text = 0.0, 1.0, "x"

        seg = TranscriptSegment.from_transcript(Bare())
        assert seg.provider is None and seg.latency is None


class TestCompositeSink:
    def test_fans_out_and_isolates_failures(self, tmp_path: Path) -> None:
        class Broken:
            def write(self, segment: TranscriptSegment) -> None:
                raise RuntimeError("disk full")

            def close(self) -> None:
                raise RuntimeError("cannot close")

        good = JsonlSink(tmp_path / "a.jsonl")
        composite = CompositeSink([Broken(), good])
        composite.write(segment("survives"))
        composite.close()
        assert "survives" in (tmp_path / "a.jsonl").read_text(encoding="utf-8")
        with pytest.raises(ValueError):  # the good sink really was closed
            good.write(segment())
