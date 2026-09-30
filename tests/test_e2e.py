"""End to end through the real command line: audio file in, transcript files and alerts out.

Only the speech-to-text is scripted (a JSONL fixture for the ``mock`` provider); capture
uses the real ffmpeg on a synthetic clip, the pipeline, the rule engine, the store, the
sinks and the webhook notifier are the real ones, and the webhook is a loopback server.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from livestream_transcriber.cli import main

from .support.loopback import serve

RULES = """\
version: 1
rules:
  - id: giveaway
    type: keyword
    keywords: [giveaway]
    severity: warning
    description: A giveaway is mentioned
    notify: [webhook]
  - id: release
    type: regex
    pattern: 'version \\d+\\.\\d+'
    severity: info
    notify: [webhook]
"""

FIXTURES = [
    {"start": 0, "end": 5, "text": "Welcome to the stream."},
    {"start": 5, "end": 10, "text": "There is a giveaway tonight, and version 2.4 is out."},
]


@pytest.fixture
def clip(synthetic_clip: Path) -> Path:
    """The synthetic tone: 6 s, so two chunks of the default 5 s file chunking."""
    return synthetic_clip


@pytest.fixture
def fixtures_file(tmp_path: Path) -> Path:
    path = tmp_path / "fixtures.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in FIXTURES) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def rules_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(RULES, encoding="utf-8")
    return path


def test_a_clip_becomes_transcript_files_and_a_webhook_alert(
    clip: Path,
    fixtures_file: Path,
    rules_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    with serve() as server:
        monkeypatch.setenv("LST_NOTIFY_WEBHOOK_URL", f"{server.url}/hook")
        code = main(
            [
                "run", "--url", str(clip), "--stt", "mock", "--mock-fixtures", str(fixtures_file),
                "--rules", str(rules_file), "--out", str(out), "--no-color",
            ]
        )  # fmt: skip
        bodies = server.json_bodies()
    assert code == 0

    # JSONL: one row per utterance, in media order.
    rows = [json.loads(line) for line in (out / "transcript.jsonl").read_text().splitlines()]
    assert [r["text"] for r in rows] == [f["text"] for f in FIXTURES]
    assert [r["start"] for r in rows] == [0.0, 5.0]

    # SubRip and WebVTT carry the same cues with their own timestamp syntax.
    srt = (out / "transcript.srt").read_text()
    assert "00:00:00,000 --> 00:00:05,000" in srt
    assert "giveaway" in srt
    vtt = (out / "transcript.vtt").read_text()
    assert vtt.startswith("WEBVTT")
    assert "00:00:05.000 --> " in vtt

    # The console shows the transcript and the alert.
    shown = capsys.readouterr().out
    assert "Welcome to the stream." in shown
    assert "giveaway" in shown

    # Exactly the two rules that matched were delivered, once each.
    assert len(bodies) == 2
    payload = json.dumps(bodies)
    assert "giveaway" in payload
    assert re.search(r"version 2\.4", payload)

    # The database has the session, the transcripts and the delivered events.
    with sqlite3.connect(out / "lst.db") as db:
        assert db.execute("select count(*) from transcripts").fetchone()[0] == 2
        events = db.execute("select rule_id from events order by rule_id").fetchall()
        assert [e[0] for e in events] == ["giveaway", "release"]


def test_record_then_replay_gives_the_same_transcript(
    clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recordings = tmp_path / "recordings"
    assert main(["record", "demo", "--url", str(clip), "--dir", str(recordings)]) == 0
    assert (recordings / "demo" / "manifest.json").is_file()
    capsys.readouterr()

    texts: list[str] = []
    for n in range(2):
        out = tmp_path / f"replay{n}"
        code = main(
            ["replay", str(recordings / "demo"), "--stt", "mock", "--out", str(out), "--no-color"]
        )
        assert code == 0
        texts.append((out / "transcript.srt").read_text())
    assert texts[0] == texts[1]
    assert "mock transcript 1" in texts[0]


def test_a_stream_url_that_does_not_exist_is_reported_not_swallowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "--url", str(tmp_path / "missing.wav"), "--stt", "mock", "--no-color"])
    assert code == 2
    assert "missing.wav" in capsys.readouterr().err
