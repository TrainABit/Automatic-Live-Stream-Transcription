"""The OpenAI provider (and OpenAI-compatible servers): multipart request and mapping."""

from __future__ import annotations

from typing import Any

import pytest

from livestream_transcriber.netutil import HttpError
from livestream_transcriber.stt.providers.openai import OpenAITranscriber
from tests.support.audio import RATE, tone

POST = "livestream_transcriber.stt.providers.openai.post_multipart"


class Server:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url, fields, files, *, headers=None, timeout=60.0):
        self.requests.append(
            {"url": url, "fields": dict(fields), "files": files, "headers": headers}
        )
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def serve(monkeypatch):
    def install(*replies: Any) -> Server:
        server = Server(*replies)
        monkeypatch.setattr(POST, server)
        return server

    return install


def make(**kwargs: Any) -> OpenAITranscriber:
    return OpenAITranscriber("your-openai-key", **kwargs)


def test_multipart_request_shape(serve):
    server = serve({"text": "hello", "language": "english"})
    got = make(language="en", prompt="Names: Ada, Grace").transcribe(
        tone(), RATE, start=10.0, end=11.0
    )
    assert got is not None and got.text == "hello"
    (req,) = server.requests
    assert req["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert req["headers"] == {"Authorization": "Bearer your-openai-key"}
    assert req["fields"] == {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "language": "en",
        "prompt": "Names: Ada, Grace",
    }
    filename, content, ctype = req["files"]["file"]
    assert (filename, ctype) == ("chunk.wav", "audio/wav")
    assert content[:4] == b"RIFF"


def test_language_and_prompt_are_optional(serve):
    server = serve({"text": "x"})
    make().transcribe(tone(), RATE, start=0, end=1)
    assert set(server.requests[0]["fields"]) == {"model", "response_format"}


def test_base_url_points_at_a_compatible_server(serve):
    server = serve({"text": "x"})
    make(base_url="http://localhost:8000/v1/").transcribe(tone(), RATE, start=0, end=1)
    assert server.requests[0]["url"] == "http://localhost:8000/v1/audio/transcriptions"


def test_word_timestamps_ask_for_word_granularity(serve):
    server = serve(
        {
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9},
            ],
        }
    )
    got = make(word_timestamps=True).transcribe(tone(), RATE, start=0, end=1)
    assert server.requests[0]["fields"]["timestamp_granularities[]"] == "word"
    assert got is not None
    assert got.words == [
        {"start": 0.0, "end": 0.4, "text": "hello"},
        {"start": 0.5, "end": 0.9, "text": "world"},
    ]


def test_segments_are_mapped_and_the_span_stays_the_chunks(serve):
    serve({"text": "one two", "segments": [{"start": 0.2, "end": 0.9, "text": "one two"}]})
    got = make().transcribe(tone(2.0), RATE, start=100.0, end=102.0)
    assert got is not None
    assert (got.start, got.end) == (100.0, 102.0)
    assert got.segments == [{"start": 0.2, "end": 0.9, "text": "one two"}]


def test_gpt4o_transcribe_models_use_plain_json(serve):
    server = serve({"text": "x"})
    stt = make(model="gpt-4o-mini-transcribe")
    stt.transcribe(tone(), RATE, start=0, end=1)
    assert server.requests[0]["fields"]["response_format"] == "json"
    assert "timestamp_granularities[]" not in server.requests[0]["fields"]


def test_a_server_without_verbose_json_is_downgraded_once(serve):
    server = serve(HttpError(400, "bad response_format"), {"text": "ok"})
    stt = make()
    got = stt.transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.text == "ok"
    assert [r["fields"]["response_format"] for r in server.requests] == ["verbose_json", "json"]
    assert stt.failures == 0


def test_estimated_cost_feeds_the_budget_guard(serve):
    serve({"text": "x"})
    stt = make()
    stt.transcribe(tone(30.0), RATE, start=0, end=30)
    assert stt.cost_usd == pytest.approx(0.003)
