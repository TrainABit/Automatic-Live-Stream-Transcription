"""The OpenRouter provider: request shape, response mapping, cost, quota, breaker.

HTTP is replaced at the ``netutil`` function the provider imported, so no socket
is opened and every request the provider makes can be inspected.
"""

from __future__ import annotations

import base64
import io
import wave
from typing import Any

import pytest

from livestream_transcriber.netutil import HttpError
from livestream_transcriber.stt.base import CircuitBreaker
from livestream_transcriber.stt.providers.openrouter import (
    DEFAULT_MODEL,
    OpenRouterTranscriber,
)
from tests.support.audio import RATE, Clock, tone

POST = "livestream_transcriber.stt.providers.openrouter.post_json"
MODEL = "openai/whisper-large-v3"


class Server:
    """A scripted endpoint: a queue of replies (or exceptions), every request recorded."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url, payload, *, headers=None, timeout=30.0):
        self.requests.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
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


def make(**kwargs: Any) -> OpenRouterTranscriber:
    return OpenRouterTranscriber("your-openrouter-key", **kwargs)


def test_request_shape(serve):
    server = serve({"text": "hello there"})
    stt = make(language="de", word_timestamps=True)
    got = stt.transcribe(tone(1.0), RATE, start=3.0, end=4.0)
    assert got is not None and got.text == "hello there"
    (req,) = server.requests
    assert req["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert req["headers"] == {"Authorization": "Bearer your-openrouter-key"}
    body = req["payload"]
    assert body["model"] == DEFAULT_MODEL == MODEL
    assert body["language"] == "de"
    assert body["response_format"] == "verbose_json"
    assert body["timestamp_granularities"] == ["segment", "word"]
    assert body["input_audio"]["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(body["input_audio"]["data"]))) as w:
        assert w.getframerate() == RATE and w.getnframes() == RATE


def test_language_is_omitted_when_auto(serve):
    server = serve({"text": "x"})
    make().transcribe(tone(), RATE, start=0, end=1)
    assert "language" not in server.requests[0]["payload"]


def test_custom_base_url(serve):
    server = serve({"text": "x"})
    make(base_url="http://localhost:9000/api/v1/").transcribe(tone(), RATE, start=0, end=1)
    assert server.requests[0]["url"] == "http://localhost:9000/api/v1/audio/transcriptions"


def test_segments_words_language_and_reported_cost_are_mapped(serve):
    serve(
        {
            "text": "good morning",
            "language": "en",
            "segments": [{"start": 0.1, "end": 1.4, "text": "good morning", "avg_logprob": -0.2}],
            "words": [
                {"word": "good", "start": 0.1, "end": 0.5},
                {"word": "morning", "start": 0.6, "end": 1.4},
            ],
            "usage": {"cost": 0.0021},
        }
    )
    stt = make()
    got = stt.transcribe(tone(2.0), RATE, start=50.0, end=52.0)
    assert got is not None
    assert (got.start, got.end) == (50.0, 52.0)  # the chunk span; segment times stay relative
    assert got.segments == [{"start": 0.1, "end": 1.4, "text": "good morning", "avg_logprob": -0.2}]
    assert [w["text"] for w in got.words] == ["good", "morning"]
    assert (got.language, got.provider, got.model) == ("en", "openrouter", MODEL)
    assert got.cost_usd == pytest.approx(0.0021)
    assert got.provider_latency is not None
    assert stt.cost_usd == pytest.approx(0.0021) and stt.requests == 1


def test_cost_is_estimated_when_the_api_reports_none(serve):
    serve({"text": "hi"})
    stt = make(usd_per_second=0.001)
    got = stt.transcribe(tone(2.0), RATE, start=0.0, end=2.0)
    assert got is not None and got.cost_usd == pytest.approx(0.002)


def test_text_is_rebuilt_from_segments_when_missing(serve):
    serve({"text": "", "segments": [{"start": 0, "end": 1, "text": " one "}, {"text": "two"}]})
    got = make().transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.text == "one two"


def test_a_paid_empty_answer_is_no_speech_not_a_failure(serve):
    serve({"text": "", "usage": {"cost": 0.0001}})
    stt = make()
    assert stt.transcribe(tone(), RATE, start=0, end=1) is None
    assert stt.successes == 1 and stt.failures == 0


def test_short_audio_is_never_sent(serve):
    server = serve({"text": "x"})
    assert make().transcribe(tone(0.25), RATE, start=0, end=0.25) is None
    assert server.requests == []


def test_verbose_json_rejection_downgrades_once_and_does_not_hurt_the_breaker(serve):
    server = serve(HttpError(400, "unsupported response_format"), {"text": "plain text"})
    breaker = CircuitBreaker(MODEL, failure_threshold=1)
    stt = make(breaker=breaker)
    got = stt.transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.text == "plain text"
    assert [r["payload"]["response_format"] for r in server.requests] == ["verbose_json", "json"]
    assert "timestamp_granularities" not in server.requests[1]["payload"]
    assert breaker.state == "closed" and stt.failures == 0
    stt.transcribe(tone(), RATE, start=1, end=2)
    assert server.requests[-1]["payload"]["response_format"] == "json"


def test_http_402_is_quota_exceeded_and_opens_the_breaker_at_once(serve):
    server = serve(HttpError(402, "payment required"))
    breaker = CircuitBreaker(MODEL)
    stt = make(breaker=breaker)
    got = stt.transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.unavailable and got.quota_exceeded
    assert breaker.state == "open" and breaker.last_failure == "http_402"
    # The open circuit refuses without a request and keeps reporting the quota.
    again = stt.transcribe(tone(), RATE, start=1, end=2)
    assert again is not None and again.quota_exceeded
    assert len(server.requests) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_bad_credentials_open_the_breaker_without_retrying(serve, status):
    server = serve(HttpError(status, "no"))
    breaker = CircuitBreaker(MODEL)
    got = make(breaker=breaker).transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.unavailable and not got.quota_exceeded
    assert breaker.state == "open"
    assert len(server.requests) == 1


def test_default_estimate_is_the_usual_whisper_price(serve):
    serve({"text": "x"})
    stt = make()
    got = stt.transcribe(tone(60.0), RATE, start=0.0, end=60.0)
    assert got is not None and got.cost_usd == pytest.approx(0.006)


def test_a_provider_that_reports_a_negative_or_garbage_cost_falls_back(serve):
    serve({"text": "x", "usage": {"cost": "free"}})
    got = make(usd_per_second=0.01).transcribe(tone(), RATE, start=0, end=1)
    assert got is not None and got.cost_usd == pytest.approx(0.01)


def test_describe_reports_counters_and_breaker(serve):
    serve({"text": "x"})
    stt = make(breaker=CircuitBreaker(MODEL, clock=Clock()))
    stt.transcribe(tone(), RATE, start=0, end=1)
    state = stt.describe()
    assert state["successes"] == 1 and state["failures"] == 0
    assert state["breaker"]["state"] == "closed"
    assert state["last_success_age_s"] is not None
