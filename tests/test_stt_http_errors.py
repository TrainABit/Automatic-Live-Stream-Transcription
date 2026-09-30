"""Cloud STT failure branches: malformed 2xx bodies, retries and time budgets.

Nothing touches the network: either urllib is replaced (for the wire-level
cases) or ``post_json`` is.
"""

from __future__ import annotations

import time
import urllib.request

import pytest

from livestream_transcriber.netutil import HttpError, NonJsonBody, post_json
from livestream_transcriber.stt.base import CircuitBreaker
from livestream_transcriber.stt.providers import _cloud
from livestream_transcriber.stt.providers.openrouter import OpenRouterTranscriber
from livestream_transcriber.stt.wrappers import CachedTranscriber
from tests.support.audio import RATE, tone

POST = "livestream_transcriber.stt.providers.openrouter.post_json"
MODEL = "openai/whisper-large-v3"


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def serve_bytes(monkeypatch, body: bytes) -> list[str]:
    seen: list[str] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        return _Resp(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def make(**kwargs) -> OpenRouterTranscriber:
    return OpenRouterTranscriber("your-openrouter-key", model=MODEL, **kwargs)


# ------------------------------------------------------------ malformed 200 --


def test_netutil_marks_a_non_json_200_body(monkeypatch):
    serve_bytes(monkeypatch, b"<html><body>502 Bad Gateway</body></html>")
    data = post_json("https://example.invalid/x", {"a": 1})
    assert isinstance(data, NonJsonBody) and isinstance(data, dict)
    assert "502" in data.raw


def test_a_non_json_200_is_a_failed_call_never_billed_or_cached(monkeypatch, tmp_path):
    seen = serve_bytes(monkeypatch, b"<html><body>502 Bad Gateway</body></html>")
    inner = make()
    cached = CachedTranscriber(inner, tmp_path, provider="openrouter", model=MODEL)
    got = cached.transcribe(tone(), RATE, start=0.0, end=1.0)
    assert seen, "the request must have gone out"
    assert got is not None and got.unavailable and got.text == ""
    assert inner.cost_usd == 0.0 and inner.requests == 0
    assert not list(tmp_path.rglob("*.json"))


def test_an_empty_200_is_a_failed_call_not_a_paid_success(monkeypatch, tmp_path):
    serve_bytes(monkeypatch, b"")
    breaker = CircuitBreaker(MODEL, failure_threshold=1)
    inner = make(breaker=breaker)
    cached = CachedTranscriber(inner, tmp_path, provider="openrouter", model=MODEL)
    got = cached.transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.unavailable
    assert inner.cost_usd == 0.0 and inner.requests == 0
    assert inner.successes == 0 and inner.last_failure == "empty_body"
    assert breaker.state == "open"
    assert not list(tmp_path.rglob("*.json"))


def test_a_json_200_transcribes_and_is_cached(monkeypatch, tmp_path):
    serve_bytes(monkeypatch, b'{"text": "good evening", "usage": {"cost": 0.0002}}')
    inner = make()
    cached = CachedTranscriber(inner, tmp_path, provider="openrouter", model=MODEL)
    got = cached.transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.text == "good evening"
    assert inner.cost_usd == pytest.approx(0.0002)
    assert len(list(tmp_path.rglob("*.json"))) == 1


# ---------------------------------------------------------- retry branches --


def test_429_backs_off_then_succeeds(monkeypatch):
    calls: list[int] = []
    sleeps: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(1)
        if len(calls) <= 2:
            raise HttpError(429, "slow down")
        return {"text": "finally", "usage": {"cost": 0.0001}}

    monkeypatch.setattr(POST, post)
    monkeypatch.setattr(_cloud.time, "sleep", sleeps.append)
    got = make().transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.text == "finally"
    assert len(calls) == 3
    assert sleeps == [2, 4]


def test_429_gives_up_after_three_retries(monkeypatch):
    calls: list[int] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(1)
        raise HttpError(429, "slow down")

    monkeypatch.setattr(POST, post)
    monkeypatch.setattr(_cloud.time, "sleep", lambda s: None)
    stt = make()
    got = stt.transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.unavailable
    assert len(calls) == 4
    assert stt.last_failure == "http_429"


def test_a_dropped_connection_is_retried_once(monkeypatch):
    calls: list[int] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(1)
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr(POST, post)
    got = make().transcribe(tone(), RATE, start=0.0, end=1.0)
    assert got is not None and got.unavailable
    assert len(calls) == 2


def test_a_timeout_is_not_retried_and_gets_the_call_budget(monkeypatch):
    timeouts: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        timeouts.append(timeout)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(POST, post)
    stt = make(timeout=90.0)
    got = stt.transcribe(tone(), RATE, start=0.0, end=1.0, timeout=7.5)
    assert got is not None and got.unavailable
    assert len(timeouts) == 1 and timeouts[0] == pytest.approx(7.5, abs=0.05)
    assert timeouts[0] <= 7.5
    assert stt.last_failure == "timeout"


def test_a_bounded_call_does_not_back_off_past_its_budget(monkeypatch):
    calls: list[float] = []
    sleeps: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(timeout)
        raise HttpError(429, "slow down")

    monkeypatch.setattr(POST, post)
    monkeypatch.setattr(_cloud.time, "sleep", sleeps.append)
    stt = make(timeout=90.0)
    got = stt.transcribe(tone(), RATE, start=0.0, end=1.0, timeout=1.0)
    assert got is not None and got.unavailable
    assert len(calls) == 1  # the 2 s back-off does not fit in a 1 s budget
    assert sleeps == []


def test_the_network_retry_gets_only_the_rest_of_the_budget(monkeypatch):
    timeouts: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            time.sleep(0.3)
            raise ConnectionResetError("reset by peer")
        raise TimeoutError("read timed out")

    monkeypatch.setattr(POST, post)
    began = time.monotonic()
    got = make(timeout=90.0).transcribe(tone(), RATE, start=0.0, end=1.0, timeout=2.0)
    assert got is not None and got.unavailable
    assert len(timeouts) == 2
    assert timeouts[1] <= 2.0 - 0.3 + 0.01  # not a fresh 2 s
    assert time.monotonic() - began < 1.0
