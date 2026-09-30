"""Circuit breaker behaviour, on its own and wired into a cloud provider.

A dead endpoint must cost a handful of requests, not one per chunk, and must be
probed again on an exponential schedule. Nothing here uses the network or a real
clock: ``post_json`` is replaced and the breaker gets an injected clock.
"""

from __future__ import annotations

import itertools
import threading
import time
import urllib.error
from typing import Any

import pytest

from livestream_transcriber.netutil import HttpError
from livestream_transcriber.stt.base import CircuitBreaker
from livestream_transcriber.stt.providers import _cloud
from livestream_transcriber.stt.providers.openrouter import OpenRouterTranscriber
from tests.support.audio import RATE, Clock, tone

POST = "livestream_transcriber.stt.providers.openrouter.post_json"
MODEL = "openai/whisper-large-v3"
REAL_SLEEP = time.sleep


class Endpoint:
    """Scripted endpoint whose behaviour can change between calls."""

    def __init__(self, behaviour: Any = "ok") -> None:
        self.behaviour = behaviour
        self.posts = 0

    def __call__(self, url, payload, *, headers=None, timeout=30.0):
        self.posts += 1
        if isinstance(self.behaviour, BaseException):
            raise self.behaviour
        return {"text": "spoken words", "usage": {"cost": 0.0001}}


@pytest.fixture
def endpoint(monkeypatch):
    ep = Endpoint()
    monkeypatch.setattr(POST, ep)
    monkeypatch.setattr(_cloud.time, "sleep", lambda s: None)
    return ep


def make(clock: Clock, **breaker_kwargs: Any) -> tuple[OpenRouterTranscriber, CircuitBreaker]:
    breaker = CircuitBreaker(MODEL, clock=clock, **breaker_kwargs)
    return OpenRouterTranscriber("your-openrouter-key", model=MODEL, breaker=breaker), breaker


def call(stt: OpenRouterTranscriber, i: int = 0, **kwargs: Any):
    return stt.transcribe(tone(), RATE, start=float(i), end=float(i) + 1.0, **kwargs)


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("read timed out"), HttpError(503, "upstream down"), HttpError(500, "boom")],
    ids=["timeout", "503", "500"],
)
def test_a_dead_endpoint_stops_being_called_after_the_threshold(endpoint, failure):
    endpoint.behaviour = failure
    stt, breaker = make(Clock(), failure_threshold=3)
    for i in range(30):
        got = call(stt, i)
        assert got is not None and got.unavailable
    assert endpoint.posts == 3
    assert breaker.state == "open"
    assert breaker.describe()["short_circuited"] == 27


def test_a_half_open_probe_closes_the_breaker_on_success(endpoint):
    endpoint.behaviour = HttpError(503, "down")
    clock = Clock()
    stt, breaker = make(clock, failure_threshold=3, cooldown_seconds=30)
    for i in range(5):
        assert call(stt, i).unavailable
    assert endpoint.posts == 3 and breaker.state == "open"
    clock.advance(29)
    assert call(stt).unavailable
    assert endpoint.posts == 3  # still cooling down
    endpoint.behaviour = "ok"
    clock.advance(1)
    got = call(stt)
    assert got is not None and got.text == "spoken words"
    assert breaker.state == "closed"


def test_failed_probes_back_off_exponentially_up_to_the_cap(endpoint):
    endpoint.behaviour = TimeoutError("still down")
    clock = Clock()
    stt, breaker = make(clock, failure_threshold=1, cooldown_seconds=30, max_cooldown_seconds=100)
    call(stt)
    probes: list[float] = []
    for _ in range(400):
        before = endpoint.posts
        call(stt)
        if endpoint.posts > before:
            probes.append(clock.now)
        clock.advance(1)
    gaps = [round(b - a) for a, b in itertools.pairwise(probes)]
    assert gaps[:3] == [60, 100, 100]
    assert breaker.describe()["cooldown_s"] == 100


@pytest.mark.parametrize(
    "failure",
    [
        OSError("connection refused"),
        urllib.error.URLError("[Errno 8] nodename nor servname provided"),
        HttpError(429, "slow down"),
    ],
    ids=["oserror", "urlerror-dns", "429"],
)
def test_a_probe_that_retries_internally_still_backs_off(monkeypatch, failure):
    """The retries inside one call must not ask the breaker again.

    A probe refused by its own breaker never reports its failure, the breaker
    stays half-open and probes at the base cool-down forever.
    """
    clock = Clock()
    calls: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(clock.now)
        raise failure

    monkeypatch.setattr(POST, post)
    monkeypatch.setattr(_cloud.time, "sleep", lambda s: None)
    stt, breaker = make(clock, failure_threshold=1, cooldown_seconds=30, max_cooldown_seconds=600)
    for i in range(3000):
        assert call(stt, i).unavailable
        clock.advance(1)
    probes = sorted(set(calls))  # a retry shares its call's instant
    gaps = [round(b - a) for a, b in itertools.pairwise(probes)]
    assert gaps == [30, 60, 120, 240, 480, 600, 600, 600]
    assert len(calls) == len(probes) * (4 if isinstance(failure, HttpError) else 2)
    assert breaker.describe()["cooldown_s"] == 600


def test_a_probe_whose_retry_succeeds_closes_the_breaker(monkeypatch):
    clock = Clock()
    script: list[Any] = [TimeoutError("stall"), OSError("reset"), "ok"]
    calls: list[float] = []

    def post(url, payload, *, headers=None, timeout=30.0):
        calls.append(clock.now)
        action = script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return {"text": "recovered", "usage": {"cost": 0.0001}}

    monkeypatch.setattr(POST, post)
    stt, breaker = make(clock, failure_threshold=1, cooldown_seconds=30)
    assert call(stt).unavailable
    assert breaker.state == "open"
    clock.advance(30)
    got = call(stt, 1)  # probe: reset, then the retry succeeds
    assert got is not None and got.text == "recovered"
    assert breaker.state == "closed" and len(calls) == 3


def test_a_probe_abandoned_in_a_429_back_off_reports_its_failure(monkeypatch):
    monkeypatch.setattr(POST, Endpoint(HttpError(429, "slow down")))
    clock = Clock()
    stt, breaker = make(clock, failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("timeout")  # open
    clock.advance(30)
    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()
    got = call(stt, cancel=cancel)
    assert got is not None and got.unavailable
    state = breaker.describe()
    assert state["state"] == "open"
    assert state["cooldown_s"] == 60  # the probe failed: doubled
    assert state["last_failure"] == "http_429"


def test_cancel_wakes_a_429_back_off_and_stops_the_retry(monkeypatch):
    ep = Endpoint(HttpError(429, "slow down"))
    monkeypatch.setattr(POST, ep)
    stt = OpenRouterTranscriber("your-openrouter-key", model=MODEL)
    cancel = threading.Event()
    threading.Timer(0.05, cancel.set).start()
    began = time.monotonic()
    got = call(stt, cancel=cancel)
    assert got is not None and got.unavailable
    assert time.monotonic() - began < 1.0  # not the 2 s back-off
    assert ep.posts == 1


def test_a_cancelled_call_sends_nothing(endpoint):
    cancel = threading.Event()
    cancel.set()
    stt, _ = make(Clock())
    got = call(stt, cancel=cancel)
    assert got is not None and got.unavailable
    assert endpoint.posts == 0


def test_success_resets_the_failure_streak(endpoint):
    stt, breaker = make(Clock(), failure_threshold=3)
    endpoint.behaviour = HttpError(503, "down")
    call(stt)
    call(stt)
    endpoint.behaviour = "ok"
    assert call(stt).text == "spoken words"
    endpoint.behaviour = HttpError(503, "down")
    call(stt)
    call(stt)
    assert breaker.state == "closed"  # two + two, never three in a row


def test_breaker_describe_when_open():
    clock = Clock()
    breaker = CircuitBreaker("m", failure_threshold=1, cooldown_seconds=30, clock=clock)
    breaker.record_failure("timeout")
    clock.advance(10)
    state = breaker.describe()
    assert (state["state"], state["retry_in_s"], state["opened"]) == ("open", 20.0, 1)
    assert not breaker.allow()
