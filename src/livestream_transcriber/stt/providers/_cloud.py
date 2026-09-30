"""Shared machinery of the HTTP speech-to-text providers.

OpenAI-compatible and OpenRouter endpoints differ in how the audio is sent and
how the answer is shaped, but they fail in the same ways and must be handled the
same way: retry a rate limit with back-off, retry a dropped connection once,
refuse quickly while a circuit breaker is open, never bill or cache an error
page served with HTTP 200, and log an outage once instead of once per chunk.
:class:`CloudTranscriber` owns that ladder; subclasses supply ``_send`` and
``_interpret``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from ...logging_setup import get_logger
from ...netutil import HttpError
from ..base import (
    CircuitBreaker,
    ConditionLog,
    Transcript,
    bad_body,
    body_snippet,
    pcm_to_wav,
    too_short,
    unavailable,
)

__all__ = ["DEFAULT_USD_PER_SECOND", "CloudTranscriber"]

log = get_logger(__name__)

#: Cost estimate (USD per audio second) used when an API reports no cost:
#: 0.006 USD per minute, the usual price of hosted Whisper. It only feeds the
#: budget guard; override it per provider when your endpoint prices differently.
DEFAULT_USD_PER_SECOND = 0.006 / 60.0

_MAX_429_RETRIES = 3


def _time_left(deadline: float | None) -> float:
    """Seconds until ``deadline`` (monotonic); unbounded when there is none."""
    return math.inf if deadline is None else deadline - time.monotonic()


def _sleep_or_cancelled(delay: float, cancel: threading.Event | None) -> bool:
    """Back off for ``delay`` s. True when ``cancel`` fired (stop retrying)."""
    if cancel is None:
        time.sleep(delay)
        return False
    return cancel.wait(delay)


class CloudTranscriber:
    """Base class of the HTTP providers. Thread-safe: several workers may share one."""

    #: Registry name, reported as ``Transcript.provider``.
    provider = "cloud"

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str,
        language: str | None = None,
        prompt: str | None = None,
        timeout: float = 60.0,
        breaker: CircuitBreaker | None = None,
        usd_per_second: float | None = None,
        word_timestamps: bool = False,
        response_format: str = "verbose_json",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.prompt = prompt
        self.timeout = timeout
        self.breaker = breaker
        self.usd_per_second = DEFAULT_USD_PER_SECOND if usd_per_second is None else usd_per_second
        self.word_timestamps = word_timestamps
        self.response_format = response_format
        """``verbose_json`` carries segments and timestamps; ``json`` is text only."""
        self.verbose_json_rejected = False
        self.cost_usd = 0.0
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.last_failure: str | None = None
        self.last_success_mono: float | None = None
        self._failure_log = ConditionLog(f"{self.provider} stt failing")
        self._retry_log = ConditionLog(f"{self.provider} stt retrying", level=logging.INFO)
        self._lock = threading.Lock()

    # ---------------------------------------------------------- subclass API --

    @property
    def endpoint(self) -> str:
        raise NotImplementedError

    def _send(self, wav: bytes, timeout: float) -> dict[str, Any]:
        """POST one request and return the decoded JSON body."""
        raise NotImplementedError

    def _interpret(
        self, data: dict[str, Any], *, start: float, end: float, cost: float
    ) -> Transcript | None:
        """Turn a valid response body into a transcript (``None`` for no speech)."""
        raise NotImplementedError

    def _reported_cost(self, data: dict[str, Any]) -> float | None:
        """The cost the API reported for this request, if it does."""
        return None

    def _adjust_after_http_error(self, exc: HttpError) -> bool:
        """Change the request after a rejection; True to retry it right away.

        Servers that speak the OpenAI protocol only partly answer HTTP 400 to
        ``verbose_json``. That is a format question, not a provider failure:
        downgrade to plain ``json`` once (losing segments and word times) and
        retry, without touching the circuit breaker.
        """
        if (
            exc.status == 400
            and self.response_format == "verbose_json"
            and not self.verbose_json_rejected
        ):
            log.warning(
                "stt verbose_json rejected; retrying as json (no timestamps)",
                extra={"provider": self.provider, "model": self.model, "error": str(exc)[:300]},
            )
            with self._lock:
                self.response_format = "json"
                self.verbose_json_rejected = True
            return True
        return False

    # ------------------------------------------------------------------ call --

    def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        start: float,
        end: float,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
    ) -> Transcript | None:
        """Transcribe one chunk.

        ``timeout`` bounds the whole call (every request, back-off and retry
        together); without it each request gets ``self.timeout``. Once ``cancel``
        is set no further retry or back-off happens.
        """
        if too_short(pcm, sample_rate):
            return None
        wav = pcm_to_wav(pcm, sample_rate)
        if cancel is not None and cancel.is_set():
            return self._unavailable(start, end)
        if self.breaker is not None and not self.breaker.allow():
            # Open circuit: no request and no log line; the breaker logged the
            # state change once.
            return self._unavailable(start, end, quota=self.breaker.last_failure == "http_402")
        budget = self.timeout if timeout is None else timeout
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        retries_429 = _MAX_429_RETRIES
        retried_network = False
        while True:
            request_timeout = min(budget, max(0.5, _time_left(deadline)))
            began = time.monotonic()
            try:
                data = self._send(wav, request_timeout)
            except TimeoutError as exc:
                return self._failed("timeout", exc, start, end)
            except HttpError as exc:
                if self._adjust_after_http_error(exc):
                    if cancel is not None and cancel.is_set():
                        return self._unavailable(start, end)
                    continue
                if exc.status == 402:
                    return self._failed("http_402", exc, start, end, immediate=True, quota=True)
                if exc.status in (401, 403):
                    return self._failed(
                        f"http_{exc.status}",
                        "authentication failed; check the API key",
                        start,
                        end,
                        immediate=True,
                    )
                delay = 2 * (_MAX_429_RETRIES + 1 - retries_429)
                if exc.status == 429 and retries_429 > 0 and _time_left(deadline) > delay:
                    self._retry_log.hit(
                        model=self.model, failure="http_429", sleep_s=delay, left=retries_429
                    )
                    if _sleep_or_cancelled(delay, cancel):
                        return self._failed("http_429", exc, start, end)
                    retries_429 -= 1
                    continue
                return self._failed(f"http_{exc.status}", exc, start, end)
            except OSError as exc:
                if (
                    not retried_network
                    and not (cancel is not None and cancel.is_set())
                    and _time_left(deadline) > 0
                ):
                    self._retry_log.hit(model=self.model, failure="network", error=str(exc)[:300])
                    retried_network = True
                    continue
                return self._failed("network", exc, start, end)
            break

        problem = bad_body(data)
        if problem is not None:
            # Not a transcript: never billed, never cached, and not a success
            # that would close a breaker.
            return self._failed(problem, body_snippet(data), start, end)
        cost = self._reported_cost(data)
        if cost is None:
            cost = self.usd_per_second * max(0.0, end - start)
        self._succeeded(cost)
        result = self._interpret(data, start=start, end=end, cost=cost)
        if result is not None:
            result.provider_latency = round(time.monotonic() - began, 3)
        return result

    # ------------------------------------------------------------ accounting --

    def _unavailable(self, start: float, end: float, *, quota: bool = False) -> Transcript:
        return unavailable(
            start, end, model=self.model, provider=self.provider, quota_exceeded=quota
        )

    def _failed(
        self,
        kind: str,
        error: object,
        start: float,
        end: float,
        *,
        immediate: bool = False,
        quota: bool = False,
    ) -> Transcript:
        """Count one failed call against this provider and log it (rate-limited)."""
        with self._lock:
            self.failures += 1
            self.last_failure = kind
        if self.breaker is not None:
            self.breaker.record_failure(kind, immediate=immediate)
        self._failure_log.hit(model=self.model, failure=kind, error=str(error)[:300])
        return self._unavailable(start, end, quota=quota)

    def _succeeded(self, cost: float) -> None:
        with self._lock:
            self.successes += 1
            self.requests += 1
            self.cost_usd += cost
            self.last_success_mono = time.monotonic()
        if self.breaker is not None:
            self.breaker.record_success()
        self._failure_log.ok(model=self.model)
        self._retry_log.ok(model=self.model)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            age = (
                None
                if self.last_success_mono is None
                else round(time.monotonic() - self.last_success_mono, 1)
            )
            out: dict[str, Any] = {
                "provider": self.provider,
                "model": self.model,
                "requests": self.requests,
                "successes": self.successes,
                "failures": self.failures,
                "cost_usd": round(self.cost_usd, 6),
                "last_failure": self.last_failure,
                "last_success_age_s": age,
            }
        if self.breaker is not None:
            out["breaker"] = self.breaker.describe()
        return out
