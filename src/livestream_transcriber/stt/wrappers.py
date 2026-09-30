"""Wrappers that add behaviour around any :class:`Transcriber`.

Each wrapper is itself a transcriber, so they stack. The factory builds them in
this order, innermost first::

    provider -> BudgetGuardTranscriber -> FallbackSttTranscriber -> CachedTranscriber

The budget guard sits *under* the cache so a cache hit never counts as spend,
and *under* the fallback so only paid calls are metered.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .base import ConditionLog, Transcriber, Transcript, unavailable

__all__ = [
    "BudgetGuardTranscriber",
    "CachedTranscriber",
    "FallbackSttTranscriber",
    "SttSpend",
    "iter_chain",
]

log = get_logger(__name__)


def _safe_token(value: str) -> str:
    """A file-name-safe rendering of a model name or tag (slashes become underscores)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


class CachedTranscriber:
    """Disk cache keyed by provider, model, tag, chunk start and PCM hash.

    Useful for replay and benchmarks: running the same recording twice must not
    pay for, or wait for, the same transcription twice. Wall-clock time and the
    session id are never part of the key. Only successful, non-empty, non-degraded
    transcripts are stored, so a failed call is retried by a later run and a
    fallback's answer never stands in for the primary's. Secrets never enter a
    cache file.

    Layout: ``<dir>/<provider>/<model>/<start>_<rate>_<sha256[:16]>[_<tag>].json``.
    ``param_tag`` should carry whatever else changes the answer (language,
    word timestamps). With ``cache_only`` a miss returns ``None`` instead of
    calling the inner transcriber, for fully offline replays.
    """

    def __init__(
        self,
        inner: Transcriber,
        cache_dir: str | Path,
        *,
        provider: str,
        model: str = "default",
        param_tag: str = "",
        cache_only: bool = False,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.provider = provider
        self.model = model
        self.param_tag = param_tag
        self.cache_only = cache_only
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def cache_path(self, pcm: bytes, sample_rate: int, start: float) -> Path:
        digest = hashlib.sha256(pcm).hexdigest()[:16]
        name = f"{start:.2f}_{sample_rate}_{digest}"
        if self.param_tag:
            name = f"{name}_{_safe_token(self.param_tag)}"
        return (
            self.cache_dir / _safe_token(self.provider) / _safe_token(self.model) / f"{name}.json"
        )

    def _read(self, path: Path, *, start: float, end: float) -> Transcript | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = str(data.get("text") or "").strip()
        except (OSError, ValueError, AttributeError) as exc:
            log.warning("stt cache entry unreadable", extra={"file": path.name, "error": str(exc)})
            return None
        if not text:
            return None
        return Transcript(
            start=float(data.get("start", start)),
            end=float(data.get("end", end)),
            text=text,
            confidence=data.get("confidence"),
            segments=data.get("segments"),
            words=data.get("words"),
            provider_latency=data.get("provider_latency"),
            cost_usd=data.get("cost_usd"),
            model=data.get("model") or self.model,
            provider=data.get("provider") or self.provider,
            language=data.get("language"),
        )

    def _write(self, path: Path, result: Transcript, sample_rate: int) -> None:
        payload: dict[str, Any] = {
            "text": result.text,
            "confidence": result.confidence,
            "provider": result.provider or self.provider,
            "model": result.model or self.model,
            "sample_rate": sample_rate,
            "start": result.start,
            "end": result.end,
        }
        for key in ("segments", "words", "provider_latency", "cost_usd", "language"):
            value = getattr(result, key)
            if value is not None:
                payload[key] = value
        if self.param_tag:
            payload["param_tag"] = self.param_tag
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a concurrent reader (or a crash) never sees half a file.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, path)
        except OSError:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        path = self.cache_path(pcm, sample_rate, start)
        if path.is_file():
            cached = self._read(path, start=start, end=end)
            if cached is not None:
                with self._lock:
                    self.hits += 1
                log.debug("stt cache hit", extra={"start": round(start, 2), "model": self.model})
                return cached
        with self._lock:
            self.misses += 1
        if self.cache_only:
            return None
        result = self.inner.transcribe(pcm, sample_rate, start=start, end=end)
        if (
            result is not None
            and result.text.strip()
            and not result.unavailable
            and not (result.degraded)
        ):
            try:
                self._write(path, result, sample_rate)
            except OSError as exc:
                log.warning("stt cache write failed", extra={"error": str(exc)})
        return result

    def describe(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "dir": str(self.cache_dir)}


class FallbackSttTranscriber:
    """Use a second provider when the primary cannot answer.

    The fallback runs when the primary returns ``STT_UNAVAILABLE`` (outage,
    open circuit breaker, exhausted credit or budget). Whatever the fallback returns
    is tagged ``degraded=True``, so downstream code can tell a fallback transcript
    from the primary's. A primary success passes through untouched.

    ``enabled`` switches the fallback off without rebuilding the chain: the
    overload guard does that when a local model cannot keep up with live audio.
    """

    def __init__(
        self,
        primary: Transcriber,
        fallback: Transcriber,
        *,
        provider_name: str = "fallback",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.provider_name = provider_name
        self.model = getattr(primary, "model", None)
        self.enabled = True
        self.fallback_hits = 0
        self.fallback_misses = 0
        self._lock = threading.Lock()
        self._log = ConditionLog("stt primary unavailable; using fallback provider")

    @property
    def cost_usd(self) -> float:
        return float(getattr(self.primary, "cost_usd", 0.0) or 0.0)

    @property
    def requests(self) -> int:
        return int(getattr(self.primary, "requests", 0) or 0)

    def _tag(self, result: Transcript) -> Transcript:
        return replace(
            result,
            degraded=True,
            provider=result.provider or self.provider_name,
            model=result.model or getattr(self.fallback, "model", None),
        )

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        result = self.primary.transcribe(pcm, sample_rate, start=start, end=end)
        if result is None or not result.unavailable:
            if result is not None:
                self._log.ok(primary=getattr(self.primary, "model", None))
            return result
        if not self.enabled:
            return result
        self._log.hit(
            primary=getattr(self.primary, "model", None), budget_tripped=result.quota_exceeded
        )
        try:
            local = self.fallback.transcribe(pcm, sample_rate, start=start, end=end)
        except Exception:
            log.exception("stt fallback provider crashed")
            local = None
            crashed = True
        else:
            crashed = False
        if local is not None and local.text.strip() and not local.unavailable:
            with self._lock:
                self.fallback_hits += 1
            return self._tag(local)
        with self._lock:
            self.fallback_misses += 1
        if local is None and not crashed:
            return None  # the fallback ran and heard nothing: an honest "no speech"
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": getattr(self.fallback, "model", None),
            "hits": self.fallback_hits,
            "misses": self.fallback_misses,
        }


class SttSpend:
    """Paid STT spend of the process, shared across capture sessions.

    A session that is rebuilt after a reconnect gets a new transcriber whose
    ``cost_usd`` starts at zero, so a cap counted per transcriber would let a
    long, auto-resuming run spend the budget once per session. One
    :class:`SttSpend` per process, handed to every session's
    :class:`BudgetGuardTranscriber`, keeps a single cap. Once crossed it stays
    crossed: one alert, no spend afterwards.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.tripped = False

    def add(self, usd: float) -> float:
        """Count ``usd`` more (never negative) and return the total."""
        with self._lock:
            if usd > 0:
                self.spent_usd += float(usd)
            return self.spent_usd

    def trip(self) -> bool:
        """Mark the cap crossed; True for the one call that crossed it."""
        with self._lock:
            if self.tripped:
                return False
            self.tripped = True
            return True


class BudgetGuardTranscriber:
    """Stop calling a paid provider once cumulative spend reaches a cap.

    Once tripped it answers "unavailable" (with ``quota_exceeded`` set), never "no
    speech": a run that can no longer listen must not look like a quiet stream. It
    neither crashes nor spends without bound, and a :class:`FallbackSttTranscriber`
    around this guard takes over. The spend counted is
    ``spend``'s: the process-wide counter when the caller shares one, this
    guard's own otherwise. The inner transcriber's ``cost_usd`` is read as a
    running total and only its growth is added.
    """

    def __init__(
        self,
        inner: Transcriber,
        budget_usd: float,
        *,
        on_exceeded: Callable[[float], None] | None = None,
        spend: SttSpend | None = None,
    ) -> None:
        self.inner = inner
        self.budget_usd = budget_usd
        self._on_exceeded = on_exceeded
        self.spend = spend if spend is not None else SttSpend()
        self._counted = float(getattr(inner, "cost_usd", 0.0) or 0.0)
        self._lock = threading.Lock()
        self.model = getattr(inner, "model", None)

    @property
    def tripped(self) -> bool:
        return self.spend.tripped

    @property
    def cost_usd(self) -> float:
        return float(getattr(self.inner, "cost_usd", 0.0) or 0.0)

    @property
    def requests(self) -> int:
        return int(getattr(self.inner, "requests", 0) or 0)

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        if self.spend.tripped:
            return unavailable(
                start,
                end,
                model=self.model,
                provider=getattr(self.inner, "provider", None),
                quota_exceeded=True,
            )
        result = self.inner.transcribe(pcm, sample_rate, start=start, end=end)
        with self._lock:
            total = self.cost_usd
            grown, self._counted = total - self._counted, total
        spent = self.spend.add(grown)
        if spent >= self.budget_usd and self.spend.trip():
            log.error(
                "stt budget reached; disabling further paid transcription",
                extra={"spent_usd": round(spent, 4), "budget_usd": self.budget_usd},
            )
            if self._on_exceeded is not None:
                try:
                    self._on_exceeded(spent)
                except Exception:
                    log.exception("stt budget callback failed")
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "tripped": self.tripped,
            "spent_usd": round(self.spend.spent_usd, 6),
            "budget_usd": self.budget_usd,
        }


def _children(node: Any) -> tuple[Any, ...]:
    """The transcribers a wrapper delegates to. Unknown types are leaves."""
    if isinstance(node, CachedTranscriber | BudgetGuardTranscriber):
        return (node.inner,)
    if isinstance(node, FallbackSttTranscriber):
        return (node.primary, node.fallback)
    return ()


def iter_chain(root: Any) -> Iterator[Any]:
    """Every transcriber in a wrapper chain, breadth first (primary before fallback)."""
    queue = [root]
    seen: set[int] = set()
    while queue:
        node = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        queue.extend(_children(node))
