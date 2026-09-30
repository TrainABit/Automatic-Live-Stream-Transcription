"""Local Whisper through faster-whisper (CTranslate2), the default provider.

Install with ``pip install 'livestream-transcriber[local]'``. The model is
loaded lazily on the first chunk, under a lock: several STT workers call
``transcribe`` concurrently and each would otherwise load its own copy of the
weights. The first load downloads the model into ``models_dir`` (or the library
default) unless it is already there.

Defaults favour a CPU that has to keep up with live audio: ``small`` at
``int8`` on one thread and greedy decoding (``beam_size=1``). Raise
``beam_size`` or the model size when latency does not matter, for example when
replaying a recording.

Times in the returned segments and words are relative to the start of the chunk.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...logging_setup import get_logger
from ...process_priority import apply_thread_nice
from ..base import ConditionLog, LoadBackoff, Transcript, too_short, unavailable

__all__ = ["FasterWhisperTranscriber"]

log = get_logger(__name__)

PROVIDER = "local"
TARGET_RATE = 16000


def _to_16k(pcm: bytes, sample_rate: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Mono float32 in [-1, 1] at 16 kHz, the input Whisper expects."""
    audio: np.ndarray[Any, np.dtype[np.float32]] = (
        np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2).astype(np.float32) / 32768.0
    )
    if sample_rate == TARGET_RATE or audio.size == 0:
        return audio
    target = max(1, round(audio.size * TARGET_RATE / sample_rate))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target, endpoint=False)
    resampled: np.ndarray[Any, np.dtype[np.float32]] = np.interp(x_new, x_old, audio).astype(
        np.float32
    )
    return resampled


class FasterWhisperTranscriber:
    """One lazily loaded faster-whisper model, shared by every worker thread."""

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int = 1,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        models_dir: str | Path | None = None,
        nice: int = 0,
        load_backoff: LoadBackoff | None = None,
    ) -> None:
        self.model_name = model
        self.model = f"faster-whisper/{model}"
        self.device = device
        self.compute_type = compute_type
        self.threads = max(0, int(threads))
        self.language = language
        self.beam_size = max(1, int(beam_size))
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.nice = nice
        self._whisper: Any = None
        self._missing_package = False
        self._backoff = load_backoff or LoadBackoff()
        self._load_lock = threading.Lock()
        self._fail_log = ConditionLog("local whisper transcription failing")

    # ------------------------------------------------------------------ load --

    def ready(self) -> bool:
        """Load the model if needed; False when it cannot be loaded right now.

        A missing package is final. Any other failure (typically the first-run download)
        is retried after a growing pause, so it does not outlive the cause.
        """
        if self._whisper is not None:
            return True
        if self._missing_package or self._backoff.blocked:
            return False
        with self._load_lock:
            if self._whisper is not None:
                return True
            if self._missing_package or self._backoff.blocked:
                return False
            return self._load()

    def _load(self) -> bool:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self._missing_package = True
            log.warning(
                "faster-whisper is not installed; local speech-to-text is unavailable "
                "(pip install 'livestream-transcriber[local]')"
            )
            return False
        kwargs: dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.threads,
            "num_workers": 1,
        }
        if self.models_dir is not None:
            kwargs["download_root"] = str(self.models_dir)
        try:
            log.info(
                "loading local whisper model (the first use downloads it)",
                extra={"model": self.model_name, "device": self.device},
            )
            self._whisper = WhisperModel(self.model_name, **kwargs)
        except Exception as exc:
            self._backoff.failed()
            log.warning(
                "local whisper model failed to load",
                extra={"model": self.model_name, "error": str(exc)[:300]},
            )
            return False
        self._backoff.succeeded()
        log.info(
            "local whisper loaded",
            extra={
                "model": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "threads": self.threads,
            },
        )
        return True

    # ------------------------------------------------------------ transcribe --

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        if too_short(pcm, sample_rate):
            return None
        if not self.ready():
            return unavailable(start, end, model=self.model, provider=PROVIDER)
        apply_thread_nice(self.nice)
        began = time.monotonic()
        try:
            # ``transcribe`` returns a generator: the decoding happens while it
            # is consumed, so consuming it belongs inside the try block.
            raw_segments, info = self._whisper.transcribe(
                _to_16k(pcm, sample_rate),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=self.word_timestamps,
            )
            segments, words = _collect(raw_segments)
        except Exception as exc:
            self._fail_log.hit(model=self.model, error=str(exc)[:300])
            return unavailable(start, end, model=self.model, provider=PROVIDER)
        self._fail_log.ok(model=self.model)
        text = " ".join(s["text"] for s in segments if s["text"]).strip()
        if not text:
            return None
        detected = getattr(info, "language", None)
        return Transcript(
            start=start,
            end=end,
            text=text,
            segments=segments,
            words=words or None,
            provider_latency=round(time.monotonic() - began, 3),
            cost_usd=0.0,
            model=self.model,
            provider=PROVIDER,
            language=detected if isinstance(detected, str) else self.language,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "model": self.model,
            "loaded": self._whisper is not None,
            "load_failed": self._missing_package or self._backoff.failing,
        }


def _collect(raw_segments: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Consume faster-whisper segments into plain dicts (chunk-relative times)."""
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for seg in raw_segments:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        rec: dict[str, Any] = {
            "start": float(getattr(seg, "start", 0.0) or 0.0),
            "end": float(getattr(seg, "end", 0.0) or 0.0),
            "text": text,
        }
        for extra in ("avg_logprob", "no_speech_prob"):
            value = getattr(seg, extra, None)
            if isinstance(value, int | float):
                rec[extra] = float(value)
        segments.append(rec)
        for w in getattr(seg, "words", None) or ():
            word = str(getattr(w, "word", "") or "").strip()
            if not word:
                continue
            item: dict[str, Any] = {
                "start": float(w.start),
                "end": float(w.end),
                "text": word,
            }
            prob = getattr(w, "probability", None)
            if isinstance(prob, int | float):
                item["confidence"] = float(prob)
            words.append(item)
    return segments, words
