"""Local ONNX speech-to-text through sherpa-onnx: NVIDIA Parakeet TDT v3
(multilingual, 25 European languages) and Moonshine (English only).

Install with ``pip install 'livestream-transcriber[onnx]'`` and fetch a model
once with ``lst models fetch``. Nothing is downloaded implicitly: ``ensure_model``
is only called by that explicit command, and a missing model is reported as an
unavailable provider rather than fetched behind the user's back.

Why offer it next to Whisper: a transducer model decodes several times faster per
CPU core and returns nothing, instead of an invented sentence, on silence.

Models are plain files (int8 ONNX graphs plus ``tokens.txt``) in
``<models_dir>/<model name>/``.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...logging_setup import get_logger
from ...process_priority import apply_thread_nice
from ..base import ConditionLog, LoadBackoff, Transcript, too_short, unavailable

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_ROOT",
    "ONNX_MODELS",
    "OnnxModelSpec",
    "SherpaOnnxTranscriber",
    "ensure_model",
    "model_path",
    "model_present",
    "release_recognizers",
]

log = get_logger(__name__)

PROVIDER = "onnx"
DEFAULT_MODEL = "parakeet-tdt-0.6b-v3"
#: Must match ``Settings.stt_models_dir``; the CLI passes the configured value.
DEFAULT_MODEL_ROOT = Path.home() / ".cache" / "livestream-transcriber" / "models"
_HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
# How far past the last token timestamp a segment is assumed to run (seconds).
_TAIL_SECONDS = 0.4


@dataclass(frozen=True)
class OnnxModelSpec:
    name: str
    kind: str
    """``nemo_transducer`` or ``moonshine``: which sherpa-onnx constructor builds it."""
    repo: str
    """Hugging Face repository the files are fetched from."""
    files: tuple[str, ...]
    languages: frozenset[str]
    """Languages the model speaks; empty means it detects any of its own."""


_MOONSHINE_FILES = (
    "preprocess.onnx",
    "encode.int8.onnx",
    "uncached_decode.int8.onnx",
    "cached_decode.int8.onnx",
    "tokens.txt",
)

ONNX_MODELS: dict[str, OnnxModelSpec] = {
    "parakeet-tdt-0.6b-v3": OnnxModelSpec(
        name="parakeet-tdt-0.6b-v3",
        kind="nemo_transducer",
        repo="csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        files=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
        languages=frozenset(),
    ),
    "moonshine-base-en": OnnxModelSpec(
        name="moonshine-base-en",
        kind="moonshine",
        repo="csukuangfj/sherpa-onnx-moonshine-base-en-int8",
        files=_MOONSHINE_FILES,
        languages=frozenset({"en"}),
    ),
    "moonshine-tiny-en": OnnxModelSpec(
        name="moonshine-tiny-en",
        kind="moonshine",
        repo="csukuangfj/sherpa-onnx-moonshine-tiny-en-int8",
        files=_MOONSHINE_FILES,
        languages=frozenset({"en"}),
    ),
}


def model_path(name: str, root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_MODEL_ROOT) / name


def model_present(name: str, root: str | Path | None = None) -> bool:
    """True when every file of the model exists and is non-empty."""
    spec = ONNX_MODELS[name]
    base = model_path(name, root)
    return all((base / f).is_file() and (base / f).stat().st_size > 0 for f in spec.files)


def ensure_model(
    name: str,
    root: str | Path | None = None,
    *,
    timeout: float = 600.0,
    on_file: Callable[[str], None] | None = None,
) -> Path:
    """Download the model's missing files (each atomically) and return its directory.

    This is the only network access of the provider, and it only happens when
    the caller asks for it (``lst models fetch``). A short read, which a
    dropped connection produces without an error, is detected by comparing the
    byte count with ``Content-Length`` and removed, so it can never pass for
    the model.
    """
    spec = ONNX_MODELS[name]
    base = model_path(name, root)
    base.mkdir(parents=True, exist_ok=True)
    for fname in spec.files:
        dest = base / fname
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        if on_file is not None:
            on_file(fname)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = _HF_URL.format(repo=spec.repo, name=fname)
        log.info("downloading stt model file", extra={"model": name, "file": fname})
        written = 0
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            expected = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
            while block := resp.read(1 << 20):
                out.write(block)
                written += len(block)
        if expected is not None and written != int(expected):
            tmp.unlink(missing_ok=True)
            raise OSError(f"{fname}: got {written} of {expected} bytes; run the fetch again")
        os.replace(tmp, dest)
    return base


# --------------------------------------------------------------------------- #
# Process-wide recognizer cache
# --------------------------------------------------------------------------- #


@dataclass
class _Shared:
    recognizer: Any
    refs: int = 0
    decode_lock: threading.Lock = field(default_factory=threading.Lock)


# One loaded recognizer per (model, dir, threads) for the whole process: the
# transcriber is rebuilt per capture session, the weights (hundreds of MB) must
# not be. Guarded by _SHARED_LOCK; each entry has its own decode lock because a
# recognizer decodes one stream at a time.
_SHARED: dict[tuple[str, str, int], _Shared] = {}
_SHARED_LOCK = threading.Lock()


def release_recognizers() -> None:
    """Drop every cached recognizer (tests, or to give the memory back)."""
    with _SHARED_LOCK:
        _SHARED.clear()


def _words_from_tokens(
    tokens: list[str], stamps: list[float], limit: float
) -> list[dict[str, Any]]:
    """Group sub-word tokens into words using sentencepiece's ``▁`` word marker."""
    if not tokens or len(tokens) != len(stamps):
        return []
    pieces: list[tuple[str, float]] = []
    for token, stamp in zip(tokens, stamps, strict=True):
        if token.startswith("▁") or not pieces:
            pieces.append((token.lstrip("▁"), float(stamp)))
        else:
            pieces[-1] = (pieces[-1][0] + token, pieces[-1][1])
    words: list[dict[str, Any]] = []
    for i, (text, began) in enumerate(pieces):
        if not text.strip():
            continue
        until = pieces[i + 1][1] if i + 1 < len(pieces) else min(limit, began + _TAIL_SECONDS)
        words.append({"start": began, "end": max(began, min(until, limit)), "text": text})
    return words


class SherpaOnnxTranscriber:
    """One sherpa-onnx offline recognizer, lazily built, one decode at a time.

    ``language`` is informational for Parakeet (it detects the language) and a
    guard for English-only models: asked for another language they answer
    ``STT_UNAVAILABLE`` rather than transcribe it as English gibberish.
    """

    provider = PROVIDER

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        model_root: str | Path | None = None,
        threads: int = 1,
        language: str | None = None,
        word_timestamps: bool = False,
        nice: int = 0,
        load_backoff: LoadBackoff | None = None,
    ) -> None:
        if model not in ONNX_MODELS:
            raise ValueError(f"unknown onnx model {model!r}; known: {sorted(ONNX_MODELS)}")
        self.spec = ONNX_MODELS[model]
        self.model = f"onnx/{model}"
        self.model_root = model_root
        self.threads = max(1, int(threads))
        self.language = language
        self.word_timestamps = word_timestamps
        self.nice = nice
        self._shared: _Shared | None = None
        self._missing_package = False
        self._backoff = load_backoff or LoadBackoff()
        self._lock = threading.Lock()
        self._fail_log = ConditionLog("onnx stt failing")

    # ------------------------------------------------------------------ load --

    def _key(self) -> tuple[str, str, int]:
        return (self.spec.name, str(model_path(self.spec.name, self.model_root)), self.threads)

    def _build(self) -> Any:
        import sherpa_onnx

        base = model_path(self.spec.name, self.model_root)
        f = {name: str(base / name) for name in self.spec.files}
        if self.spec.kind == "nemo_transducer":
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=f["encoder.int8.onnx"],
                decoder=f["decoder.int8.onnx"],
                joiner=f["joiner.int8.onnx"],
                tokens=f["tokens.txt"],
                model_type="nemo_transducer",
                num_threads=self.threads,
            )
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=f["preprocess.onnx"],
            encoder=f["encode.int8.onnx"],
            uncached_decoder=f["uncached_decode.int8.onnx"],
            cached_decoder=f["cached_decode.int8.onnx"],
            tokens=f["tokens.txt"],
            num_threads=self.threads,
        )

    def ready(self) -> bool:
        """Load the recognizer if needed; False when it cannot be loaded right now.

        A missing package is final. Missing or unloadable model files are retried after a
        growing pause, so fetching the model while the process runs is enough.
        """
        if self._shared is not None:
            return True
        if self._missing_package or self._backoff.blocked:
            return False
        with self._lock:
            if self._shared is not None:
                return True
            if self._missing_package or self._backoff.blocked:
                return False
            if not model_present(self.spec.name, self.model_root):
                self._backoff.failed()
                log.warning(
                    "onnx model files are missing; run `lst models fetch`",
                    extra={"model": self.spec.name},
                )
                return False
            try:
                # Built under the process-wide lock: two sessions never load
                # two ~1 GB copies of the same model.
                with _SHARED_LOCK:
                    shared = _SHARED.get(self._key())
                    if shared is None:
                        shared = _Shared(recognizer=self._build())
                        _SHARED[self._key()] = shared
                    shared.refs += 1
                self._shared = shared
            except ImportError:
                self._missing_package = True
                log.warning(
                    "sherpa-onnx is not installed (pip install 'livestream-transcriber[onnx]')"
                )
                return False
            except Exception as exc:
                self._backoff.failed()
                log.warning(
                    "onnx model failed to load",
                    extra={"model": self.spec.name, "error": str(exc)[:300]},
                )
                return False
        self._backoff.succeeded()
        log.info("onnx model loaded", extra={"model": self.spec.name, "threads": self.threads})
        return True

    def close(self) -> None:
        """Release this transcriber's hold on the shared recognizer.

        The weights are freed once the last transcriber using them closes.
        """
        with self._lock:
            shared, self._shared = self._shared, None
        if shared is None:
            return
        with _SHARED_LOCK:
            shared.refs -= 1
            if shared.refs <= 0 and _SHARED.get(self._key()) is shared:
                del _SHARED[self._key()]

    # ------------------------------------------------------------ transcribe --

    def transcribe(
        self, pcm: bytes, sample_rate: int, *, start: float, end: float
    ) -> Transcript | None:
        if too_short(pcm, sample_rate):
            return None
        if self.spec.languages and self.language and self.language not in self.spec.languages:
            return unavailable(start, end, model=self.model, provider=PROVIDER)
        if not self.ready() or self._shared is None:
            return unavailable(start, end, model=self.model, provider=PROVIDER)
        apply_thread_nice(self.nice)
        shared = self._shared
        began = time.monotonic()
        try:
            audio = np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2).astype(np.float32)
            audio /= 32768.0
            with shared.decode_lock:
                stream = shared.recognizer.create_stream()
                stream.accept_waveform(sample_rate, audio)
                shared.recognizer.decode_stream(stream)
                result = stream.result
            text = str(getattr(result, "text", "") or "").strip()
            stamps = [float(t) for t in (getattr(result, "timestamps", None) or [])]
            tokens = [str(t) for t in (getattr(result, "tokens", None) or [])]
        except Exception as exc:
            self._fail_log.hit(model=self.model, error=str(exc)[:300])
            return unavailable(start, end, model=self.model, provider=PROVIDER)
        self._fail_log.ok(model=self.model)
        if not text:
            return None
        limit = max(0.0, end - start)
        segments = None
        if stamps:
            seg_start = stamps[0]
            seg_end = min(limit, stamps[-1] + _TAIL_SECONDS)
            if seg_end > seg_start:
                segments = [{"start": seg_start, "end": seg_end, "text": text}]
        words = _words_from_tokens(tokens, stamps, limit) if self.word_timestamps else []
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
            language=self.language,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": PROVIDER,
            "model": self.spec.name,
            "loaded": self._shared is not None,
            "load_failed": self._missing_package or self._backoff.failing,
            "threads": self.threads,
        }
