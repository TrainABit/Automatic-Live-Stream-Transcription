"""Run speech-to-text providers over a directory of clips and report accuracy and speed.

A clips directory holds audio files with an optional reference transcript of the same
name::

    clips/
      interview.wav      interview.txt
      talk.mp3           talk.srt
      noise.flac                          # no reference: latency only

Each clip is decoded to 16 kHz mono with ffmpeg, cut into chunks of ``chunk_seconds``
(the pipeline's own unit of work, so latency and accuracy are measured the way a live
run experiences them), and transcribed chunk by chunk. Silent chunks are skipped by the
same energy gate the pipeline uses.

Timing is honest about model loading: before the measured pass, one *cold pass*
transcribes the first chunk and reports its time separately. Without it the first
request would pay for loading the model and inflate that provider's p95 and RTF.

Providers run one after another, never concurrently, so they do not compete for the
CPU while being timed.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import ConfigError, Settings
from ..logging_setup import get_logger
from ..stt.base import MIN_AUDIO_SECONDS, SILENCE_DBFS, Transcriber, Transcript, pcm_is_silent
from ..stt.factory import build_transcriber, close_transcriber
from .metrics import (
    ErrorCounts,
    aggregate,
    cer,
    is_hallucination,
    latency_stats,
    real_time_factor,
    wer,
)

log = get_logger(__name__)

__all__ = [
    "BenchError",
    "BenchReport",
    "Clip",
    "ClipResult",
    "ProviderResult",
    "ProviderSpec",
    "decode_audio",
    "discover_clips",
    "parse_provider_specs",
    "read_reference",
    "run_bench",
    "split_pcm",
    "write_report",
]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".mp4", ".webm")
REFERENCE_EXTENSIONS = (".txt", ".srt", ".vtt")
SAMPLE_RATE = 16000

_TIMING_LINE = re.compile(r"-->")
_CUE_NUMBER = re.compile(r"^\d+$")
_MARKUP = re.compile(r"<[^>]+>|\{[^}]*\}")
_VTT_META = ("WEBVTT", "NOTE", "STYLE", "REGION", "Kind:", "Language:")


class BenchError(RuntimeError):
    """The benchmark cannot run (no clips, ffmpeg missing, unreadable audio)."""


@dataclass(frozen=True, slots=True)
class Clip:
    name: str
    audio: Path
    reference: Path | None = None


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One provider to benchmark, optionally with a model (``local:tiny``)."""

    provider: str
    model: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.model else self.provider


def parse_provider_specs(text: str) -> list[ProviderSpec]:
    """``"local:tiny,openai"`` -> two specs. Empty items and duplicates are ignored."""
    specs: list[ProviderSpec] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        provider, _, model = item.partition(":")
        spec = ProviderSpec(provider.strip().lower(), model.strip() or None)
        if spec not in specs:
            specs.append(spec)
    if not specs:
        raise BenchError("no providers given")
    return specs


# --------------------------------------------------------------------------- #
# Clips and references
# --------------------------------------------------------------------------- #


def discover_clips(directory: str | Path) -> list[Clip]:
    """Audio files in ``directory`` paired with a same-named reference, sorted by name."""
    base = Path(directory)
    if not base.is_dir():
        raise BenchError(f"clips directory not found: {base}")
    clips: list[Clip] = []
    for audio in sorted(base.iterdir()):
        if audio.suffix.lower() not in AUDIO_EXTENSIONS or not audio.is_file():
            continue
        reference = next(
            (
                audio.with_suffix(ext)
                for ext in REFERENCE_EXTENSIONS
                if audio.with_suffix(ext).is_file()
            ),
            None,
        )
        clips.append(Clip(audio.stem, audio, reference))
    if not clips:
        raise BenchError(
            f"no audio clips in {base} (looked for {', '.join(AUDIO_EXTENSIONS)}); "
            "put <name>.wav next to <name>.txt"
        )
    return clips


def read_reference(path: str | Path) -> str:
    """The reference text of a ``.txt``, ``.srt`` or ``.vtt`` file, as one string."""
    file = Path(path)
    raw = file.read_text(encoding="utf-8-sig")
    if file.suffix.lower() == ".txt":
        return " ".join(raw.split())
    words: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or _TIMING_LINE.search(line) or _CUE_NUMBER.match(line):
            continue
        if line.startswith(_VTT_META):
            continue
        cleaned = _MARKUP.sub("", line).strip()
        if cleaned:
            words.append(cleaned)
    return " ".join(" ".join(words).split())


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def decode_audio(
    path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    ffmpeg: str = "ffmpeg",
    timeout: float = 600.0,
) -> bytes:
    """Decode any audio file to mono signed 16-bit PCM at ``sample_rate`` with ffmpeg."""
    binary = shutil.which(ffmpeg)
    if binary is None:
        raise BenchError(f"{ffmpeg} not found on PATH (macOS: brew install ffmpeg)")
    command = [
        binary, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"decoding {Path(path).name} timed out") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no audio"]
        raise BenchError(f"could not decode {Path(path).name}: {detail[0]}")
    return proc.stdout


def split_pcm(
    pcm: bytes, sample_rate: int, chunk_seconds: float
) -> list[tuple[float, float, bytes]]:
    """Cut PCM into ``(start, end, bytes)`` chunks of ``chunk_seconds``.

    A final piece shorter than the providers' minimum is merged into the previous chunk
    instead of being sent alone (models hallucinate on fragments).
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    step = max(1, round(sample_rate * chunk_seconds)) * 2
    pieces = [pcm[i : i + step] for i in range(0, len(pcm), step)]
    min_bytes = int(sample_rate * MIN_AUDIO_SECONDS) * 2
    if len(pieces) > 1 and len(pieces[-1]) < min_bytes:
        tail = pieces.pop()
        pieces[-1] += tail
    out: list[tuple[float, float, bytes]] = []
    offset = 0
    for piece in pieces:
        start = offset / 2 / sample_rate
        offset += len(piece)
        out.append((start, offset / 2 / sample_rate, piece))
    return out


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ClipResult:
    name: str
    audio_seconds: float
    processing_seconds: float = 0.0
    chunks: int = 0
    skipped_silent: int = 0
    failures: int = 0
    latencies: list[float] = field(default_factory=list)
    hypothesis: str = ""
    reference: str | None = None
    wer: ErrorCounts | None = None
    cer: ErrorCounts | None = None
    hallucinations: int = 0
    cost_usd: float = 0.0

    @property
    def rtf(self) -> float | None:
        return real_time_factor(self.processing_seconds, self.audio_seconds)

    def describe(self) -> dict[str, Any]:
        rtf = self.rtf
        return {
            "name": self.name,
            "audio_seconds": round(self.audio_seconds, 2),
            "chunks": self.chunks,
            "skipped_silent": self.skipped_silent,
            "failures": self.failures,
            "hallucinations": self.hallucinations,
            "wer": None if self.wer is None else self.wer.describe(),
            "cer": None if self.cer is None else self.cer.describe(),
            "latency": _rounded(latency_stats(self.latencies)),
            "rtf": None if rtf is None else round(rtf, 3),
            "cost_usd": round(self.cost_usd, 6),
            "hypothesis": self.hypothesis,
            "reference": self.reference,
        }


@dataclass(slots=True)
class ProviderResult:
    spec: ProviderSpec
    model: str = ""
    clips: list[ClipResult] = field(default_factory=list)
    cold_start_seconds: float | None = None
    error: str | None = None
    """Why the provider could not run at all (missing extra or key); the run went on."""

    @property
    def label(self) -> str:
        return f"{self.spec.provider}:{self.model}" if self.model else self.spec.provider

    @property
    def wer(self) -> ErrorCounts | None:
        counts = [c.wer for c in self.clips if c.wer is not None]
        return aggregate(counts) if counts else None

    @property
    def cer(self) -> ErrorCounts | None:
        counts = [c.cer for c in self.clips if c.cer is not None]
        return aggregate(counts) if counts else None

    @property
    def latencies(self) -> list[float]:
        return [value for clip in self.clips for value in clip.latencies]

    @property
    def audio_seconds(self) -> float:
        return sum(c.audio_seconds for c in self.clips)

    @property
    def processing_seconds(self) -> float:
        return sum(c.processing_seconds for c in self.clips)

    @property
    def rtf(self) -> float | None:
        return real_time_factor(self.processing_seconds, self.audio_seconds)

    @property
    def failures(self) -> int:
        return sum(c.failures for c in self.clips)

    @property
    def hallucinations(self) -> int:
        return sum(c.hallucinations for c in self.clips)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.clips)

    def describe(self) -> dict[str, Any]:
        rtf = self.rtf
        wer_counts, cer_counts = self.wer, self.cer
        return {
            "provider": self.spec.provider,
            "model": self.model,
            "error": self.error,
            "clips": len(self.clips),
            "audio_seconds": round(self.audio_seconds, 2),
            "wer": None if wer_counts is None else wer_counts.describe(),
            "cer": None if cer_counts is None else cer_counts.describe(),
            "latency": _rounded(latency_stats(self.latencies)),
            "rtf": None if rtf is None else round(rtf, 3),
            "cold_start_seconds": (
                None if self.cold_start_seconds is None else round(self.cold_start_seconds, 3)
            ),
            "failures": self.failures,
            "hallucinations": self.hallucinations,
            "cost_usd": round(self.cost_usd, 6),
            "per_clip": [clip.describe() for clip in self.clips],
        }


def _rounded(stats: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}


def environment(chunk_seconds: float, sample_rate: int) -> dict[str, Any]:
    """Where and how the numbers were produced. No host names, no user names."""
    versions: dict[str, str] = {}
    for package in ("faster-whisper", "sherpa-onnx", "ctranslate2", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    ffmpeg_version = None
    binary = shutil.which("ffmpeg")
    if binary:
        try:
            out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=10)
            ffmpeg_version = out.stdout.splitlines()[0] if out.stdout else None
        except (OSError, subprocess.SubprocessError):
            ffmpeg_version = None
    return {
        "date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "livestream_transcriber": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ffmpeg": ffmpeg_version,
        "packages": versions,
        "chunk_seconds": chunk_seconds,
        "sample_rate": sample_rate,
        "silence_gate_dbfs": SILENCE_DBFS,
    }


@dataclass(slots=True)
class BenchReport:
    environment: dict[str, Any]
    providers: list[ProviderResult]
    clips: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "clips": self.clips,
            "providers": [p.describe() for p in self.providers],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Two tables: one row per provider, then one per provider and clip."""
        env = self.environment
        lines = [
            "# Benchmark",
            "",
            f"{len(self.clips)} clip(s), {env['chunk_seconds']:g} s chunks, "
            f"{env['python']} on {env['platform']}, {env['date']}.",
            "",
            "| Provider | Model | WER | CER | p50 latency | p95 latency | RTF | Cold start | "
            "Failures | Cost (USD) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for result in self.providers:
            if result.error:
                lines.append(
                    f"| {result.spec.provider} | {result.model or '-'} | unavailable: "
                    f"{_cell(result.error)} | | | | | | | |"
                )
                continue
            lat = latency_stats(result.latencies)
            lines.append(
                f"| {result.spec.provider} | {result.model or '-'} | {_pct(result.wer)} | "
                f"{_pct(result.cer)} | {_sec(lat['p50'])} | {_sec(lat['p95'])} | "
                f"{_num(result.rtf)} | {_sec(result.cold_start_seconds)} | "
                f"{result.failures} | {result.cost_usd:.4f} |"
            )
        lines += [
            "",
            "WER and CER are micro-averaged over all clips with a reference. RTF is "
            "processing time divided by audio time (below 1 keeps up with a live stream). "
            "Latency is per chunk request; the cold start is the first request, which "
            "includes loading the model.",
            "",
            "## Per clip",
            "",
            "| Provider | Clip | Audio (s) | WER | CER | RTF | Hallucinations |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for result in self.providers:
            for clip in result.clips:
                lines.append(
                    f"| {result.label} | {clip.name} | {clip.audio_seconds:.1f} | "
                    f"{_pct(clip.wer)} | {_pct(clip.cer)} | {_num(clip.rtf)} | "
                    f"{clip.hallucinations} |"
                )
        return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _pct(counts: ErrorCounts | None) -> str:
    rate = None if counts is None else counts.rate
    return "-" if rate is None else f"{rate * 100:.1f}%"


def _sec(value: float | int | None) -> str:
    return "-" if value is None else f"{value:.2f} s"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

TranscriberFactory = Callable[[Settings, ProviderSpec], Transcriber]
Decoder = Callable[[Path], bytes]
Progress = Callable[[str], None]


def _default_factory(settings: Settings, spec: ProviderSpec) -> Transcriber:
    # fallback="none" and no cache: a benchmark measures one provider, not a chain.
    return build_transcriber(settings, provider=spec.provider, model=spec.model, fallback="none")


def _transcribe_one(
    transcriber: Transcriber, pcm: bytes, sample_rate: int, start: float, end: float
) -> tuple[Transcript | None, float]:
    began = time.perf_counter()
    result = transcriber.transcribe(pcm, sample_rate, start=start, end=end)
    return result, time.perf_counter() - began


def _run_clip(
    transcriber: Transcriber,
    clip: Clip,
    chunks: Sequence[tuple[float, float, bytes]],
    *,
    sample_rate: int,
    skip_silence: bool,
) -> ClipResult:
    result = ClipResult(name=clip.name, audio_seconds=chunks[-1][1] if chunks else 0.0)
    texts: list[str] = []
    for start, end, pcm in chunks:
        result.chunks += 1
        if skip_silence and pcm_is_silent(pcm):
            result.skipped_silent += 1
            continue
        transcript, elapsed = _transcribe_one(transcriber, pcm, sample_rate, start, end)
        result.processing_seconds += elapsed
        if transcript is None:
            result.latencies.append(elapsed)
            continue
        if transcript.unavailable:
            result.failures += 1
            continue
        result.latencies.append(elapsed)
        result.cost_usd += transcript.cost_usd or 0.0
        text = transcript.text.strip()
        if text:
            texts.append(text)
            if is_hallucination(text):
                result.hallucinations += 1
    result.hypothesis = " ".join(texts)
    if clip.reference is not None:
        reference = read_reference(clip.reference)
        result.reference = reference
        result.wer = wer(reference, result.hypothesis)
        result.cer = cer(reference, result.hypothesis)
    return result


def run_bench(
    clips_dir: str | Path,
    specs: Sequence[ProviderSpec],
    settings: Settings,
    *,
    chunk_seconds: float = 5.0,
    cold_pass: bool = True,
    skip_silence: bool = True,
    transcriber_factory: TranscriberFactory | None = None,
    decoder: Decoder | None = None,
    progress: Progress | None = None,
) -> BenchReport:
    """Benchmark ``specs`` on every clip in ``clips_dir``.

    A provider that cannot start (missing extra, missing key) is reported with its
    reason and skipped; it does not abort the others. Everything else that goes wrong
    raises :class:`BenchError`.
    """
    say = progress or (lambda _msg: None)
    clips = discover_clips(clips_dir)
    sample_rate = settings.capture_sample_rate
    decode = decoder or (
        lambda path: decode_audio(
            path, sample_rate=sample_rate, ffmpeg=settings.capture_ffmpeg_binary
        )
    )
    factory = transcriber_factory or _default_factory

    # Decode once; every provider sees identical audio.
    audio: list[tuple[Clip, list[tuple[float, float, bytes]]]] = []
    for clip in clips:
        say(f"decoding {clip.audio.name}")
        audio.append((clip, split_pcm(decode(clip.audio), sample_rate, chunk_seconds)))

    results: list[ProviderResult] = []
    for spec in specs:
        result = ProviderResult(spec, model=spec.model or settings.model_for(spec.provider))
        results.append(result)
        try:
            transcriber = factory(settings, spec)
        except ConfigError as exc:
            result.error = str(exc)
            say(f"{spec.label}: skipped ({exc})")
            continue
        try:
            if cold_pass:
                first = next(
                    (
                        chunk
                        for _, chunks in audio
                        for chunk in chunks
                        if not pcm_is_silent(chunk[2])
                    ),
                    None,
                )
                if first is not None:
                    say(f"{spec.label}: cold pass")
                    _, result.cold_start_seconds = _transcribe_one(
                        transcriber, first[2], sample_rate, first[0], first[1]
                    )
            for clip, chunks in audio:
                say(f"{spec.label}: {clip.name}")
                result.clips.append(
                    _run_clip(
                        transcriber,
                        clip,
                        chunks,
                        sample_rate=sample_rate,
                        skip_silence=skip_silence,
                    )
                )
        finally:
            close_transcriber(transcriber)
    return BenchReport(
        environment=environment(chunk_seconds, sample_rate),
        providers=results,
        clips=[clip.name for clip in clips],
    )


def write_report(
    report: BenchReport, json_path: str | Path | None, md_path: str | Path | None
) -> None:
    """Write the JSON and/or markdown report, creating parent directories."""
    for path, text in ((json_path, report.to_json()), (md_path, report.to_markdown())):
        if path is None:
            continue
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
