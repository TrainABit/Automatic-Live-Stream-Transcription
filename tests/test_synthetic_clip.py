"""The synthetic audio fixtures are real, decodable audio (and the ffmpeg fixture skips cleanly)."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np


def test_wav_clip_is_a_six_second_440hz_tone(synthetic_clip: Path):
    with wave.open(str(synthetic_clip), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.getnframes() == 6 * 16000
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples))
    peak_hz = np.argmax(spectrum) * 16000 / len(samples)
    assert abs(peak_hz - 440) < 1.0


def test_mp4_clip_decodes_to_pcm(ffmpeg_bin: str, synthetic_clip_mp4: Path):
    out = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-i", str(synthetic_clip_mp4), "-f", "s16le", "-ac", "1",
         "-ar", "16000", "pipe:1"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    seconds = len(out) / 2 / 16000
    assert 5.5 < seconds < 6.5
