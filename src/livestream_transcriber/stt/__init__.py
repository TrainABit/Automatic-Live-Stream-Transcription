"""Speech-to-text interface, providers and wrappers.

Import the interface from here; the providers stay behind
:func:`livestream_transcriber.stt.factory.build_transcriber`, so an optional
dependency that is not installed never breaks an import.
"""

from .base import (
    STT_OK,
    STT_UNAVAILABLE,
    CircuitBreaker,
    FixtureTranscriber,
    MockTranscriber,
    NullTranscriber,
    Transcriber,
    Transcript,
    pcm_is_silent,
    transcribe_chunk,
)

__all__ = [
    "STT_OK",
    "STT_UNAVAILABLE",
    "CircuitBreaker",
    "FixtureTranscriber",
    "MockTranscriber",
    "NullTranscriber",
    "Transcriber",
    "Transcript",
    "pcm_is_silent",
    "transcribe_chunk",
]
