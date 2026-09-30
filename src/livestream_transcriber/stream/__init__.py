"""Audio capture: resolving, ffmpeg supervision, fallback, recording and replay."""

from .auto_resume import ProbeBackoff, SourceStatusMonitor, wait_until_live
from .base import StreamError, StreamNotLiveError, StreamResolutionError, StreamSource
from .fallback import (
    LiveSourceSelector,
    SourceCandidate,
    SourceSelection,
    decide_source,
    maybe_failover,
)
from .queues import DropOldestQueue
from .recorder import Recorder
from .replay import RecordingError, ReplayStreamSource, load_audio_chunks, read_manifest
from .resolver import resolve_stream, resolve_stream_sync
from .source import CaptureOptions, LiveStreamSource

__all__ = [
    "CaptureOptions",
    "DropOldestQueue",
    "LiveSourceSelector",
    "LiveStreamSource",
    "ProbeBackoff",
    "Recorder",
    "RecordingError",
    "ReplayStreamSource",
    "SourceCandidate",
    "SourceSelection",
    "SourceStatusMonitor",
    "StreamError",
    "StreamNotLiveError",
    "StreamResolutionError",
    "StreamSource",
    "decide_source",
    "load_audio_chunks",
    "maybe_failover",
    "read_manifest",
    "resolve_stream",
    "resolve_stream_sync",
    "wait_until_live",
]
