# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-29

First public release.

### Added

- `lst` command line: `doctor`, `run`, `record`, `replay`, `bench`, `models fetch`, and
  `rules test`, with stable exit codes (0 ok, 1 command failed, 2 configuration, 3 not live,
  4 no data, 5 stream lost, 130 interrupted).
- Sources: YouTube and any other yt-dlp URL, HLS (`.m3u8`) and plain HTTP media URLs, and
  local files, with ordered fallback sources.
- Audio-only capture through ffmpeg with a supervisor: full-jitter exponential
  reconnects, an audio stall watchdog, and auto-resume that waits for an offline stream to
  return.
- Speech-to-text providers: local faster-whisper, local sherpa-onnx (Parakeet, Moonshine),
  OpenAI and OpenAI-compatible servers, OpenRouter, plus `mock` and `none`. Circuit breaker
  per cloud provider, cloud-to-local fallback and a spend budget guard.
- Pipeline with bounded queues (drop-oldest for live, lossless for files and replay), a
  RAM queue with disk spill for STT jobs, an overload guard that pauses and probes, a media
  order gate and a speech stream that stitches overlapping transcripts.
- Event rules in YAML: keyword, regex and optional LLM matchers with cooldown, fuzzy
  de-duplication, severity escalation and shared groups, all on stream time.
- Notifiers: console, generic webhook (JSON, Slack and Discord payloads) and Telegram, with
  durable-before-notify delivery, per-target retry and rate limiting.
- Outputs: JSONL, SRT, WebVTT and SQLite. Subtitle cues follow the provider's own segment
  times when it reports them, and a run never overwrites the transcript files of an earlier
  one in the same directory.
- Optional transcript cache for repeated audio (`LST_STT_CACHE_DIR`).
- Recording and deterministic replay of captured audio.
- `lst bench`: WER, CER, latency percentiles, real-time factor and cold start per
  provider, with JSON and Markdown reports.
- Secret redaction on every log handler, ffmpeg child environment allowlist, process
  heartbeat line, optional memory limit, and configuration validated at startup.
- Docker image, example `docker-compose.yml` and systemd unit.
