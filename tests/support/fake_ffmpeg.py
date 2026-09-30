#!/usr/bin/env python3
"""Stand-in for ffmpeg used by capture-pipe tests.

Writes raw PCM to the last ``pipe:N`` argument (``pipe:1`` is stdout) and floods
stderr, so a blocked stderr drain would deadlock the writer. It is not a
decoder: the audio is a fixed byte pattern.

Behaviour can be tuned through environment variables so tests can script
failure modes without a second script:

``FAKE_FFMPEG_CHUNKS``   number of PCM writes (default 12)
``FAKE_FFMPEG_BYTES``    size of each write in bytes (default 8000)
``FAKE_FFMPEG_DELAY``    seconds between writes (default 0.01)
``FAKE_FFMPEG_EXIT``     process exit status (default 0)
``FAKE_FFMPEG_HANG``     when ``1``, sleep forever after the writes (simulates a stall)
"""

from __future__ import annotations

import os
import sys
import time


def _audio_fd(argv: list[str]) -> int:
    """File descriptor to write PCM to: the last ``pipe:N`` argument, else stdout."""
    fd = 1
    for token in argv:
        if token.startswith("pipe:"):
            fd = int(token.split(":", 1)[1])
    return fd


def main(argv: list[str]) -> int:
    fd = _audio_fd(argv)
    chunks = int(os.environ.get("FAKE_FFMPEG_CHUNKS", "12"))
    size = int(os.environ.get("FAKE_FFMPEG_BYTES", "8000"))
    delay = float(os.environ.get("FAKE_FFMPEG_DELAY", "0.01"))
    payload = b"\x01\x00" * (size // 2)
    for i in range(chunks):
        for _ in range(80):
            sys.stderr.write(("e" * 120) + f"-{i}\n")
        sys.stderr.flush()
        os.write(fd, payload)
        time.sleep(delay)
    if os.environ.get("FAKE_FFMPEG_HANG") == "1":
        while True:
            time.sleep(3600)
    return int(os.environ.get("FAKE_FFMPEG_EXIT", "0"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
