"""Executable stand-ins for ffmpeg, so capture tests need no real decoder.

Each stub is a tiny Python script with the interpreter's path in its shebang. The
capture code launches it exactly as it would launch ffmpeg (same argv, allowlisted
environment), so stubs are configured by baking parameters into their source, never
through environment variables.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

_FAKE = Path(__file__).resolve().parent / "fake_ffmpeg.py"

# Writes ``chunks`` PCM blocks to stdout, closes it, lingers ``exit_delay`` seconds
# and exits with ``rc``. The readers see EOF at once; the exit code exists only once
# the stub has exited.
_PCM_THEN_EXIT = """#!{python}
import os, sys, time
for _ in range({chunks}):
    os.write(1, b"\\x01\\x00" * {samples})
os.close(1)
time.sleep({exit_delay})
sys.exit({rc})
"""

# Writes a few PCM blocks, then goes quiet: a stalled input. SIGTERM is handled the way
# real ffmpeg handles it: close the output at once (the pump sees EOF), then tear down
# and exit 255 ``exit_delay`` seconds later.
_STALLED = """#!{python}
import os, signal, sys, time

def on_term(_signum, _frame):
    os.close(1)
    time.sleep({exit_delay})
    os._exit(255)

signal.signal(signal.SIGTERM, on_term)
for _ in range({chunks}):
    os.write(1, b"\\x01\\x00" * {samples})
time.sleep(60)
"""

# Ignores SIGTERM (the handler only sets a flag, so a write() blocked on a full pipe is
# restarted and the process keeps running) and floods stdout.
_STUBBORN = """#!{python}
import os, signal, sys
signal.signal(signal.SIGTERM, lambda *_: None)
sys.stderr.write("stubborn stub up\\n")
sys.stderr.flush()
block = b"\\x01\\x00" * {samples}
while True:
    os.write(1, block)
"""

# Records its own argv and environment into ``child.json`` next to itself, then exits.
_DUMP = """#!{python}
import json, os, sys
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child.json")
with open(out, "w") as fh:
    json.dump({{"argv": sys.argv, "env": dict(os.environ)}}, fh)
"""


def write_script(directory: Path, name: str, body: str) -> str:
    """Write ``body`` as an executable script and return its path."""
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def stub_fake_ffmpeg(directory: Path) -> str:
    """The shared flood-stderr stub (``tests/support/fake_ffmpeg.py``) with its defaults."""
    lines = _FAKE.read_text(encoding="utf-8").splitlines()[1:]  # drop its shebang
    return write_script(directory, "fake_ffmpeg", f"#!{sys.executable}\n" + "\n".join(lines))


def stub_pcm_then_exit(
    directory: Path, *, rc: int = 0, chunks: int = 4, samples: int = 1600, exit_delay: float = 0.0
) -> str:
    body = _PCM_THEN_EXIT.format(
        python=sys.executable, rc=rc, chunks=chunks, samples=samples, exit_delay=exit_delay
    )
    return write_script(directory, f"pcm_stub_rc{rc}", body)


def stub_stalled(
    directory: Path, *, chunks: int = 3, samples: int = 1600, exit_delay: float = 0.1
) -> str:
    body = _STALLED.format(
        python=sys.executable, chunks=chunks, samples=samples, exit_delay=exit_delay
    )
    return write_script(directory, "stalled_stub", body)


def stub_stubborn(directory: Path, *, samples: int = 60_000) -> str:
    body = _STUBBORN.format(python=sys.executable, samples=samples)
    return write_script(directory, "stubborn_stub", body)


def stub_dump(directory: Path) -> str:
    return write_script(directory, "dump_stub", _DUMP.format(python=sys.executable))
