"""One ffmpeg process that turns any input into raw mono PCM on stdout.

Why counters and not timestamp parsing
--------------------------------------
The audio output is decoded to constant-rate ``s16le`` PCM, so sample *n* sits at
``n / sample_rate`` exactly. Chunk timestamps therefore come from counting bytes,
with no PTS parsing and no drift. That holds across gaps in the input only because
``aresample=async=1`` fills a hole in the audio timestamps with silence (and trims
an overlap) instead of letting the samples go missing and every later chunk drift
early. Silence costs nothing downstream: silent chunks never reach STT.

What the child sees
-------------------
ffmpeg's argv is world-readable (``/proc/<pid>/cmdline``, ``ps``), so it carries no
credentials: a proxy reaches ffmpeg through its environment only (``http_proxy``,
which ffmpeg's http *and* tls protocols read), and secret-bearing request headers
are never put on the command line. The child also gets a small allowlisted
environment instead of ours, so API keys and bot tokens never sit in an ffmpeg
process.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field

from ..logging_setup import get_logger
from ..redact import describe_command

log = get_logger(__name__)

__all__ = [
    "FFmpegAudioPipe",
    "FFmpegSpec",
    "build_ffmpeg_command",
    "ffmpeg_available",
    "ffmpeg_child_env",
    "is_http_url",
    "looks_like_hls",
]

# Read in small pieces so a chunk is assembled with low latency; the chunker
# upstream decides the actual chunk length.
_AUDIO_READ_BYTES = 8192

# Parent variables ffmpeg may need to run: search path, locale, time zone, home
# (~/.ffmpeg presets), temp dir, and a library path or CA bundle on odd installs.
# Everything else (API keys, bot tokens) stays in the parent.
_CHILD_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
)
_PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
_LOOPBACK_NO_PROXY = "127.0.0.1,localhost,::1"

# Request headers that identify the caller. They are not put on the command line.
_SECRET_HEADERS = frozenset({"cookie", "authorization", "proxy-authorization"})


@dataclass(frozen=True, slots=True)
class FFmpegSpec:
    """Everything needed to build the capture command."""

    url: str
    """What ffmpeg opens: a media URL, an HLS manifest or a local file path."""
    sample_rate: int = 16000
    binary: str = "ffmpeg"
    loglevel: str = "warning"
    hls_live_start_index: int | None = -3
    """Where in a live playlist to begin, relative to the live edge (HLS only)."""
    input_seek: float | None = None
    """``-ss`` on the input, to start a recorded file part-way in."""
    duration: float | None = None
    """``-t`` on the output; mainly for tests and short recordings."""
    realtime: bool = False
    """``-re``: read the input at its native rate. Only meaningful for files."""
    extra_input_args: Sequence[str] = ()
    thread_queue_size: int | None = None
    user_agent: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    """HTTP headers the resolver says the media host wants."""
    http_proxy: str | None = None
    """Proxy for the ffmpeg child only, passed in its environment (see
    :func:`ffmpeg_child_env`). Not a process-wide variable: STT and webhook calls
    must not ride the capture egress."""


def is_http_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def _is_loopback(url: str) -> bool:
    """Local HLS windows are served on 127.0.0.1."""
    host = url.split("://", 1)[-1].split("/", 1)[0].split("@")[-1]
    host = host.split("]")[-1]
    host = host.rsplit(":", 1)[0].strip("[]")
    return host in {"127.0.0.1", "localhost", "::1"}


def looks_like_hls(url: str | None) -> bool:
    """Heuristic: live renditions are ``.m3u8`` manifests."""
    if not url:
        return False
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(".m3u8") or "/m3u8" in path or "manifest/hls" in path


def ffmpeg_available(binary: str = "ffmpeg") -> str | None:
    """Absolute path to the ffmpeg binary, or None."""
    return shutil.which(binary)


def ffmpeg_child_env(
    http_proxy: str | None,
    parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for an ffmpeg child: an allowlist, never a copy.

    With ``http_proxy`` set, ffmpeg reaches every non-loopback URL through it: its
    http protocol and its tls protocol both read ``http_proxy`` when no
    ``-http_proxy`` option is given, and both honour ``no_proxy``. That covers the
    remote playlist *and* the absolute segment URLs inside a local window, which an
    input option never did. The loopback playlist server is exempt via ``no_proxy``.

    Without one, a proxy the operator set in our own environment is passed on.
    """
    source = os.environ if parent is None else parent
    env = {
        key: value
        for key, value in source.items()
        if key in _CHILD_ENV_KEYS or key.startswith("LC_")
    }
    if http_proxy:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env[key] = http_proxy
        env["no_proxy"] = _LOOPBACK_NO_PROXY
        env["NO_PROXY"] = _LOOPBACK_NO_PROXY
    else:
        env.update({key: source[key] for key in _PROXY_ENV_KEYS if key in source})
    return env


def _header_block(headers: Mapping[str, str], *, gzip: bool) -> str:
    """The ``-headers`` value: safe request headers, CRLF-terminated."""
    lines = [
        f"{name}: {value}"
        for name, value in headers.items()
        if name.lower() not in _SECRET_HEADERS
        and name.lower() not in {"user-agent", "accept-encoding"}
        and "\n" not in name + value
        and "\r" not in name + value
    ]
    if gzip:
        lines.append("Accept-Encoding: gzip")
    return "".join(f"{line}\r\n" for line in lines)


def build_ffmpeg_command(spec: FFmpegSpec) -> list[str]:
    """Build the argv: one input, one raw PCM output on ``pipe:1``."""
    url = spec.url
    cmd = [
        spec.binary,
        "-hide_banner",
        "-nostdin",
        # The periodic "size= time= speed=" progress line is noise in the stderr
        # tail kept for diagnostics.
        "-nostats",
        "-loglevel",
        spec.loglevel,
    ]
    if spec.realtime:
        cmd.append("-re")
    # These are private options of the http protocol and the hls demuxer. ffmpeg
    # *errors out* when they are passed for an input that does not accept them, so
    # they are applied by URL shape rather than unconditionally.
    if is_http_url(url):
        # Let ffmpeg ride out transient HTTP hiccups itself before we escalate to a
        # full reconnect: a new ffmpeg run after a back-off and a fresh resolve.
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "5",
            "-rw_timeout", "15000000",  # microseconds
        ]  # fmt: skip
        if spec.user_agent:
            cmd += ["-user_agent", spec.user_agent]
        # A remote HLS playlist can be several MB of text; ask for it compressed. The
        # trailing CRLF is required by the http protocol option.
        block = _header_block(spec.headers, gzip=looks_like_hls(url) and not _is_loopback(url))
        if block:
            cmd += ["-headers", block]
        # No -http_proxy here: the proxy URL may carry credentials and the argv is
        # world-readable. ffmpeg_child_env() sets it instead.
        if spec.hls_live_start_index is not None and looks_like_hls(url):
            cmd += ["-live_start_index", str(spec.hls_live_start_index)]
    if spec.input_seek is not None:
        cmd += ["-ss", f"{spec.input_seek:.3f}"]
    if spec.thread_queue_size is not None:
        cmd += ["-thread_queue_size", str(spec.thread_queue_size)]
    cmd += [*spec.extra_input_args, "-i", url]

    if spec.duration is not None:
        cmd += ["-t", f"{spec.duration:.3f}"]
    cmd += [
        "-map", "0:a:0",
        "-vn",
        "-filter:a", "aresample=async=1",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(spec.sample_rate),
        "-ac", "1",
        "pipe:1",
    ]  # fmt: skip
    return cmd


class FFmpegAudioPipe:
    """Runs one ffmpeg process and exposes its raw PCM stream."""

    def __init__(self, spec: FFmpegSpec) -> None:
        self.spec = spec
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: list[str] = []
        self._command: list[str] = []
        self.diagnostics: dict[str, float | int | None] = {}
        self._t0 = 0.0
        self._returncode: int | None = None

    @property
    def command(self) -> list[str]:
        return list(self._command)

    @property
    def returncode(self) -> int | None:
        """Exit code; still there after :meth:`stop` dropped the process."""
        if self._proc is not None:
            return self._proc.returncode
        return self._returncode

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    async def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("already started")
        if ffmpeg_available(self.spec.binary) is None:
            raise FileNotFoundError(
                f"ffmpeg binary {self.spec.binary!r} not found on PATH "
                "(macOS: brew install ffmpeg, Debian/Ubuntu: apt install ffmpeg)"
            )
        self._command = build_ffmpeg_command(self.spec)
        self._t0 = time.monotonic()
        self.diagnostics = {
            "ffmpeg_started_s": None,
            "first_audio_byte_s": None,
            "audio_bytes": 0,
            "stderr_lines": 0,
            "ffmpeg_rc": None,
        }
        # Never the raw argv: input URLs may be signed and carry our public IP.
        log.debug("starting ffmpeg", extra=describe_command(self._command))
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=ffmpeg_child_env(self.spec.http_proxy),
            limit=1 << 20,
        )
        assert self._proc.stderr is not None
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._proc.stderr), name="ffmpeg-stderr"
        )
        self.diagnostics["ffmpeg_started_s"] = round(time.monotonic() - self._t0, 3)
        log.info(
            "ffmpeg process started", extra={"started_s": self.diagnostics["ffmpeg_started_s"]}
        )

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        # Holds the reader, not self._proc: stop() drops the process first and this
        # keeps draining until EOF, so ffmpeg's last words reach the tail. Without a
        # reader ffmpeg blocks in write() once the stderr pipe fills.
        try:
            while line := await stderr.readline():
                text = line.decode("utf-8", "replace").rstrip()
                if not text:
                    continue
                self._stderr_tail.append(text)
                del self._stderr_tail[:-40]
                self.diagnostics["stderr_lines"] = (
                    int(self.diagnostics.get("stderr_lines") or 0) + 1
                )
                log.debug("ffmpeg: %s", text)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            log.debug("ffmpeg stderr drain stopped", exc_info=True)

    async def read_audio(self) -> AsyncIterator[bytes]:
        """Yield PCM byte runs until EOF. Boundaries are arbitrary."""
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            try:
                data = await stdout.read(_AUDIO_READ_BYTES)
            except (BrokenPipeError, ConnectionResetError):
                return
            if not data:
                return
            if self.diagnostics.get("first_audio_byte_s") is None:
                self.diagnostics["first_audio_byte_s"] = round(time.monotonic() - self._t0, 3)
                log.info(
                    "ffmpeg first audio byte",
                    extra={"t_s": self.diagnostics["first_audio_byte_s"]},
                )
            self.diagnostics["audio_bytes"] = int(self.diagnostics.get("audio_bytes") or 0) + len(
                data
            )
            yield data

    async def wait(self) -> int:
        assert self._proc is not None
        return await self._proc.wait()

    async def stop(self, *, timeout: float = 5.0, grace: float = 0.0) -> None:
        """Terminate ffmpeg, reap it and release every pipe. Idempotent.

        ``grace`` first gives ffmpeg that long to exit on its own. Pass it once the
        output hit EOF: a clean end then keeps its real exit code instead of racing
        our SIGTERM. The code survives in :attr:`returncode` and
        ``diagnostics["ffmpeg_rc"]``.
        """
        proc, self._proc = self._proc, None
        try:
            if proc is not None:
                try:
                    if grace > 0 and proc.returncode is None:
                        with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                            await asyncio.wait_for(proc.wait(), grace)
                finally:
                    await self._reap(proc, timeout)
        finally:
            await self._finish_stderr()

    async def _reap(self, proc: asyncio.subprocess.Process, timeout: float) -> None:
        """SIGTERM, then SIGKILL; every wait is bounded.

        Stdout is drained while waiting. asyncio's ``wait()`` returns only once every pipe
        of the child is closed, and a pipe nobody reads any more (the pump was cancelled)
        stays paused before EOF while ffmpeg sits in ``write()`` on the full pipe and
        cannot act on SIGTERM. Reading and discarding lets it exit and EOF arrive.
        """
        discard = asyncio.create_task(_discard(proc.stdout), name="ffmpeg-stdout-discard")
        try:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout)
            except TimeoutError:
                if proc.returncode is None:
                    log.warning("ffmpeg did not exit, killing")
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                if proc.returncode is None:
                    proc.kill()
            raise
        finally:
            discard.cancel()
            self._returncode = proc.returncode
            self.diagnostics["ffmpeg_rc"] = proc.returncode

    async def _finish_stderr(self) -> None:
        task, self._stderr_task = self._stderr_task, None
        if task is None:
            return
        if not task.done():
            # The child is gone, so EOF is at most a moment away.
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 1.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _discard(reader: asyncio.StreamReader | None) -> None:
    """Read ``reader`` to EOF and throw the bytes away."""
    if reader is None:
        return
    with contextlib.suppress(Exception):
        while await reader.read(_AUDIO_READ_BYTES):
            pass
