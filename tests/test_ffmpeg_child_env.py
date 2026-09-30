"""What an ffmpeg child can see: its argv and its environment.

The argv of every process is world-readable (``/proc/<pid>/cmdline``, ``ps``) and a
child's environment holds whatever we pass it. A proxy URL carries credentials; the
parent environment holds STT keys and chat-bot tokens. None of that may reach ffmpeg
except the proxy, and the proxy only through the environment.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from livestream_transcriber.stream import ffmpeg as ffmpeg_mod
from livestream_transcriber.stream.ffmpeg import FFmpegAudioPipe, FFmpegSpec
from tests.support.ffmpeg_stubs import stub_dump

PROXY = "http://proxy-user:your-proxy-password@proxy.example.test:8877"

# Secret-looking variables the process's own environment really holds, plus a couple
# of generic ones: none of them has any business inside ffmpeg.
PARENT_SECRETS = {
    "OPENROUTER_API_KEY": "your-openrouter-key",
    "OPENAI_API_KEY": "your-openai-key",
    "LST_NOTIFY_TELEGRAM_TOKEN": "123456:your-telegram-bot-token",
    "LST_NOTIFY_WEBHOOK_URL": "https://hooks.example.test/your-webhook-path",
    "LST_CAPTURE_PROXY": PROXY,
    "LST_CAPTURE_COOKIES_FILE": "/secrets/cookies.txt",
    "AWS_SECRET_ACCESS_KEY": "your-aws-secret",
    "SOME_PASSWORD": "your-password",
}

_ALLOWED = {
    "PATH", "HOME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "LD_LIBRARY_PATH",
    "SSL_CERT_FILE", "SSL_CERT_DIR",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY",
}  # fmt: skip


async def _spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, proxy: str | None) -> Any:
    for key, value in PARENT_SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    spec = FFmpegSpec(
        url="https://cdn.example.test/audio/index.m3u8",
        binary=stub_dump(tmp_path),
        hls_live_start_index=None,
        http_proxy=proxy,
    )
    pipe = FFmpegAudioPipe(spec)
    await pipe.start()
    # The stub exits by itself once it has written what it saw.
    await asyncio.wait_for(pipe.wait(), 20)
    await pipe.stop()
    return json.loads((tmp_path / "child.json").read_text(encoding="utf-8"))


def _unexpected(env: dict[str, str]) -> set[str]:
    # The OS may add its own launch variables to every child (macOS:
    # __CF_USER_TEXT_ENCODING); nothing else outside the allowlist.
    return {key for key in env if key not in _ALLOWED and not key.startswith(("LC_", "__CF"))}


async def test_child_sees_no_parent_secret_and_no_credentials_in_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = await _spawn(tmp_path, monkeypatch, proxy=PROXY)
    env, argv = seen["env"], seen["argv"]

    for key, value in PARENT_SECRETS.items():
        assert key not in env, key
        if value != PROXY:
            assert all(value not in v for v in env.values()), key
    assert not any("your-proxy-password" in tok for tok in argv)
    assert "-http_proxy" not in argv

    # The proxy arrives where ffmpeg's http and tls protocols look for it, and the
    # loopback playlist server is exempt.
    assert env["http_proxy"] == PROXY
    assert "127.0.0.1" in env["no_proxy"]
    # What stays is plumbing, not configuration.
    assert env["PATH"]
    assert env["LC_ALL"] == "C.UTF-8"
    assert _unexpected(env) == set()


async def test_child_without_a_proxy_gets_no_secrets_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(key, raising=False)
    seen = await _spawn(tmp_path, monkeypatch, proxy=None)
    env = seen["env"]
    for key in PARENT_SECRETS:
        assert key not in env, key
    assert "http_proxy" not in env
    assert _unexpected(env) == set()


def test_child_env_is_an_allowlist() -> None:
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "LANG": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "TZ": "UTC",
        **PARENT_SECRETS,
    }
    env = ffmpeg_mod.ffmpeg_child_env(PROXY, parent)
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/home/someone"
    assert env["LC_CTYPE"] == "C.UTF-8" and env["TZ"] == "UTC"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert env[key] == PROXY
    assert env["no_proxy"] == env["NO_PROXY"] == "127.0.0.1,localhost,::1"
    assert not set(PARENT_SECRETS) & set(env)


def test_operator_proxy_in_our_env_is_kept_when_none_is_configured() -> None:
    parent = {
        "PATH": "/usr/bin",
        "https_proxy": "http://corp.example.test:3128",
        "no_proxy": "internal",
    }
    env = ffmpeg_mod.ffmpeg_child_env(None, parent)
    assert env["https_proxy"] == "http://corp.example.test:3128"
    assert env["no_proxy"] == "internal"


@pytest.mark.slow
async def test_real_ffmpeg_sends_https_through_the_env_proxy(ffmpeg_bin: str) -> None:
    """The claim the argv design rests on: with no -http_proxy option, real ffmpeg
    still tunnels an https input through ``http_proxy`` from its environment,
    credentials included. The "proxy" is a loopback socket that challenges once
    (ffmpeg sends credentials only after a 407) and records the requests; nothing
    leaves the machine."""
    requests: list[bytes] = []
    got = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        requests.append(head)
        if b"Proxy-Authorization:" in head:
            got.set()
        else:
            writer.write(
                b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                b'Proxy-Authenticate: Basic realm="proxy"\r\n'
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    proxy = f"http://proxy-user:your-proxy-password@127.0.0.1:{port}"
    spec = FFmpegSpec(
        url="https://media.example.invalid/audio/index.m3u8",
        binary=ffmpeg_bin,
        hls_live_start_index=None,
        http_proxy=proxy,
    )
    pipe = FFmpegAudioPipe(spec)
    try:
        await pipe.start()
        await asyncio.wait_for(got.wait(), timeout=20)
    finally:
        await pipe.stop(timeout=2.0)
        server.close()
        await server.wait_closed()
    head = requests[-1].decode("latin-1")
    assert head.startswith("CONNECT media.example.invalid:443 ")
    expected = base64.b64encode(b"proxy-user:your-proxy-password").decode()
    assert f"Proxy-Authorization: Basic {expected}" in head
    assert not any("your-proxy-password" in tok for tok in pipe.command)
