"""Tiny HTTP helpers on top of the standard library.

Everything that talks to a web API (cloud speech-to-text, webhooks, Telegram)
goes through here, so there is one place to guard in tests and no third-party
HTTP client to install.
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request
from typing import Any

__all__ = ["HttpError", "NonJsonBody", "get_json", "post_json", "post_multipart"]


class HttpError(RuntimeError):
    """A non-2xx response. ``body`` is kept whole; the message is truncated."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class NonJsonBody(dict[str, Any]):
    """A 2xx response whose body was not JSON, kept as ``{"text": body}``.

    Callers that only need the request to have gone through (a webhook, a chat
    message) keep working. A caller that reads *content* from the body must
    check for this type: a proxy's HTML error page served with HTTP 200 is not
    a transcript, and must not be transcribed, billed or cached as one.
    """

    @property
    def raw(self) -> str:
        return str(self.get("text") or "")


# Headers that carry credentials. urllib keeps ordinary headers across a redirect, even to
# another host, so these are attached as "unredirected" ones, which it sends only on the
# original request.
_CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "x-api-key"})


def _add_headers(req: urllib.request.Request, headers: dict[str, str] | None) -> None:
    for key, value in (headers or {}).items():
        if key.lower() in _CREDENTIAL_HEADERS:
            req.add_unredirected_header(key, value)
        else:
            req.add_header(key, value)


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    _add_headers(req, headers)
    return _read(req, timeout)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    _add_headers(req, headers)
    return _read(req, timeout)


def post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST ``multipart/form-data``.

    ``files`` maps a form field to ``(filename, content, content_type)``.
    """
    boundary = "----lst-" + secrets.token_hex(12)
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(value.encode("utf-8") + b"\r\n")
    for name, (filename, content, ctype) in files.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    _add_headers(req, headers)
    return _read(req, timeout)


def _read(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise HttpError(exc.code, body) from exc
    if not raw:
        return {}
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return NonJsonBody(text=text)
    return parsed if isinstance(parsed, dict) else {"data": parsed}
