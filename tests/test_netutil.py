"""``netutil`` against a real HTTP server on loopback (the only network tests allow)."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from livestream_transcriber.netutil import (
    HttpError,
    NonJsonBody,
    get_json,
    post_json,
    post_multipart,
)


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.requests.append(
            {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body}
        )
        status, payload, ctype = {
            "/json": (200, b'{"ok": true}', "application/json"),
            "/list": (200, b"[1, 2]", "application/json"),
            "/html": (200, b"<html>proxy error</html>", "text/html"),
            "/empty": (200, b"", "text/plain"),
            "/boom": (500, b"upstream exploded", "text/plain"),
        }.get(self.path, (404, b"nope", "text/plain"))
        self.send_response(302 if self.path == "/redirect" else status)
        if self.path == "/redirect":
            self.send_header("Location", "/json")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _serve

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    handler = type("Handler", (_Handler,), {"requests": []})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", handler.requests
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_get_json_sends_headers(server: tuple[str, list[dict[str, Any]]]):
    base, seen = server
    assert get_json(f"{base}/json", headers={"X-Test": "1"}) == {"ok": True}
    assert seen[0]["headers"]["X-Test"] == "1"


def test_post_json_serialises_the_payload(server: tuple[str, list[dict[str, Any]]]):
    base, seen = server
    assert post_json(f"{base}/json", {"text": "Willkommen"}, headers={"Authorization": "Bearer x"})
    request = seen[0]
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["Authorization"] == "Bearer x"
    assert json.loads(request["body"]) == {"text": "Willkommen"}


def test_top_level_json_arrays_are_wrapped(server: tuple[str, list[dict[str, Any]]]):
    assert get_json(f"{server[0]}/list") == {"data": [1, 2]}


def test_empty_body_is_an_empty_dict(server: tuple[str, list[dict[str, Any]]]):
    assert get_json(f"{server[0]}/empty") == {}


def test_a_200_with_an_html_body_is_marked_as_non_json(
    server: tuple[str, list[dict[str, Any]]],
):
    out = get_json(f"{server[0]}/html")
    assert isinstance(out, NonJsonBody)
    assert out.raw == "<html>proxy error</html>"
    assert out["text"] == out.raw


def test_http_errors_carry_status_and_body(server: tuple[str, list[dict[str, Any]]]):
    with pytest.raises(HttpError) as info:
        post_json(f"{server[0]}/boom", {})
    assert info.value.status == 500
    assert info.value.body == "upstream exploded"
    assert "HTTP 500" in str(info.value)


def test_multipart_body_is_well_formed(server: tuple[str, list[dict[str, Any]]]):
    base, seen = server
    post_multipart(
        f"{base}/json",
        {"model": "whisper-1", "language": "de"},
        {"file": ("chunk.wav", b"RIFF\x00\x01binary", "audio/wav")},
        headers={"Authorization": "Bearer x"},
    )
    request = seen[0]
    ctype = request["headers"]["Content-Type"]
    assert ctype.startswith("multipart/form-data; boundary=----lst-")
    boundary = ctype.split("boundary=", 1)[1].encode()
    body: bytes = request["body"]
    assert body.count(b"--" + boundary + b"\r\n") == 3
    assert body.endswith(b"--" + boundary + b"--\r\n")
    assert b'name="model"\r\n\r\nwhisper-1\r\n' in body
    assert b'name="file"; filename="chunk.wav"\r\nContent-Type: audio/wav\r\n\r\nRIFF' in body
    assert request["headers"]["Authorization"] == "Bearer x"


def test_the_boundary_is_random(server: tuple[str, list[dict[str, Any]]]):
    base, seen = server
    for _ in range(2):
        post_multipart(f"{base}/json", {"a": "b"}, {})
    first, second = (r["headers"]["Content-Type"] for r in seen)
    assert first != second


def test_the_test_suite_blocks_remote_hosts():
    """The autouse guard turns an accidental real request into an immediate failure."""
    with pytest.raises(OSError, match="network blocked"):
        urllib.request.urlopen("https://example.com/")
    with pytest.raises(OSError, match="network blocked"):
        get_json("http://203.0.113.9/status")


def test_credentials_are_not_forwarded_across_a_redirect(
    server: tuple[str, list[dict[str, Any]]],
) -> None:
    base, requests = server
    get_json(
        f"{base}/redirect",
        headers={"Authorization": "Bearer your-api-key", "X-Trace": "abc"},
    )
    first, second = requests
    assert (
        first["path"] == "/redirect" and first["headers"]["Authorization"] == "Bearer your-api-key"
    )
    assert second["path"] == "/json"
    assert "Authorization" not in second["headers"]
    assert second["headers"]["X-Trace"] == "abc"
