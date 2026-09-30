"""A tiny HTTP server on a loopback port for tests of HTTP clients."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["Recorded", "serve"]


class Recorded:
    """Requests the server has seen, and the scripted replies it hands out."""

    def __init__(self, replies: list[tuple[int, bytes]]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.url = ""

    def json_bodies(self) -> list[Any]:
        return [json.loads(r["body"]) for r in self.requests]


@contextmanager
def serve(replies: list[tuple[int, bytes]] | None = None) -> Iterator[Recorded]:
    """Serve ``replies`` in order (the last one repeats); record every request."""
    state = Recorded(replies or [(200, b"{}")])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            state.requests.append(
                {"path": self.path, "headers": dict(self.headers.items()), "body": body}
            )
            index = min(len(state.requests) - 1, len(state.replies) - 1)
            status, payload = state.replies[index]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


ReplyFactory = Callable[[], list[tuple[int, bytes]]]
