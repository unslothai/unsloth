"""Extended fake llama-server for simulation tests.

Builds on the base shim used by tests/studio_load_freeze/. Adds these
knobs for full failure-mode coverage:

  - ``tok_status``      : HTTP status to return on /tokenize (default 200)
  - ``tok_body``        : raw bytes to write as the /tokenize response
                          (overrides JSON serialisation); use to inject
                          malformed JSON, partial bodies, etc.
  - ``tok_reset``       : if True, write a partial body then close the
                          socket abruptly (simulates connection reset
                          during recv)
  - ``tok_response_map``: dict mapping a request body (``content`` field)
                          to a token list (so the test can synthesise
                          matches for snac / csm / bicodec / dac /
                          whisper / audio_vlm)
  - ``detok_status``    : same shape as tok_status but for /detokenize
  - ``detok_body``      : raw bytes override for /detokenize
  - ``detok_map``       : token-id -> string mapping for /detokenize
                          responses (used by detect_audio_type to test
                          for the "<custom_token_..." prefix that
                          triggers the snac match)

All other behaviour matches the base shim.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


LLAMA_SERVER_STDOUT_TEMPLATE = """\
0.00.040.600 W Setting 'enable_thinking' via --chat-template-kwargs is deprecated.
0.00.198.766 I srv          main: loading model
0.00.198.817 I srv    load_model: loading model '{model_path}'
0.05.583.299 I srv          main: model loaded
0.05.583.301 I srv          main: server is listening on http://127.0.0.1:{port}
0.05.583.315 I srv  update_slots: all slots are idle
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_raw(
        self, status: int, body: bytes, *, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_reset(self, partial: bytes) -> None:
        """Write a partial body and slam the connection. Simulates a
        crashed llama-server returning a RemoteProtocolError to httpx."""
        # Don't call send_response -- write a half-finished response.
        try:
            self.wfile.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 9999\r\n\r\n"
            )
            self.wfile.write(partial)
            self.wfile.flush()
        except Exception:
            pass
        try:
            # Use socket-level shutdown so the next read sees a reset.
            sock = self.connection
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\1\0\0\0\0\0\0\0")
            sock.close()
        except Exception:
            pass

    def do_GET(self) -> None:  # noqa: N802
        srv: "FakeLlamaServer._Server" = self.server  # type: ignore[assignment]
        path = self.path.split("?", 1)[0]
        if path == "/health":
            time.sleep(srv.config.health_delay)
            if srv.config.health_fail:
                self._send_json(503, {"status": "unavailable"})
            else:
                self._send_json(200, {"status": "ok"})
            return
        if path == "/props":
            self._send_json(200, {"chat_template": "", "total_slots": 1})
            return
        self._send_json(404, {"error": f"unknown route {path}"})

    def do_POST(self) -> None:  # noqa: N802
        srv: "FakeLlamaServer._Server" = self.server  # type: ignore[assignment]
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            body = {}

        path = self.path.split("?", 1)[0]
        if path == "/tokenize":
            time.sleep(srv.config.tok_delay)
            if srv.config.tok_reset:
                self._send_reset(partial = b'{"toke')
                return
            if srv.config.tok_body is not None:
                self._send_raw(srv.config.tok_status, srv.config.tok_body)
                return
            content = str(body.get("content", ""))
            # tok_response_map lets the test inject a specific token count
            # for a specific input text. Used to synthesise "this text
            # tokenises to exactly one token" for the csm / bicodec / dac
            # detection branches.
            if content in srv.config.tok_response_map:
                tokens = list(srv.config.tok_response_map[content])
            else:
                tokens = list(range(max(1, len(content.split()) or 1)))
            self._send_json(srv.config.tok_status, {"tokens": tokens})
            return
        if path == "/detokenize":
            time.sleep(srv.config.detok_delay)
            if srv.config.detok_body is not None:
                self._send_raw(srv.config.detok_status, srv.config.detok_body)
                return
            tids = body.get("tokens") or []
            content = "".join(
                srv.config.detok_map.get(int(t), f"<tok_{t}>") for t in tids
            )
            self._send_json(srv.config.detok_status, {"content": content})
            return
        if path == "/completion":
            time.sleep(srv.config.completion_delay)
            self._send_json(200, {"content": "", "tokens_predicted": 0})
            return
        self._send_json(404, {"error": f"unknown route {path}"})


class FakeLlamaServer:
    class _Config:
        __slots__ = (
            "health_delay",
            "health_fail",
            "tok_delay",
            "tok_status",
            "tok_body",
            "tok_reset",
            "tok_response_map",
            "detok_delay",
            "detok_status",
            "detok_body",
            "detok_map",
            "completion_delay",
        )

        def __init__(
            self,
            *,
            health_delay: float,
            health_fail: bool,
            tok_delay: float,
            tok_status: int,
            tok_body: Optional[bytes],
            tok_reset: bool,
            tok_response_map: dict,
            detok_delay: float,
            detok_status: int,
            detok_body: Optional[bytes],
            detok_map: dict,
            completion_delay: float,
        ) -> None:
            self.health_delay = health_delay
            self.health_fail = health_fail
            self.tok_delay = tok_delay
            self.tok_status = tok_status
            self.tok_body = tok_body
            self.tok_reset = tok_reset
            self.tok_response_map = tok_response_map
            self.detok_delay = detok_delay
            self.detok_status = detok_status
            self.detok_body = detok_body
            self.detok_map = detok_map
            self.completion_delay = completion_delay

    class _Server(ThreadingHTTPServer):
        config: "FakeLlamaServer._Config"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        health_delay: float = 0.0,
        health_fail: bool = False,
        tok_delay: float = 0.0,
        tok_status: int = 200,
        tok_body: Optional[bytes] = None,
        tok_reset: bool = False,
        tok_response_map: Optional[dict] = None,
        detok_delay: float = 0.0,
        detok_status: int = 200,
        detok_body: Optional[bytes] = None,
        detok_map: Optional[dict] = None,
        completion_delay: float = 0.0,
        # Cosmetic only -- only appears in the synthesised stdout
        # template's "loading model '<path>'" line. The production code
        # we drive from the tests does not parse this value. Default is
        # OS-portable so the shim does not bake any one developer's
        # machine path into the test surface (gemini-code-assist review
        # on PR #5669).
        model_path: str = "<test-fixture>/gemma-4.gguf",
    ) -> None:
        self.host = host
        self._requested_port = port
        self.model_path = model_path
        self.config = FakeLlamaServer._Config(
            health_delay = health_delay,
            health_fail = health_fail,
            tok_delay = tok_delay,
            tok_status = tok_status,
            tok_body = tok_body,
            tok_reset = tok_reset,
            tok_response_map = tok_response_map or {},
            detok_delay = detok_delay,
            detok_status = detok_status,
            detok_body = detok_body,
            detok_map = detok_map or {},
            completion_delay = completion_delay,
        )
        self._server: Optional[FakeLlamaServer._Server] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FakeLlamaServer":
        # Pass requested port directly (port=0 lets ThreadingHTTPServer pick
        # a free port atomically, avoiding the find-port-then-bind race the
        # gemini-code-assist review on PR #5669 flagged). After bind, read
        # the actually-bound port back via server_address[1].
        self._server = FakeLlamaServer._Server(
            (self.host, self._requested_port), _Handler
        )
        self._server.config = self.config
        bound_port = self._server.server_address[1]
        self._thread = threading.Thread(
            target = self._server.serve_forever,
            daemon = True,
            name = f"fake-llama-{bound_port}",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout = 5.0)
            self._thread = None

    def __enter__(self) -> "FakeLlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
