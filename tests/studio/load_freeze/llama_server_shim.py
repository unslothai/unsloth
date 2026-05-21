"""Fake llama-server for orchestrator / event-loop tests.

Stands up a stdlib ``ThreadingHTTPServer`` that answers the subset of
endpoints Unsloth Studio's ``LlamaCppBackend`` touches during a model
load (``/health``, ``/props``, ``/tokenize``, ``/detokenize``,
``/completion``). Every endpoint has a per-instance delay knob so a
test can model the post-``_wait_for_health`` block in issue #5642
deterministically without a real GPU.

Use as:
    - context manager from a test
        with FakeLlamaServer(tok_delay=5.0) as srv:
            httpx.get(f"http://127.0.0.1:{srv.port}/health")
    - or standalone
        python llama_server_shim.py --port 52495 --tok-delay 5
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


# Verbatim stdout sequence from issue #5642 user log, with ``{model_path}``
# and ``{port}`` placeholders. The drain thread and _classify_gpu_offload
# parse these lines, so the shim emits a faithful reproduction so the
# orchestrator test exercises the same code paths.
LLAMA_SERVER_STDOUT_TEMPLATE = """\
0.00.040.600 W Setting 'enable_thinking' via --chat-template-kwargs is deprecated. Use --reasoning on / --reasoning off instead.
0.00.040.891 I log_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.040.895 I device_info:
0.00.184.630 I   - CUDA0   : NVIDIA GeForce RTX 4060 Ti (16379 MiB, 15223 MiB free)
0.00.184.639 I   - CPU     : AMD Ryzen 5 PRO 4650G with Radeon Graphics (32064 MiB, 13696 MiB free)
0.00.184.697 I system_info: n_threads = 12 (n_threads_batch = 12) / 12 | CUDA : ARCHS = 890 | USE_GRAPHS = 1
0.00.184.729 I srv          init: running without SSL
0.00.184.744 I srv          init: using 11 threads for HTTP server
0.00.184.847 I srv         start: binding port with default address family
0.00.198.766 I srv          main: loading model
0.00.198.817 I srv    load_model: loading model '{model_path}'
0.04.071.607 I common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.05.523.103 I srv    load_model: initializing slots, n_slots = 1
0.05.566.139 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 131072
0.05.583.299 I srv          init: init: chat template, thinking = 1
0.05.583.264 I srv          init: init: chat template, thinking = 1
0.05.583.299 I srv          main: model loaded
0.05.583.301 I srv          main: server is listening on http://127.0.0.1:{port}
0.05.583.315 I srv  update_slots: all slots are idle
"""


def _free_port() -> int:
    """Bind a tcp socket on 127.0.0.1:0 to learn a free port, then release."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


class _Handler(BaseHTTPRequestHandler):
    """Per-request handler. Pulls its delay configuration from the server
    instance to keep state out of the handler class."""

    # Silence the noisy default logging; tests prefer clean stdout.
    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        return

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ----- GETs -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
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
            time.sleep(srv.config.props_delay)
            self._send_json(200, {"chat_template": "", "total_slots": 1})
            return
        self._send_json(404, {"error": f"unknown route {path}"})

    # ----- POSTs ------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 (stdlib API)
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
            # Mirror llama-server: return one token per word-ish.
            content = str(body.get("content", ""))
            tokens = list(range(max(1, len(content.split()) or 1)))
            self._send_json(200, {"tokens": tokens})
            return
        if path == "/detokenize":
            time.sleep(srv.config.detok_delay)
            # If the configured fake-vocab maps the requested token to a
            # specific string, return it; otherwise return a generic
            # filler that won't match Studio's codec heuristics.
            tids = body.get("tokens") or []
            mapped = srv.config.detok_map
            content = "".join(mapped.get(int(t), f"<tok_{t}>") for t in tids)
            self._send_json(200, {"content": content})
            return
        if path == "/completion":
            time.sleep(srv.config.completion_delay)
            self._send_json(200, {"content": "", "tokens_predicted": 0})
            return
        self._send_json(404, {"error": f"unknown route {path}"})


class FakeLlamaServer:
    """Test-friendly facade around the stdlib HTTP server.

    Each instance is single-use. Use as a context manager:
        with FakeLlamaServer(tok_delay=5.0) as srv:
            ...  # srv.port, srv.url, srv.stdout_text
    or call ``start()`` / ``stop()`` manually.
    """

    class _Config:
        __slots__ = (
            "health_delay",
            "health_fail",
            "props_delay",
            "tok_delay",
            "detok_delay",
            "completion_delay",
            "detok_map",
        )

        def __init__(
            self,
            *,
            health_delay: float,
            health_fail: bool,
            props_delay: float,
            tok_delay: float,
            detok_delay: float,
            completion_delay: float,
            detok_map: dict,
        ) -> None:
            self.health_delay = health_delay
            self.health_fail = health_fail
            self.props_delay = props_delay
            self.tok_delay = tok_delay
            self.detok_delay = detok_delay
            self.completion_delay = completion_delay
            self.detok_map = detok_map

    class _Server(ThreadingHTTPServer):
        # Attach the Config so handlers can read it without globals.
        config: "FakeLlamaServer._Config"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        health_delay: float = 0.0,
        health_fail: bool = False,
        props_delay: float = 0.0,
        tok_delay: float = 0.0,
        detok_delay: float = 0.0,
        completion_delay: float = 0.0,
        detok_map: Optional[dict] = None,
        model_path: str = r"C:\Users\Admin\.cache\hf\models--unsloth--gemma-4-E2B-it-GGUF\gemma-4-E2B-it-UD-Q4_K_XL.gguf",
    ) -> None:
        self.host = host
        self._requested_port = port
        self.model_path = model_path
        self.config = FakeLlamaServer._Config(
            health_delay = health_delay,
            health_fail = health_fail,
            props_delay = props_delay,
            tok_delay = tok_delay,
            detok_delay = detok_delay,
            completion_delay = completion_delay,
            detok_map = detok_map or {},
        )
        self._server: Optional[FakeLlamaServer._Server] = None
        self._thread: Optional[threading.Thread] = None

    # ----- lifecycle --------------------------------------------------

    def start(self) -> "FakeLlamaServer":
        port = self._requested_port or _free_port()
        self._server = FakeLlamaServer._Server((self.host, port), _Handler)
        self._server.config = self.config
        self._thread = threading.Thread(
            target = self._server.serve_forever, daemon = True, name = f"fake-llama-{port}"
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

    # ----- accessors --------------------------------------------------

    @property
    def port(self) -> int:
        assert self._server is not None, "FakeLlamaServer not started"
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def stdout_text(self) -> str:
        """Return the verbatim llama-server stdout sequence parameterised
        for this instance. Use to feed Studio's ``_drain_stdout``."""
        return LLAMA_SERVER_STDOUT_TEMPLATE.format(
            model_path = self.model_path, port = self.port
        )


# ----- CLI for standalone smoke tests -----------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description = "Fake llama-server for tests.")
    parser.add_argument("--port", type = int, default = 0)
    parser.add_argument("--host", default = "127.0.0.1")
    parser.add_argument("--health-delay", type = float, default = 0.0)
    parser.add_argument("--health-fail", action = "store_true")
    parser.add_argument("--tok-delay", type = float, default = 0.0)
    parser.add_argument("--detok-delay", type = float, default = 0.0)
    parser.add_argument("--props-delay", type = float, default = 0.0)
    parser.add_argument("--completion-delay", type = float, default = 0.0)
    # Swallow any unrecognised flag so the shim can also be used as a
    # drop-in replacement for the real llama-server binary in tests
    # that spawn it via subprocess.Popen with the production flag set.
    args, _unknown = parser.parse_known_args()

    srv = FakeLlamaServer(
        host = args.host,
        port = args.port,
        health_delay = args.health_delay,
        health_fail = args.health_fail,
        props_delay = args.props_delay,
        tok_delay = args.tok_delay,
        detok_delay = args.detok_delay,
        completion_delay = args.completion_delay,
    )
    srv.start()
    print(srv.stdout_text, flush = True)
    print(f"fake-llama-server listening on {srv.url}", flush = True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
