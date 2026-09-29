# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional background HTTP server exposing `/metrics` for Prometheus scraping."""

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from unsloth.metrics.prometheus import (
    enable_prometheus_metrics,
    generate_prometheus_metrics,
    get_metrics_content_type,
)


class MetricsHandler(BaseHTTPRequestHandler):
    def _reply(
        self,
        status,
        body,
        content_type = "text/plain; charset=utf-8",
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            try:
                self._reply(200, generate_prometheus_metrics(), get_metrics_content_type())
            except Exception as e:
                self._reply(500, f"Error generating metrics: {e}\n".encode())
        elif path in ("", "/"):
            self._reply(200, b"Unsloth Metrics Server\n/metrics - Prometheus metrics endpoint\n")
        else:
            self._reply(404, b"Not Found\n")

    def log_message(self, format, *args):
        pass


_metrics_server: Optional[ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_server_lock = threading.Lock()


def start_metrics_server(host: str = "127.0.0.1", port: int = 9090):
    """Serve `/metrics` on host:port in a daemon thread. Loopback by default;
    pass host="0.0.0.0" to expose it on every interface. port=0 picks a free port.
    Raises OSError if the port cannot be bound."""
    global _metrics_server, _server_thread
    with _server_lock:
        if _metrics_server is not None:
            return _server_thread
        enable_prometheus_metrics()
        # Bind here, not in the thread, so a busy port raises to the caller.
        server = ThreadingHTTPServer((host, port), MetricsHandler)
        server.daemon_threads = True
        thread = threading.Thread(
            target = server.serve_forever, daemon = True, name = "UnslothMetricsServer"
        )
        thread.start()
        _metrics_server, _server_thread = server, thread
        bound_host, bound_port = server.server_address[:2]
        print(f"Unsloth: metrics server started at http://{bound_host}:{bound_port}/metrics")
        return thread


def stop_metrics_server():
    global _metrics_server, _server_thread
    with _server_lock:
        if _metrics_server is None:
            return
        _metrics_server.shutdown()
        _metrics_server.server_close()
        if _server_thread is not None:
            _server_thread.join(timeout = 5)
        _metrics_server = _server_thread = None


def is_metrics_server_running() -> bool:
    return _metrics_server is not None


def get_metrics_server_port() -> Optional[int]:
    server = _metrics_server
    return server.server_address[1] if server is not None else None


def test_metrics_server(port: Optional[int] = None) -> bool:
    """True if the running server answers GET /metrics with 200."""
    port = port or get_metrics_server_port()
    if not is_metrics_server_running() or port is None:
        return False
    host = _metrics_server.server_address[0]
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout = 2) as r:
            return r.status == 200
    except Exception:
        return False
