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

"""
Optional HTTP server for exposing Prometheus metrics.
"""

import threading
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

from unsloth.metrics.prometheus import (
    generate_prometheus_metrics,
    get_metrics_content_type,
    enable_prometheus_metrics,
)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics endpoint."""

    def do_GET(self):
        """Handle GET requests to /metrics endpoint."""
        if self.path == "/metrics":
            try:
                metrics_output = generate_prometheus_metrics()
                self.send_response(200)
                self.send_header("Content-Type", get_metrics_content_type())
                self.send_header("Content-Length", str(len(metrics_output)))
                self.end_headers()
                self.wfile.write(metrics_output)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error generating metrics: {str(e)}".encode())
        elif self.path == "/":
            # Simple health check endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Unsloth Metrics Server\n/metrics - Prometheus metrics endpoint")
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


_metrics_server: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None


def start_metrics_server(host: str = "0.0.0.0", port: int = 9090):
    """
    Start a background HTTP server to expose Prometheus metrics.

    Args:
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 9090)

    Returns:
        Thread object running the server
    """
    global _metrics_server, _server_thread

    if _metrics_server is not None:
        return _server_thread

    # Enable Prometheus metrics
    enable_prometheus_metrics()

    def run_server():
        global _metrics_server
        _metrics_server = HTTPServer((host, port), MetricsHandler)
        _metrics_server.serve_forever()

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    print(f"📊 Unsloth metrics server started at http://{host}:{port}/metrics")

    return _server_thread


def stop_metrics_server():
    """Stop the metrics server."""
    global _metrics_server, _server_thread

    if _metrics_server is not None:
        _metrics_server.shutdown()
        _metrics_server = None
        _server_thread = None
