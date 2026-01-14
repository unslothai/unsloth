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
import socket
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
        try:
            if self.path == "/metrics":
                try:
                    metrics_output = generate_prometheus_metrics()
                    self.send_response(200)
                    self.send_header("Content-Type", get_metrics_content_type())
                    self.send_header("Content-Length", str(len(metrics_output)))
                    self.end_headers()
                    self.wfile.write(metrics_output)
                    self.wfile.flush()
                except Exception as e:
                    error_msg = f"Error generating metrics: {str(e)}\n"
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(error_msg)))
                    self.end_headers()
                    self.wfile.write(error_msg.encode())
                    self.wfile.flush()
            elif self.path == "/" or self.path == "":
                # Simple health check endpoint
                response = (
                    b"Unsloth Metrics Server\n/metrics - Prometheus metrics endpoint\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                self.wfile.flush()
            else:
                response = b"Not Found\n"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                self.wfile.flush()
        except Exception as e:
            # Handle any errors in request processing
            try:
                error_msg = f"Internal server error: {str(e)}\n"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(error_msg)))
                self.end_headers()
                self.wfile.write(error_msg.encode())
                self.wfile.flush()
            except:
                pass  # Connection might be closed

    def log_message(self, format, *args):
        """Suppress default logging."""


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
        print(f"📊 Metrics server already running at http://{host}:{port}/metrics")
        return _server_thread

    # Enable Prometheus metrics
    enable_prometheus_metrics()

    def run_server():
        global _metrics_server
        try:
            _metrics_server = HTTPServer((host, port), MetricsHandler)
            # Set socket options to allow reuse
            _metrics_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Give the server a moment to bind
            import time

            time.sleep(0.1)
            _metrics_server.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e) or "already in use" in str(e).lower():
                print(
                    f"⚠️  Port {port} is already in use. Please use a different port or stop the other service."
                )
            else:
                print(f"⚠️  Failed to start metrics server: {e}")
            _metrics_server = None
        except Exception as e:
            print(f"⚠️  Error starting metrics server: {e}")
            import traceback

            traceback.print_exc()
            _metrics_server = None

    # Use daemon=False in some cases to keep server alive
    # But daemon=True is better for cleanup when main program exits
    _server_thread = threading.Thread(
        target = run_server, daemon = True, name = "UnslothMetricsServer"
    )
    _server_thread.start()

    # Give the thread a moment to start and bind
    import time

    max_wait = 2.0
    waited = 0.0
    while _metrics_server is None and waited < max_wait:
        time.sleep(0.1)
        waited += 0.1

    # Check if server started successfully
    if _metrics_server is not None:
        # Verify the server is actually listening
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                print(
                    f"📊 Unsloth metrics server started at http://localhost:{port}/metrics"
                )
                print(f"   (Also accessible at http://{host}:{port}/metrics)")
            else:
                print(f"⚠️  Server thread started but port {port} is not accessible")
                print(f"   Waiting a bit longer for server to bind...")
                time.sleep(0.5)
                # Try one more time
                sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result2 = sock2.connect_ex(("localhost", port))
                sock2.close()
                if result2 == 0:
                    print(
                        f"📊 Server is now accessible at http://localhost:{port}/metrics"
                    )
                else:
                    print(
                        f"⚠️  Server still not accessible. Try restarting or use a different port."
                    )
        except Exception as e:
            print(f"⚠️  Could not verify server: {e}")
    else:
        print(f"⚠️  Failed to start metrics server. Check if port {port} is available.")
        print(f"   Try: start_metrics_server(port=9091) to use a different port")

    return _server_thread


def stop_metrics_server():
    """Stop the metrics server."""
    global _metrics_server, _server_thread

    if _metrics_server is not None:
        _metrics_server.shutdown()
        _metrics_server = None
        _server_thread = None
        print("📊 Metrics server stopped")


def is_metrics_server_running() -> bool:
    """Check if the metrics server is currently running."""
    global _metrics_server
    return _metrics_server is not None


def test_metrics_server(port: int = 9090):
    """Test if the metrics server is accessible."""
    import urllib.request
    import urllib.error

    # First check if server object exists
    if not is_metrics_server_running():
        print(f"❌ Metrics server is not running")
        print(f"   Call start_metrics_server() first")
        return False

    # Check if port is listening
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result != 0:
            print(f"❌ Port {port} is not listening")
            print(f"   Server object exists but port is not accessible")
            return False
    except Exception as e:
        print(f"❌ Could not check port {port}: {e}")
        sock.close()
        return False

    # Try HTTP request
    try:
        url = f"http://localhost:{port}/metrics"
        response = urllib.request.urlopen(url, timeout = 2)
        print(f"✅ Metrics server is running and accessible at {url}")
        print(f"   Status: {response.getcode()}")
        print(f"   Content-Length: {response.headers.get('Content-Length', 'unknown')}")
        return True
    except urllib.error.URLError as e:
        print(
            f"❌ Could not connect to metrics server at http://localhost:{port}/metrics"
        )
        print(f"   Error: {e}")
        print(f"   Error code: {getattr(e, 'code', 'unknown')}")
        print(f"   Error reason: {getattr(e, 'reason', 'unknown')}")
        return False
    except Exception as e:
        print(f"❌ Error testing metrics server: {e}")
        return False
