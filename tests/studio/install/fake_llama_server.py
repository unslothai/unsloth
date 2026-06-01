#!/usr/bin/env python3
"""A fake llama-server for GPU-offload validation tests (no GPU required).

It accepts the arguments install_llama_prebuilt.py's validate_server passes
(``-m`` / ``--host`` / ``--port`` / ``--n-gpu-layers`` / ...), prints a canned
llama.cpp startup log chosen by the ``FAKE_LLAMA_MODE`` env var, then serves
HTTP 200 from ``/completion`` until killed. This lets CI exercise the real
validate_server subprocess + HTTP + log-classifier path on GPU-less Windows /
macOS / Linux runners: the same binary "starts and serves 200" while its log
says CPU-only or GPU, which is exactly the #5807 / #5830 situation.

FAKE_LLAMA_MODE (default "cuda"):
  cuda           device_info enumerates CUDA0 (GPU offload confirmed)
  cuda_buffer    older "CUDA0 model buffer size" + offloaded 33/33 lines
  cpu            device_info enumerates only CPU (the silent CPU fallback)
  offloaded_zero "offloaded 0/33 layers to GPU" (definite CPU-only)
  no_signal      a log with no offload evidence (validator must not reject)
"""

import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


LOGS = {
    "cuda": (
        "0.00 I device_info:\n"
        "0.01 I   - CUDA0 : NVIDIA GeForce RTX 5070 (12282 MiB, 11000 MiB free)\n"
        "0.01 I   - CPU : Generic CPU (32000 MiB free)\n"
        "0.01 I system_info: n_threads = 8 | CUDA : ARCHS = 1200 | CPU : AVX2 = 1\n"
        "0.02 I srv llama_server: model loaded\n"
    ),
    "cuda_buffer": (
        "load_tensors: offloaded 33/33 layers to GPU\n"
        "load_tensors:        CUDA0 model buffer size = 21000.0 MiB\n"
        "load_tensors:   CPU_Mapped model buffer size =     0.6 MiB\n"
        "srv llama_server: model loaded\n"
    ),
    "cpu": (
        "0.00 I device_info:\n"
        "0.00 I   - CPU : Generic CPU (32000 MiB free)\n"
        "0.00 I system_info: n_threads = 8 | CPU : AVX2 = 1\n"
        "0.01 I srv llama_server: model loaded\n"
    ),
    "offloaded_zero": (
        "load_tensors: offloaded 0/33 layers to GPU\n"
        "load_tensors:   CPU_Mapped model buffer size = 21000.0 MiB\n"
        "srv llama_server: model loaded\n"
    ),
    "no_signal": (
        "INFO [main] starting server\n"
        "load_tensors: file format = GGUF V3\n"
        "srv llama_server: model loaded\n"
    ),
}


def _arg(name, default = None):
    argv = sys.argv
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return default


class _Handler(BaseHTTPRequestHandler):
    def _ok(self):
        body = b'{"content": "x", "tokens_predicted": 1}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self._ok()

    def do_GET(self):
        self._ok()

    def log_message(self, *args):
        pass  # keep stdout clean for the classifier


def main() -> int:
    mode = os.environ.get("FAKE_LLAMA_MODE", "cuda")
    sys.stdout.write(LOGS.get(mode, LOGS["cuda"]))
    sys.stdout.flush()

    host = _arg("--host", "127.0.0.1")
    port = int(_arg("--port", "8080"))
    server = HTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
