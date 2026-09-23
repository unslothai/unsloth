# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fake lemond 11.9.0 with authenticated requests, model state and download progress.

Accepts LemonadeServer's arguments and logs requests to cache/requests.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CATALOG = [
    {
        "id": "qwen3-0.6b-FLM",
        "checkpoint": "qwen3:0.6b",
        "labels": ["reasoning", "chat"],
        "size": 0.66,
        "max_context_window": 40960,
    },
    {
        "id": "qwen3-it-4b-FLM",
        "checkpoint": "qwen3-it:4b",
        "labels": ["tool-calling", "chat"],
        "size": 3.1,
        "max_context_window": 40960,
    },
    {
        "id": "gemma3-4b-FLM",
        "checkpoint": "gemma3:4b",
        "labels": ["vision", "chat"],
        "size": 4.5,
        "max_context_window": 131072,
    },
    {
        "id": "embed-gemma-300m-FLM",
        "checkpoint": "embed-gemma:300m",
        "labels": ["embeddings"],
        "size": 0.62,
    },
    {
        "id": "Qwen3-0.6B-GGUF",
        "checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0",
        "labels": ["chat"],
        "recipe": "llamacpp",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir")
    parser.add_argument("config_dir")
    parser.add_argument("--port", type = int, required = True)
    parser.add_argument("--host", default = "127.0.0.1")
    parser.add_argument("--no-broadcast", action = "store_true")
    parser.add_argument("--log-file", default = "auto")
    args = parser.parse_args()

    if os.environ.get("FAKE_LEMOND_IGNORE_SIGTERM") == "1":
        # A lemond that never acts on SIGTERM, so only a SIGKILL death signal ends it.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    key = os.environ.get("LEMONADE_API_KEY", "")
    cache = Path(args.cache_dir)
    record = cache / "requests.jsonl"
    downloaded = set(json.loads(os.environ.get("FAKE_LEMOND_DOWNLOADED", "[]")))
    loaded: dict[str, dict] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:
            return

        def _send(self, status: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if self.path == "/live":
                return True
            if self.headers.get("Authorization") != f"Bearer {key}":
                self._send(401, {"error": "Invalid or missing API key"})
                return False
            return True

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            with open(record, "a", encoding = "utf-8") as handle:
                handle.write(json.dumps({"path": self.path, "body": body}) + "\n")
            return body

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            if self.path == "/live":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "all_models_loaded": [
                            {
                                "model_name": name,
                                "device": "npu",
                                "loaded": True,
                                "recipe": "flm",
                                "recipe_options": options,
                            }
                            for name, options in loaded.items()
                        ],
                    },
                )
            elif self.path.startswith("/v1/models"):
                self._send(
                    200,
                    {
                        "data": [
                            {"recipe": "flm", **row, "downloaded": row["id"] in downloaded}
                            for row in CATALOG
                        ]
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            body = self._body()
            name = body.get("model_name")
            if self.path == "/v1/install":
                flm = cache / "bin" / "flm" / "npu" / "flm"
                flm.parent.mkdir(parents = True, exist_ok = True)
                report = os.environ.get("FAKE_FLM_VALIDATE", '{"ready": true}')
                flm.write_text(f"#!/bin/sh\necho '{report}'\n", encoding = "utf-8")
                flm.chmod(0o755)
                self._send(200, {"status": "success", "recipe": "flm", "backend": "npu"})
            elif self.path == "/v1/pull":
                downloaded.add(name)
                events = (
                    'data: {"file":"model.q4nx","file_index":2,"total_files":4,"percent":40,"bytes_downloaded":4,"bytes_total":10}\n\n'
                    "event: complete\n"
                    'data: {"file":"","file_index":4,"total_files":4,"percent":100}\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(events)))
                self.end_headers()
                self.wfile.write(events)
            elif self.path == "/v1/load":
                # A load FastFlowLM takes this long to answer, to exercise cancelling one.
                time.sleep(float(os.environ.get("FAKE_LEMOND_LOAD_SECONDS", "0")))
                if name not in downloaded:
                    self._send(404, {"error": {"message": f"Model '{name}' was not found."}})
                    return
                loaded.clear()
                loaded[name] = {"ctx_size": body.get("ctx_size"), "flm_args": ""}
                self._send(200, {"status": "success", "model_name": name, "recipe": "flm"})
            elif self.path == "/v1/unload":
                if os.environ.get("FAKE_LEMOND_UNLOAD_FAILS") == "1":
                    self._send(500, {"error": {"message": "unload failed"}})
                    return
                if name:
                    loaded.pop(name, None)
                else:
                    loaded.clear()
                self._send(200, {"status": "success"})
            elif self.path == "/v1/delete":
                downloaded.discard(name)
                self._send(200, {"status": "success", "message": f"Deleted model: {name}"})
            else:
                self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fake lemond listening on {args.port}", flush = True)
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
