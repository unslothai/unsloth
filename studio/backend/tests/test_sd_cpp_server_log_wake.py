# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd-server polls wake on server log lines: a real loopback HTTP fake plus a piped stdout.

Upstream sd-server (examples/server/async_jobs.cpp) logs ``generate_image completed`` and then encodes
the PNG before the job flips to ``completed``; ``listening on`` is logged just before the port binds.
Neither flip has its own line, so the poll must follow a line up, not trust it.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from core.inference import sd_cpp_server as srv
from core.inference.sd_cpp_server import SdCppServer

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class _PipeProc:
    """A live child whose stdout is a real pipe the test writes server log lines into."""

    def __init__(self):
        r, w = os.pipe()
        self.stdout = io.TextIOWrapper(io.FileIO(r, "r"), encoding = "utf-8", errors = "replace")
        self._w = io.FileIO(w, "w")
        self.pid = 4242
        self.returncode = None

    def log(self, line: str) -> None:
        self._w.write((line + "\n").encode())

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        try:
            self._w.close()
        except OSError:
            pass

    def kill(self):
        self.terminate()

    def wait(self, timeout = None):
        return self.returncode


class _FakeSdServer:
    """Loopback job API: one img_gen job that flips to completed ``gap`` after its last log line."""

    def __init__(
        self,
        proc: _PipeProc,
        *,
        step_lines = 3,
        step_s = 0.07,
        gap_s = 0.03,
    ):
        self.proc = proc
        self.step_lines, self.step_s, self.gap_s = step_lines, step_s, gap_s
        self.requests: list[tuple[str, str, bytes]] = []
        self.status = "queued"
        self.flipped_at: float | None = None
        fake = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.requests.append(("POST", self.path, body))
                threading.Thread(target = fake._run_job, daemon = True).start()
                self._send(202, {"id": "job1", "status": "queued"})

            def do_GET(self):
                fake.requests.append(("GET", self.path, b""))
                if self.path == "/v1/models":
                    return self._send(200, {"data": []})
                if fake.status != "completed":
                    return self._send(200, {"id": "job1", "status": fake.status, "result": None})
                b64 = base64.b64encode(_PNG).decode()
                self._send(
                    200,
                    {
                        "id": "job1",
                        "status": "completed",
                        "result": {"images": [{"index": 0, "b64_json": b64}]},
                    },
                )

        self._handler = _H
        self.httpd: ThreadingHTTPServer | None = None

    def bind(self, port: int) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler)
        threading.Thread(target = self.httpd.serve_forever, daemon = True).start()

    def _run_job(self):
        self.status = "generating"
        for i in range(self.step_lines):
            time.sleep(self.step_s)
            self.proc.log(f"|=====| {i + 1}/{self.step_lines} - 14.2it/s")
        self.proc.log("generate_image completed in 0.25s")
        time.sleep(self.gap_s)  # PNG encode + base64 before the status flip, with no line
        self.flipped_at = time.monotonic()
        self.status = "completed"

    def close(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _attach(proc: _PipeProc, port: int) -> SdCppServer:
    s = SdCppServer("sd-server")
    s._process = proc
    s.port = port
    s._stdout_thread = threading.Thread(target = s._drain_stdout, args = (proc,), daemon = True)
    s._stdout_thread.start()
    return s


def _one_job(on_step):
    proc = _PipeProc()
    fake = _FakeSdServer(proc)
    port = _free_port()
    fake.bind(port)
    s = _attach(proc, port)
    try:
        images = s.img_gen({"prompt": "a cat", "seed": 7}, on_step = on_step)
        done = time.monotonic()
        return done - fake.flipped_at, images, fake.requests
    finally:
        s.stop()
        fake.close()


def test_job_completion_is_seen_soon_after_the_last_log_line():
    lags, steps = [], []
    for _ in range(5):
        lag, images, requests = _one_job(steps.append)
        lags.append(lag)
        assert images == [_PNG]  # same bytes as served
        posts = [r for r in requests if r[0] == "POST"]
        assert [(m, p, json.loads(b)) for m, p, b in posts] == [
            ("POST", "/sdcpp/v1/img_gen", {"prompt": "a cat", "seed": 7})
        ]
        assert all(p == "/sdcpp/v1/jobs/job1" for m, p, _ in requests if m == "GET")
        # Log-driven polling stays bounded: at most one GET per floor over the job.
        assert len(requests) < 60
    # Every server line reached the progress callback, in order, before the result.
    assert steps == [
        line
        for _ in range(5)
        for line in (
            "|=====| 1/3 - 14.2it/s",
            "|=====| 2/3 - 14.2it/s",
            "|=====| 3/3 - 14.2it/s",
            "generate_image completed in 0.25s",
        )
    ]
    median = statistics.median(lags)
    print(
        f"job flip -> img_gen return lag: median {median * 1000:.1f} ms, all {[round(x * 1000) for x in lags]}"
    )
    assert median < 0.1, lags


def test_ready_is_seen_soon_after_listening_line(monkeypatch):
    monkeypatch.setattr(SdCppServer, "_port_is_ours", lambda self: True)
    lags = []
    for _ in range(5):
        proc = _PipeProc()
        fake = _FakeSdServer(proc)
        port = _free_port()
        s = _attach(proc, port)
        bound: list[float] = []

        def _serve():
            time.sleep(0.12)  # model load
            proc.log(f"listening on: http://127.0.0.1:{port}")
            time.sleep(0.005)
            fake.bind(port)
            bound.append(time.monotonic())

        threading.Thread(target = _serve, daemon = True).start()
        try:
            assert s._wait_ready(10.0)
            lags.append(time.monotonic() - bound[0])
        finally:
            s.stop()
            fake.close()
    median = statistics.median(lags)
    print(
        f"bind -> ready lag: median {median * 1000:.1f} ms, all {[round(x * 1000) for x in lags]}"
    )
    assert median < 0.1, lags


def test_timer_still_drives_polls_without_log_lines():
    """No line at all: the fallback timer alone finishes the job (pre-change behaviour)."""
    wait = srv._LogPacedWait(threading.Event(), 0.05)
    t0 = time.monotonic()
    wait.wait()
    assert 0.04 <= time.monotonic() - t0 < 0.5


def test_follow_up_backs_off_to_the_timer():
    wake = threading.Event()
    wait = srv._LogPacedWait(wake, 0.4, floor = 0.02)
    wake.set()
    wait.wait()  # line: one floor
    follows = []
    for _ in range(6):
        follows.append(wait._follow)
        wait.wait()
    assert follows[:4] == [0.02, 0.04, 0.08, 0.16]
    assert follows[5] is None  # back on the plain interval timer
