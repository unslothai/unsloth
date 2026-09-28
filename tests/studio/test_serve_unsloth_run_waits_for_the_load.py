# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""serve-unsloth-run.sh waits for the model load, not a fixed 30 seconds.

`unsloth run` answers /api/health before it loads the model and prints its banner, API key
included, only once the load is done. The script used to give the banner 30 one-second polls
after health. That is about 10s of work normally, but a 4.8 GB GGUF on a busy CPU runner took
longer, and the `connection (opencode, stable)` leg failed with "could not parse an API key
from the banner" while the model was still loading.

The script runs here against a stand-in `unsloth` on PATH, and a stand-in `sleep` that returns
at once and counts its calls, so the script's seconds are counted polls. The stand-in prints
the banner 40 polls after it first answers health, past the old window, in well under a second
of real time.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "serve-unsloth-run.sh"
KEY = "sk-unsloth-standin0123456789"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("setsid") is None or shutil.which("jq") is None,
    reason = "the script needs bash, setsid and jq, as on the Linux runner",
)

# `unsloth run -H HOST -p PORT ...`: healthy at once, banner BANNER_AFTER sleeps after the first
# health answer, or exit that many sleeps after it without a banner when BANNER_AFTER is negative.
STANDIN_UNSLOTH = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import json, os, sys, threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    port = int(sys.argv[sys.argv.index("-p") + 1])
    after = int(os.environ["BANNER_AFTER"])
    counter = os.environ["SLEEP_COUNTER"]

    healthy_at = []  # the poll count when /api/health first answered

    def polls():
        try:
            return len(open(counter).read())
        except FileNotFoundError:
            return 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = {{"status": "healthy"}}
            if self.path.startswith("/api/health") and not healthy_at:
                healthy_at.append(polls())
            if self.path.startswith("/v1/models"):
                if self.headers.get("Authorization") != "Bearer {KEY}":
                    self.send_response(401); self.end_headers(); return
                body = {{"data": [{{"id": "standin-model"}}]}}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    # A slow start: health is not answered until the script has polled it this many times.
    while polls() < int(os.environ["START_AFTER"]):
        time.sleep(0.005)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()

    # Counted from health, so a slow start cannot spend the banner's polls before the wait begins.
    while not healthy_at:
        time.sleep(0.005)
    with open(os.environ["HEALTHY_AT"], "w") as out:
        out.write(str(healthy_at[0]))
    if after < 0:
        while polls() < healthy_at[0] - after:
            time.sleep(0.005)
        sys.exit(1)
    while polls() < healthy_at[0] + after:
        time.sleep(0.005)
    print("  API Key:      {KEY}", flush = True)
    time.sleep(600)
    """
)

# Counts a call and returns: the script's `sleep 1` becomes one poll.
STANDIN_SLEEP = '#!/bin/sh\nprintf x >> "$SLEEP_COUNTER"\nexec /bin/sleep 0.01\n'


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(
    tmp_path: Path,
    banner_after: int,
    start_after: int = 0,
) -> tuple[subprocess.CompletedProcess, int]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("unsloth", STANDIN_UNSLOTH), ("sleep", STANDIN_SLEEP)):
        path = bin_dir / name
        path.write_text(body, encoding = "utf-8")
        path.chmod(0o755)
    counter = tmp_path / "sleeps"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
    env.update(
        PATH = f"{bin_dir}{os.pathsep}{env['PATH']}",
        BANNER_AFTER = str(banner_after),
        START_AFTER = str(start_after),
        SLEEP_COUNTER = str(counter),
        HEALTHY_AT = str(tmp_path / "healthy_at"),
        STUDIO_HOME = str(tmp_path / "studio"),
    )
    port = _free_port()
    try:
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--gguf-file",
                "/standin/model.gguf",
                "--port",
                str(port),
                "--log-dir",
                str(tmp_path / "logs"),
            ],
            env = env,
            capture_output = True,
            text = True,
            timeout = 60,
        )
    finally:
        # The script leaves the server running for the steps after it, as CI wants.
        subprocess.run(["pkill", "-f", f"{bin_dir}/unsloth run "], capture_output = True)
    polls = len(counter.read_text()) if counter.exists() else 0
    healthy_at = tmp_path / "healthy_at"
    # Polls spent waiting for the banner, not starting the server.
    if healthy_at.exists():
        polls -= int(healthy_at.read_text())
    return result, polls


def test_a_banner_after_the_old_30_second_window_is_still_read(tmp_path):
    # 15 polls to come up healthy, then 40 more to load: the load alone outlasts the old window.
    result, polls = _serve(tmp_path, banner_after = 40, start_after = 15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"UNSLOTH_API_KEY={KEY}" in result.stdout
    assert "UNSLOTH_MODEL_ID=standin-model" in result.stdout
    assert polls >= 40, f"the banner came after {polls} polls, so the stand-in did not delay it"


def test_a_server_that_dies_while_loading_fails_at_once_and_says_so(tmp_path):
    result, polls = _serve(tmp_path, banner_after = -5)
    assert result.returncode == 1
    assert "process exited before printing its banner" in result.stderr
    # Not the whole banner budget: the script noticed the exit within a few polls.
    assert polls < 30, f"the script kept polling a dead server for {polls} polls"
