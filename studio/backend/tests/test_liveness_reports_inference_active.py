# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/liveness says whether the backend is generating, and stays cheap.

The desktop health watchdog probes this route every 15s with a 10s budget and declares the
backend dead after 3 consecutive misses. Startup is not the only window where a healthy
backend misses three in a row: a host serving a model far larger than it can hold runs at
fractions of a token per second, and the loop feeding those streams goes quiet the same way
the warm thread's `import torch` makes it go quiet. Killing there ends a response the user
is still waiting on, and the window reports it as "Server stopped unexpectedly" (#8945).

So liveness carries an `inference_active` marker and the watchdog widens its failure budget
while the last answered probe was generating. Only for probes that time out: a refused
connection means the port is gone, and that is still reported at three strikes.

The marker must not cost what health costs: it is a len() under a lock already held for
microseconds, never a wait on the generations themselves.

CPU-only, no network, no GPU, no weights.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_COMMANDS_RS = _BACKEND_DIR.parent / "src-tauri" / "src" / "commands.rs"


_SNIPPET = r"""
import json, os, threading, time

# Keep the real warm out of the way: it would import the ML stack, and this file is about
# the busy marker, not the warm one.
os.environ["UNSLOTH_STUDIO_DISABLE_TORCH_WARM"] = "1"

import main
from fastapi import FastAPI
from fastapi.testclient import TestClient
from state import active_generations

hw = main._hw_module
hw.DEVICE = hw.DeviceType.CPU
hw.CHAT_ONLY = True
hw.CHAT_ONLY_REASON = "no_accelerator"
hw.DETECTION_COMPLETE.set()

def must_not_run(*_):
    raise AssertionError("liveness started hardware detection")

hw.ensure_hardware_detected = must_not_run

app = FastAPI()
app.add_api_route("/api/liveness", main.liveness_check, methods = ["GET"])
client = TestClient(app)

def probe():
    started = time.perf_counter()
    response = client.get("/api/liveness")
    elapsed = time.perf_counter() - started
    body = response.json()
    return {
        "status_code": response.status_code,
        "status": body.get("status"),
        "service": body.get("service"),
        "elapsed": elapsed,
        "inference_active": body.get("inference_active"),
        "has_busy_key": "inference_active" in body,
    }

idle_before = probe()
with active_generations.ActiveGeneration(threading.Event(), thread_id = "t1", kind = "messages"):
    busy = probe()
idle_after = probe()

print("RESULT" + json.dumps({
    "idle_before": idle_before,
    "busy": busy,
    "idle_after": idle_after,
}))
"""


def _probe() -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd = str(_BACKEND_DIR),
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert (
        proc.returncode == 0
    ), f"probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


def test_liveness_reports_a_generation_in_flight():
    """The regression: nothing in the reply distinguished a backend stalled under four
    concurrent generations from one that had exited, so the watchdog killed both."""
    result = _probe()

    assert result["busy"]["status_code"] == 200
    assert result["busy"]["status"] == "alive"
    assert result["busy"]["service"] == "Unsloth UI Backend"
    assert result["busy"]["inference_active"] is True, (
        "liveness does not say the backend is generating; the watchdog cannot tell a busy "
        "backend from a dead one and kills the stream at three missed probes"
    )


def test_the_marker_disappears_once_nothing_is_generating():
    """The wide budget is for a backend that is producing tokens. Leaving the marker lit
    after the last stream ends would arm it for the rest of the session, and a genuinely
    hung backend would sit unreported."""
    result = _probe()

    assert not result["idle_before"][
        "has_busy_key"
    ], "liveness reports inference_active before anything was registered"
    assert not result["idle_after"]["has_busy_key"], (
        "liveness still reports inference_active after the generation finished; the "
        "watchdog would keep the widened budget armed against a hung backend"
    )


def test_the_marker_costs_nothing_to_read():
    """A probe every 15s cannot pay for anything that waits, which is why the route reads
    a registry len() rather than asking the backend what it is doing."""
    result = _probe()

    for state, sample in result.items():
        assert sample["elapsed"] < 0.5, (
            f"/api/liveness took {sample['elapsed']:.2f}s while {state}; it must read the "
            f"registry rather than wait on the generations in it"
        )


def test_the_desktop_watchdog_still_reads_the_marker():
    """Cross-language guard: the marker only does anything because commands.rs reads it,
    and either side can be changed without the other."""
    assert _COMMANDS_RS.is_file(), f"{_COMMANDS_RS} moved; update this guard"
    rust = _COMMANDS_RS.read_text(encoding = "utf-8")
    probe = rust[rust.index("async fn check_health_inner") :]
    end = probe.find("\n}\n")
    if end != -1:
        probe = probe[: end + 3]

    assert '"inference_active"' in probe, (
        "the watchdog probe no longer reads inference_active, so a backend stalled mid "
        "generation is killed at three missed probes again"
    )
    assert "HEALTH_WATCHDOG_MAX_FAILURES_BUSY" in rust, (
        "commands.rs no longer defines a widened failure budget; reading the marker "
        "without acting on it changes nothing"
    )
    assert "fn watchdog_failure_budget" in rust, (
        "the budget is no longer chosen in one place; the busy case and the dead-port "
        "case have to stay distinguishable"
    )
