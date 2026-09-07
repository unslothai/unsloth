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

Media jobs do not enter `active_generations`, so the marker also checks video and both
image engines.

The marker must not cost what health costs: it is a len() under a lock already held for
microseconds plus a bool off each resident media backend, never a wait on the work itself
and never an import of the ML stack.

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
import json, os, sys, threading, time, types


def _raise():
    raise RuntimeError("backend module is mid-teardown")

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

# Keep this independent of main._MEDIA_BACKEND_MODULES so omissions are detected.
media_modules = (
    "core.inference.video",
    "core.inference.diffusion",
    "core.inference.sd_cpp_backend",
    "routes.inference",
)
# routes.inference is imported at startup regardless, so the ML-stack question is only
# about the three engines.
media_import_free = [
    name for name in media_modules if name.startswith("core.") and name in sys.modules
]
scanned = list(getattr(main, "_MEDIA_BACKEND_MODULES", ()) or ())

probes = {"idle_before": idle_before, "busy": busy, "idle_after": idle_after}
for name in media_modules:
    short = name.rsplit(".", 1)[-1]
    module = types.ModuleType(name)
    module.generation_in_flight = lambda: False
    sys.modules[name] = module
    probes[short + "_idle"] = probe()
    module.generation_in_flight = lambda: True
    probes[short + "_rendering"] = probe()
    module.generation_in_flight = lambda: False
    probes[short + "_done"] = probe()

# Backend probe failures must not fail liveness.
sys.modules["core.inference.video"].generation_in_flight = _raise
probes["broken"] = probe()

print("RESULT" + json.dumps({
    "probes": probes,
    "media_import_free": media_import_free,
    "media_modules": list(media_modules),
    "scanned": scanned,
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
    result = _probe()["probes"]

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
    result = _probe()["probes"]

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
    result = _probe()["probes"]

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


def test_liveness_reports_a_media_job_in_flight():
    """Media jobs publish the same busy marker as chat generation."""
    result = _probe()["probes"]

    for backend in ("video", "diffusion", "sd_cpp_backend"):
        assert result[f"{backend}_rendering"]["inference_active"] is True, (
            f"liveness does not say the backend is busy while {backend} renders; the "
            f"watchdog kills the job at three missed probes and reports the app as crashed"
        )
        assert not result[f"{backend}_idle"]["has_busy_key"], f"{backend} reported busy while idle"
        assert not result[f"{backend}_done"]["has_busy_key"], (
            f"{backend} still reports busy after the job ended; the widened budget would "
            f"stay armed for the rest of the session"
        )


def test_the_probe_does_not_import_the_media_backends():
    """The liveness check must not import media backends."""
    result = _probe()

    assert result["media_import_free"] == [], (
        f"/api/liveness imported {result['media_import_free']} to answer; the marker must "
        f"read backends that already exist and say 'not busy' for the rest"
    )


def test_a_broken_media_backend_still_answers_the_probe():
    """A backend probe failure must not fail liveness."""
    result = _probe()["probes"]

    assert result["broken"]["status_code"] == 200
    assert result["broken"]["status"] == "alive"


def test_every_media_backend_is_scanned():
    """Every supported media backend must be scanned."""
    result = _probe()

    assert result["scanned"] == result["media_modules"], (
        f"/api/liveness scans {result['scanned']}, this file exercises "
        f"{result['media_modules']}; a media backend in neither list renders invisibly"
    )


def test_liveness_covers_the_image_persist_tail():
    """An image job is not over when the engine's marker clears: the route is still writing
    the gallery records the response is built from, and on a saturated host that write is
    exactly when probes start missing. generate-progress already calls that window active
    (routes/inference.py diffusion_generate_progress); liveness disagreeing with it is how a
    request that is still running gets the idle three-strike budget."""
    import importlib

    routes_inference = importlib.import_module("routes.inference")
    assert routes_inference.generation_in_flight() is False

    routes_inference._diffusion_persist_active += 1
    try:
        assert routes_inference.generation_in_flight() is True
    finally:
        routes_inference._diffusion_persist_active -= 1
    assert routes_inference.generation_in_flight() is False


def test_both_image_routes_publish_the_persist_marker():
    """The Unsloth route and the OpenAI-compatible one write the gallery on separate paths.
    Only the first used to count, so an OpenAI client's persist was invisible to both
    generate-progress and liveness."""
    import inspect
    import importlib

    source = inspect.getsource(importlib.import_module("routes.inference"))
    assert source.count("_diffusion_persist_active += 1") == 2, (
        "an image route persists gallery records without publishing the marker, so liveness "
        "reports idle while that request is still in flight"
    )
    assert source.count("_diffusion_persist_active -= 1") == 2
