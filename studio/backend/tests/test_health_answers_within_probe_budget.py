# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/health answers inside the desktop launcher's probe timeout,
detected hardware or not.

studio/src-tauri/src/preflight/backend.rs probes with a 2s client timeout right after
TAURI_PORT is emitted, and a timeout is not retried: it falls through to
"desktop_owned_backend_starting", a dead end the user has to clear by hand. That was safe
while the lifespan detected inline, since TAURI_PORT came after detection; detection runs on
the warm thread now and TAURI_PORT precedes it, so an unbounded wait inside health puts a
cold `import torch` in front of that deadline. Nothing the launcher reads from health
depends on detection, only chat_only does.

CPU-only, no network, no GPU, no weights: the subprocess tests stub detection.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_MAIN_SRC = _BACKEND_DIR / "main.py"
_PROBE_RS = _BACKEND_DIR.parent / "src-tauri" / "src" / "preflight" / "backend.rs"


def _run(snippet: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd = str(_BACKEND_DIR),
        capture_output = True,
        text = True,
        timeout = 900,
    )


def _main_constant(name: str) -> float:
    """Read a module-level float from main.py without importing it."""
    tree = ast.parse(_MAIN_SRC.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise AssertionError(f"main.py no longer defines {name}")


def test_the_budget_stays_under_the_desktop_probe_timeout():
    """Cross-language guard: the budget is only correct relative to the Rust one, and
    either side can be changed without the other."""
    assert _PROBE_RS.is_file(), f"{_PROBE_RS} moved; update this guard"
    rust = _PROBE_RS.read_text(encoding = "utf-8")
    probe = rust[rust.index("fn probe_ownerless_spawned_backend") :]
    # Bound to this function; a later one must not be the source of the number.
    end = probe.find("\n}\n")
    if end != -1:
        probe = probe[: end + 3]
    # The builder's .timeout() and the shared loopback_http::client() constructor both
    # take the client timeout as a whole-seconds Duration, and the probe sets exactly
    # one. Matching the Duration rather than either call site keeps this guard working
    # across that refactor while still failing if the unit stops being seconds.
    match = re.search(r"Duration::from_secs\((\d+)\)", probe)
    assert match, (
        "probe_ownerless_spawned_backend no longer sets a whole-seconds client "
        "timeout; re-derive the health budget from whatever replaced it"
    )
    probe_timeout = float(match.group(1))
    budget = _main_constant("_HEALTH_DETECT_BUDGET_S")
    assert budget < probe_timeout, (
        f"/api/health waits up to {budget}s for detection but the desktop probe "
        f"gives up at {probe_timeout}s"
    )
    # Connect, routing and JSON share the same 2s, and the budget overruns whenever a
    # C-extension import holds the GIL past it (0.24s measured at 1.5s). A budget that
    # only just fits is one slow host away from the dead end.
    assert probe_timeout - budget >= 0.9, (
        f"only {probe_timeout - budget}s of headroom between the health budget "
        f"and the {probe_timeout}s probe timeout"
    )


_SNIPPET = r"""
import asyncio, json, os, sys, threading, time

# UNSLOTH_STUDIO_DISABLE_TORCH_WARM is deliberately NOT set here. It used to be,
# to keep a real warm out of the way, but nothing below runs the lifespan -- the
# app is built by hand with one route -- so no warm ever starts anyway. Setting
# it now would suppress the very kick these tests measure: health does not start
# detection when the switch is on, which is what makes the switch mean anything.
os.environ.pop("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", None)

import main
from fastapi import FastAPI
from fastapi.testclient import TestClient

hw = main._hw_module
DETECT_SECONDS = %(detect_seconds)s

hw.DEVICE = None
hw.CHAT_ONLY = True
hw.CHAT_ONLY_REASON = None
# This stub stands in for ensure_hardware_detected, so it owns the same
# completion protocol. DEVICE alone no longer means "finished": the branches
# assign it partway through and keep probing, so the health wait polls
# DETECTION_COMPLETE instead. Start it clear to model "detection has not run".
hw.DETECTION_COMPLETE.clear()

def slow_detect(*_):  # start_background_detection binds the spawn-time epoch
    # Stands in for the torch import the warm thread is running.
    time.sleep(DETECT_SECONDS)
    hw.DEVICE = hw.DeviceType.CUDA
    hw.CHAT_ONLY = False
    hw.CHAT_ONLY_REASON = None
    hw.DETECTION_COMPLETE.set()
    return hw.DEVICE

hw.ensure_hardware_detected = slow_detect

app = FastAPI()
app.add_api_route("/api/health", main.health_check, methods = ["GET"])
client = TestClient(app)

started = time.perf_counter()
response = client.get("/api/health")
elapsed = time.perf_counter() - started
body = response.json()

# Same request with a bearer, which is the one config/env.ts caches on.
# health_check imports get_current_subject inside the function, so patching the
# module attribute is enough and keeps this test off the real JWT/storage path.
import auth.authentication as _authmod

async def _subject(_creds):
    return "tester"

_authmod.get_current_subject = _subject
authed = client.get("/api/health", headers = {"Authorization": "Bearer probe"}).json()

print("RESULT" + json.dumps({
    "status": response.status_code,
    "elapsed": elapsed,
    "chat_only": body.get("chat_only"),
    "hardware_detecting": body.get("hardware_detecting"),
    "authed_has_device_type": "device_type" in authed,
    "authed_has_chat_only_reason": "chat_only_reason" in authed,
    "authed_hardware_detecting": authed.get("hardware_detecting"),
    "authed_has_version": "version" in authed,
}))
"""


def _probe(detect_seconds: float) -> dict:
    proc = _run(_SNIPPET % {"detect_seconds": detect_seconds})
    assert (
        proc.returncode == 0
    ), f"probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT") :])


# A pass that sets CHAT_ONLY False and then degrades, the shape every accelerator
# branch has: the flag is assigned before the device-name probe that can raise, and
# detect_hardware() restores the previous verdict on the way out.
_MID_PASS_SNIPPET = r"""
import json, os, threading, time

os.environ.pop("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", None)

import main
from fastapi import FastAPI
from fastapi.testclient import TestClient

hw = main._hw_module
hw.DEVICE = None
hw.CHAT_ONLY = True
hw.CHAT_ONLY_REASON = None
hw.DETECTION_COMPLETE.clear()

def degrading_detect(*_):
    # The XPU branch verbatim: DEVICE and CHAT_ONLY assigned, then a probe that hangs
    # and would raise. DETECTION_COMPLETE is never set, so the verdict is not settled.
    hw.DEVICE = hw.DeviceType.XPU
    hw.CHAT_ONLY = False
    time.sleep(30.0)
    return hw.DEVICE

hw.ensure_hardware_detected = degrading_detect

app = FastAPI()
app.add_api_route("/api/health", main.health_check, methods = ["GET"])
client = TestClient(app)

# start_background_detection() runs the stub, which flips the global well inside the
# budget; the reply lands while the pass is still in flight.
body = client.get("/api/health").json()

print("RESULT" + json.dumps({
    "chat_only": body.get("chat_only"),
    "hardware_detecting": body.get("hardware_detecting"),
    "global_chat_only": hw.CHAT_ONLY,
    "complete": hw.DETECTION_COMPLETE.is_set(),
}))
"""


def test_an_unsettled_pass_is_never_published_as_training_capable():
    """chat_only with no snapshot must be the literal True, not the live global.

    Every accelerator branch of _detect_hardware_locked assigns CHAT_ONLY = False and then
    keeps probing (torch.xpu.get_device_name, the MLX stack check), and a raise there
    degrades the host to CPU. DETECTION_COMPLETE stays clear throughout, so the reply is
    marked provisional; with the kill switch on it is also marked deferred, and
    hardware-verdict.ts stores data.chat_only verbatim for those. Reading the global
    therefore flashes Train and Export on a host that ends up chat-only."""
    proc = _run(_MID_PASS_SNIPPET)
    assert (
        proc.returncode == 0
    ), f"probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    result = json.loads(line[len("RESULT") :])

    assert not result["complete"], "the stub settled detection; the window is not modelled"
    assert result["global_chat_only"] is False, "the mid-pass flip did not happen"
    assert result["hardware_detecting"] is True, "expected a provisional reply"
    assert result["chat_only"] is True, (
        "health published chat_only=False from a detection pass still in flight; the "
        "frontend enables Train and Export until the real verdict lands"
    )


def test_health_answers_within_the_budget_while_detection_is_slow():
    """The regression: detection outlasts the probe, health must not. Against an unbounded
    `await asyncio.to_thread(ensure_hardware_detected)` this returns in ~10s and the desktop
    launch dead-ends."""
    budget = _main_constant("_HEALTH_DETECT_BUDGET_S")
    result = _probe(10.0)

    assert result["status"] == 200
    assert result["elapsed"] < budget + 0.5, (
        f"/api/health took {result['elapsed']:.2f}s with detection still "
        f"running; the budget is {budget}s and the desktop probe gives up at 2s"
    )
    assert result["hardware_detecting"] is True, (
        "health returned a provisional chat_only without saying so; a client "
        "cannot tell it apart from a measured one"
    )
    # Conservative direction: never offer Train/Export on a host that may not have it.
    assert result["chat_only"] is True


def test_health_still_waits_when_detection_finishes_inside_the_budget():
    """The budget is a ceiling, not a floor. The wait exists so a GPU host is not
    published as chat-only for a second; bounding it must not make every early health
    call provisional."""
    result = _probe(0.3)

    assert result["status"] == 200
    assert (
        result["chat_only"] is False
    ), "health published chat_only=True on a host whose detection completed well inside the budget"
    assert "hardware_detecting" not in result or result["hardware_detecting"] is None


def test_health_does_not_await_detection_unbounded():
    """Static guard: health_check must go through the bounded helper. `await
    asyncio.to_thread(ensure_hardware_detected)` reads like the obvious fix and passes every
    test that pins DEVICE first; it only fails on a cold desktop launch."""
    tree = ast.parse(_MAIN_SRC.read_text(encoding = "utf-8"))
    health = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "health_check"
    )
    called = {
        node.func.id
        for node in ast.walk(health)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_await_hardware_detection" in called, (
        "health_check no longer waits for detection through "
        "_await_hardware_detection; whatever replaced it must still answer "
        "inside the desktop probe timeout"
    )
    assert "ensure_hardware_detected" not in called, (
        "health_check calls ensure_hardware_detected directly again; that wait "
        "is unbounded and the desktop probe gives up at 2s"
    )


def test_a_provisional_reply_is_not_cacheable_by_the_frontend():
    """The provisional reply must not look authoritative to config/env.ts.

    fetchDeviceType() sets ``fetched = data.device_type !== undefined`` and every later
    non-forced call short-circuits on ``fetched``, so a provisional authed reply carrying
    device_type pins chat_only=true for the rest of the SPA session on a GPU host: Train
    hidden, /studio redirected to /chat. The sidebar's recovery poll runs only for
    chat_only_reason === "mlx_unavailable", so it does not save it either."""
    result = _probe(10.0)

    assert result["authed_hardware_detecting"] is True, "expected a provisional reply"
    assert not result["authed_has_device_type"], (
        "the provisional authed reply carries device_type, so the frontend caches "
        "it as authoritative and never re-reads the measured chat_only"
    )
    assert not result[
        "authed_has_chat_only_reason"
    ], "chat_only_reason is meaningless before detection has run"
    # The launcher-facing fields are unaffected.
    assert result["authed_has_version"]


def test_a_measured_reply_still_carries_the_authoritative_fields():
    """The omission is scoped to the provisional case."""
    result = _probe(0.3)

    assert not result["hardware_detecting"]
    assert result["authed_has_device_type"]
    assert result["authed_has_chat_only_reason"]


def test_a_mid_detection_assignment_is_not_treated_as_finished():
    """DEVICE goes non-None before detection has settled. The XPU branch assigns DEVICE and
    CHAT_ONLY=False, then calls torch.xpu.get_device_name(0); if that raises, the host
    degrades to CPU/chat-only. A waiter keyed on "DEVICE is not None" would have published
    the intermediate value, reporting training available on a chat-only host."""
    import importlib
    import threading

    hw = importlib.import_module("utils.hardware.hardware")

    assert hasattr(
        hw, "DETECTION_COMPLETE"
    ), "detection needs a completion signal distinct from the DEVICE assignment"
    assert isinstance(hw.DETECTION_COMPLETE, threading.Event)

    src = _MAIN_SRC.read_text(encoding = "utf-8")
    start = src.index("async def _await_hardware_detection")
    body = src[start : start + 1800]
    assert (
        "DETECTION_COMPLETE" in body
    ), "the health wait must poll the completion signal, not the DEVICE assignment"
    assert (
        "while _hw_module.DEVICE is None" not in body
    ), "the health wait is keyed on the mid-detection assignment again"
