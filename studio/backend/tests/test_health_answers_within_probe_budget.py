# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/health answers inside the desktop launcher's probe timeout,
detected hardware or not.

studio/src-tauri/src/preflight/backend.rs builds its probe client with a 2s
timeout and calls backend_health() from probe_ownerless_spawned_backend(), which
runs immediately after TAURI_PORT is emitted. A timeout is not a retry:
probe_ownerless_spawned_backend() returns Missing,
choose_ownerless_spawned_preflight() falls through to
ExternalConflict/"desktop_owned_backend_starting", and use-tauri-backend.ts maps
that to setBackendError("The desktop-owned Unsloth backend is still starting.
Wait a moment, then try again.") -- a dead end the user has to retry by hand.

That was safe while the lifespan detected hardware inline: TAURI_PORT came after
detection, so health answered instantly. Detection runs on the warm thread now
and TAURI_PORT is emitted before it finishes, so an unbounded wait on detection
inside health puts a cold `import torch` (1.5s here, longer on a cold page
cache) directly in front of that 2s deadline.

Nothing the launcher reads from health depends on detection -- status, service,
the protocol and manageability versions, the auth and ownership bits,
studio_root_id, desktop_owner, version. Only chat_only does.

CPU-only, no network, no GPU, no weights. The subprocess tests stub detection;
they never import torch.
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
    """Cross-language guard: the budget is only correct relative to the Rust one.

    Either side can be changed without the other; this fails when they drift.
    """
    assert _PROBE_RS.is_file(), f"{_PROBE_RS} moved; update this guard"
    rust = _PROBE_RS.read_text(encoding = "utf-8")
    probe = rust[rust.index("fn probe_ownerless_spawned_backend") :]
    match = re.search(r"\.timeout\(Duration::from_secs\((\d+)\)\)", probe)
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
    # C-extension import holds the GIL past it (0.24s measured at a 1.5s budget). A
    # budget that only just fits is one slow host away from the dead end.
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

def slow_detect():
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


def test_health_answers_within_the_budget_while_detection_is_slow():
    """The regression: detection outlasts the probe, health must not.

    Against an unbounded `await asyncio.to_thread(ensure_hardware_detected)`
    this returns in ~10s and the desktop launch dead-ends.
    """
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
    # Provisional, and in the conservative direction: never offer Train/Export
    # on a host that turns out not to support them.
    assert result["chat_only"] is True


def test_health_still_waits_when_detection_finishes_inside_the_budget():
    """The budget is a ceiling, not a floor.

    The wait exists so a GPU host is not published as chat-only for a second;
    bounding it must not turn every early health call into a provisional one.
    """
    result = _probe(0.3)

    assert result["status"] == 200
    assert (
        result["chat_only"] is False
    ), "health published chat_only=True on a host whose detection completed well inside the budget"
    assert "hardware_detecting" not in result or result["hardware_detecting"] is None


def test_health_does_not_await_detection_unbounded():
    """Static guard: health_check must go through the bounded helper.

    `await asyncio.to_thread(ensure_hardware_detected)` reads like the obvious
    fix and passes every test that pins DEVICE first, because then it returns
    immediately. It only fails on a cold desktop launch.
    """
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

    fetchDeviceType() sets ``fetched = data.device_type !== undefined`` and every
    later non-forced call short-circuits on ``fetched``. So an authenticated
    health request answered inside the warm window with device_type present
    pins the provisional chat_only=true for the rest of the SPA session: Train
    hidden, /studio redirected to /chat, on a GPU host, until a reload. The
    sidebar's recovery poll does not save it either -- that only runs for
    chat_only_reason === "mlx_unavailable", and a provisional reply has no
    reason.
    """
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
    """DEVICE goes non-None before detection has settled.

    The XPU branch assigns DEVICE and CHAT_ONLY=False and only then calls
    torch.xpu.get_device_name(0); if that raises, ensure_hardware_detected's
    handler degrades the host to CPU/chat-only. A waiter keyed on "DEVICE is not
    None" would have already published the intermediate value, so health could
    report training as available on a host that ends up chat-only, and
    config/env.ts would cache that for the SPA session.
    """
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
