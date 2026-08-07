# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: /api/liveness says whether the backend is still warming up, and stays cheap.

The desktop health watchdog probes this route every 15s and kills the backend after 3
consecutive misses. It cannot use /api/health for that -- health awaits hardware detection,
so a probe is billed for the warm thread's `import torch` -- and it cannot treat one reply
as "startup finished" either, because those C-extension imports hold the GIL and stall the
next probes on a process that is perfectly healthy. So liveness carries a
`torch_warm_in_progress` marker and the watchdog holds its startup grace open until a reply
omits it. See studio/src-tauri/src/commands.rs.

That marker tracks the whole coordinated warm, not hardware detection alone: detection is
only the first of utils/torch_warmup.py's stages and the inference_backend, transformers,
datasets and unsloth_zoo imports that follow it are the ones that stall a probe. The older
`hardware_detecting` marker stays exactly what it was, a "this verdict is provisional"
signal the frontend reads, and is still published beside it.

The markers must not cost what health costs: liveness reads settled snapshots, it must
never start detection or wait on it.

CPU-only, no network, no GPU, no weights: the subprocess tests stub detection.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_COMMANDS_RS = _BACKEND_DIR.parent / "src-tauri" / "src" / "commands.rs"


_SNIPPET = r"""
import json, os, time

# The warm is what would settle the verdict, and these tests decide by hand whether it is
# settled, so keep a real one out of the way. DEFERRED picks the kill switch on or off.
if %(deferred)s:
    os.environ["UNSLOTH_STUDIO_DISABLE_TORCH_WARM"] = "1"
else:
    os.environ.pop("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", None)

import main
from fastapi import FastAPI
from fastapi.testclient import TestClient

hw = main._hw_module

# The real warm imports the ML stack, which is the cost these tests exist to keep off the
# route, so drive its published state by hand instead. WARM picks which of the four states
# the thread can be observed in.
import utils.torch_warmup as tw

class _FakeWarmThread:
    def __init__(self, alive):
        self._alive = alive
    def is_alive(self):
        return self._alive

# started / finished / thread, per state:
#   running  -- mid-stage, the state the startup grace is for
#   finished -- every stage done, so startup really is over
#   never    -- the kill switch is on and no warm was ever started
#   retired  -- a shutdown stopped it between stages; it will never set finished
tw._status["started"], tw._status["finished"], tw._thread = {
    "running": (True, False, _FakeWarmThread(True)),
    "finished": (True, True, _FakeWarmThread(False)),
    "never": (False, False, None),
    "retired": (True, False, _FakeWarmThread(False)),
}[%(warm)r]

if %(settled)s:
    hw.DEVICE = hw.DeviceType.CPU
    hw.CHAT_ONLY = True
    hw.CHAT_ONLY_REASON = "no_accelerator"
    hw.DETECTION_COMPLETE.set()
else:
    # "detection has not run": DEVICE alone is not the completion signal, the branches
    # assign it partway through, so clear the event too.
    hw.DEVICE = None
    hw.CHAT_ONLY = True
    hw.CHAT_ONLY_REASON = None
    hw.DETECTION_COMPLETE.clear()

def must_not_run(*_):
    raise AssertionError("liveness started hardware detection")

hw.ensure_hardware_detected = must_not_run

app = FastAPI()
app.add_api_route("/api/liveness", main.liveness_check, methods = ["GET"])
client = TestClient(app)

started = time.perf_counter()
response = client.get("/api/liveness")
elapsed = time.perf_counter() - started
body = response.json()

print("RESULT" + json.dumps({
    "status_code": response.status_code,
    "elapsed": elapsed,
    "status": body.get("status"),
    "service": body.get("service"),
    "hardware_detecting": body.get("hardware_detecting"),
    "hardware_detection_deferred": body.get("hardware_detection_deferred"),
    "has_detecting_key": "hardware_detecting" in body,
    "torch_warm_in_progress": body.get("torch_warm_in_progress"),
    "has_warm_key": "torch_warm_in_progress" in body,
    "studio_root_id": body.get("studio_root_id"),
}))
"""


def _probe(
    settled: bool,
    deferred: bool = False,
    warm: str = "running",
) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _SNIPPET % {"settled": settled, "deferred": deferred, "warm": warm},
        ],
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


def test_an_unsettled_verdict_is_published_as_still_warming_up():
    """Without this the watchdog ends its startup grace on the first reply, and the next
    GIL stall inside the torch import reads as three dead probes."""
    result = _probe(settled = False)

    assert result["status_code"] == 200
    assert result["status"] == "alive"
    assert result["service"] == "Unsloth UI Backend"
    assert result["torch_warm_in_progress"] is True
    assert result["hardware_detecting"] is True
    assert result["hardware_detection_deferred"] is None, (
        "the warm is running here; a deferred marker would tell the watchdog the verdict "
        "will never settle"
    )


def test_a_late_warm_stage_still_holds_the_startup_grace_open():
    """The regression: hardware detection is _STAGES[0], and inference_backend,
    transformers, datasets and unsloth_zoo import after it.

    So hardware_detecting is already gone while the warm is at its most expensive. A
    watchdog reading only that marker ends its grace mid-warm and the next GIL stall,
    which is exactly what the reported "torch warm finished in 8190.4ms" timeline shows
    happening well after detection, is billed as three dead probes against a healthy
    backend."""
    result = _probe(settled = True, warm = "running")

    assert not result["has_detecting_key"], (
        "the point of this case is a settled verdict; if detection still reads unsettled "
        "the late-stage window is not being modelled"
    )
    assert result["torch_warm_in_progress"] is True, (
        "liveness reports startup finished while the warm is still importing transformers, "
        "datasets and unsloth_zoo; the watchdog ends its grace and the next GIL stall kills "
        "a backend that is starting normally"
    )


def test_a_settled_verdict_carries_no_warming_up_marker():
    """The markers are the whole signal, so they have to disappear once the warm is done, or
    the watchdog never counts a real failure until the grace period expires."""
    result = _probe(settled = True, warm = "finished")

    assert result["status"] == "alive"
    assert not result["has_detecting_key"], (
        "liveness still reports hardware_detecting after detection settled; the desktop "
        "watchdog would hold its startup grace open for the full 5 minutes"
    )
    assert not result["has_warm_key"], (
        "liveness still reports torch_warm_in_progress after every stage finished; the "
        "watchdog would never count a failure against a genuinely hung backend"
    )


def test_a_deferred_warm_says_so_instead_of_looking_like_a_slow_start():
    """With UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 nothing will settle the verdict, so a bare
    hardware_detecting would be indistinguishable from a backend still importing torch."""
    result = _probe(settled = False, deferred = True, warm = "never")

    assert result["hardware_detecting"] is True
    assert result["hardware_detection_deferred"] is True
    assert not result["has_warm_key"], (
        "no warm was started, so nothing will ever finish one; a warm marker here would "
        "hold the watchdog's startup grace open until it expired on its own"
    )


def test_a_warm_retired_mid_stage_is_not_reported_as_warming_forever():
    """A shutdown stops the warm at a stage boundary, so it never sets finished and its
    thread is gone. Deriving the marker from "not finished" alone would leave it lit for
    the rest of the process, which is the deferred trap by another route."""
    result = _probe(settled = False, warm = "retired")

    assert not result["has_warm_key"], (
        "a warm whose thread has exited is reported as still running; the watchdog would "
        "hold its startup grace open for the full 5 minutes and never trust the backend"
    )


def test_liveness_answers_immediately_and_never_starts_detection():
    """The route exists because health's detection wait is too expensive to probe every
    15s. The stub raises if detection is started, so returning at all proves it was not."""
    result = _probe(settled = False)

    assert result["elapsed"] < 0.5, (
        f"/api/liveness took {result['elapsed']:.2f}s; it must read the settled snapshot "
        f"rather than wait for one"
    )
    # Still the full port-validation payload the launcher matches on. The key must be
    # present because the launcher reads it; its value is environment-derived and is
    # legitimately empty on a bare CI runner, so presence is what this asserts.
    assert "studio_root_id" in result


def test_the_desktop_watchdog_still_reads_these_fields():
    """Cross-language guard: the marker only does anything because commands.rs reads it,
    and either side can be changed without the other."""
    assert _COMMANDS_RS.is_file(), f"{_COMMANDS_RS} moved; update this guard"
    rust = _COMMANDS_RS.read_text(encoding = "utf-8")
    probe = rust[rust.index("async fn check_health_inner") :]
    end = probe.find("\n}\n")
    if end != -1:
        probe = probe[: end + 3]

    assert '"/api/liveness"' in probe, (
        "the watchdog probe no longer asks for /api/liveness; it must not go back to "
        "/api/health, which awaits hardware detection"
    )
    assert '"torch_warm_in_progress"' in probe, (
        "the watchdog probe no longer reads torch_warm_in_progress, so one early reply ends "
        "the startup grace again and a GIL stall can kill a healthy backend"
    )
    assert '"hardware_detecting"' in probe, (
        "the watchdog probe dropped its hardware_detecting fallback; a backend older than "
        "torch_warm_in_progress then gets no startup grace at all"
    )
    assert '"hardware_detection_deferred"' in probe

    match = re.search(r"const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs\((\d+)\)", rust)
    assert match, "commands.rs no longer sets a whole-seconds probe timeout"
    interval = re.search(
        r"const HEALTH_WATCHDOG_INTERVAL: Duration = Duration::from_secs\((\d+)\)", rust
    )
    assert interval, "commands.rs no longer sets a whole-seconds watchdog interval"
    assert int(match.group(1)) < int(interval.group(1)), (
        f"a {match.group(1)}s probe outlives the {interval.group(1)}s watchdog interval, "
        f"so the next tick starts on top of the last one"
    )
