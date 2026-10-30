# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Liveness reports inference_active, so a generating backend gets a wider budget on timeouts only."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
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

# A route that does nothing, served by the same app through the same TestClient. The
# claim here is "liveness costs about what answering costs", and a bare answer is the
# only honest zero: everything a runner charges the real probe, it charges this too.
async def _nothing():
    return {"ok": True}

app.add_api_route("/api/nothing", _nothing, methods = ["GET"])
client = TestClient(app)

def _timed(path):
    started = time.perf_counter()
    response = client.get(path)
    return time.perf_counter() - started, response

def probe():
    # Interleaved, and best of three on each side. Taken in one batch after the fact, a
    # control shares no scheduler delay with what it is compared against, so a pause that
    # lands on the measured request alone is not divided out. Alternating them gives each
    # side the same chance of being unlucky, and a minimum over three is not moved by a
    # pause that has to hit all three to count.
    mine, controls, response = [], [], None
    for _ in range(3):
        control_elapsed, _ = _timed("/api/nothing")
        elapsed, response = _timed("/api/liveness")
        controls.append(control_elapsed)
        mine.append(elapsed)
    body = response.json()
    return {
        "status_code": response.status_code,
        "status": body.get("status"),
        "service": body.get("service"),
        "elapsed": min(mine),
        "control": min(controls),
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
    "core.inference.mlx_diffusion_backend",
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
    """The wide budget is for token production; a marker left lit after the stream would hide a real
    hang."""
    result = _probe()["probes"]

    assert not result["idle_before"][
        "has_busy_key"
    ], "liveness reports inference_active before anything was registered"
    assert not result["idle_after"]["has_busy_key"], (
        "liveness still reports inference_active after the generation finished; the "
        "watchdog would keep the widened budget armed against a hung backend"
    )


def _watchdog_probe_budget_s() -> float:
    """Reads HEALTH_PROBE_TIMEOUT from the Rust launcher so the liveness ceiling cannot drift past it."""
    assert _COMMANDS_RS.is_file(), f"{_COMMANDS_RS} moved; update this guard"
    match = re.search(
        r"const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs\((\d+)\)",
        _COMMANDS_RS.read_text(encoding = "utf-8"),
    )
    assert match, "commands.rs no longer sets a whole-seconds probe timeout"
    return float(match.group(1))


def test_the_marker_costs_nothing_to_read():
    """A probe every 15s cannot wait, so the route reads a registry len() instead of asking the backend."""
    result = _probe()["probes"]

    # relative bound vs a trivial route catches new I/O; absolute bound is the watchdog budget
    for state, sample in result.items():
        control = sample["control"]
        relative = max(control * 40, 0.05)
        ceiling = min(relative, _watchdog_probe_budget_s() / 2)
        assert sample["elapsed"] < ceiling, (
            f"/api/liveness took {sample['elapsed'] * 1000:.1f}ms while {state}, against "
            f"{control * 1000:.1f}ms to answer a route that does nothing; it must read "
            f"the registry rather than wait on the generations in it"
        )


def test_the_desktop_watchdog_still_reads_the_marker():
    """Cross-language guard: commands.rs must keep reading the marker, or the watchdog ignores it."""
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
    """Liveness must stay active through the image persist tail, as generate-progress already does."""
    import importlib

    routes_inference = importlib.import_module("routes.inference")
    assert routes_inference.generation_in_flight() is False

    routes_inference._diffusion_persist_active += 1
    try:
        assert routes_inference.generation_in_flight() is True
    finally:
        routes_inference._diffusion_persist_active -= 1
    assert routes_inference.generation_in_flight() is False


_PERSIST_COUNTER = "_diffusion_persist_active"
# The two image routes' run bodies: the Unsloth /images/generate and the OpenAI-compatible /images/generations.
_PERSIST_ROUTES = {"_generate_diffusion_image_run", "_generate_openai_images"}


def _persist_marker_violations(source: str) -> list[str]:
    """Each image route raises the persist marker, and every raise is dropped on every exit.

    A raise is balanced when the statement after it is a ``try`` whose ``finally`` drops it, or (a hold handed to a
    later persist) when it sits under ``if C:`` and its ``try`` drops it in an ``except BaseException`` that re-raises,
    and a later ``try`` in the same block drops ``2 if C else 1`` in its ``finally``, with nothing that can return or
    await in between."""
    import ast

    def is_counter(node, op):
        return (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == _PERSIST_COUNTER
            and isinstance(node.op, op)
        )

    def is_one(value):
        return isinstance(value, ast.Constant) and value.value == 1

    def two_if(value):
        """The condition C of ``2 if C else 1``, else None."""
        if (
            isinstance(value, ast.IfExp)
            and isinstance(value.body, ast.Constant)
            and value.body.value == 2
            and is_one(value.orelse)
        ):
            return ast.dump(value.test)
        return None

    def finally_drop(node):
        drops = [s for s in node.finalbody if is_counter(s, ast.Sub)]
        return drops[0] if len(drops) == 1 else None

    def reraising_drop(node):
        for handler in node.handlers:
            catches_all = handler.type is None or (
                isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
            )
            body = handler.body
            if (
                catches_all
                and body
                and is_counter(body[0], ast.Sub)
                and is_one(body[0].value)
                and isinstance(body[-1], ast.Raise)
                and body[-1].exc is None
            ):
                return body[0]
        return None

    tree = ast.parse(source)
    problems: list[str] = []
    mutators: set[str] = set()
    for func in tree.body:
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = func.name
        parents = {}
        for node in ast.walk(func):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for index, child in enumerate(value):
                        if isinstance(child, ast.AST):
                            parents[id(child)] = (node, value, index)
        raises = [n for n in ast.walk(func) if is_counter(n, ast.Add)]
        drops = [n for n in ast.walk(func) if is_counter(n, ast.Sub)]
        if not raises and not drops:
            continue
        mutators.add(name)
        accounted: set[int] = set()
        handoffs: dict[str, tuple] = {}
        releases: dict[str, tuple] = {}
        for stmt in raises:
            owner, body, index = parents[id(stmt)]
            follow = body[index + 1] if index + 1 < len(body) else None
            if not is_one(stmt.value) or not isinstance(follow, ast.Try):
                problems.append(
                    f"{name}:{stmt.lineno}: the raise is not followed by the try that drops it"
                )
                continue
            drop = finally_drop(follow)
            if drop is not None and is_one(drop.value):
                accounted.add(id(drop))
                continue
            if drop is not None and two_if(drop.value) is not None:
                accounted.add(id(drop))
                releases[two_if(drop.value)] = (stmt, body, index)
                continue
            handed = reraising_drop(follow)
            if (
                drop is None
                and handed is not None
                and isinstance(owner, ast.If)
                and body is owner.body
            ):
                accounted.add(id(handed))
                handoffs[ast.dump(owner.test)] = (stmt, owner)
                continue
            problems.append(
                f"{name}:{stmt.lineno}: a raise is not dropped on every exit of its try"
            )
        for drop in drops:
            if id(drop) not in accounted:
                problems.append(f"{name}:{drop.lineno}: a drop with no matching raise")
        for cond, (stmt, guard) in handoffs.items():
            if cond not in releases:
                problems.append(f"{name}:{stmt.lineno}: a handed-off hold no later persist drops")
                continue
            _, release_body, release_index = releases.pop(cond)
            _, guard_body, guard_index = parents[id(guard)]
            if guard_body is not release_body or guard_index >= release_index:
                problems.append(
                    f"{name}:{stmt.lineno}: the hold's release is not later in its block"
                )
                continue
            for between in guard_body[guard_index + 1 : release_index]:
                if isinstance(between, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue  # defined here, run inside the release's try
                for node in ast.walk(between):
                    if isinstance(node, (ast.Return, ast.Raise, ast.Await, ast.Yield)):
                        problems.append(
                            f"{name}:{node.lineno}: an exit between a held marker and its release"
                        )
        for cond, (stmt, _, _) in releases.items():
            problems.append(f"{name}:{stmt.lineno}: a persist drops a hold nothing raised")
    for route in sorted(_PERSIST_ROUTES - mutators):
        problems.append(f"{route}: persists gallery records without publishing the marker")
    for other in sorted(mutators - _PERSIST_ROUTES):
        problems.append(f"{other}: moves the persist marker outside the image routes")
    return problems


def test_both_image_routes_publish_the_persist_marker():
    """Both image routes must raise the persist marker and drop it on every exit, or liveness goes stale."""
    import inspect
    import importlib

    source = inspect.getsource(importlib.import_module("routes.inference"))
    assert _persist_marker_violations(source) == [], (
        "an image route persists gallery records without a balanced marker, so liveness "
        "misreports that request"
    )


def test_the_persist_marker_check_catches_each_imbalance():
    """The check above fails on each way the marker can go wrong, so its pass means something."""
    import inspect
    import importlib

    source = inspect.getsource(importlib.import_module("routes.inference"))
    detailer_drop = "_diffusion_persist_active -= 2 if detailer is not None else 1"
    handoff_drop = "        except BaseException:\n            _diffusion_persist_active -= 1\n            raise\n"
    openai_raise = '    global _diffusion_persist_active\n    _diffusion_persist_active += 1\n    try:\n        with account_access.media_generation("diffusion"):\n            data ='
    assert source.count(detailer_drop) == 1
    assert source.count(handoff_drop) == 1
    assert source.count(openai_raise) == 1
    mutants = {
        "the persist no longer drops the Detailer's hold": source.replace(
            detailer_drop, "_diffusion_persist_active -= 1"
        ),
        "a failed Detailer pass leaks its hold": source.replace(
            handoff_drop, "        except BaseException:\n            raise\n"
        ),
        "the OpenAI route stops publishing": source.replace(
            openai_raise,
            openai_raise.replace("    _diffusion_persist_active += 1\n", "    pass\n"),
        ),
        "the OpenAI route never drops its marker": source.replace(
            "    finally:\n        _diffusion_persist_active -= 1\n\n    return ImageGen",
            "    finally:\n        pass\n\n    return ImageGen",
        ),
        "a return between the hold and its release": source.replace(
            "    created_at = time.time()\n    per_image_seeds",
            "    if not result:\n        return None\n    created_at = time.time()\n    per_image_seeds",
        ),
    }
    for why, mutant in mutants.items():
        assert mutant != source, why
        assert _persist_marker_violations(mutant), why
