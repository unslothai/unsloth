# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime contract of routes.inference._offline_guarded.

Driven in a clean interpreter rather than in-process: the backend test suite installs
stubs for structlog, fastapi and others into sys.modules at collection time, so loading
the route module in-process passes or fails depending on collection order. One subprocess
keeps the check honest and order-independent. All three assertions share it so the
interpreter and torch import are paid once.
"""

import os
import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

_DRIVER = r"""
import contextlib, importlib.util, sys
sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("inference_route_under_test", "routes/inference.py")
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)

events = []

@contextlib.contextmanager
def _tracking_guard():
    events.append("open")
    try:
        yield
    finally:
        events.append("close")

# Patched on the ROUTE module: _offline_guarded deliberately uses the module-level symbol
# so route tests can intercept it, rather than re-importing from llama_cpp.
route._hf_offline_if_unreachable = _tracking_guard

# 1. the wrapped work runs INSIDE the window, not after it closes
def _work(a, b, *, kw):
    assert events == ["open"], events
    return (a, b, kw)

assert route._offline_guarded("org/model", _work, 1, 2, kw = 3) == (1, 2, 3)
assert events == ["open", "close"], events

# 2. a wrapped call may take its own model_identifier kwarg
# (_guard_chat_load_against_training does), so the helper's params are positional-only
events.clear()
seen = {}

def _guard(config, *, model_identifier, hf_token = None):
    seen.update(config = config, model_identifier = model_identifier)

route._offline_guarded("org/model", _guard, "cfg", model_identifier = "org/model")
assert seen == {"config": "cfg", "model_identifier": "org/model"}, seen
assert events == ["open", "close"], events

# 3. an exception inside the window still closes it
events.clear()
try:
    route._offline_guarded("org/model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
except RuntimeError:
    pass
assert events == ["open", "close"], events

# 4. keyed on what is READ: a local-only read skips the probe, but a local adapter whose
# base resolves remotely still opens the window
import tempfile
with tempfile.TemporaryDirectory() as local:
    events.clear()
    route._offline_guarded(local, lambda: None)
    assert events == [], events

    events.clear()
    route._offline_guarded((local, local), lambda: None)
    assert events == [], events

    events.clear()
    route._offline_guarded((local, "org/base"), lambda: None)
    assert events == ["open", "close"], events

print("OFFLINE_GUARDED_OK")
"""


def test_offline_guarded_runtime_contract():
    # utf-8 both ways: the child prints Unsloth's non-ASCII banner, and a Windows runner
    # would otherwise decode it as cp1252 and fail on the banner rather than the contract.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd = _BACKEND_ROOT,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        timeout = 300,
    )
    assert (
        "OFFLINE_GUARDED_OK" in proc.stdout
    ), f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
