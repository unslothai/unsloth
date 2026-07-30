# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime contract of routes.inference._offline_guarded.

Driven in a clean interpreter rather than in-process: the backend test suite installs
stubs for structlog, fastapi and others into sys.modules at collection time, so loading
the route module in-process passes or fails depending on collection order. One subprocess
keeps the check honest and order-independent. All three assertions share it so the
interpreter and torch import are paid once.
"""

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

import core.inference.llama_cpp as llama_cpp

# 1. the wrapped work runs INSIDE the window, not after it closes
opened = []

@contextlib.contextmanager
def _tracking_guard(model_id):
    opened.append(model_id)
    yield
    opened.append("closed:" + model_id)

llama_cpp._hf_offline_if_unreachable_for = _tracking_guard

def _work(a, b, *, kw):
    assert opened == ["org/model"], opened
    return (a, b, kw)

assert route._offline_guarded("org/model", _work, 1, 2, kw = 3) == (1, 2, 3)
assert opened == ["org/model", "closed:org/model"], opened

# 2. a wrapped call may take its own model_identifier kwarg
# (_guard_chat_load_against_training does), so the helper's params are positional-only
llama_cpp._hf_offline_if_unreachable_for = lambda _m: contextlib.nullcontext()
seen = {}

def _guard(config, *, model_identifier, hf_token = None):
    seen.update(config = config, model_identifier = model_identifier)

route._offline_guarded("org/model", _guard, "cfg", model_identifier = "org/model")
assert seen == {"config": "cfg", "model_identifier": "org/model"}, seen

# 3. an exception inside the window still closes it
closed = []

@contextlib.contextmanager
def _closing_guard(_model_id):
    try:
        yield
    finally:
        closed.append(True)

llama_cpp._hf_offline_if_unreachable_for = _closing_guard
try:
    route._offline_guarded("org/model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
except RuntimeError:
    pass
assert closed == [True], closed

print("OFFLINE_GUARDED_OK")
"""


def test_offline_guarded_runtime_contract():
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd = _BACKEND_ROOT,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert (
        "OFFLINE_GUARDED_OK" in proc.stdout
    ), f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
