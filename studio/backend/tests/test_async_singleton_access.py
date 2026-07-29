# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async handlers must not build the inference singleton on the event loop.

The construction runs get_default_models() -> hw.get_device(), so the first
caller waits for the background warm. An async handler doing that inline holds
the event-loop thread for the whole torch import, stalling login, liveness and
the deadline-bound desktop health probe.

The offload has to stay at the call site, passing the route module's own
`get_inference_backend` to a thread. A helper living in orchestrator.py would
resolve that module's global instead, so tests and callers that patch
`routes.inference.get_inference_backend` would be silently bypassed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_ROUTE_FILES = ("routes/inference.py", "routes/models.py")


def _async_call_sites(rel: str) -> list[str]:
    """Bare get_inference_backend() invocations inside an async def.

    `asyncio.to_thread(get_inference_backend)` passes the function object, so it
    is an ast.Name and never an ast.Call here. Only a real on-loop invocation is
    reported.
    """
    tree = ast.parse((_BACKEND / rel).read_text())
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for sub in ast.walk(fn):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
                continue
            if sub.func.id == "get_inference_backend":
                found.append(f"{rel}:{sub.lineno} in async {fn.name}")
    return found


def test_no_async_handler_builds_the_singleton_inline():
    offenders = [s for rel in _ROUTE_FILES for s in _async_call_sites(rel)]
    assert not offenders, "async handlers building the singleton inline:\n  " + "\n  ".join(
        offenders
    )


def test_the_offload_is_actually_present():
    """Guard against the sweep passing because the calls simply vanished."""
    total = 0
    for rel in _ROUTE_FILES:
        text = (_BACKEND / rel).read_text()
        total += text.count("await asyncio.to_thread(get_inference_backend)")
    assert total >= 14, f"expected the offloaded call sites to survive, found {total}"


def test_the_offload_stays_at_the_call_site():
    """No orchestrator-level async helper: it would bypass patched route globals.

    tests/test_orchestrator_unload_cancel.py patches
    routes.inference.get_inference_backend. An accessor defined in
    orchestrator.py resolves orchestrator's own global, so the patch would not
    take and the test hangs on a load gate that never opens.
    """
    orch = (_BACKEND / "core/inference/orchestrator.py").read_text()
    assert "async def get_inference_backend_async" not in orch, (
        "an async accessor in orchestrator.py bypasses callers that patch the "
        "route module's get_inference_backend"
    )
