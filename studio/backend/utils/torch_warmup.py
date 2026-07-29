# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import the ML stack on a background thread while the backend finishes booting.

torch (plus the sympy/scipy/pandas it drags in) used to be imported as a side
effect of `import main`, so the port could not bind until it finished. Those
imports are deferred now, which would only move the cost to the first request
that needs them -- this module pays it concurrently instead.

Started from the last line of main.py's lifespan, deliberately: everything
above it in the lifespan is on the critical path to binding the socket, and a
warm thread started earlier would compete with that work for the GIL. Uvicorn
binds as soon as the lifespan returns, so the warm overlaps the serving window
instead of the boot.

Contract:
  * idempotent -- repeat calls are no-ops, one thread per process
  * never fatal -- every stage is isolated; a failure is logged and leaves the
    stage un-warmed, to be retried by whoever actually needs it
  * no half-initialised state -- each stage delegates to the module that owns
    the cache (utils.hardware, model_config), all of which cache under a lock
    and only on success, so a request racing the warm waits rather than
    duplicating or observing a partial result

What this does NOT do: make the endpoints that need torch cheap while it runs.
Anything reaching get_device() blocks until the hardware stage finishes, so
`async def` handlers on that path must go through asyncio.to_thread or they
stall the event loop for the whole import (see main.py's /api/health).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
import time
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_TORCH_WARM"

_start_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_status: dict = {"started": False, "finished": False, "stages": {}}


def _torch_installed() -> bool:
    """True if torch is importable, without importing it.

    find_spec resolves the module but never runs it, so this stays free. Used to
    skip the stages that hard-require torch on a --no-torch install (install.sh
    --no-torch, and the api-smoke CI job) rather than log an import error for a
    dependency that host deliberately does not have.
    """
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _run_stage(name: str, fn) -> None:
    started = time.perf_counter()
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - a warm failure must be visible, not fatal
        _status["stages"][name] = {"ok": False, "error": repr(exc)}
        # warning, not debug: the stage stays cold and the first request pays
        # for it, so this needs to be greppable when someone reports a slow
        # first inference.
        logger.warning("torch warm stage %r failed: %r", name, exc)
    else:
        _status["stages"][name] = {
            "ok": True,
            "seconds": round(time.perf_counter() - started, 3),
        }


def _warm_hardware() -> None:
    # Imports torch and enumerates devices. The lifespan waits on this same
    # call, so it either finds the result cached or blocks on the lock the warm
    # thread holds -- never a second, racing detection.
    from utils.hardware import ensure_hardware_detected

    ensure_hardware_detected()


def _warm_transformers() -> None:
    # The registry read that used to happen at `import main`. Kept ahead of the
    # unsloth_zoo stage: that is the order the eager imports ran in, and
    # unsloth_zoo patches transformers on import.
    from utils.models.model_config import _detection_sets

    _detection_sets()


def _warm_unsloth_zoo() -> None:
    # Backs the download stall watchdog (utils.hf_xet_fallback). It was imported
    # at startup before, so warming it keeps the first download's cost where it
    # used to be. Skipped without torch: unsloth_zoo hard-requires it, and the
    # shim already degrades to its own stubs on such a host.
    if not _torch_installed():
        return
    importlib.import_module("unsloth_zoo")


_STAGES = (
    ("hardware", _warm_hardware),
    ("transformers", _warm_transformers),
    ("unsloth_zoo", _warm_unsloth_zoo),
)


def _warm() -> None:
    started = time.perf_counter()
    for name, fn in _STAGES:
        _run_stage(name, fn)
    _status["finished"] = True
    _status["seconds"] = round(time.perf_counter() - started, 3)
    logger.info("torch warm finished in %.1fms", (time.perf_counter() - started) * 1000)


def start_background_warm() -> bool:
    """Start the warm thread once. Returns True iff this call started it.

    Runs on every host, torch or not: the first stage is hardware detection,
    which the lifespan used to do inline and which still has to happen without
    waiting for a request (it is what prints the "Hardware detected: ..." line
    and what /api/health reports as chat_only).
    """
    global _thread
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    with _start_lock:
        if _thread is not None:
            return False
        _thread = threading.Thread(target = _warm, daemon = True, name = "torch-warm")
        _status["started"] = True
        _thread.start()
        return True


def warm_status() -> dict:
    """Snapshot of the warm for diagnostics and tests."""
    return {
        "started": _status["started"],
        "finished": _status["finished"],
        "alive": bool(_thread is not None and _thread.is_alive()),
        "stages": dict(_status["stages"]),
        "seconds": _status.get("seconds"),
    }


def join_background_warm(timeout: Optional[float] = None) -> bool:
    """Wait for the warm thread. Returns True if it is done (or never ran)."""
    thread = _thread
    if thread is None:
        return True
    thread.join(timeout)
    return not thread.is_alive()
