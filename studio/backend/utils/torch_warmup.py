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
    the cache (utils.hardware, model_config, hf_xet_fallback), all of which
    cache under a lock and only on success, so a request racing the warm waits
    rather than duplicating or observing a partial result
  * same end state -- the stages import what `import main` used to import, in
    the order it used to import them, and through the same call sites. A stage
    that shortcuts to a bare `import x` can behave differently from the edge it
    replaced; see _warm_unsloth_zoo.

What this does NOT do: make the endpoints that need torch cheap while it runs.
Anything reaching get_device() blocks until the hardware stage finishes, so
`async def` handlers on that path must go through asyncio.to_thread or they
stall the event loop for the whole import (see main.py's /api/health).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
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


def _is_extension_module(name: str) -> bool:
    """True if sys.modules[name] is a compiled extension, not Python source."""
    module = sys.modules.get(name)
    origin = getattr(getattr(module, "__spec__", None), "origin", None) or getattr(
        module, "__file__", None
    )
    if not isinstance(origin, str):
        return False
    return origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


def purge_partial_import(package: str) -> list:
    """Drop the submodules a failed package import left behind in sys.modules.

    When ``package/__init__.py`` raises, CPython removes only the parent from
    sys.modules and keeps every submodule it had already executed. The next
    import then re-runs ``__init__`` while each of its ``from .x import y`` is
    served from that cache, so the attributes are never rebound: the package
    imports "successfully" but is missing pieces (the bitsandbytes case fixed in
    #7580). The warm makes this reachable -- it imports these packages on a
    thread and swallows the failure, so the retry is somebody else's request.

    Only acts on that exact signature (parent gone, submodules present), so a
    concurrent, still-running import is left alone. Returns what it removed.

    Refuses to touch a package that has already-initialised C extension
    submodules. Evicting one only makes the next import re-run its module init,
    and a second registration of the same pybind11 type calls std::terminate:
    purging torch.* on this box and re-importing killed the process outright
    with 'generic_type: type "GradBucket" is already registered!'. Leaving the
    zombie gives a torch that is missing attributes, which is bad; SIGABRT in
    the middle of serving is worse.
    """
    if package in sys.modules:
        return []
    prefix = package + "."
    stale = [name for name in list(sys.modules) if name.startswith(prefix)]
    compiled = sorted(name for name in stale if _is_extension_module(name))
    if compiled:
        logger.warning(
            "not purging %s: %d of its submodule(s) are loaded C extensions and "
            "re-importing one aborts the process (%s). The next import will reuse "
            "the cached submodules and may be missing attributes.",
            package,
            len(compiled),
            ", ".join(compiled[:4]),
        )
        return []
    for name in stale:
        sys.modules.pop(name, None)
    if stale:
        logger.warning(
            "purged %d half-imported %s submodule(s) so the next import re-runs clean: %s",
            len(stale),
            package,
            ", ".join(sorted(stale)[:8]),
        )
    return stale


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


def _warm_datasets() -> None:
    # utils/datasets/raw_text.py used to import this for an annotation alone,
    # and `datasets` reaches torch through datasets.formatting.torch_formatter.
    # Nothing depends on it being imported early, but it was, so warm it and
    # keep the first dataset operation as cheap as it used to be. 0.3s once
    # transformers is loaded. Ungated: datasets imports without torch.
    importlib.import_module("datasets")


def _warm_unsloth_zoo() -> None:
    """Prime the download stall watchdog, the way the eager import primed it.

    Through utils.hf_xet_fallback rather than a bare ``import unsloth_zoo``,
    because the edge this replaces was orchestrator.py's ``from
    utils.hf_xet_fallback import DownloadStallError``. The shim does not just
    import: when unsloth_zoo's GPU init raises it retries under
    UNSLOTH_ZOO_DISABLE_GPU_INIT=1, which makes unsloth_zoo skip that init and
    inject its triton/bitsandbytes stubs. A bare import skips the retry, so on a
    host whose bitsandbytes wheel cannot find libcudart -- where unsloth_zoo
    raises "CUDA Setup failed despite GPU being available" -- the warm reported
    a failed stage for something startup used to complete. Measured on this box
    before the fix; a bare import fails there and the shim succeeds.

    Skipped without torch: unsloth_zoo hard-requires it, and the shim already
    degrades to its own stubs on such a host.
    """
    if not _torch_installed():
        return
    # Private, deliberately: this is the exact function the removed eager import
    # drove, and the public names reach it only by resolving an attribute whose
    # degraded fallback is indistinguishable from success.
    from utils.hf_xet_fallback import _load_shared

    if not _load_shared():
        # _load_shared() has already logged why and left the shim on its
        # degraded stubs, so downloads still work -- but the stage is not warm,
        # and that has to show up in warm_status(). Leave nothing half-imported
        # behind first: whoever imports unsloth_zoo next must re-run __init__
        # against an empty cache. See purge_partial_import().
        purge_partial_import("unsloth_zoo")
        raise RuntimeError("unsloth_zoo unavailable; the download stall watchdog stays degraded")


# Order matters: it is the order the eager imports ran in. transformers before
# unsloth_zoo because unsloth_zoo patches transformers on import, and datasets
# between them because that is where `import main` reached it.
_STAGES = (
    ("hardware", _warm_hardware),
    ("transformers", _warm_transformers),
    ("datasets", _warm_datasets),
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
