# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import the ML stack on a background thread while the backend finishes booting.

torch (plus the sympy/scipy/pandas it drags in) used to be imported by `import
main`, so the port could not bind until it finished. Deferring it alone would just
move that cost to the first request; this module pays it concurrently instead.

Started from the last line of main.py's lifespan: everything above is on the
critical path to binding the socket and would contend for the GIL. Uvicorn binds as
soon as the lifespan returns, so the warm overlaps serving, not boot.

Contract:
  * idempotent -- one thread per process, repeat calls are no-ops
  * never fatal -- a failed stage is logged and left cold, retried by whoever needs it
  * no half-initialised state -- stages delegate to the module owning the cache
    (utils.hardware, model_config, hf_xet_fallback), which caches under a lock and
    only on success, so a racing request waits rather than sees a partial
  * same end state -- same imports, order and call sites as `import main`. A stage
    shortcutting to a bare `import x` can behave differently; see _warm_unsloth_zoo.

This does NOT make torch-dependent endpoints cheap while it runs: anything reaching
get_device() blocks until the hardware stage finishes, so `async def` handlers on
that path must use asyncio.to_thread (see main.py's /api/health).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import time
from contextlib import contextmanager
from functools import partial
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_TORCH_WARM"

_start_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
# Detection epoch of the live warm: "already warmed this lifespan" vs. one whose lifespan ended.
_thread_epoch: Optional[int] = None
_status: dict = {"started": False, "finished": False, "stages": {}}


def _torch_installed() -> bool:
    """True if torch is importable without importing it, so torch-requiring stages are
    skipped on a --no-torch install rather than logging an error for a missing-on-purpose dep."""
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

    When ``package/__init__.py`` raises, CPython evicts only the parent and keeps
    every submodule it already executed. The next import re-runs ``__init__`` with
    each ``from .x import y`` served from that cache, so attributes are never
    rebound: the package imports "successfully" but is missing pieces (the
    bitsandbytes case fixed in #7580). The warm makes this reachable -- it imports on
    a thread and swallows the failure, so the retry is somebody else's request.

    Acts only on that exact signature (parent gone, submodules present), so a
    concurrent, still-running import is left alone. Returns what it removed.

    Declines when any submodule is a loaded C extension: evicting one re-runs its
    module init, and pybind11 answers a duplicate type registration with
    std::terminate. A torch missing attributes is bad; SIGABRT mid-serve is worse.
    """
    if package in sys.modules:
        return []
    prefix = package + "."
    stale = [name for name in list(sys.modules) if name.startswith(prefix)]
    # Re-check before touching anything: a retrying importer publishes the parent as soon
    # as its __init__ begins, and popping submodules out from under it produces the very
    # half-initialised package this prevents. Narrows the window; CPython's lock is private.
    if package in sys.modules:
        logger.info(
            "not purging %s: another importer republished it while collecting its "
            "leftovers, so that import owns them now",
            package,
        )
        return []
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
    # Track what actually went: a partway bail must not report a clean slate that never happened.
    removed = []
    for name in stale:
        # Same race, per pop: bail the moment the parent is back.
        if package in sys.modules:
            logger.warning(
                "stopped purging %s partway: another importer republished it. The "
                "submodules already removed will be re-executed by that import.",
                package,
            )
            break
        if sys.modules.pop(name, None) is not None:
            removed.append(name)
    if removed:
        logger.warning(
            "purged %d half-imported %s submodule(s) so the next import re-runs clean: %s",
            len(removed),
            package,
            ", ".join(sorted(removed)[:8]),
        )
    return removed


# Stage -> package to purge on failure. inference_backend is absent: it imports nothing.
_STAGE_PACKAGE = {
    "hardware": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "unsloth_zoo": "unsloth_zoo",
}


def _run_stage(name: str, fn) -> None:
    started = time.perf_counter()
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - a warm failure must be visible, not fatal
        _status["stages"][name] = {"ok": False, "error": repr(exc)}
        # warning, not debug: the stage stays cold and the first request pays for it.
        logger.warning("torch warm stage %r failed: %r", name, exc)
        package = _STAGE_PACKAGE.get(name)
        if package:
            purge_partial_import(package)
    else:
        _status["stages"][name] = {
            "ok": True,
            "seconds": round(time.perf_counter() - started, 3),
        }


def _warm_hardware(epoch: Optional[int] = None) -> None:
    # Requests hit the same call, so they reuse the cache or block on this thread's lock,
    # never race a second detection. The epoch rides along: a shutdown landing between
    # _warm()'s check and _DETECT_LOCK would else publish for the lifespan that just ended.
    from utils.hardware import ensure_hardware_detected
    ensure_hardware_detected(epoch)


def _warm_transformers() -> None:
    # Ahead of unsloth_zoo: that is the eager order, and unsloth_zoo patches it on import.
    from utils.models.model_config import _detection_sets
    _detection_sets()


def _warm_datasets() -> None:
    # `import main` pulled it in; keep the first dataset op as cheap. Ungated: no torch needed.
    importlib.import_module("datasets")


def _warm_unsloth_zoo() -> None:
    """Prime the download stall watchdog, the way the eager import primed it.

    Through utils.hf_xet_fallback, not a bare ``import unsloth_zoo``: the edge this
    replaces was orchestrator.py's ``from utils.hf_xet_fallback import
    DownloadStallError``, and the shim does more. When unsloth_zoo's GPU init raises it
    retries under UNSLOTH_ZOO_DISABLE_GPU_INIT=1 and injects triton/bitsandbytes stubs;
    a bare import skips that retry, so a host whose bitsandbytes cannot find libcudart
    would fail a stage startup used to complete.

    Skipped without torch: unsloth_zoo hard-requires it and the shim degrades to stubs.
    """
    if not _torch_installed():
        return
    # Private deliberately: the exact function the eager import drove. The public names
    # reach it only via an attribute whose degraded fallback looks like success.
    from utils.hf_xet_fallback import _load_shared

    if not _load_shared():
        # Downloads still work on the degraded stubs, but the stage is cold and
        # warm_status() must say so. Purge so the next importer re-runs __init__ clean;
        # the shim's negative cache stays pinned, as the raise site and `except` must
        # resolve to one DownloadStallError class or the stall handler is bypassed.
        purge_partial_import("unsloth_zoo")
        raise RuntimeError("unsloth_zoo unavailable; the download stall watchdog stays degraded")


# _STAGES follows the eager import order: transformers before unsloth_zoo (which patches
# it on import), datasets between them because that is where `import main` reached it.
def _warm_inference_backend() -> None:
    # Its constructor reaches hw.get_device(), so whoever builds it first pays for detection
    # -- lazily that is some request, and sync helpers call the getter inline from async
    # handlers. Building it here makes the getter a dict read. After hardware, to reuse it.
    from core.inference import get_inference_backend
    get_inference_backend()


_STAGES = (
    ("hardware", _warm_hardware),
    ("inference_backend", _warm_inference_backend),
    ("transformers", _warm_transformers),
    ("datasets", _warm_datasets),
    ("unsloth_zoo", _warm_unsloth_zoo),
)


def _warm(epoch: Optional[int] = None) -> None:
    started = time.perf_counter()
    if epoch is None:
        epoch = _detection_epoch()
    # The boundary checks below only catch a shutdown between stages. Inside one, the
    # orchestrator constructor reaches get_device(), which takes no epoch; the scope binds
    # it to this pass so a mid-stage shutdown discards it instead of republishing DEVICE.
    with _owning_epoch(epoch):
        for name, fn in _STAGES:
            if epoch is not None and _detection_epoch() != epoch:
                # Checked before the first stage too: start_background_warm() reads the
                # epoch before start(), so a shutdown in that gap retires this thread
                # while it is still scheduled, with nothing yet run.
                logger.info("torch warm stopped before %s: its lifespan ended", name)
                return
            # Only the real stage takes the epoch; a patched _STAGES entry is called bare.
            _run_stage(name, partial(fn, epoch) if fn is _warm_hardware else fn)
            if epoch is not None and _detection_epoch() != epoch:
                # Shutdown retired this lifespan's detection. Later stages build the
                # orchestrator, which reaches get_device() and would start a fresh
                # detection, republishing DEVICE after teardown cleared it.
                logger.info("torch warm stopped after %s: its lifespan ended", name)
                return
    _status["finished"] = True
    _status["seconds"] = round(time.perf_counter() - started, 3)
    logger.info("torch warm finished in %.1fms", (time.perf_counter() - started) * 1000)


@contextmanager
def _owning_epoch(epoch: Optional[int]):
    """hardware.owning_detection_epoch(), a no-op when hardware is not importable: a
    --no-torch host still runs the warm and each stage reports its own absence."""
    try:
        from utils.hardware import hardware as _hw
        scope = _hw.owning_detection_epoch(epoch)
    except Exception:
        yield
        return
    with scope:
        yield


def _detection_epoch() -> Optional[int]:
    """The current detection epoch, or None if hardware is not importable."""
    try:
        from utils.hardware import hardware as _hw
        return _hw.current_detection_epoch()
    except Exception:
        return None


def _warm_after(previous: threading.Thread, epoch: Optional[int]) -> None:
    """Wait out a retired warm, then warm for ``epoch``. One importer at a time."""
    previous.join()
    _warm(epoch)


def start_background_warm() -> bool:
    """Start the warm thread once. Returns True iff this call started it.

    Runs on every host, torch or not: stage one is hardware detection, which must not
    wait for a request (it feeds /api/health's chat_only).

    A finished thread from an earlier lifespan does not count as one already running:
    reset_background_warm() declines mid-warm, so a shutdown then leaves the object in
    place, and treating that as "already started" would skip the warm over hardware
    state the same shutdown just cleared.
    """
    global _thread
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    global _thread_epoch
    # Epoch read before start(): the child may not run for a while, and a shutdown in that
    # gap retires this lifespan. Reading it in the thread would adopt the post-shutdown one.
    epoch = _detection_epoch()
    with _start_lock:
        target, args = _warm, (epoch,)
        if _thread is not None:
            # A warm holds the latch while its own lifespan is current, so repeat calls
            # are no-ops. Once shutdown retires that epoch the next lifespan warms again.
            if _thread_epoch is not None and epoch == _thread_epoch:
                return False
            if _thread.is_alive():
                # Stale but mid-stage: it stops at the next boundary and nothing retries,
                # so this lifespan would serve cold. Hand off; the successor joins it
                # first, so only one thread imports.
                target, args = _warm_after, (_thread, epoch)
            else:
                _clear_finished_warm_locked()
        _thread = threading.Thread(
            target = target,
            args = args,
            daemon = True,
            name = "torch-warm",
        )
        _thread_epoch = epoch
        _status["started"] = True
        _thread.start()
        return True


def reset_background_warm() -> bool:
    """Let a later lifespan in this process start a fresh warm. True iff reset.

    The same app can start twice (repeated ASGI lifespans, an embedded restart) and
    shutdown clears the hardware state the first warm produced; leaving the finished
    thread in place would make the second lifespan skip the warm and hand detection back
    to the first request, the stall this module removes.

    Declines while the previous warm runs, so two warms never share the same imports.
    Detection self-heals then: shutdown clears DETECTION_COMPLETE and /api/health kicks
    start_background_detection().
    """
    with _start_lock:
        thread = _thread
        if thread is not None and thread.is_alive():
            return False
        _clear_finished_warm_locked()
        return True


def _clear_finished_warm_locked() -> None:
    """Drop the finished warm and its status. Caller holds ``_start_lock``."""
    global _thread, _thread_epoch
    _thread = None
    _thread_epoch = None
    _status["started"] = False
    _status["finished"] = False
    _status["stages"] = {}
    _status.pop("seconds", None)


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
