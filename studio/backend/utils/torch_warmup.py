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
    (utils.hardware, model_config), which caches under a lock and only on success,
    so a racing request waits rather than sees a partial
  * optional GPU consumers stay cold -- Hub downloads load the Xet/Unsloth Zoo
    integration on demand, and RAG operations load their embedding backend on demand

This does NOT make torch-dependent endpoints cheap while it runs: anything reaching
get_device() blocks until the hardware stage finishes, so `async def` handlers on
that path must use asyncio.to_thread (see main.py's /api/health).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import threading
import time
from contextlib import contextmanager
from functools import partial, wraps
from importlib._bootstrap import _ModuleLockManager
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_TORCH_WARM"

_start_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
# Detection epoch of the live warm: "already warmed this lifespan" vs. one whose lifespan ended.
_thread_epoch: Optional[int] = None
_status: dict = {"started": False, "finished": False, "stages": {}}


def _is_extension_module(name: str) -> bool:
    """True if sys.modules[name] is a compiled extension, not Python source."""
    module = sys.modules.get(name)
    origin = getattr(getattr(module, "__spec__", None), "origin", None) or getattr(
        module, "__file__", None
    )
    if not isinstance(origin, str):
        return False
    return origin.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))


_DATASETS_ARROW_EXTENSION_TYPES = tuple(
    f"datasets.features.features.Array{dimensions}DExtensionType" for dimensions in range(2, 6)
)


def _clear_external_import_state(package: str) -> list[str]:
    """Undo native registrations made by a pure-Python module before it failed."""
    if package != "datasets":
        return []
    pyarrow = sys.modules.get("pyarrow")
    unregister = getattr(pyarrow, "unregister_extension_type", None)
    if unregister is None:
        return []
    cleared: list[str] = []
    for type_name in _DATASETS_ARROW_EXTENSION_TYPES:
        try:
            unregister(type_name)
        except KeyError:
            continue
        cleared.append(type_name)
    if cleared:
        logger.warning(
            "unregistered %d PyArrow extension type(s) left by the failed %s "
            "import so its modules can be executed again",
            len(cleared),
            package,
        )
    return cleared


def _synchronize_with_imports(fn):
    """Run cleanup under the same per-module lock used by CPython imports."""

    @wraps(fn)
    def synchronized(package: str):
        with _ModuleLockManager(package):
            return fn(package)

    return synchronized


@_synchronize_with_imports
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
    Known native registries populated by pure-Python modules are reset only after
    every stale module has been removed and no importer has republished the parent.
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
    fully_purged = package not in sys.modules and not any(name in sys.modules for name in stale)
    if fully_purged:
        _clear_external_import_state(package)
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
}

# Hold the import lock across a bare import and its failure cleanup. Locking only
# the purge leaves a window where a queued importer can reuse stale submodules.
# Hardware and transformers acquire additional locks, so exclude them to avoid
# lock-order inversions.
_BARE_IMPORT_STAGES = frozenset({"datasets"})


@contextmanager
def _held_import_lock(name: str, package: Optional[str]):
    """Hold ``package``'s import lock for a bare-import stage; a no-op for the rest."""
    if package is None or name not in _BARE_IMPORT_STAGES:
        yield
        return
    with _ModuleLockManager(package):
        yield


def _run_stage(name: str, fn) -> None:
    package = _STAGE_PACKAGE.get(name)
    started = time.perf_counter()
    with _held_import_lock(name, package):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - a warm failure must be visible, not fatal
            _status["stages"][name] = {"ok": False, "error": repr(exc)}
            # warning, not debug: the stage stays cold and the first request pays for it.
            logger.warning("torch warm stage %r failed: %r", name, exc)
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
    from utils.models.model_config import _detection_sets
    _detection_sets()


def _warm_datasets() -> None:
    # `import main` pulled it in; keep the first dataset op as cheap. Ungated: no torch needed.
    importlib.import_module("datasets")


# Keep metadata and framework registries ready without importing optional GPU consumers.
# Unsloth Zoo is loaded by utils.hf_xet_fallback only when a Hub operation needs it.
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
