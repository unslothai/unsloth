# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import the ML stack on a background thread while the backend finishes booting.

torch (plus the sympy/scipy/pandas it drags in) used to be imported by `import main`, so the port could not bind until it finished; deferring it alone would only move the cost to the first request, so this module pays it concurrently. Started from the last line of main.py's lifespan, since everything above is on the critical path to binding the socket and would contend for the GIL, and uvicorn binds as soon as the lifespan returns, so the warm overlaps serving rather than boot.

Contract: idempotent (one thread per process); never fatal (a failed stage is logged, left cold and retried by whoever needs it); no half-initialised state, since stages delegate to the module owning the cache (utils.hardware, model_config) which caches under a lock and only on success; optional GPU consumers stay cold, with Hub downloads loading the Xet/Unsloth Zoo integration and RAG its embedding backend on demand.

This does NOT make torch-dependent endpoints cheap while it runs: anything reaching get_device() blocks until the hardware stage finishes, so `async def` handlers on that path must use asyncio.to_thread (see main.py's /api/health).
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

    When ``package/__init__.py`` raises, CPython evicts only the parent and keeps every submodule it already executed, so the next import re-runs ``__init__`` with each ``from .x import y`` served from that cache and attributes never rebound: the package imports "successfully" while missing pieces (the bitsandbytes case fixed in #7580). The warm makes this reachable, importing on a thread and swallowing the failure so the retry is somebody else's request.

    Acts only on that exact signature (parent gone, submodules present), so a concurrent still-running import is left alone; returns what it removed.

    Declines when any submodule is a loaded C extension: evicting one re-runs its module init, and pybind11 answers a duplicate type registration with std::terminate. A torch missing attributes is bad, SIGABRT mid-serve is worse. Known native registries populated by pure-Python modules are reset only after every stale module has been removed and no importer has republished the parent.
    """
    if package in sys.modules:
        return []
    prefix = package + "."
    stale = [name for name in list(sys.modules) if name.startswith(prefix)]
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

# Hold the import lock across a bare import AND its failure cleanup: locking only the purge leaves a window where a queued importer reuses stale submodules.
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
    # Its constructor reaches hw.get_device(), so whoever builds it first pays for detection, which lazily is some request, and sync helpers call the getter inline from async handlers. Building it here makes the getter a dict read. After hardware, to reuse it.
    from core.inference import get_inference_backend

    get_inference_backend()
    # Before _prime_nvlink_topology, which is the first thread this module starts: once that
    # exists, "the first torch._dynamo import is single-threaded" stops being true. Folded into
    # this stage rather than added as a fifth one on purpose -- _STAGES carries a purge-on-
    # failure contract (_STAGE_PACKAGE) whose only sensible entry here would be torch, and
    # purging torch is never right. ensure_dynamo_imported never raises, so this stage's
    # failure semantics are unchanged.
    ensure_dynamo_imported()
    _prime_nvlink_topology()


def _prime_nvlink_topology() -> Optional[threading.Thread]:
    """Build the P2P gate's interconnect matrix off the load path. Returns the thread,
    for tests to join.

    Fire and forget on its own thread: the probe can spend its NVML bound plus the
    shell-out timeout, and a warm stage blocking that long delays every stage behind
    it. Nothing here is critical-path, so a prime that never finishes costs one load
    its head start. prime_nvlink_topology stays NVML-only and success-only: a
    subprocess running its full timeout perturbs the process, and a miss cached this
    early would keep P2P off for the life of it (#10613)."""

    def _probe() -> None:
        try:
            from core.inference.llama_cpp import LlamaCppBackend

            # Opted out, so the answer could never be used; the load path skips it too.
            if os.environ.get("UNSLOTH_DISABLE_DC_TUNING") == "1":
                return
            if LlamaCppBackend._p2p_user_opted_out():
                return
            if LlamaCppBackend._effective_gpu_count() < 2:
                return
            if not LlamaCppBackend._all_selected_gpus_match(
                LlamaCppBackend._NVLINK_FABRIC_GPU_RE, None
            ):
                return
            LlamaCppBackend.prime_nvlink_topology()
        except Exception as e:  # noqa: BLE001 -- a warm miss costs latency, never correctness
            logger.debug("NVLink topology prime skipped: %r", e)

    worker = threading.Thread(target = _probe, daemon = True, name = "nvlink-topology-prime")
    worker.start()
    return worker


_dynamo_lock = threading.Lock()
_dynamo_done = False


def ensure_dynamo_imported() -> bool:
    """Finish ``import torch._dynamo`` on ONE thread. True iff dynamo is importable.

    ``torch/__init__.py`` makes ``_dynamo`` a LAZY submodule, so any ``torch._dynamo.X``
    calls ``importlib.import_module``, which hands back whatever is in ``sys.modules`` -- a
    still-initialising module included, since ``_lock_unlock_module`` swallows ``_DeadlockError``
    to avoid deadlocking. The import takes ~0.6-3.9s and binds ``.config`` at ~0.007s but
    ``.utils`` only at ~0.284s, so for most of it the module is present and ``.utils`` is not.
    Anything reading it in that window raises ``partially initialized module 'torch._dynamo'
    has no attribute 'utils'`` (#10350, #10963).

    The window is opened by ordinary loads, not by torch.compile: ``diffusers.hooks``
    evaluates ``@torch.compiler.disable()`` at class-body time and ``torch/compiler/__init__.py``
    is ``import torch._dynamo``, so the CPU-offload step of any diffusion load opens it well
    outside every compile gate. Accelerate hit the same thing and wrapped it lazily; diffusers
    does not.

    Doing the first import once, early and single-threaded is what closes it: measured on torch
    2.10.0 (the version both reports carry), 4-8 threads entering the dynamo/inductor cycle
    together failed 19/20 runs without this and 0/20 with it.

    Honest limit: this makes the first import single-threaded IN PRACTICE by getting there
    first. It cannot make a genuinely concurrent first import safe -- CPython's per-module lock
    is the thing being contended, and third-party code reaches dynamo without passing through
    here. Never fatal: a host with no torch, or a dynamo that legitimately fails to import,
    reports False and leaves callers on their existing eager fallbacks."""
    global _dynamo_done
    if _dynamo_done:
        return True
    with _dynamo_lock:
        if _dynamo_done:
            return True
        try:
            import torch  # noqa: PLC0415
            import torch._dynamo  # noqa: PLC0415
            import torch._dynamo.utils  # noqa: F401, PLC0415

            # By ATTRIBUTE, not just by import: a submodule already in sys.modules is returned
            # by `import` without being bound on its parent, which is the broken state itself.
            # torch's own compile stack reads it this way (_functorch/aot_autograd.py).
            if getattr(torch._dynamo, "utils", None) is None:
                return False
        except Exception as exc:  # noqa: BLE001 -- no torch, or a dynamo that cannot import
            logger.debug("torch._dynamo warm skipped: %r", exc)
            return False
        _dynamo_done = True
        return True


def close_dynamo_import_window(log) -> bool:
    """``ensure_dynamo_imported()`` plus the breadcrumb, for a caller about to import diffusers.

    Every media load path reaches diffusers, and `import diffusers` is itself a dynamo importer,
    so each one owes this call in front of its first such import. The warning is deliberately not
    a retry: measured on torch 2.10, a process that has lost this race does not recover (0 of 14
    retries resolved), and evicting the half-built package to re-import is the C-extension purge
    that ``purge_partial_import`` already refuses. What is useful is a breadcrumb, so that if
    anything below dies on dynamo the log already says the condition predates the load.

    Call sites still wrap the IMPORT of this module as well as the call: this file reaches
    ``importlib._bootstrap._ModuleLockManager``, a private CPython name, and a build lacking it
    must not take a load down. Best-effort throughout, never a new failure."""
    if ensure_dynamo_imported():
        return True
    log.warning(
        "torch._dynamo is not importable in this process; "
        "if this load fails on a dynamo import, restart Unsloth"
    )
    return False


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
    # The boundary checks below only catch a shutdown between stages. Inside one, the orchestrator constructor reaches get_device(), which takes no epoch; the scope binds it to this pass so a mid-stage shutdown discards it instead of republishing DEVICE.
    with _owning_epoch(epoch):
        for name, fn in _STAGES:
            if epoch is not None and _detection_epoch() != epoch:
                # Checked before the first stage too: start_background_warm() reads the epoch before start(), so a shutdown in that gap retires this thread while it is still scheduled, with nothing yet run.
                logger.info("torch warm stopped before %s: its lifespan ended", name)
                return
            # Only the real stage takes the epoch; a patched _STAGES entry is called bare.
            _run_stage(name, partial(fn, epoch) if fn is _warm_hardware else fn)
            if epoch is not None and _detection_epoch() != epoch:
                # Shutdown retired this lifespan's detection. Later stages build the orchestrator, which reaches get_device() and would start a fresh detection, republishing DEVICE after teardown cleared it.
                logger.info("torch warm stopped after %s: its lifespan ended", name)
                return
    _status["finished"] = True
    _status["seconds"] = round(time.perf_counter() - started, 3)
    logger.info("torch warm finished in %.1fms", (time.perf_counter() - started) * 1000)


@contextmanager
def _owning_epoch(epoch: Optional[int]):
    """hardware.owning_detection_epoch(), a no-op when hardware is not importable: a --no-torch host still runs the warm and each stage reports its own absence."""
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

    Runs on every host, torch or not: stage one is hardware detection, which must not wait for a request (it feeds /api/health's chat_only). A finished thread from an earlier lifespan does not count as one already running, because reset_background_warm() declines mid-warm, so a shutdown leaves the object in place and treating that as "already started" would skip the warm over hardware state the same shutdown just cleared.
    """
    global _thread
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    global _thread_epoch
    # Epoch read before start(): the child may not run for a while, and a shutdown in that gap retires this lifespan. Reading it in the thread would adopt the post-shutdown one.
    epoch = _detection_epoch()
    with _start_lock:
        target, args = _warm, (epoch,)
        if _thread is not None:
            # A warm holds the latch while its own lifespan is current, so repeat calls are no-ops. Once shutdown retires that epoch the next lifespan warms again.
            if _thread_epoch is not None and epoch == _thread_epoch:
                return False
            if _thread.is_alive():
                # Stale but mid-stage: it stops at the next boundary and nothing retries, so this lifespan would serve cold. Hand off; the successor joins it first, so only one thread imports.
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

    The same app can start twice (repeated ASGI lifespans, an embedded restart) and shutdown clears the hardware state the first warm produced, so leaving the finished thread in place would make the second lifespan skip the warm and hand detection back to the first request, the stall this module removes.

    Declines while the previous warm runs, so two warms never share the same imports. Detection self-heals then: shutdown clears DETECTION_COMPLETE and /api/health kicks start_background_detection().
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


DIFFUSERS_PREWARM_DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_DIFFUSERS_PREWARM"

# The catalog's own task identifiers, which _build_index compares with ==. Anything else
# (a friendly "image"/"video") silently builds an empty index and reads as "no models here",
# so the gate would refuse forever. Pinned against the catalog by test_diffusers_prewarm.py.
_VIDEO_TASK = "text-to-video"
_MEDIA_PREWARM_TASKS = ("text-to-image", _VIDEO_TASK)

_diffusers_prewarm_lock = threading.Lock()
_diffusers_prewarmed = False


def _a_local_model_would_load_through_diffusers() -> bool:
    """Whether any indexed media model would actually load through DIFFUSERS on this host.

    Presence alone is the wrong question. A CPU or MPS host with a runnable native binary, or
    any host with ``UNSLOTH_DIFFUSION_ENGINE=sd_cpp``, routes a supported GGUF to sd.cpp, which
    imports no diffusers at all -- so prewarming there would add ~316 MB to exactly the
    low-memory installs that can least afford it, and it would never be reclaimed by a load.

    ``predict_engine`` is the same predicate selection and the download planner use, and it is
    documented to activate nothing and install nothing, so asking it here cannot perturb a
    resident model. A non-GGUF pick short-circuits it: only a GGUF can go native.

    The family comes from ``detected_image_family``, the resolver the listing and locality
    routes already share, rather than ``detect_family`` on the id: a local GGUF can carry its
    family only in the FILENAME (``/models/custom/model.gguf`` holding ``z-image``), which
    ``detect_family`` cannot see, and treating that as unknown would prewarm on exactly the
    sd.cpp host this gate exists to spare."""
    from core.inference.diffusion_engine_router import (  # noqa: PLC0415
        ENGINE_DIFFUSERS,
        predict_engine,
    )
    from core.inference.media_locality import detected_image_family  # noqa: PLC0415
    from core.inference.media_model_index import (  # noqa: PLC0415
        available_media_model_ids,
        resolve_local_media_model,
    )

    # Scope limit, deliberate: the media index is keyed on current_account_id(), and this runs on
    # a boot thread with no bound account, so it answers for the owner. On a multi-user install
    # where only a non-owner has an image model, the gate says no and that user's first load
    # still pays the import. Not worth fixing here: building an index per configured account
    # means a filesystem scan per account on the boot thread, which is the cost this gate exists
    # to avoid, and the failure mode is only the absence of a speedup, exactly as today.
    for task in _MEDIA_PREWARM_TASKS:
        for model_id in available_media_model_ids(task):
            pick = resolve_local_media_model(model_id, task = task)
            if pick is None:
                continue
            kind = pick.model_kind or ("gguf" if pick.gguf_filename else None)
            if kind != "gguf":
                return True  # only a GGUF can go native, by either backend
            if task == _VIDEO_TASK:
                # The video backend has its own native path and its own family type, so the
                # image resolver and the image router cannot answer for it: an H3 GGUF returns
                # from _run_load_h3_native before video.py's own `import diffusers`, while every
                # other video load reaches it.
                if _is_native_video_pick(pick):
                    continue
                return True
            family = detected_image_family(pick)
            if family is None:
                return True  # unknown family: diffusers is where the load would land
            if predict_engine(family, model_kind = "gguf") == ENGINE_DIFFUSERS:
                return True
    return False


def _is_native_video_pick(pick) -> bool:
    """Whether *pick* is the one video combination that never imports diffusers.

    ``VideoBackend.load_pipeline`` asks ``is_h3_native(fam, kind)`` and returns through
    ``_run_load_h3_native`` before its own ``import diffusers``; everything else falls through
    to it. Asking the video backend's own predicate keeps this from drifting away from it."""
    from core.inference.video_families import detect_video_family  # noqa: PLC0415
    from core.inference.video_minimax_h3 import is_h3_native  # noqa: PLC0415

    gguf = getattr(pick, "gguf_filename", None)
    for base in (pick.model_path, pick.model_id):
        if not base:
            continue
        # Repo id first, then repo id + picked filename, which is the order and the pair
        # video.py's own _detect_load_family uses: a local directory or a generically named repo
        # often carries the family token only in the checkpoint filename.
        for needle in (base, f"{base}/{gguf}" if gguf else None):
            if not needle:
                continue
            try:
                family = detect_video_family(needle)
            except Exception:  # noqa: BLE001 -- a probe failure must not decide "native"
                continue
            if family is not None:
                return bool(is_h3_native(family, "gguf"))
    # Not covered on purpose: _detect_load_family also reads general.architecture out of a
    # renamed GGUF's header. That is file IO on a boot thread to save memory, and guessing wrong
    # in this direction only costs the prewarm, so an unidentifiable GGUF falls through to the
    # safe default and prewarms.
    return False


def prewarm_diffusers_if_image_models_exist() -> bool:
    """Import diffusers off the first image load. True iff this call did the import.

    Measured on this stack, the first diffusion load pays roughly 5.3s of pure import before it
    touches a weight: ``diffusers`` 1.6s, ``diffusers.hooks`` 2.4s and the pipeline classes 1.3s,
    for about 316 MB. None of it depends on which model was picked, so it is the same cost every
    first load in a fresh process, and all of it can be paid earlier by a thread nobody is
    waiting on.

    Gated on the install actually having a local image or video model, which is the whole point:
    a chat-only or training-only user never pays the 316 MB. The gate itself is stdlib only (it
    does not import torch or diffusers) and its index is cached and needed by the Images page
    anyway, so building it here is work moved earlier rather than work added.

    Called from the POST-warm worker, after ``join_background_warm()``, so it cannot delay any
    coordinated warm stage or the socket bind. Concurrency was measured rather than assumed: 4 to
    8 threads importing diffusers submodules together failed 0 of 16 trials, with and without
    dynamo already imported, because these are ordinary package imports that CPython's per module
    lock serialises, unlike the dynamo/inductor cycle in #10350.

    Never fatal, and opt out with ``UNSLOTH_STUDIO_DISABLE_DIFFUSERS_PREWARM=1``."""
    global _diffusers_prewarmed
    if _diffusers_prewarmed:
        return False
    if os.environ.get(DIFFUSERS_PREWARM_DISABLE_ENV_VAR) == "1":
        return False
    # The torch opt-out covers this too. start_background_warm() declines under DISABLE_ENV_VAR,
    # but join_background_warm() reports True when no worker ever ran, so the post-warm thread
    # arrives here regardless; importing diffusers imports torch, which is precisely what that
    # variable exists to prevent. Checked here rather than at the call site so every caller gets
    # it, and so the warm-window tests that assert no unsolicited torch import keep holding.
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    with _diffusers_prewarm_lock:
        if _diffusers_prewarmed:
            return False
        try:
            if not _a_local_model_would_load_through_diffusers():
                # Nothing diffusers would serve, so the import is pure cost. Not latched: a
                # model downloaded later should let the next lifespan reconsider.
                logger.debug("diffusers prewarm skipped: no local model routes to diffusers")
                return False
        except Exception as exc:  # noqa: BLE001 -- a gate that cannot answer means skip, not crash
            logger.debug("diffusers prewarm gate unavailable: %r", exc)
            return False

        try:
            # core.inference.diffusion installs these at module scope, above its own lazy
            # `import diffusers`, because on Windows ROCm diffusers reaches xformers and torchao
            # and both land on an absent distributed backend. This prewarm can be the first
            # importer in the process, so it owes the same three installs; they are idempotent.
            from core._torchao_stub import (  # noqa: PLC0415
                install_torchao_windows_rocm_stub,
                install_xformers_windows_rocm_stub,
            )
            from core.inference.diffusion_torchao_patches import (  # noqa: PLC0415
                install_torchao_int_mm_patch,
            )

            install_xformers_windows_rocm_stub()
            install_torchao_windows_rocm_stub()
            install_torchao_int_mm_patch()
        except Exception as exc:  # noqa: BLE001 -- importing unprotected is the hazard; skip
            logger.debug("diffusers prewarm skipped: stubs unavailable: %r", exc)
            return False

        started = time.perf_counter()
        # One lock at a time, never both. Each scope holds that package's own lock across its
        # import AND its cleanup, the way _held_import_lock does for the bare warm stages,
        # because releasing between the two is the bug: CPython drops the lock the moment the
        # import raises, so a request already waiting on it wakes up, re-imports against the
        # submodules that import left behind, and republishes the malformed package -- at which
        # point purge_partial_import declines, since those leftovers now belong to a live
        # importer. Reentrant per thread, so the nested acquire inside the purge is free.
        #
        # NOT nested, which is the part that is easy to get backwards. `import diffusers.hooks`
        # makes CPython take the CHILD lock first and import the parent from inside it
        # (_find_and_load -> _ModuleLockManager(name) -> _find_and_load_unlocked -> import
        # parent). Holding parent-then-child here would invert that against any concurrent
        # `from diffusers.hooks import ...` and produce a lock cycle, which surfaces as the
        # _DeadlockError that _lock_unlock_module swallows -- i.e. exactly the partially
        # initialised module this whole change exists to prevent. Sequential scopes have no
        # cycle: by the time the hooks scope runs, the parent is published, so importing the
        # child acquires nothing else.
        # The try INSIDE each with, not around it: exiting the scope on the exception would
        # release the lock and the handler would reacquire it, and that gap is the whole bug.
        # A request already waiting on the lock wakes up in it, re-imports against the submodules
        # the failed import left behind, and republishes the malformed package -- at which point
        # purge_partial_import declines, because those leftovers now belong to a live importer.
        # Reentrant per thread, so the nested acquire inside the purge is free.
        with _ModuleLockManager("diffusers"):
            try:
                import diffusers  # noqa: F401, PLC0415
            except Exception as exc:  # noqa: BLE001 -- the load path imports it again and reports
                logger.debug("diffusers prewarm skipped: %r", exc)
                purge_partial_import("diffusers")
                return False

        # A separate scope, never nested inside the one above. `import diffusers.hooks` makes
        # CPython take the CHILD lock first and import the parent from inside it (_find_and_load
        # -> _ModuleLockManager(name) -> _find_and_load_unlocked -> import parent). Holding
        # parent-then-child here would invert that against a concurrent `from diffusers.hooks
        # import ...` and produce a lock cycle, which surfaces as the _DeadlockError that
        # _lock_unlock_module swallows: exactly the partially initialised module this change
        # exists to prevent. Sequentially there is no cycle, because the parent is already
        # published by the time this runs, so importing the child acquires nothing else.
        with _ModuleLockManager("diffusers.hooks"):
            try:
                import diffusers.hooks  # noqa: F401, PLC0415
            except Exception as exc:  # noqa: BLE001 -- the load path imports it again and reports
                logger.debug("diffusers prewarm skipped: %r", exc)
                # The parent stays: it imported cleanly, and purging it would be a no-op anyway
                # since it is in sys.modules and belongs to nobody. What has to go is the hook
                # submodules that did execute, or the load path's own `from diffusers.hooks
                # import ...` rebuilds an incomplete package from them (#7580, one level down).
                purge_partial_import("diffusers.hooks")
                return False

        # Outside both locks: it imports nothing under diffusers. diffusers hard-codes
        # _tqdm_active = True at import and honours no env var, so a prewarm that skipped this
        # would let "Loading pipeline components..." draw straight onto the structlog stream,
        # mid-record. Not fatal to the prewarm: the imports above already succeeded, which is the
        # work this exists to do, and the load path calls the same idempotent helper anyway.
        try:
            from loggers.config import quiet_third_party_progress_bars  # noqa: PLC0415

            quiet_third_party_progress_bars()
        except Exception as exc:  # noqa: BLE001 -- cosmetic only
            logger.debug("quieting third-party progress bars failed: %r", exc)

        _diffusers_prewarmed = True
        logger.info(
            "diffusers prewarmed in %.0fms; the first image load skips that import",
            (time.perf_counter() - started) * 1000,
        )
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
