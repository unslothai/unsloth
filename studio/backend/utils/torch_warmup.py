# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import the ML stack on a background thread while the backend finishes booting.

Started from the LAST line of main.py's lifespan: everything above is on the critical path to binding the socket and would contend for the GIL, and uvicorn binds as soon as the lifespan returns, so the warm overlaps serving rather than boot. Idempotent, never fatal (a failed stage is logged, left cold and retried by whoever needs it), and never half-initialised, since stages delegate to the module owning the cache.

This does NOT make torch-dependent endpoints cheap while it runs: anything reaching get_device() blocks until the hardware stage finishes, so `async def` handlers there must use asyncio.to_thread.
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

    When ``package/__init__.py`` raises, CPython evicts only the parent and keeps every submodule it executed, so the next import re-runs ``__init__`` with each ``from .x import y`` served from that cache: the package imports "successfully" while missing pieces (#7580).

    Acts only on that exact signature (parent gone, submodules present), so a concurrent still-running import is left alone; returns what it removed.

    Declines when any submodule is a loaded C extension: evicting one re-runs its module init, and pybind11 answers a duplicate type registration with std::terminate. Known native registries populated by pure-Python modules are reset only after every stale module has been removed and no importer has republished the parent.
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

# Hold the import lock across the import AND its cleanup, or a queued importer sees stale
# submodules in between.
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
    from core.inference import get_inference_backend

    get_inference_backend()
    # Must precede _prime_nvlink_topology: once that thread exists, the first dynamo import
    # is no longer single-threaded (#10350).
    ensure_dynamo_imported()
    _prime_nvlink_topology()


def _prime_nvlink_topology() -> Optional[threading.Thread]:
    """Build the P2P gate's interconnect matrix off the load path. Returns the thread,
    for tests to join.

    Fire and forget, or its timeouts delay every stage behind it. Success-only: a miss cached
    this early keeps P2P off for the life of the process (#10613)."""

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

    ``_dynamo`` is a LAZY submodule, so ``torch._dynamo.X`` hands back a still-initialising
    module: ``.config`` binds early and ``.utils`` late, and a read in between raises
    ``partially initialized module ... has no attribute 'utils'`` (#10350, #10963). Ordinary
    loads open that window, not torch.compile: ``diffusers.hooks`` evaluates
    ``@torch.compiler.disable()`` at class-body time. Wins only by getting there first."""
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

    `import diffusers` is itself a dynamo importer, so every media load path owes this call in
    front of its first one. A warning, not a retry: a process that lost the race does not
    recover. Wrap the IMPORT of this module too, since it reaches a private CPython name."""
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
    # These checks catch only a shutdown BETWEEN stages; the scope binds the epoch so a
    # mid-stage shutdown discards this pass rather than republishing DEVICE.
    with _owning_epoch(epoch):
        for name, fn in _STAGES:
            if epoch is not None and _detection_epoch() != epoch:
                # Before the first stage too: a shutdown between the epoch read and start().
                logger.info("torch warm stopped before %s: its lifespan ended", name)
                return
            # Only the real stage takes the epoch; a patched _STAGES entry is called bare.
            _run_stage(name, partial(fn, epoch) if fn is _warm_hardware else fn)
            if epoch is not None and _detection_epoch() != epoch:
                # Later stages reach get_device(), republishing DEVICE after teardown.
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

    Runs on every host, torch or not: stage one is hardware detection, which feeds /api/health's chat_only. A FINISHED thread from an earlier lifespan does not count as one already running: reset_background_warm() declines mid-warm, so a shutdown leaves the object in place and treating that as "already started" skips the warm over hardware state the same shutdown cleared.
    """
    global _thread
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    global _thread_epoch
    # Epoch read before start(): reading it in the child would adopt the post-shutdown one.
    epoch = _detection_epoch()
    with _start_lock:
        target, args = _warm, (epoch,)
        if _thread is not None:
            if _thread_epoch is not None and epoch == _thread_epoch:
                return False
            if _thread.is_alive():
                # Stale but mid-stage: nothing retries it, so hand off to a successor that
                # joins it first, keeping one importer.
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

    The same app can start twice, and shutdown clears the hardware state the first warm produced, so leaving the finished thread in place hands detection back to the first request, which is the stall this module removes.

    Declines while the previous warm runs, so two warms never share the same imports; detection self-heals then, because /api/health kicks start_background_detection().
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

    Presence alone is the wrong question: a CPU or MPS host with a native binary, or
    ``UNSLOTH_DIFFUSION_ENGINE=sd_cpp``, routes a supported GGUF to sd.cpp and imports no
    diffusers. Family detection is pick-aware because a local GGUF can name it only in the
    FILENAME."""
    from core.inference.diffusion_engine_router import (  # noqa: PLC0415
        ENGINE_DIFFUSERS,
        predict_engine,
    )
    from core.inference.media_locality import detected_image_family  # noqa: PLC0415
    from core.inference.media_model_index import (  # noqa: PLC0415
        available_media_model_ids,
        resolve_local_media_model,
    )

    # Deliberate scope limit: the media index is keyed on current_account_id() and this boot
    # thread has none, so it answers for the owner; the failure mode is only no speedup.
    for task in _MEDIA_PREWARM_TASKS:
        for model_id in available_media_model_ids(task):
            pick = resolve_local_media_model(model_id, task = task)
            if pick is None:
                continue
            kind = pick.model_kind or ("gguf" if pick.gguf_filename else None)
            if kind != "gguf":
                return True  # only a GGUF can go native, by either backend
            if task == _VIDEO_TASK:
                # The image resolver cannot answer for video; see _is_native_video_pick.
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

    ``VideoBackend.load_pipeline`` returns through ``_run_load_h3_native`` before its own
    ``import diffusers``; every other video load reaches that import."""
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
    # renamed GGUF's header, which is file IO on a boot thread.
    return False


def prewarm_diffusers_if_image_models_exist() -> bool:
    """Import diffusers off the first image load. True iff this call did the import.

    Gated on the install having a local image or video model, so a chat-only or a
    training-only user never pays it. The gate itself is stdlib only.

    Called from the POST-warm worker, after ``join_background_warm()``, so it cannot delay a
    warm stage or the socket bind. Concurrent submodule imports are safe here, unlike the
    dynamo cycle in #10350: CPython's per-module lock serialises ordinary imports.

    Never fatal, and opt out with ``UNSLOTH_STUDIO_DISABLE_DIFFUSERS_PREWARM=1``."""
    global _diffusers_prewarmed
    if _diffusers_prewarmed:
        return False
    if os.environ.get(DIFFUSERS_PREWARM_DISABLE_ENV_VAR) == "1":
        return False
    # join_background_warm() reports True when no worker ran, so the post-warm thread arrives
    # here even under DISABLE_ENV_VAR, and diffusers imports torch.
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    with _diffusers_prewarm_lock:
        if _diffusers_prewarmed:
            return False
        try:
            if not _a_local_model_would_load_through_diffusers():
                # Not latched: a model downloaded later lets the next lifespan reconsider.
                logger.debug("diffusers prewarm skipped: no local model routes to diffusers")
                return False
        except Exception as exc:  # noqa: BLE001 -- a gate that cannot answer means skip, not crash
            logger.debug("diffusers prewarm gate unavailable: %r", exc)
            return False

        try:
            # On Windows ROCm diffusers reaches xformers and torchao, both landing on an
            # absent distributed backend, so any first importer owes these stubs.
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
        # The try goes INSIDE each `with`: leaving the scope first frees the lock for a waiter
        # to republish the malformed package before the purge runs.
        with _ModuleLockManager("diffusers"):
            try:
                import diffusers  # noqa: F401, PLC0415
            except Exception as exc:  # noqa: BLE001 -- the load path imports it again and reports
                logger.debug("diffusers prewarm skipped: %r", exc)
                purge_partial_import("diffusers")
                return False

        # A separate scope, NEVER nested in the one above: CPython takes the CHILD lock first
        # here, so parent-then-child would invert that against a concurrent importer.
        with _ModuleLockManager("diffusers.hooks"):
            try:
                import diffusers.hooks  # noqa: F401, PLC0415
            except Exception as exc:  # noqa: BLE001 -- the load path imports it again and reports
                logger.debug("diffusers prewarm skipped: %r", exc)
                # The parent stays; the hook submodules that ran must go, or a later
                # `from diffusers.hooks import ...` rebuilds from them (#7580).
                purge_partial_import("diffusers.hooks")
                return False

        # Outside both locks: it imports nothing under diffusers. diffusers hard-codes
        # diffusers hard-codes _tqdm_active = True and honours no env var, so without this
        # its bars draw onto the structlog stream mid-record.
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
