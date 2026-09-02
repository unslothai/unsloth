# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Hardware detection — run once at startup, read everywhere.

Usage:
    # At FastAPI lifespan startup:
    from utils.hardware import detect_hardware
    detect_hardware()

    # Anywhere else:
    from utils.hardware import DEVICE, DeviceType, is_apple_silicon
    if DEVICE == DeviceType.CUDA:
        import torch
        ...
"""

import copy
import gc
import glob
import os
import platform
import re
import subprocess
import sys
import threading
import types
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
import structlog
from loggers import get_logger
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

logger = get_logger(__name__)


# ── GPU index ordering ──────────────────────────────────────────────────────
# CUDA defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST, numbering GPUs by compute
# performance. nvidia-smi -- and every free-VRAM probe in Unsloth -- numbers GPUs
# by PCI bus id instead. On a mixed-GPU host (e.g. an RTX 5090 alongside an RTX
# PRO 6000) the two orderings disagree, so an index picked from nvidia-smi data
# ("the emptiest card is GPU 1") gets written into CUDA_VISIBLE_DEVICES and then
# reinterpreted by CUDA against FASTEST_FIRST -- landing the model on a different
# physical GPU than the one selected. Pinning PCI_BUS_ID makes torch, nvidia-smi,
# and CUDA_VISIBLE_DEVICES share a single index space, matching what users see in
# `nvidia-smi -L`. Set at import (before any torch.cuda call latches the order
# at context creation) and inherited by child processes, since the llama-server
# and spawn workers copy os.environ. setdefault so an explicit user override wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Unsloth workers can import MLX without importing unsloth first, so mirror the
# package bootstrap here. Keep an explicit user value authoritative.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")


# ========== Device Enum ==========


class DeviceType(str, Enum):
    """Supported compute backends. str subclass for clean JSON serialization."""

    CUDA = "cuda"
    XPU = "xpu"
    MLX = "mlx"
    CPU = "cpu"


# ========== Global State (set once by detect_hardware) ==========

DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True  # No CUDA GPU -> GGUF chat only (Mac, CPU-only, etc.)
# Why CHAT_ONLY is True (Train/Export disabled). None when training is enabled.
# "mlx_unavailable": Apple Silicon but the MLX stack is missing, too old, or broken
# (the usual cause of "Train/Export greyed out" on Macs after a reinstall dropped MLX);
# "intel_mac": Intel Mac (no PyTorch/MLX); "no_gpu": CPU-only non-Mac host.
CHAT_ONLY_REASON: Optional[str] = None
# What exactly blocked the reason above, when there is something specific to say. Only
# "mlx_unavailable" sets it today: the gate is all-or-nothing across mlx, mlx-lm and
# mlx-vlm, so "run `unsloth studio update`" was the whole message even to someone who
# had just run it. Naming the package that is missing, too old, or refusing to import
# is the difference between a dead end and a fix. Never shown on its own.
CHAT_ONLY_DETAIL: Optional[str] = None
IS_ROCM: bool = False  # True when running on AMD ROCm (HIP) -- routes GPU monitoring to amd.py

# Detection has concurrent callers (the warm thread plus any early get_device()).
# Unlocked, two runs interleave on the globals above and a reader between the reset and
# the CUDA branch sees "chat only" on a GPU host. Re-entrant: get_device() nests.
_DETECT_LOCK = threading.RLock()

# Bumped by shutdown so a detection still inside the torch import cannot publish over
# the reset, leaving the next lifespan a non-None DEVICE that makes it skip detection.
# An epoch, not _DETECT_LOCK: taking that lock would park teardown behind the import.
_EPOCH_LOCK = threading.Lock()
DETECTION_EPOCH = 0


def invalidate_detection() -> int:
    """Retire any detection in flight. Returns the new epoch."""
    global DETECTION_EPOCH
    with _EPOCH_LOCK:
        DETECTION_EPOCH += 1
        return DETECTION_EPOCH


def current_detection_epoch() -> int:
    with _EPOCH_LOCK:
        return DETECTION_EPOCH


# The epoch nested detections on this thread belong to. get_device() takes no epoch and
# the warm builds the orchestrator, whose constructor reaches it; without this a shutdown
# mid-stage lets the nested pass republish DEVICE over the teardown, so the next lifespan
# skips detection. Thread-local, so only the warm's own call stack is bound.
_OWNING_EPOCH = threading.local()


@contextmanager
def owning_detection_epoch(epoch: Optional[int]):
    """Bind epoch-less detections on this thread to ``epoch`` for the block. Nested, not
    assigned: restoring the previous value keeps concurrent scopes on other threads apart."""
    previous = getattr(_OWNING_EPOCH, "value", None)
    _OWNING_EPOCH.value = epoch
    try:
        yield
    finally:
        _OWNING_EPOCH.value = previous


def _discard_detection_locked() -> None:
    """Drop a verdict produced for an epoch that has been retired."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM
    DEVICE = None
    CHAT_ONLY = True
    CHAT_ONLY_REASON = None
    CHAT_ONLY_DETAIL = None
    IS_ROCM = False
    DETECTION_COMPLETE.clear()


# Set once detection has a settled answer, including its CPU/chat-only fallback. Poll
# this, not DEVICE: DEVICE is assigned mid-detection and can be revised.
DETECTION_COMPLETE = threading.Event()
# Bumped every time detection settles. Detection is not once-per-process (the MLX
# self-heal re-detects and flips CHAT_ONLY), so snapshot holders can spot staleness.
DETECTION_GENERATION = 0

# Drives start_background_detection(). Separate from _DETECT_LOCK: never held across the import.
_DETECT_KICK_LOCK = threading.Lock()
_DETECT_THREAD: Optional[threading.Thread] = None


def start_background_detection() -> None:
    """Run detection on a daemon thread if nothing is running it yet.

    For callers on a deadline that cannot await ensure_hardware_detected(), such as
    /api/health under the launcher's 2s timeout. They poll DEVICE against their own budget;
    this guarantees someone is filling it in even when the warm is past its hardware stage
    or a shutdown cleared the verdict. Callers skip it under
    UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1, which means no background import at all.

    At most one thread, and none once DEVICE is set, so a route polling "still detecting"
    cannot pile them up. Not the asyncio executor: a to_thread outliving its awaiter holds
    a slot, and a polled endpoint would exhaust the pool during a slow import.
    """
    global _DETECT_THREAD
    if DEVICE is not None:
        return
    with _DETECT_KICK_LOCK:
        if DEVICE is not None:
            return
        if _DETECT_THREAD is not None and _DETECT_THREAD.is_alive():
            return
        # Epoch read before start(): the thread can be scheduled after a shutdown
        # retires this epoch, and it must lose to that, not adopt it.
        _DETECT_THREAD = threading.Thread(
            target = ensure_hardware_detected,
            args = (current_detection_epoch(),),
            daemon = True,
            name = "hardware-detect",
        )
        _DETECT_THREAD.start()


def _backend_label(device: DeviceType) -> str:
    """Return the user-facing backend name for API responses.

    ROCm hosts stay ``DeviceType.CUDA`` internally (ROCm reuses ``torch.cuda.*``),
    but "cuda" is misleading in JSON, so swap to ``"rocm"`` when ``IS_ROCM`` is set.
    """
    if IS_ROCM and device == DeviceType.CUDA:
        return "rocm"
    return device.value


# ========== Detection ==========


def is_apple_silicon() -> bool:
    """True on Apple Silicon (pure platform check, no ML imports)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# Set by _has_torch() when torch is installed but its import blew up (unresolved CUDA
# libs). The CPU fallback reports that as a detection failure, not as "no GPU".
TORCH_IMPORT_ERROR: Optional[str] = None


def _has_torch() -> bool:
    """True if PyTorch is importable.

    Any failure counts as "no torch", not just ImportError: ensure_hardware_detected()
    re-runs while DEVICE is None, so an escaping OSError would make every request retry the
    import. Take the CPU path, but record the error, or the host is told to install the
    torch it already has.
    """
    global TORCH_IMPORT_ERROR
    try:
        import torch
        TORCH_IMPORT_ERROR = None
        return True
    except Exception as exc:
        # ImportError does NOT mean "not installed": a wheel with unresolved native libs
        # raises it from torch's own __init__ (OSError on Windows). Only ModuleNotFoundError
        # naming torch itself is absent; a failed submodule is a broken install. Both purge.
        absent = isinstance(exc, ModuleNotFoundError) and exc.name == "torch"
        TORCH_IMPORT_ERROR = None if absent else repr(exc)
        if TORCH_IMPORT_ERROR is not None:
            logger.error("torch is installed but failed to import: %r", exc)
        # A part-way failure leaves submodules cached under an evicted parent, so the next
        # importer gets a torch missing pieces. purge_partial_import() clears that.
        try:
            from utils.torch_warmup import purge_partial_import
        except Exception:
            # Also exec'd standalone (tests/python/test_e2e_no_torch_sandbox.py); nothing to purge.
            pass
        else:
            purge_partial_import("torch")
        return False


def _torch_mps_available() -> bool:
    """True when torch exposes a usable Metal (MPS) device.

    Apple Silicon alone is not enough: a torch built without MPS, or one that fails to import,
    leaves the pipelines nowhere to run. Never raises; a failed probe reads as no MPS.
    """
    if not _has_torch():
        return False
    try:
        import torch
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def _has_mlx() -> bool:
    """True if MLX is importable."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


# What the last gate call measured, so the CPU fallback can name a blocker without
# running the mlx imports a second time. This module already treats those imports as
# able to park indefinitely on a broken stack, and that is exactly the host that needs
# the detail, so a second run there can double detection latency or keep the pass from
# reaching the repair scheduler. Written and consumed inside one locked detection pass.
_MLX_BLOCKERS_MEASURED: Optional[list[str]] = None


def _has_usable_mlx_stack() -> bool:
    """True only when the FULL Unsloth MLX training/export stack is usable
    (mlx + mlx-lm + mlx-vlm at the minimum versions unsloth-zoo requires), not
    just a bare ``import mlx.core``. A backtracked/old mlx-vlm still imports but
    breaks VLM Train/Export, so the training gate must match the self-heal's own
    criterion (utils.mlx_repair) -- otherwise detect_hardware would enable
    Train/Export on exactly the inadequate stack the MLX self-heal is trying to
    repair, leaving the user with greyed-in-but-broken buttons.

    Asked as "no blockers" rather than through mlx_stack_available(), which is the
    same question: both run the version checks before the imports, in the same order,
    and stop at the first failure. Reading the list is what lets the answer be
    explained without measuring it again."""
    global _MLX_BLOCKERS_MEASURED
    _MLX_BLOCKERS_MEASURED = None
    try:
        from utils.mlx_repair import mlx_stack_blockers
        blockers = mlx_stack_blockers()
    except Exception as exc:
        # mlx_repair should always import; if it somehow cannot, fall back to the
        # bare import check rather than forcing a working host into chat-only.
        logger.debug("MLX stack availability check failed, using bare import: %s", exc)
        return _has_mlx()
    _MLX_BLOCKERS_MEASURED = blockers
    return not blockers


def _mlx_stack_detail() -> Optional[str]:
    """One line naming what the MLX gate is unhappy about, or None if it cannot tell.

    Never raises and never re-runs the gate's own verdict: this only describes a
    verdict already reached, so a failure here costs a sentence, not Train.

    Takes what the gate measured when there is any. Measuring again is the fallback
    for a caller that reached here without one, e.g. a test driving this alone.
    """
    global _MLX_BLOCKERS_MEASURED
    blockers = _MLX_BLOCKERS_MEASURED
    # Consumed, not kept: a list left behind by an earlier pass describes a stack that
    # has since been re-measured, and the whole point of the detail is that it belongs
    # to the verdict beside it.
    _MLX_BLOCKERS_MEASURED = None
    if blockers is None:
        try:
            from utils.mlx_repair import mlx_stack_blockers
            blockers = mlx_stack_blockers()
        except Exception as exc:
            logger.debug("MLX blocker detail unavailable: %s", exc)
            return None
    if not blockers:
        # The gate said no and the detail says yes, which means the stack changed
        # under us. Saying nothing beats naming a blocker that is no longer there.
        return None
    return "; ".join(blockers[:3])


def verdict_pending_mlx_repair(chat_only: bool, reason: Optional[str]) -> bool:
    """True when this settled verdict is one the MLX self-heal is about to overturn.

    Detection gets its answer before utils.mlx_repair gets its turn, so an Apple Silicon
    host whose MLX stack is missing or unreadable settles chat-only first and flips only
    once the background reinstall lands. Published as final, that greys Train behind a
    "run `unsloth studio update`" tooltip the repair makes wrong a minute later, and the row
    then enables itself on the frontend's recovery poll -- the reported "greyed out, then
    they come out". Callers report it as still-detecting instead. Video is unaffected either
    way: it runs on Metal without MLX and reads its own capability verdict.

    The "mlx_unavailable" check is also what lets mlx_repair_in_flight() be cheap: that
    reason means this pass has just measured the stack as unusable, so the self-heal only
    has to report whether it has finished, not re-probe whether it is needed.

    Takes the verdict as arguments rather than reading the globals, so a caller that
    already holds a consistent snapshot does not re-read them mid-pass."""
    if not chat_only or reason != "mlx_unavailable":
        return False
    if not is_apple_silicon():
        return False
    try:
        from utils.mlx_repair import mlx_repair_in_flight
        return mlx_repair_in_flight()
    except Exception as exc:
        # A self-heal we cannot even ask about is one that cannot be relied on, so let the
        # verdict settle rather than hold Train and Video spinning for the whole session.
        logger.debug("MLX repair progress check failed, treating the verdict as final: %s", exc)
        return False


def verdict_blames_the_mlx_stack() -> bool:
    """Unlocked deliberately: _DETECT_LOCK spans a whole detection pass, imports included, so
    taking it would park the post-warm worker behind an early request's first import. The
    overturn re-reads under the lock, so a straddling read costs one needless measurement."""
    return bool(CHAT_ONLY) and CHAT_ONLY_REASON == "mlx_unavailable"


def overturn_the_mlx_verdict(epoch: Optional[int] = None) -> bool:
    """For a caller that has just measured the stack as usable.

    Read and re-detect share one locked section, or a forced pass landing between them loses
    its answer to this one. ``epoch`` predates the measurement, so a shutdown since discards
    the pass instead of republishing for a dead lifespan. True is /api/health's three-part
    settled read, not "a re-detect ran": callers announce it, and shutdown clears DEVICE
    before the event and the verdict."""
    with _DETECT_LOCK:
        if not CHAT_ONLY or CHAT_ONLY_REASON != "mlx_unavailable":
            return False
        with owning_detection_epoch(epoch):
            detect_hardware()
        return DEVICE is not None and DETECTION_COMPLETE.is_set() and not CHAT_ONLY


def _print_cuda_device_list(is_rocm: bool) -> None:
    """List every visible CUDA/ROCm GPU with its index at startup.

    The "Hardware detected" banner names only device 0, which hides the other
    cards on a multi-GPU host. This lists the full visible set in CUDA-ordinal
    order, matching `nvidia-smi -L` when no CUDA_VISIBLE_DEVICES mask is set
    (under a mask the indices are visible ordinals, not physical PCI ids).
    CUDA_DEVICE_ORDER governs only CUDA, so it is shown for CUDA but not ROCm.
    No-ops on single-GPU hosts and never raises -- it is purely informational.
    """
    try:
        import torch

        count = torch.cuda.device_count()
        if count <= 1:
            return
        if is_rocm:
            header = f"ROCm devices ({count}):"
        else:
            order = os.environ.get("CUDA_DEVICE_ORDER", "default")
            header = f"CUDA devices ({count}, CUDA_DEVICE_ORDER={order}):"
        lines = [header]
        for i in range(count):
            try:
                name = torch.cuda.get_device_properties(i).name
            except Exception as e:
                logger.debug("CUDA device %d property probe failed: %s", i, e)
                name = "<unavailable>"
            lines.append(f"  [{i}] {name}")
        print("\n".join(lines))
    except Exception:
        return  # purely informational; never disrupt startup


def detect_hardware() -> DeviceType:
    """
    Detect the best compute device and set the module-level DEVICE global.

    Call once at FastAPI lifespan startup; idempotent.

    Detection order:
      1. XPU-preferred hint: only on an unambiguous "prefer XPU" signal
         (CUDA hidden via ``CUDA_VISIBLE_DEVICES="" / "-1"``,
         ``UNSLOTH_FORCE_XPU=1``, or CUDA unavailable) AND a non-empty
         ``ZE_AFFINITY_MASK`` AND ``torch.xpu`` reports a device. A stray
         inherited mask is not enough: CUDA still wins on hybrid hosts.
      2. CUDA  (NVIDIA GPU, requires torch)
      3. XPU   (Intel GPU, requires torch with XPU support)
      4. MLX   (Apple Silicon via MLX framework)
      5. CPU   (fallback)
    """
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM, DETECTION_GENERATION
    with _DETECT_LOCK:
        # A forced pass mutates the globals partway through; leaving the event set lets
        # /api/health serve that as settled, so the sidebar MLX poll caches reason=None,
        # stops, and leaves Train hidden on a repaired host. Republished once it settles.
        was_complete = DETECTION_COMPLETE.is_set()
        # Snapshot the whole verdict, not just the event: a raise mid-pass leaves a
        # half-written answer the autorepair path swallows, and losing "mlx_unavailable"
        # stops the sidebar poll for good.
        published = (DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM)
        # Owning epoch first, current only as a fallback. The MLX self-heal calls this
        # after a pip install that can outlast the lifespan; reading current would adopt
        # the epoch shutdown moved to, so the next lifespan finds DEVICE set and skips
        # its own detection.
        epoch = getattr(_OWNING_EPOCH, "value", None)
        if epoch is None:
            epoch = current_detection_epoch()
        elif current_detection_epoch() != epoch:
            # Retired before this pass began, so leave the running lifespan alone. Going on
            # would clear DETECTION_COMPLETE over a settled verdict, probe, then discard: a
            # late repair would erase the restart's verdict, not merely fail to add its own.
            return DEVICE
        DETECTION_COMPLETE.clear()
        try:
            device = _detect_hardware_locked()
        except BaseException:
            if current_detection_epoch() != epoch:
                # Shutdown ran mid-pass. Restoring would put back the verdict it just
                # cleared, and the next lifespan would treat that as measured.
                _discard_detection_locked()
                raise
            DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM = published
            # Restore rather than leave it clear: start_background_detection() declines
            # once DEVICE is set, so an unset event keeps health provisional forever.
            if was_complete:
                DETECTION_COMPLETE.set()
            raise
        if current_detection_epoch() != epoch:
            # Shutdown ran mid-pass; this verdict belongs to a lifespan that ended.
            _discard_detection_locked()
            return device
        DETECTION_GENERATION += 1
        DETECTION_COMPLETE.set()
        return device


def ensure_hardware_detected(epoch: Optional[int] = None) -> DeviceType:
    """Detect once, from any thread. Prefer this to detect_hardware() unless you want a
    forced re-detect: it collapses the warm thread and an early request into one pass, and
    a caller arriving mid-detection waits rather than starts a second.

    Never raises. A raise on the warm thread is swallowed and leaves DEVICE None, so every
    later request retries the failing import and /api/health, which waits on this, would
    500. Record CPU + chat-only with a reason instead.

    ``epoch`` is the epoch this pass belongs to; a spawner passes the one it read before
    Thread.start(), since the thread can be scheduled after a shutdown retired it and
    reading it here would bind the pass to the retirement it must lose to. Direct callers
    pass nothing and own the current epoch."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, DETECTION_GENERATION
    with _DETECT_LOCK:
        if epoch is None:
            # A nested read inside an owning scope belongs to that pass, not to whatever
            # is current by the time it reaches here. See owning_detection_epoch().
            epoch = getattr(_OWNING_EPOCH, "value", None)
        if epoch is None:
            epoch = current_detection_epoch()
        elif DEVICE is None and current_detection_epoch() != epoch:
            # Retired before this worker reached the lock. Probing would import torch for a
            # stopped lifespan and the next warm would queue behind it only to re-detect.
            # Mid-probe shutdown: below.
            return DEVICE
        produced_here = DEVICE is None
        if produced_here:
            # Same reason detect_hardware() clears it: the pass below assigns DEVICE and
            # CHAT_ONLY before probes that can still fall back to CPU. A stale set event
            # (shutdown cleared DEVICE while a cached waiter went on to set it) would
            # publish the XPU candidate as settled. Republished below once final.
            DETECTION_COMPLETE.clear()
            try:
                _detect_hardware_locked()
            except BaseException as exc:  # noqa: BLE001 - degrade, never 500 the health check
                logger.error("Hardware detection failed; falling back to CPU: %r", exc)
                DEVICE = DeviceType.CPU
                CHAT_ONLY = True
                CHAT_ONLY_REASON = "detection_failed"
                # The pass may have got as far as recording one for a different reason.
                CHAT_ONLY_DETAIL = None
            # Inside the branch: the orchestrator rebuilds its curated defaults whenever this
            # counter moves, so bumping on the cached path caused needless rebuilds. Forced
            # detect_hardware() bumps it too.
            DETECTION_GENERATION += 1
        if produced_here and current_detection_epoch() != epoch:
            # See detect_hardware(): a retired pass must not publish -- but only what this
            # call produced. A retired worker that waited out the lock and found DEVICE set
            # is looking at the new lifespan's verdict; discarding it would leave the
            # restart provisional.
            _discard_detection_locked()
            return DEVICE
        # Set only here, where a final value is guaranteed: a non-None DEVICE means only that
        # a candidate was picked (the XPU branch assigns before a probe that can raise), so a
        # waiter trusting it could publish training-enabled for a CPU/chat-only host.
        # Unconditional, unlike the counter: re-setting is a no-op and a late waiter needs it.
        DETECTION_COMPLETE.set()
        return DEVICE


def _detect_hardware_locked() -> DeviceType:
    """detect_hardware() body. Call only with _DETECT_LOCK held."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM
    global _MLX_BLOCKERS_MEASURED
    CHAT_ONLY = True  # reset -- only CUDA/ROCm/XPU/MLX sets it to False
    CHAT_ONLY_REASON = None
    CHAT_ONLY_DETAIL = None
    _MLX_BLOCKERS_MEASURED = None
    IS_ROCM = False

    # Probe torch once per pass: a failed probe is expensive and a second can disagree.
    torch_ok = _has_torch()

    # --- CUDA / ROCm / XPU: try PyTorch ---
    if torch_ok:
        import torch

        # --- Explicit-XPU hint ---
        # Prefer XPU on UNSLOTH_FORCE_XPU=1, or ZE_AFFINITY_MASK set + CUDA
        # hidden/unavailable. A bare mask alone is NOT enough (can leak from
        # unrelated Intel tooling); torch.xpu must report a device.
        ze_mask = os.environ.get("ZE_AFFINITY_MASK")
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        cuda_hidden = cvd is not None and cvd.strip() in ("", "-1")
        force_xpu = os.environ.get("UNSLOTH_FORCE_XPU") == "1"
        try:
            cuda_unavailable = not torch.cuda.is_available()
        except Exception:
            cuda_unavailable = True

        prefer_xpu = force_xpu or (bool(ze_mask) and (cuda_hidden or cuda_unavailable))
        if prefer_xpu:
            try:
                xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
            except Exception:
                xpu_ok = False
            if xpu_ok:
                # Forced XPU on a hybrid host: unsloth's device_type picks
                # CUDA before XPU and ignores this Unsloth-only env var, so
                # hide CUDA or spawned workers would silently train on CUDA.
                if force_xpu and not cuda_hidden and not cuda_unavailable:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                DEVICE = DeviceType.XPU
                CHAT_ONLY = False
                CHAT_ONLY_REASON = None
                device_name = torch.xpu.get_device_name(0)
                if force_xpu and not ze_mask:
                    reason = "UNSLOTH_FORCE_XPU=1"
                elif force_xpu:
                    reason = "UNSLOTH_FORCE_XPU=1 + ZE_AFFINITY_MASK"
                else:
                    reason = "ZE_AFFINITY_MASK hint honoured"
                print(f"Hardware detected: XPU -- {device_name} ({reason})")
                return DEVICE

        # --- CUDA: NVIDIA GPU ---
        if torch.cuda.is_available():
            DEVICE = DeviceType.CUDA
            CHAT_ONLY = False
            try:
                device_name = torch.cuda.get_device_properties(0).name
            except Exception as e:
                logger.debug("CUDA device 0 property probe failed: %s", e)
                device_name = "<unavailable>"

            # Distinguish ROCm from CUDA for display only (DeviceType stays CUDA).
            # AMD SDK wheels don't set torch.version.hip, so fall back to __version__.
            _hip_ver = getattr(torch.version, "hip", None)
            if _hip_ver is not None or "rocm" in torch.__version__.lower():
                IS_ROCM = True
                _hip_label = _hip_ver or torch.__version__
                print(f"Hardware detected: ROCm (HIP {_hip_label}) -- {device_name}")
            else:
                print(f"Hardware detected: CUDA -- {device_name}")
            _print_cuda_device_list(IS_ROCM)
            return DEVICE

    # --- XPU: Intel GPU ---
    if torch_ok:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            DEVICE = DeviceType.XPU
            CHAT_ONLY = False
            device_name = torch.xpu.get_device_name(0)
            print(f"Hardware detected: XPU — {device_name}")
            return DEVICE

    # --- MLX: Apple Silicon ---
    # Require the full mlx/mlx-lm/mlx-vlm stack (not a bare `import mlx.core`) so
    # the gate matches utils.mlx_repair: a partial/backtracked stack stays
    # chat-only (reason "mlx_unavailable") and the background self-heal repairs it.
    if is_apple_silicon() and _has_usable_mlx_stack():
        DEVICE = DeviceType.MLX
        CHAT_ONLY = False
        # Use platform.machine() ("arm64"); platform.processor() returns "i386"
        # on universal2 / Rosetta builds even on native arm64.
        chip = platform.machine() or "arm64"
        print(f"Hardware detected: MLX — Apple Silicon ({chip})")
        return DEVICE

    # --- Fallback ---
    DEVICE = DeviceType.CPU
    # CHAT_ONLY is still True here (every training-capable branch returned early),
    # so record WHY so the UI can explain the greyed-out Train/Export instead of
    # silently disabling them.
    if is_apple_silicon():
        # Reached the CPU fallback on Apple Silicon, so the MLX stack is missing,
        # too old, or broken. This is usually an environment problem recoverable
        # with `unsloth studio update`.
        CHAT_ONLY_REASON = "mlx_unavailable"
        CHAT_ONLY_DETAIL = _mlx_stack_detail()
        logger.warning(
            "Apple Silicon detected but the MLX stack is incomplete or too old; "
            "Train/Export disabled (chat-only)%s Run `unsloth studio update` to "
            "restore MLX training.",
            f" ({CHAT_ONLY_DETAIL})." if CHAT_ONLY_DETAIL else ".",
        )
    elif TORCH_IMPORT_ERROR is not None:
        # torch installed but broken, so this host was never measured. "no_gpu" would lie.
        CHAT_ONLY_REASON = "detection_failed"
    elif platform.system() == "Darwin":
        CHAT_ONLY_REASON = "intel_mac"  # Intel Mac: no PyTorch/MLX -> GGUF-only by design.
    else:
        CHAT_ONLY_REASON = "no_gpu"
    print("Hardware detected: CPU training backend (no PyTorch/MLX GPU backend available)")
    return DEVICE


# ========== Convenience helpers ==========


def get_device() -> DeviceType:
    """
    Return the detected device, auto-detecting if detect_hardware() hasn't run.
    Prefer calling detect_hardware() explicitly at startup.
    """
    return ensure_hardware_detected()


def export_capability() -> dict:
    """Whether model export can run here, with a torch-aware reason when it cannot.

    Export runs through Unsloth, which hard-requires an accelerator (it calls ``torch.cuda`` at
    import and has no CPU path), so it is supported iff ``get_device() in {CUDA, XPU, MLX}``. The
    reason distinguishes a --no-torch install from a bare-CPU host. Safe to call without torch.

    Returns {export_supported, export_unsupported_reason, export_unsupported_message}.
    """
    if get_device() in (DeviceType.CUDA, DeviceType.XPU, DeviceType.MLX):
        return {
            "export_supported": True,
            "export_unsupported_reason": None,
            "export_unsupported_message": None,
        }
    # No accelerator: name the blocker. Detection failure first -- the branches below all
    # describe a measured host, so a broken probe would tell a GPU box to install PyTorch.
    if CHAT_ONLY_REASON == "detection_failed":
        reason = "detection_failed"
        message = (
            "Hardware detection failed on this host, so export is disabled. The server log records "
            "the underlying error; restart Unsloth Studio to retry detection."
        )
    elif is_apple_silicon():
        reason = "mlx_unavailable"
        message = (
            "Export on Apple Silicon requires the MLX stack, which is unavailable or too old. Run "
            "`unsloth studio update` to restore MLX and enable export."
        )
    elif not _has_torch():
        reason = "pytorch_not_installed"
        message = (
            "PyTorch is not installed. Model export requires PyTorch with a supported accelerator "
            "(NVIDIA, AMD, or Intel GPU) or Apple Silicon (MLX). Install PyTorch to enable export."
        )
    else:
        reason = "no_accelerator"
        message = (
            "Export requires an NVIDIA, AMD, or Intel GPU, or Apple Silicon (MLX). No supported "
            "accelerator was found on this host. (PyTorch is installed, but Unsloth cannot export "
            "on CPU only.)"
        )
    return {
        "export_supported": False,
        "export_unsupported_reason": reason,
        "export_unsupported_message": message,
    }


def video_capability() -> dict:
    """Whether video generation can run here, with a torch-aware reason when it cannot.

    Supported on CUDA and XPU, and on Apple Silicon with a usable MPS device. The pipelines in
    core/inference/video.py are device-neutral -- they resolve the device through the shared
    diffusion device target, whose capability flags already decline the CUDA-only options -- so
    Metal needs no branches of its own. Safe to call without torch.

    Returns {video_supported, video_unsupported_reason, video_unsupported_message}.
    """
    if get_device() in (DeviceType.CUDA, DeviceType.XPU):
        return {
            "video_supported": True,
            "video_unsupported_reason": None,
            "video_unsupported_message": None,
        }
    # Detection failure first, as in export_capability: the branches below all describe a
    # measured host, so a broken probe would tell a GPU box to go buy a GPU.
    if CHAT_ONLY_REASON == "detection_failed":
        reason = "detection_failed"
        message = (
            "Hardware detection failed on this host, so video generation is disabled. The server "
            "log records the underlying error; restart Unsloth Studio to retry detection."
        )
    elif is_apple_silicon() or get_device() == DeviceType.MLX:
        # The MLX arm covers an Apple host whose platform probe somehow disagrees.
        if _torch_mps_available():
            return {
                "video_supported": True,
                "video_unsupported_reason": None,
                "video_unsupported_message": None,
            }
        if TORCH_IMPORT_ERROR is not None:
            # Installed but broken reads as no torch below, and detect_hardware() records
            # mlx_unavailable for this host, so neither the branch above nor the one below sees
            # it -- and the host would be told to install the PyTorch it already has.
            reason = "detection_failed"
            message = (
                "PyTorch is installed but fails to import on this host, so the video pipelines "
                "cannot start. The server log records the error; reinstall PyTorch to fix it."
            )
        elif not _has_torch():
            reason = "pytorch_not_installed"
            message = (
                "PyTorch is not installed. Video generation on Apple Silicon requires PyTorch "
                "with Metal (MPS) support. Install PyTorch to enable video generation."
            )
        else:
            reason = "mps_unavailable"
            message = (
                "This PyTorch build exposes no Metal (MPS) device, so the video pipelines have "
                "nowhere to run. Reinstall PyTorch with MPS support to enable video generation."
            )
    elif platform.system() == "Darwin":
        # Ahead of the torch/GPU branches below: neither installing torch nor adding a GPU
        # enables video on an Intel Mac, so it must not be told to try either.
        reason = "macos_unsupported"
        message = (
            "Video generation requires Apple Silicon. This Intel Mac has no Metal (MPS) device "
            "for the video pipelines to run on."
        )
    elif not _has_torch():
        reason = "pytorch_not_installed"
        message = (
            "PyTorch is not installed. Video generation requires PyTorch with an NVIDIA, AMD or "
            "Intel GPU. Install PyTorch to enable video generation."
        )
    else:
        reason = "no_accelerator"
        message = (
            "Video generation requires an NVIDIA, AMD or Intel GPU. No supported accelerator was "
            "found on this host. (PyTorch is installed, but the video pipelines cannot run on CPU "
            "only.)"
        )
    return {
        "video_supported": False,
        "video_unsupported_reason": reason,
        "video_unsupported_message": message,
    }


def clear_gpu_cache():
    """
    Clear GPU memory cache for the current device.
    Safe on any platform — no-ops gracefully.
    """
    gc.collect()

    device = get_device()

    if device == DeviceType.CUDA:
        import torch

        # Nothing reserved anywhere means no allocator was ever built, so there is
        # nothing to wait on. Skip only the synchronize: it attaches a primary
        # context (~612 MiB, never returned) to idle on an empty stream.
        # memory_reserved needs no context; is_initialized() is useless here, since
        # reading device properties already flips it. Summed over devices so one
        # that allocated off the current device still drains. Deliberately NOT
        # wrapped: the unload paths need a sticky CUDA fault to propagate.
        if torch.cuda.is_available() and any(
            torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())
        ):
            torch.cuda.synchronize()
        # Both are no-ops without an allocator and neither creates a context.
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == DeviceType.XPU:
        # Guard synchronize/empty_cache: older torch-xpu builds may lack
        # them, and an unguarded AttributeError would propagate to callers.
        # torch.xpu has no ipc_collect(), so do not call it here.
        try:
            import torch
            if hasattr(torch, "xpu"):
                if hasattr(torch.xpu, "synchronize"):
                    torch.xpu.synchronize()
                if hasattr(torch.xpu, "empty_cache"):
                    torch.xpu.empty_cache()
        except Exception as e:
            logger.debug("Failed to clear XPU cache: %s", e)
    elif device == DeviceType.MLX:
        _clear_mps_cache()
    elif is_apple_silicon():
        # An Apple Silicon host whose MLX stack is unavailable reports CPU, but diffusion and
        # video still run on Metal (see diffusion_device._mps_or_cpu_target), so the MPS
        # allocator needs the same teardown the MLX branch gets.
        _clear_mps_cache()


def _clear_mps_cache() -> None:
    """Return torch's MPS reservations to the shared pool.

    MLX manages its own memory, but Apple Silicon also runs torch MPS (diffusion/video), whose
    caching allocator keeps freed buffers reserved. Those bytes read as used system memory, so
    skipping this leaves the next load budgeting against a pool that looks smaller than it is.
    """
    try:
        import torch
        empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception as e:
        logger.debug("Failed to clear MPS cache: %s", e)


def _rocm_visibility_masks_are_stacked() -> bool:
    """Whether a ROCr mask is composed with a higher HIP-layer mask."""
    if sys.platform == "win32" or os.environ.get("ROCR_VISIBLE_DEVICES") is None:
        return False
    return (
        os.environ.get("HIP_VISIBLE_DEVICES") is not None
        or os.environ.get("CUDA_VISIBLE_DEVICES") is not None
    )


def _rocm_device_ordinal_active() -> bool:
    """Whether GPU_DEVICE_ORDINAL renumbers HIP devices.

    ROCclr-layer, so it applies on Windows too, and no visibility spec here reads
    it: a torch ordinal cannot be paired with a physical id while it is set.
    """
    return bool(os.environ.get("GPU_DEVICE_ORDINAL", "").strip())


def _cuda_order_matches_smi() -> bool:
    """Whether torch ordinals and nvidia-smi rows share one index space.

    CUDA enumerates FASTEST_FIRST by default and nvidia-smi reports PCI order;
    PCI_BUS_ID is only a setdefault here, so an explicit override survives and
    the two disagree. Equal-sized cards defeat the total-scope check, so nothing
    else catches it. Same gate the llama.cpp SM probe applies. One GPU is exempt:
    every ordering is the identity there.
    """
    if IS_ROCM or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
        return True
    # The count must be the SMI's. The torch fallback counts VISIBLE devices, so
    # a mask of one on a multi-GPU host reads as single-GPU and would grant the
    # exemption the mask is exactly the reason to withhold. It is cached, so one
    # transient `nvidia-smi -L` timeout would otherwise disable this for the
    # life of the process.
    count = get_physical_gpu_count()
    if _physical_gpu_count_from_smi and count <= 1:
        return True
    if not _physical_gpu_count_from_smi:
        logger.debug("Skipping SMI VRAM query: physical GPU count is not SMI-confirmed")
        return False
    logger.debug("Skipping SMI VRAM query: CUDA_DEVICE_ORDER is not PCI_BUS_ID")
    return False


def _amd_smi_ids_for_hip_ids(hip_ids: Optional[list[int]]) -> Optional[list[int]]:
    """Translate visible HIP ordinals to amd-smi physical GPU IDs."""
    if hip_ids is None or not hip_ids:
        return hip_ids
    if _rocm_device_ordinal_active():
        logger.debug("Skipping amd-smi VRAM query: GPU_DEVICE_ORDINAL filters HIP devices")
        return None
    if _rocm_visibility_masks_are_stacked():
        logger.debug("Skipping amd-smi VRAM query: ROCr and HIP visibility masks are stacked")
        return None

    from . import amd

    smi_to_hip = amd.get_hip_id_by_gpu_index()
    if smi_to_hip is None:
        # Identity is still provable on a host with exactly one physical GPU.
        if hip_ids == [0] and amd.get_physical_gpu_count() == 1:
            return [0]
        logger.debug("Skipping amd-smi VRAM query: HIP GPU mapping is unavailable")
        return None

    torch_visible_count = _torch_get_physical_gpu_count()
    if torch_visible_count is None or torch_visible_count != len(hip_ids):
        logger.debug("Skipping amd-smi VRAM query: amd-smi and HIP visible counts differ")
        return None
    has_standard_mask = any(
        os.environ.get(name) is not None
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    )
    if not has_standard_mask and len(smi_to_hip) != torch_visible_count:
        logger.debug("Skipping amd-smi VRAM query: amd-smi and HIP inventories differ")
        return None

    hip_to_smi = {hip_id: smi_id for smi_id, hip_id in smi_to_hip.items()}
    if any(hip_id not in hip_to_smi for hip_id in hip_ids):
        logger.debug("Skipping amd-smi VRAM query: HIP GPU mapping is incomplete")
        return None
    return [hip_to_smi[hip_id] for hip_id in hip_ids]


def _free_in_torch_scope(total_bytes: int, used_gb: float) -> int:
    """Driver used-memory turned into free bytes, in torch's allocatable scope.

    Subtract from torch's total, never the driver's: NVIDIA's total also spans a
    reserved framebuffer that ``used`` excludes and torch can never hand out
    (726 MiB on a B200), so ``driver_total - used`` reports a full card as free.

    Clamped both ends: the parsers accept signed values, and a negative used
    would otherwise advertise more free than the card has, which utilization_pct
    and the training-method policy both read.
    """
    return min(total_bytes, max(0, total_bytes - round(used_gb * (1024**3))))


def _context_free_cuda_memory_info(
    idx: int,
    total_bytes: int,
    unified: bool = False,
) -> Optional[int]:
    """System-wide free bytes without attaching a CUDA/HIP primary context.

    ``unified`` marks a ROCm APU, whose *total* must still come from HIP but whose
    *used* is available here; see ``_rocm_windows_unified_used_bytes``.
    """
    parent_visible_spec = _get_parent_visible_gpu_spec()

    # A unified part goes straight to the WDDM counters. The vendor CLI and DRM
    # sysfs both report the dedicated carve-out for it, which the total-tolerance
    # check below would reject anyway, and Linux hipMemGetInfo is already
    # system-wide there -- so this exists for Windows, where it is not.
    if unified:
        if platform.system() != "Windows" or _rocm_device_ordinal_active():
            return None
        used_bytes = _rocm_windows_unified_used_bytes()
        if used_bytes is None:
            return None
        return _free_in_torch_scope(total_bytes, used_bytes / (1024**3))

    # Prefer the vendor CLI. Both nvidia-smi and amd-smi run out of process, so
    # querying an idle backend does not leave a context resident in this process.
    visible_ids = parent_visible_spec["numeric_ids"]
    if IS_ROCM:
        visible_ids = _amd_smi_ids_for_hip_ids(visible_ids)
    # Without numeric ids ROCm cannot map at all, while an NVIDIA UUID mask names
    # its devices absolutely: CUDA enumerates them in the order listed, so there
    # is no ordinal space to share and the order gate does not apply.
    may_query = not IS_ROCM if visible_ids is None else _cuda_order_matches_smi()
    result = None
    if may_query:
        result = _smi_query(
            "get_visible_gpu_utilization",
            visible_ids,
            parent_cuda_visible_devices = parent_visible_spec["raw"],
        )
    if result is not None:
        for device in result.get("devices", []):
            if device.get("visible_ordinal") != idx:
                continue
            used_gb = device.get("vram_used_gb")
            driver_total_gb = device.get("vram_total_gb")
            if used_gb is None or driver_total_gb is None:
                break
            driver_total_bytes = round(driver_total_gb * (1024**3))
            total_tolerance = max(total_bytes // 100, 16 * 1024**2)
            if abs(driver_total_bytes - total_bytes) > total_tolerance:
                logger.debug("Skipping whole-GPU VRAM telemetry for a partitioned GPU device")
                break
            return _free_in_torch_scope(total_bytes, used_gb)

    if not IS_ROCM:
        return None

    # Linux DRM sysfs is system-wide and context-free. Build the complete
    # physical inventory because the resolver intentionally rejects partial or
    # visibility-masked sets whose ordinals cannot be matched safely.
    if platform.system() == "Linux":
        numeric_ids = parent_visible_spec.get("numeric_ids")
        if numeric_ids is not None and 0 <= idx < len(numeric_ids):
            mod, _ = _torch_get_device_module()
            probe = []
            if mod is not None:
                try:
                    for ordinal, physical_idx in enumerate(numeric_ids):
                        props = mod.get_device_properties(ordinal)
                        probe.append(
                            {
                                "index": physical_idx,
                                "vram_total_gb": props.total_memory / (1024**3),
                            }
                        )
                except Exception as e:
                    logger.debug("ROCm context-free inventory failed: %s", e)
                    probe = []
            resolved = _rocm_system_wide_vram_by_index(probe)
            entry = resolved.get(numeric_ids[idx])
            if entry is not None:
                used_gb, _sysfs_total_gb = entry
                return _free_in_torch_scope(total_bytes, used_gb)

    # Native Windows ROCm exposes per-adapter dedicated usage without entering
    # HIP. Reuse the same conservative mapping as the System telemetry route.
    # GPU_DEVICE_ORDINAL renumbers torch ordinals but not the visible spec, so
    # device_ids[idx] would be a different card; the Linux path declines already.
    if platform.system() == "Windows" and not _rocm_device_ordinal_active():
        numeric_ids = parent_visible_spec.get("numeric_ids")
        device_ids = (
            numeric_ids if numeric_ids else list(range(_torch_get_physical_gpu_count() or 0))
        )
        # Only when the counter instances ARE the visible set. With extras present
        # the capacity ranking may hand a hidden adapter's smaller reading to a
        # busier visible card: simulated over 89/89/8 GiB, a card really holding
        # 86 GiB reports 10, overstating free by 76 GiB. The System tab can carry
        # that best-effort mapping, but free_gb feeds training-method selection,
        # where overstating free is what OOMs. Same cardinality assumption
        # _rocm_windows_aggregate_used_bytes already rests on.
        adapters = _rocm_windows_perf_counter_vram_by_adapter()
        if adapters is None or len(adapters) != len(device_ids):
            return None
        devices, _aggregate = _rocm_windows_per_device_vram(device_ids, adapters)
        for device in devices:
            if device.get("visible_ordinal") != idx:
                continue
            used_gb = device.get("used_gb")
            driver_total_gb = device.get("total_gb")
            if used_gb is None or driver_total_gb is None:
                break
            return _free_in_torch_scope(total_bytes, used_gb)

    return None


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory info.
    Supports CUDA (NVIDIA), MLX (Apple Silicon), and CPU-only.
    """
    device = get_device()

    # ---- CUDA path ----
    if device == DeviceType.CUDA:
        try:
            import torch

            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)

            total = props.total_memory
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)

            # Driver-level free includes torch's cache and other processes. Try
            # context-free telemetry first: mem_get_info pins a primary context
            # for the life of this backend.
            #
            # A ROCm APU needs its GTT *total* from HIP, which the context-free
            # probes cannot supply, so it pays for the context regardless. That is
            # a reason to take the total from HIP, not a reason to take the free
            # figure from it too: hipMemGetInfo is process-local on Windows WDDM
            # and blind to other processes there. So resolve the total first, then
            # still prefer telemetry for used.
            driver_total_needed = _rocm_props_total_is_carve_out(props)
            free = None
            if driver_total_needed:
                try:
                    free, driver_total = trusted_mem_get_info(idx)
                    # Only adopt a driver total that is usable: utilization_pct
                    # divides by it, so a zero would lose the whole report.
                    if driver_total:
                        total = driver_total
                except Exception as e:
                    logger.debug("mem_get_info probe failed; free VRAM from reserved: %s", e)
                    free = max(0, total - reserved)
                # The counters only apply where UMA is POSITIVELY identified.
                # _rocm_props_total_is_carve_out also answers True for an
                # uncertain device (old HIP, unreadable flag), on the principle
                # that a too-small total hides models. That justifies the driver
                # total, not summing Shared Usage: shared system memory is not
                # part of a discrete card's props.total_memory, so on a discrete
                # GPU misread as uncertain it would understate free.
                telemetry_free = None
                if _rocm_props_are_positively_unified(props):
                    try:
                        telemetry_free = _context_free_cuda_memory_info(idx, total, unified = True)
                    except Exception as e:
                        logger.debug("context-free free-VRAM probe failed: %s", e)
                if telemetry_free is not None:
                    free = telemetry_free
            else:
                try:
                    free = _context_free_cuda_memory_info(idx, total)
                except Exception as e:
                    logger.debug("context-free free-VRAM probe failed: %s", e)
                try:
                    if free is None:
                        free, _driver_total = trusted_mem_get_info(idx)
                except Exception as e:
                    logger.debug("mem_get_info probe failed; free VRAM from reserved: %s", e)
                    free = max(0, total - reserved)

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": idx,
                "device_name": props.name,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": free / (1024**3),
                "utilization_pct": (allocated / total) * 100,
            }
        except Exception as e:
            logger.error(f"Error getting CUDA GPU info: {e}")
            return {
                "available": False,
                "backend": _backend_label(device),
                "error": str(e),
            }

    # ---- XPU path (Intel GPU) ----
    if device == DeviceType.XPU:
        try:
            import torch

            idx = torch.xpu.current_device()
            props = torch.xpu.get_device_properties(idx)

            total = props.total_memory
            allocated = torch.xpu.memory_allocated(idx)
            reserved = torch.xpu.memory_reserved(idx)

            # Same rationale as the CUDA path: driver free, reserved as the
            # fallback bound (see above).
            try:
                free, _driver_total = trusted_mem_get_info(idx, module = torch.xpu)
            except Exception as e:
                logger.debug("xpu mem_get_info probe failed; free VRAM from reserved: %s", e)
                free = max(0, total - reserved)

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": idx,
                "device_name": props.name,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": free / (1024**3),
                "utilization_pct": (allocated / total) * 100,
            }
        except Exception as e:
            logger.error("Error getting XPU GPU info: %s", e)
            return {
                "available": False,
                "backend": _backend_label(device),
                "error": str(e),
            }

    # ---- MLX path (Apple Silicon) ----
    if device == DeviceType.MLX:
        try:
            import mlx.core as mx
            import psutil

            # Unified memory: total = system RAM, GPU used from IORegistry AGX.
            total = psutil.virtual_memory().total
            agx = _read_apple_gpu_stats()
            allocated = agx.get("vram_used_bytes", 0) if agx else 0

            try:
                info = mx.device_info()
                # prefer machine(); processor() can return "i386" on native arm64.
                gpu_name = info.get("device_name") or platform.machine() or "arm64"
            except Exception:
                gpu_name = platform.machine() or "arm64"

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": 0,
                "device_name": f"Apple Silicon ({gpu_name})",
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": allocated / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total else 0,
            }
        except Exception as e:
            logger.error(f"Error getting MLX GPU info: {e}")
            return {
                "available": False,
                "backend": _backend_label(device),
                "error": str(e),
            }

    # ---- CPU-only ----
    return {"available": False, "backend": "cpu"}


def log_gpu_memory(context: str):
    """Log GPU memory usage with context."""
    memory_info = get_gpu_memory_info()
    if memory_info.get("available"):
        backend = memory_info.get("backend", "unknown").upper()
        device_name = memory_info.get("device_name", "")
        label = f"{backend}" + (f" ({device_name})" if device_name else "")
        logger.info(
            f"GPU Memory [{context}] {label}: "
            f"{memory_info['allocated_gb']:.2f}GB/{memory_info['total_gb']:.2f}GB "
            f"({memory_info['utilization_pct']:.1f}% used, "
            f"{memory_info['free_gb']:.2f}GB free)"
        )
    else:
        logger.info(f"GPU Memory [{context}]: No GPU available (CPU-only)")


# ========== GPU Summary & Package Versions ==========


def get_gpu_summary() -> Dict[str, Any]:
    """
    Return a compact summary of the primary GPU.

    Returns dict with keys:
        gpu_name      – e.g. "NVIDIA L4" (or None)
        vram_total_gb – e.g. 22.17       (or None)
    """
    mem = get_gpu_memory_info()
    if mem.get("available"):
        return {
            "gpu_name": mem.get("device_name"),
            "vram_total_gb": round(mem.get("total_gb", 0), 2),
            "vram_free_gb": round(mem.get("free_gb", 0), 2),
        }
    return {"gpu_name": None, "vram_total_gb": None, "vram_free_gb": None}


def get_package_versions() -> Dict[str, Optional[str]]:
    """
    Return installed versions of key ML packages.

    Uses importlib.metadata (stdlib), no subprocess. CUDA version from
    torch.version.cuda. Returns dict keyed unsloth/torch/transformers/cuda;
    missing packages yield None.
    """
    packages = ("unsloth", "torch", "transformers")
    versions: Dict[str, Optional[str]] = {}

    for name in packages:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = None

    # GPU runtime versions bundled with torch (CUDA, ROCm/HIP, Intel XPU)
    try:
        import torch

        versions["cuda"] = getattr(torch.version, "cuda", None)
        versions["rocm"] = getattr(torch.version, "hip", None)
        # Isolated probe: a broken Intel runtime raising in is_available()
        # must not blank the already-read cuda/rocm versions.
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # torch.version.xpu may be None on modern builds; fall back to
                # "available" so the UI distinguishes present-but-unknown from
                # "package not found".
                xpu_ver = getattr(torch.version, "xpu", None)
                versions["xpu"] = xpu_ver if xpu_ver is not None else "available"
        except Exception:
            versions["xpu"] = None
    except Exception:
        versions["cuda"] = None
        versions["rocm"] = None
        versions["xpu"] = None

    return versions


# ========== Torch-based GPU fallbacks (AMD ROCm, Intel XPU, nvidia-smi missing) ==========


def _torch_get_device_module():
    """Return the appropriate torch device module (cuda or xpu) and its name."""
    device = get_device()
    import torch

    if device == DeviceType.CUDA:
        return torch.cuda, "cuda"
    if device == DeviceType.XPU and hasattr(torch, "xpu"):
        return torch.xpu, "xpu"
    return None, None


def _torch_get_physical_gpu_count() -> Optional[int]:
    mod, _ = _torch_get_device_module()
    if mod is None:
        return None
    try:
        return mod.device_count()
    except Exception:
        return None


def rocm_windows_free_is_untrusted() -> bool:
    """Whether ``mem_get_info``'s FREE half must be treated as an over-report.

    AMD documents this: the hipMemGetInfo reference warns "On Windows, the free
    memory only accounts for memory allocated by this process and may be optimistic."
    WDDM virtualises video memory, so a process is told its own budget rather than
    the card's residency and a fresh process sees free at or near total whatever else
    is resident. An AMD engineer confirms in ROCm/librocdxg#57 that this is the
    intended Windows model rather than a defect, measuring 24410 MiB of 24560
    reported free on a deliberately filled card; ROCm/TheRock#3724 is the same
    symptom, torch OOM while reporting 52.71 GiB of a 53.92 GiB card free.

    Near ``total``, not equal to it, so callers cap instead of testing for a
    sentinel. The TOTAL half is fine. Every other platform is left alone, WSL
    included and deliberately: AMD keeps the WSL2 reading consistent with native
    Linux, where free tracks physical residency, and ``sys.platform`` is "linux"
    there, so this is False and the accurate figure passes through uncapped. One
    predicate for the whole backend (#7452 reporting side, #8403 guard side).
    """
    return sys.platform == "win32" and IS_ROCM


def trusted_mem_get_info(device: Any = None, *, module: Any = None) -> tuple[int, int]:
    """``mem_get_info`` with the Windows ROCm free over-report capped (#8403).

    Guards that budget against free VRAM (the image activation refusal, llama.cpp
    slot fitting, the video preflight) cannot be handed a figure that says the
    whole card is free while a model is resident: on Windows WDDM the overflow
    does not raise, the driver satisfies it from host RAM, and the process grows
    past the card instead of failing, so an optimistic reading removes the only
    protection there is.

    Free is capped at what this process's own torch allocator has NOT reserved,
    which is a true upper bound on free VRAM. It still cannot see another
    process's allocations, so it is a ceiling, not a measurement; a caller that
    adds torch's reclaimable cache back (as the diffusion snapshot does) recovers
    exactly ``total - allocated``. Off Windows ROCm this returns the driver's own
    numbers untouched.

    ``module`` defaults to ``torch.cuda``; pass ``torch.xpu`` or the resolved
    device module to probe another backend. Exceptions propagate, so callers keep
    their existing "unreadable card decides nothing" handling.
    """
    import torch

    mod = module if module is not None else torch.cuda
    free_bytes, total_bytes = mod.mem_get_info() if device is None else mod.mem_get_info(device)
    free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    if not rocm_windows_free_is_untrusted():
        return free_bytes, total_bytes
    try:
        reserved = int(mod.memory_reserved() if device is None else mod.memory_reserved(device))
    except Exception as e:
        # No allocator accounting to cap against: the driver figure is all there is.
        logger.debug("memory_reserved probe failed while capping free VRAM: %s", e)
        return free_bytes, total_bytes
    return min(free_bytes, max(0, total_bytes - reserved)), total_bytes


# clr fills hipDeviceProp_t.integrated from the HSA "agent is APU" flag only from
# ROCm 6.1.2 on; before that an APU answers 0 exactly like a discrete card. HIP's
# last version field is a build number, so 6.1.2 cannot be told from 6.1.0 and the
# whole 6.1 line stays untrusted.
_HIP_INTEGRATED_FLAG_MIN = (6, 2)


def _hip_runtime_version() -> Optional[tuple[int, int]]:
    """(major, minor) of the HIP runtime torch was built against, None if unreadable."""
    try:
        import torch

        raw = getattr(torch.version, "hip", None)
        if raw:
            parts = str(raw).split(".")
            return (int(parts[0]), int(parts[1]))
        # AMD SDK / Radeon wheels leave version.hip unset; the tag is in __version__.
        match = re.search(r"rocm(\d+)\.(\d+)", getattr(torch, "__version__", "") or "")
        return (int(match.group(1)), int(match.group(2))) if match else None
    except Exception as e:
        logger.debug("HIP runtime version probe failed: %s", e)
        return None


# APU architectures that are unified memory but sit outside the shared
# classifier's positive set. That set is the list of parts whose SHARED POOL SIZE
# drives a lower set_per_process_memory_fraction cap, and it names anything else
# from hipDeviceProp_t::integrated; a Phoenix / Hawk Point iGPU on a runtime that
# leaves that flag at 0 is therefore in neither, and reads as "not unified".
#
# Harmless for the cap and for the total, and wrong for a NUMERATOR: clr never
# assigned deviceProps.integrated before tag rocm-6.2.0 (so HIP SDK for Windows
# 5.5.1 / 5.7.1 / 6.1.2 leave it 0 in a zero-initialised struct, which is what
# _HIP_INTEGRATED_FLAG_MIN above already encodes), and the legacy R0000 property
# ABI still does not assign it on any version -- while paldevice.cpp inflates
# globalMemSize_ off settings().apuSystem_ in both cases.
#
# gfx1103 Phoenix / Hawk Point is an integrated part and never shipped as a
# discrete board, so naming it here can only ever upgrade a part whose
# props.total_memory really does span host memory.
_ROCM_UNNAMED_APU_ARCHES = frozenset({"gfx1103"})


def _rocm_props_unified_status(props: Any) -> Optional[bool]:
    try:
        from core.training.worker import _rocm_classify_unified_memory

        classification = _rocm_classify_unified_memory(props)
        arch = str(classification[0] or "")
        return bool(classification[1]) or (
            arch.split(":")[0].strip().lower() in _ROCM_UNNAMED_APU_ARCHES
        )
    except Exception as e:
        logger.debug("ROCm unified-memory classification failed: %s", e)
        return None


def _rocm_props_are_positively_unified(props: Any) -> bool:
    """Whether this part is KNOWN to be unified memory, not merely unclassified.

    ``_rocm_props_total_is_carve_out`` folds "uncertain" in with "unified" on
    purpose, because a total that is too small hides models. Anything that adds
    host-shared memory to a used figure needs the stricter question: on a
    discrete card, shared bytes are not part of ``props.total_memory``.

    "Uncertain" is what this must not accept. A part the classifier can NAME as
    an APU is not uncertain, whichever way the classifier's own answer went: on
    Windows the backend is PAL, and paldevice.cpp adds the WDDM shared heap to
    globalMemSize_ for any ``Pal::GpuType::Integrated`` part, which is a property
    of the DEVICE and not of whether HIP filled in the props struct's integrated
    field. So a gfx1103 Phoenix on a pre-6.2 runtime already carries a
    pool-scoped total, and answering False here pairs that pool with Dedicated
    Usage alone, which plateaus at the BIOS carve-out. Measured shape: 17.0 GB
    total, 8 GiB resident, published as 1.90 used and 15.10 free.

    """
    if not IS_ROCM:
        return False
    return _rocm_props_unified_status(props) is True


def _rocm_props_total_is_carve_out(props: Any) -> bool:
    """True when ``props.total_memory`` may understate what torch can actually use.

    On some stacks ``props.total_memory`` is the dedicated carve-out while
    ``hipMemGetInfo``'s total spans the GTT pool. MAY, not does: measured on a
    Windows gfx1151 (driver 32.0.21041.1000, torch 2.11.0+rocm7.13.0) the two
    agree at 89.47 GB against a 32 GB BIOS carve-out, and on a Linux gfx1151 they
    agree at 64 GB, so on both APUs tested props.total_memory already spans the
    pool and this returning True buys nothing. Callers must therefore compare the
    two and adopt only a larger driver total, never adopt it outright.

    The gap this module DOES have provenance for is a different pair:
    _apply_unified_memory_correction adopts torch's total over amd-smi's, because
    amd-smi reports the carve-out. That is torch against amd-smi, not
    props.total_memory against hipMemGetInfo, and it is not evidence for this one.
    A stack where these two disagree is inferred from reports of an understated
    total, not yet reproduced; #8862 and #7449 are about used VRAM reading
    Unknown, which is a different fault.

    There is no context-free source for the GTT total, so only APUs pay
    mem_get_info; every discrete device keeps the free inventory. Same classifier
    the training worker and the llama.cpp backend use, so all three agree on what
    an APU is.

    "Not unified" is not the same answer as "discrete". That classifier knows an APU
    by the driver's integrated flag or by a hardcoded arch set, so a gfx1103 Phoenix
    iGPU on a runtime that leaves the flag at 0 reads as discrete and would publish
    its carve-out as the whole device. Only a runtime that fills the flag in gets to
    settle it; on anything older, and when the classifier itself fails, the driver
    total is worth a context, because a total that is too small hides models.
    """
    if not IS_ROCM:
        return False
    if _rocm_props_unified_status(props) is not False:
        return True
    hip_version = _hip_runtime_version()
    return (
        getattr(props, "is_integrated", None) is None
        or hip_version is None
        or hip_version < _HIP_INTEGRATED_FLAG_MIN
    )


def _torch_get_device_inventory(device_indices: list[int]) -> list[Dict[str, Any]]:
    """Per-GPU name and total VRAM only, without creating a driver context.

    INVENTORY, not occupancy. ``get_device_properties`` is answered from the
    driver's device list, so it costs nothing; ``mem_get_info`` has to attach a
    primary context to the device, which is ~612 MiB the process never gives back
    (a context is only torn down at exit). A telemetry poll every few seconds must
    not be what pins that, so callers that need name and capacity but not live
    occupancy come here instead of _torch_get_per_device_info.

    ``props.total_memory`` is the same number ``mem_get_info`` returns as its total
    half everywhere except a ROCm APU (see _rocm_props_total_is_carve_out), so
    totals are unchanged. ``used_gb`` is always None, the value this module already
    uses for "telemetry unavailable".
    """
    mod, _ = _torch_get_device_module()
    if mod is None:
        return []

    devices = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            # torch ordinals are 0-based relative to CUDA_VISIBLE_DEVICES.
            props = mod.get_device_properties(ordinal)
            props_total_bytes = int(props.total_memory)
            total_bytes = props_total_bytes
            known_unified = _rocm_props_are_positively_unified(props)
            shared_memory = known_unified and platform.system() == "Windows"
            shared_memory_host_backed_bytes = None
            try:
                if _rocm_props_total_is_carve_out(props) and hasattr(mod, "mem_get_info"):
                    # Only a WIDER total, same as _rocm_windows_per_device_vram.
                    # The classifier fails open, so a discrete card reaches here
                    # too, and a total below props.total_memory would hide models
                    # this device can hold.
                    driver_total_bytes = int(mod.mem_get_info(ordinal)[1])
                    shared_memory = known_unified and (
                        driver_total_bytes > int(total_bytes)
                        or (
                            platform.system() == "Windows"
                            and driver_total_bytes == int(total_bytes)
                        )
                    )
                    if shared_memory and driver_total_bytes > props_total_bytes:
                        shared_memory_host_backed_bytes = driver_total_bytes - props_total_bytes
                    total_bytes = max(driver_total_bytes, int(total_bytes))
            except Exception as e:
                # Keep the carve-out rather than dropping the device: an
                # understated total still beats no device at all.
                logger.debug("ROCm APU driver total failed for ordinal %d: %s", ordinal, e)
            try:
                rocm_gfx = str(getattr(props, "gcnArchName", "") or "")
            except Exception:
                rocm_gfx = ""
            devices.append(
                {
                    "index": phys_idx,
                    "visible_ordinal": ordinal,
                    "name": props.name,
                    "total_gb": round(total_bytes / (1024**3), 2),
                    "used_gb": None,
                    "shared_memory": shared_memory,
                    "shared_memory_host_backed_gb": (
                        round(shared_memory_host_backed_bytes / (1024**3), 2)
                        if shared_memory_host_backed_bytes is not None
                        else None
                    ),
                    "_rocm_known_unified": known_unified,
                    "_rocm_gfx": rocm_gfx,
                }
            )
        except Exception as e:
            logger.debug("torch inventory probe failed for ordinal %d: %s", ordinal, e)
    return devices


def _torch_get_per_device_info(device_indices: list[int]) -> list[Dict[str, Any]]:
    """Query torch for per-GPU name, total VRAM, and used VRAM.

    Creates a driver context on CUDA/HIP (see _torch_get_device_inventory). Only
    call this when live occupancy is actually consumed.

    ``used_gb`` is ``None`` on Windows ROCm when the driver reports ``free ==
    total``: that 0 means unknown, not empty. This is the DISPLAY path, so unknown
    rather than a pessimistic ceiling, which is the right answer for a refusal and
    the wrong one for a number shown to the user as measured.
    """
    mod, _ = _torch_get_device_module()
    if mod is None:
        return []

    device = get_device()
    # free==total is a Windows-ROCm-only quirk.
    _win_rocm = rocm_windows_free_is_untrusted()
    devices = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            # torch ordinals are 0-based relative to CUDA_VISIBLE_DEVICES.
            props = mod.get_device_properties(ordinal)
            total_bytes = props.total_memory
            used_bytes: Optional[int]
            # Prefer mem_get_info (system-wide) so auto-select sees other consumers.
            if hasattr(mod, "mem_get_info"):
                try:
                    free_bytes, total_bytes = mod.mem_get_info(ordinal)
                    used_bytes = total_bytes - free_bytes
                except Exception as e:
                    if device != DeviceType.XPU:
                        raise
                    # Arc B580 and Lunar Lake can report properties while
                    # rejecting free-memory queries. Preserve the usable
                    # device and its total memory with unknown utilization.
                    logger.debug(
                        "XPU free-memory query failed for ordinal %d: %s",
                        ordinal,
                        e,
                    )
                    used_bytes = None
                else:
                    # free==total is the broken-API sentinel, not an idle GPU.
                    if _win_rocm and free_bytes == total_bytes:
                        used_bytes = None
            elif device == DeviceType.XPU:
                # XPU without mem_get_info: memory_allocated() is process-local
                # and misleading for placement, so return None for the
                # selector's no-telemetry fallback.
                used_bytes = None
            else:
                used_bytes = mod.memory_allocated(ordinal)
            devices.append(
                {
                    "index": phys_idx,
                    "visible_ordinal": ordinal,
                    "name": props.name,
                    "total_gb": round(total_bytes / (1024**3), 2),
                    "used_gb": (
                        round(used_bytes / (1024**3), 2) if used_bytes is not None else None
                    ),
                }
            )
        except Exception as e:
            logger.debug("torch device query failed for ordinal %d: %s", ordinal, e)
    return devices


# ========== Live GPU Utilization ==========


def _xpu_hierarchy_is_composite() -> bool:
    """Return True iff Level Zero is running in COMPOSITE device hierarchy.

    COMPOSITE: numeric ``ZE_AFFINITY_MASK`` entries address root GPU IDs
    (tiles use ``N.M``). FLAT (the oneAPI default; also assumed when
    ``ZE_FLAT_DEVICE_HIERARCHY`` is unset): entries address tile/device
    handles, so mapping them back to root GPU IDs is unsafe. Only COMPOSITE
    gives stable root-ID semantics.
    """
    hierarchy = (os.environ.get("ZE_FLAT_DEVICE_HIERARCHY") or "FLAT").strip().upper()
    return hierarchy == "COMPOSITE"


def _parse_ze_mask_roots(mask: str) -> list[int]:
    """Parse a ``ZE_AFFINITY_MASK`` value into an ordered list of root device IDs.

    One root ID per mask token, preserving order and duplicates so logical
    ordinals map 1-to-1 to physical root IDs (e.g. ``"0.0,0.1"`` -> ``[0, 0]``,
    ``"2.0,0.1,0.2"`` -> ``[2, 0, 0]``); empty list if no parseable digits.
    Only meaningful in COMPOSITE hierarchy -- callers needing a stable
    root-ID mapping must gate on ``_xpu_hierarchy_is_composite()``.
    """
    roots: list[int] = []
    if not mask:
        return roots
    for token in mask.split(","):
        token = token.strip()
        if not token:
            continue
        root = token.split(".", 1)[0]
        # isdecimal() (not isdigit()) rejects Unicode superscripts like
        # "²"/"³", which pass isdigit() but crash int() with ValueError.
        if root.isdecimal():
            roots.append(int(root))
    return roots


def _smi_query(func_name: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
    """Query the appropriate SMI backend (amd-smi or nvidia-smi).

    Returns the result dict if available, else None.
    """
    if IS_ROCM:
        backend_name = "amd-smi"
        try:
            from . import amd as _backend
        except Exception as e:
            logger.warning("%s import failed: %s", backend_name, e)
            return None
    else:
        backend_name = "nvidia-smi"
        try:
            from . import nvidia as _backend
        except Exception as e:
            logger.warning("%s import failed: %s", backend_name, e)
            return None
    try:
        func = getattr(_backend, func_name)
        result = func(*args, **kwargs)
        if isinstance(result, dict) and result.get("available"):
            return result
    except Exception as e:
        logger.warning("%s %s query failed: %s", backend_name, func_name, e)
    return None


def _read_apple_gpu_stats() -> Dict[str, Any]:
    """Query macOS IORegistry for AGX (Apple GPU) live stats. No sudo needed.

    Returns dict with utilization_pct, vram_used_bytes (system-wide GPU
    memory), or empty dict on failure.
    """
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator"],
            capture_output = True,
            timeout = 2,
        )
        text = result.stdout.decode("utf-8", errors = "replace")
    except Exception:
        return {}

    # PerformanceStatistics block has GPU utilization and in-use memory
    m = re.search(r'"PerformanceStatistics" = \{([^}]+)\}', text)
    if not m:
        return {}
    stats_str = m.group(1)
    pairs = re.findall(r'"([^"]+)"=(\d+)', stats_str)
    stats = {k: int(v) for k, v in pairs}

    return {
        "utilization_pct": stats.get("Device Utilization %", 0),
        "vram_used_bytes": stats.get("In use system memory", 0),
    }


# ── CPU frequency on Apple Silicon ──────────────────────────────────────────
# psutil divides the pmgr "voltage-statesN-sram" IORegistry tables by 1e6 to
# reach MHz, but Apple switched them from Hz to kHz on M4, so psutil <= 7.2.2
# shows a 4.5 GHz M4 Pro as "4 MHz" in Settings > System (issue #8519). Upstream
# fix is giampaolo/psutil#2824, merged and unreleased; until it ships we read the
# tables ourselves through ioreg with that PR's heuristics, else rescale psutil's
# value. A fixed psutil is already plausible, so neither correction runs.

# Apple clocks are 0.6-4.6 GHz, so a raw Hz entry sits above 1e8 and kHz below.
_CPU_FREQ_UNIT_THRESHOLD = 100_000_000
_MIN_PLAUSIBLE_CPU_MHZ = 500
_MAX_PLAUSIBLE_CPU_MHZ = 20000
# Below this a table is a GPU/NPU rail: above every Apple GPU peak so far, under
# the slowest CPU cluster shipped (M1 E-core, 2064 MHz).
_CPU_CLUSTER_MIN_PEAK_MHZ = 2000
_VOLTAGE_STATES_KEY = re.compile(r"^voltage-states\d+-sram$")

# Fixed for the life of the host and /api/system polls every few seconds, so
# probe once. The sentinel separates "not probed yet" from "probed, unavailable".
_apple_cpu_peak_mhz: Any = "unprobed"
_apple_cpu_peak_lock = threading.Lock()


def _voltage_state_freqs_mhz(blob: bytes) -> list:
    """Plausible MHz from a voltage-statesN-sram blob.

    Each entry is 8 bytes: little-endian uint32 frequency then uint32 voltage.
    """
    freqs = []
    for offset in range(0, len(blob) - 7, 8):
        raw = int.from_bytes(blob[offset : offset + 4], "little")
        if raw == 0:
            continue
        mhz = raw / 1e6 if raw > _CPU_FREQ_UNIT_THRESHOLD else raw / 1e3
        if _MIN_PLAUSIBLE_CPU_MHZ <= mhz <= _MAX_PLAUSIBLE_CPU_MHZ:
            freqs.append(mhz)
    return freqs


def _peak_cpu_mhz_from_ioreg_entries(entries) -> Optional[float]:
    """Highest CPU-cluster peak across pmgr voltage-state tables, or None."""
    peaks = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if not isinstance(value, (bytes, bytearray)) or not _VOLTAGE_STATES_KEY.match(str(key)):
                continue
            freqs = _voltage_state_freqs_mhz(bytes(value))
            # M5 renumbered the indexes, so classify by peak, not by index.
            if freqs and max(freqs) >= _CPU_CLUSTER_MIN_PEAK_MHZ:
                peaks.append(max(freqs))
    return max(peaks) if peaks else None


def _read_apple_cpu_peak_mhz() -> Optional[float]:
    """Read peak CPU MHz from the pmgr IORegistry node. None if unavailable."""
    global _apple_cpu_peak_mhz
    if _apple_cpu_peak_mhz != "unprobed":
        return _apple_cpu_peak_mhz

    # /api/system is polled from several worker threads; unlocked, a burst of
    # first requests spawns one ioreg each and a slow failing probe landing last
    # poisons the cache for the rest of the run.
    with _apple_cpu_peak_lock:
        if _apple_cpu_peak_mhz != "unprobed":
            return _apple_cpu_peak_mhz

        peak = None
        try:
            import plistlib

            result = subprocess.run(
                ["ioreg", "-a", "-r", "-c", "AppleARMIODevice", "-d", "1"],
                capture_output = True,
                # Same budget as the AGX probe above. The call is made once per
                # process but from inside a /api/system request, so it must not
                # be able to hold a worker thread for long.
                timeout = 2,
            )
            entries = plistlib.loads(result.stdout) if result.stdout else []
            if isinstance(entries, dict):
                entries = [entries]
            peak = _peak_cpu_mhz_from_ioreg_entries(entries)
        except Exception as e:
            logger.debug("Apple CPU frequency ioreg probe failed: %s", e)

        _apple_cpu_peak_mhz = peak
        return peak


def cpu_frequency_mhz() -> Optional[float]:
    """Current CPU clock in MHz, corrected for the psutil Apple Silicon unit bug.

    Returns None when no frequency is available (psutil reports nothing inside
    many containers and VMs).
    """
    freq = None
    try:
        import psutil
        freq = psutil.cpu_freq()
    except Exception as e:
        # Not fatal on Apple Silicon: the IORegistry read below stands in. psutil
        # raises here on M5, whose tables are not at the indexes it hardcodes.
        logger.debug("Failed to get CPU frequency: %s", e)

    current = getattr(freq, "current", None) if freq else None
    usable = isinstance(current, (int, float)) and current == current and current > 0

    if not is_apple_silicon():
        return round(float(current), 2) if usable else None
    if usable and current >= _MIN_PLAUSIBLE_CPU_MHZ:
        return round(float(current), 2)

    exact = _read_apple_cpu_peak_mhz()
    if exact is not None:
        return round(exact, 2)
    if not usable:
        return None
    # No tables: recover the magnitude from psutil's kHz-as-Hz reading. It
    # truncates in integer arithmetic, so this lands on the GHz step, not the peak.
    return round(float(current) * 1000, 2)


def _rocm_linux_sysfs_gpu_busy_pct() -> Optional[float]:
    """Query AMD GPU compute utilization via Linux DRM sysfs gpu_busy_percent."""
    if platform.system() != "Linux":
        return None
    try:
        files = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        if not files:
            return None
        values = [int(open(f, encoding = "utf-8").read().strip()) for f in files]
        return round(sum(values) / len(values), 1)
    except Exception:
        return None


def _rocm_linux_sysfs_temp_c() -> Optional[float]:
    """Query AMD GPU edge temperature via Linux DRM hwmon sysfs (temp1_input, millidegrees C)."""
    if platform.system() != "Linux":
        return None
    try:
        files = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input")
        if not files:
            return None
        temps = [int(open(f, encoding = "utf-8").read().strip()) / 1000.0 for f in files]
        return round(max(temps), 1)
    except Exception:
        return None


def _rocm_linux_sysfs_power_w() -> Optional[float]:
    """Query AMD GPU average power draw via Linux DRM hwmon sysfs (microwatts)."""
    if platform.system() != "Linux":
        return None
    try:
        for pattern in (
            "/sys/class/drm/card*/device/hwmon/hwmon*/power1_average",
            "/sys/class/drm/card*/device/hwmon/hwmon*/power1_input",
        ):
            files = glob.glob(pattern)
            if files:
                watts = sum(
                    int(open(f, encoding = "utf-8").read().strip()) / 1_000_000.0 for f in files
                )
                return round(watts, 1)
        return None
    except Exception:
        return None


def _engine_instance_luid(instance_name: str) -> Optional[int]:
    """Same two halves ``_parse_adapter_luid`` reads, behind a ``pid_<pid>_`` prefix."""
    head = instance_name.lower().find("luid_0x")
    if head < 0:
        return None
    return _parse_adapter_luid(instance_name[head:])


def _rocm_windows_perf_counter_gpu_util_pct(luid: Optional[int] = None) -> Optional[float]:
    """Query AMD GPU compute utilization via Windows Performance Counters (3D engine nodes).

    ``luid`` narrows the sum to one adapter's engines, matched here rather than in
    the counter path so tests can reach it, as in ``..._vram_by_adapter``. The
    engine type stays in the path: it is not what this narrows, and it is what
    keeps the sample set to one type per process rather than all of them.
    """
    if platform.system() != "Windows":
        return None
    try:
        ps = (
            "$s=(Get-Counter '\\GPU Engine(*engtype_3D*)\\Utilization Percentage'"
            " -ErrorAction SilentlyContinue).CounterSamples;"
            "if($s){$s|ForEach-Object{'{0}|{1}' -f $_.InstanceName,$_.CookedValue}}"
            "else{'__NONE__'}"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        total = 0.0
        matched = False
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line or line == "__NONE__" or "|" not in line:
                continue
            instance, _, raw = line.rpartition("|")
            instance = instance.strip()
            # Hosts spell it engtype_3D and engtype_3d both.
            if "engtype_3d" not in instance.lower():
                continue
            if luid is not None and _engine_instance_luid(instance) != luid:
                continue
            try:
                value = float(raw.strip())
            except (ValueError, TypeError):
                continue
            # 0.0 is an idle engine, so a reading; NaN / inf / negative is not.
            if value != value or value in (float("inf"), float("-inf")) or value < 0:
                continue
            total += value
            matched = True
        if not matched:
            return None
        return round(min(total, 100.0), 1)
    except Exception:
        return None


def _rocm_linux_sysfs_vram_gb() -> tuple[Optional[float], Optional[float]]:
    """Query system-wide AMD GPU VRAM via Linux DRM sysfs.

    Reads /sys/class/drm/card*/device/mem_info_vram_*, which the kernel
    updates in real-time across all processes. No tools required.
    Returns (used_gb, total_gb) or (None, None) on failure.
    """
    if platform.system() != "Linux":
        return None, None
    try:
        used_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        total_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        if not used_files or not total_files:
            return None, None
        used_bytes = sum(int(open(f, encoding = "utf-8").read().strip()) for f in used_files)
        total_bytes = sum(int(open(f, encoding = "utf-8").read().strip()) for f in total_files)
        if total_bytes == 0:
            return None, None
        return round(used_bytes / (1024**3), 2), round(total_bytes / (1024**3), 2)
    except Exception:
        return None, None


# 0x1002. NVIDIA's open kernel module also registers KFD nodes (vendor_id 0x10DE);
# a non-AMD node is not a HIP device and must never take an ordinal.
_AMD_PCI_VENDOR_ID = 4098


def _rocm_kfd_gpu_pci_ids() -> list[str]:
    """PCI addresses of the GPUs ROCm enumerates, in HIP device order.

    Reads /sys/class/kfd/kfd/topology/nodes/<N>/properties, the topology ROCm
    itself enumerates from: AMD GPU nodes (simd_count > 0 excludes CPUs,
    vendor_id == AMD excludes NVIDIA) in node-id order are HIP's device order, so
    position N is ROCm physical device N. Unlike DRM sysfs, an amdgpu adapter HIP
    cannot enumerate has no node here, so it never consumes an ordinal.

    Returns [] (disabling the overlay) when KFD is absent, and FAILS CLOSED the
    same way on any unreadable node or an AMD node with no location_id: dropping
    one would shift every later ordinal and let a similar-capacity GPU pass the
    total-size guard while showing another card's usage.

    location_id is the kernel's (bus << 8) | devfn; domain is separate.
    """
    nodes: list[tuple[int, str]] = []
    try:
        node_dirs = glob.glob("/sys/class/kfd/kfd/topology/nodes/*")
    except Exception:
        return []
    for node_dir in node_dirs:
        m = re.fullmatch(r".*/(\d+)", node_dir)
        if m is None:
            continue
        props: dict[str, int] = {}
        try:
            with open(os.path.join(node_dir, "properties"), encoding = "utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            props[parts[0]] = int(parts[1])
                        except ValueError:
                            continue
        except (OSError, UnicodeDecodeError):
            return []  # unreadable node could be a GPU: fail closed, don't shift
        if props.get("simd_count", 0) <= 0:
            continue  # CPU node, not a GPU
        if props.get("vendor_id") != _AMD_PCI_VENDOR_ID:
            continue  # non-AMD GPU node (NVIDIA open driver): not a HIP device
        location_id = props.get("location_id")
        if location_id is None:
            return []  # an AMD GPU we cannot place: fail closed for the whole map
        domain = props.get("domain", 0)
        bus = (location_id >> 8) & 0xFF
        devfn = location_id & 0xFF
        bdf = f"{domain:04x}:{bus:02x}:{(devfn >> 3) & 0x1F:02x}.{devfn & 0x7}"
        nodes.append((int(m.group(1)), bdf))
    nodes.sort(key = lambda n: n[0])
    return [bdf for _node_id, bdf in nodes]


def _rocm_linux_amdgpu_cards() -> list[tuple[str, int, str]]:
    """The amdgpu-bound DRM cards in PCI order: ``(pci_bdf, card_no, device_dir)``.

    Membership is by the BOUND DRIVER, not the VRAM sysfs files: an AMD device
    with incomplete sysfs support (some APUs expose no mem_info_vram_*) still
    consumes a ROCm ordinal, and dropping it would shift every later card down.
    PCI order is HIP's default enumeration order, so list position is the ROCm
    ordinal; card_no is a stable tiebreak when the BDF cannot be resolved.

    NOTE this is a superset of the ROCm-visible set (a HIP-unsupported amdgpu
    adapter appears too), so callers must check the counts agree before assuming
    a 1:1 mapping onto torch devices.
    """
    if platform.system() != "Linux":
        return []
    amd_cards: list[tuple[str, int, str]] = []
    try:
        for card_path in glob.glob("/sys/class/drm/card*"):
            # Match card<N> exactly so connector nodes (card0-DP-1) are skipped.
            m = re.fullmatch(r".*/card(\d+)", card_path)
            if m is None:
                continue
            dev_dir = os.path.join(card_path, "device")
            try:
                driver = os.path.basename(os.path.realpath(os.path.join(dev_dir, "driver")))
            except OSError:
                continue
            if driver != "amdgpu":
                continue  # foreign adapter: not a ROCm device, takes no ordinal
            try:
                bdf = os.path.basename(os.path.realpath(dev_dir))
            except OSError:
                bdf = ""
            amd_cards.append((bdf, int(m.group(1)), dev_dir))
    except Exception:
        return []
    amd_cards.sort(key = lambda c: (c[0], c[1]))
    return amd_cards


def _rocm_linux_sysfs_vram_by_pci_gb() -> dict[str, tuple[float, float]]:
    """System-wide AMD VRAM via Linux DRM sysfs, keyed by the card's PCI address.

    Reads each card's mem_info_vram_{used,total} (kernel-updated across all
    processes) so every GPU gets its own figure, unlike _rocm_linux_sysfs_vram_gb
    which sums the host. Keyed by PCI address, not an ordinal, so the caller can
    join it to _rocm_kfd_gpu_pci_ids() by identity: DRM card numbers include
    foreign adapters and this set includes cards HIP does not enumerate, so any
    ordinal from this list alone can be shifted relative to ROCm's. A card with
    missing/unreadable/zero-total figures simply has no entry. Empty off Linux.
    """
    if platform.system() != "Linux":
        return {}

    try:
        by_pci: dict[str, tuple[float, float]] = {}
        for bdf, _card_no, dev_dir in _rocm_linux_amdgpu_cards():
            if not bdf:
                continue
            try:
                with open(os.path.join(dev_dir, "mem_info_vram_used"), encoding = "utf-8") as f:
                    used_bytes = int(f.read().strip())
                with open(os.path.join(dev_dir, "mem_info_vram_total"), encoding = "utf-8") as f:
                    total_bytes = int(f.read().strip())
            except (OSError, ValueError):
                continue
            if total_bytes <= 0:
                continue
            by_pci[bdf.lower()] = (
                round(used_bytes / (1024**3), 2),
                round(total_bytes / (1024**3), 2),
            )
        return by_pci
    except Exception:
        return {}


# ── Windows AMD/ROCm per-adapter VRAM (issue #7072) ──────────────────────────
# amd-smi is disabled and hipMemGetInfo reports free==total, so read used from the
# per-LUID "GPU Adapter Memory" perf counters and take each total from torch, so
# every GPU shows instead of one fake device with GPU 0's total.
# Placeholder adapters (Basic Render Driver / idle iGPU) drop only when they would
# outnumber the real torch devices.
_ROCM_WIN_ADAPTER_MIN_BYTES = 64 * 1024 * 1024  # 64 MiB


def _rocm_windows_perf_counter_vram_by_adapter(
    counter: str = "Dedicated Usage",
) -> Optional[list[tuple[str, float]]]:
    """Per-adapter VRAM usage on Windows via Performance Counters.

    ``counter`` selects the ``GPU Adapter Memory`` field. Dedicated Usage is the
    default and the only one safe to select adapters on; see
    ``_rocm_windows_unified_used_bytes`` for why Shared Usage is read separately
    rather than folded in here.

    Returns ``[(instance_name, used_bytes)]`` (one per LUID-named adapter), or
    ``None`` when the counter is unavailable/localized/empty so callers fall back.
    """
    if platform.system() != "Windows":
        return None
    try:
        # Emit "<InstanceName>|<CookedValue>" per sample, or a __NONE__ sentinel.
        ps = (
            f"$s=(Get-Counter '\\GPU Adapter Memory(*)\\{counter}'"
            " -ErrorAction SilentlyContinue).CounterSamples;"
            "if($s){$s|ForEach-Object{'{0}|{1}' -f $_.InstanceName,[int64]$_.CookedValue}}"
            "else{'__NONE__'}"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        adapters: list[tuple[str, float]] = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line or line == "__NONE__" or "|" not in line:
                continue
            instance, _, raw = line.rpartition("|")
            try:
                used = float(raw.strip())
            except (ValueError, TypeError):
                continue
            if used < 0:
                continue
            adapters.append((instance.strip(), used))
        return adapters or None
    except Exception:
        return None


# DirectX writes one record per adapter here, keyed by a GUID, holding the same
# AdapterLuid the counter instances are named after alongside the Description
# string torch reports as props.name and the gfx target it reports as
# gcnArchName. Those are the join keys capacity ranking lacks.
_WINDOWS_DIRECTX_KEY = r"SOFTWARE\Microsoft\DirectX"


def _parse_adapter_luid(instance_name: str) -> Optional[int]:
    """The 64-bit LUID in a ``GPU Adapter Memory`` instance name, or None.

    Instances are named ``luid_0x<high>_0x<low>_phys_<n>``; DirectX stores the
    same value as one 64-bit ``AdapterLuid``, so recombine the halves.
    """
    m = re.match(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)", instance_name.strip(), re.IGNORECASE)
    if m is None:
        return None
    try:
        return (int(m.group(1), 16) << 32) | int(m.group(2), 16)
    except ValueError:
        return None


# hipDeviceProp_tR0600 opens with name[256], hipUUID[16], luid[8],
# luidDeviceNodeMask[4]. The R0600 suffix IS the ABI version, so that prefix is
# fixed by the same contract that named it; the name is read back and compared
# against the device's own anyway, so a layout that ever moved is caught rather
# than trusted.
_HIP_PROPS_NAME = slice(0, 256)
_HIP_PROPS_LUID = slice(272, 280)
_HIP_PROPS_NODE_MASK = slice(280, 284)
# Oversized on purpose: HIP writes the whole struct, ~2 KiB, and only the prefix
# above is read back.
_HIP_PROPS_BUFFER_BYTES = 64 * 1024


def _rocm_windows_hip_adapter_ids(
    ordinals: list[int], names: list[str]
) -> Optional[list[tuple[int, int]]]:
    """The ``(luid, node_mask)`` HIP itself reports for each visible ordinal.

    ``hipDeviceProp_tR0600`` carries the same DXGI LUID the ``GPU Adapter
    Memory`` counter instances are named after, so asking HIP joins the two
    sides on the adapter itself rather than on anything about it. Windows
    reassigns LUIDs across a reboot or a driver restart, and all three sources
    move together, so the value is only ever compared within one poll.

    ``node_mask`` is ``luidDeviceNodeMask``: which nodes of a linked adapter
    this ordinal owns, and 0 on the ordinary adapter that is the whole thing.

    Returns one pair per ordinal, or None when the runtime cannot be asked at
    all (off Windows, the DLL not loaded into this process, the symbol absent,
    an ordinal that will not answer, an all-zero LUID, or a name that does not
    read back) so the caller falls back to the DirectX join. All or nothing for
    the same reason that map is: a partially resolved set would let one card's
    counter answer for another.

    Idea and the R0600 route from @pablo86gr in #8793.
    """
    if platform.system() != "Windows" or not ordinals:
        return None
    try:
        import ctypes

        torch = sys.modules.get("torch")
        major = str(getattr(getattr(torch, "version", None), "hip", "")).split(".", 1)[0]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        # Only a module already in this process: torch loaded the runtime it is
        # built against, and LoadLibrary could pull in a different one.
        hip = None
        for dll in dict.fromkeys(
            [f"amdhip64_{major}.dll" if major.isdigit() else "", "amdhip64.dll"]
        ):
            handle = kernel32.GetModuleHandleW(dll) if dll else None
            if handle:
                hip = ctypes.WinDLL(dll, handle = handle)
                break
        if hip is None:
            return None
        get_properties = hip.hipGetDevicePropertiesR0600
        get_properties.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_properties.restype = ctypes.c_int

        identities: list[tuple[int, int]] = []
        for ordinal, name in zip(ordinals, names):
            raw = ctypes.create_string_buffer(_HIP_PROPS_BUFFER_BYTES)
            if get_properties(ctypes.byref(raw), ordinal) != 0:
                return None
            blob = bytes(raw)
            if blob[_HIP_PROPS_NAME].rstrip(b"\x00").decode("utf-8", "replace") != name:
                # Not the struct this reads, so the offsets below mean nothing.
                logger.debug("HIP properties prefix did not read back ordinal %d", ordinal)
                return None
            luid_bytes = blob[_HIP_PROPS_LUID]
            if not any(luid_bytes):
                return None  # the runtime has no LUID for this device
            identities.append(
                (
                    int.from_bytes(luid_bytes, "little"),
                    int.from_bytes(blob[_HIP_PROPS_NODE_MASK], "little"),
                )
            )
        return identities
    except Exception as e:
        logger.debug("HIP adapter identity probe unavailable: %s", e)
        return None


def _match_adapter_used_by_hip_luid(
    adapters: list[tuple[str, float]], dev_meta: list[Dict[str, Any]]
) -> Optional[tuple[list[Optional[float]], float, list[Optional[int]]]]:
    """Attribute per-adapter used bytes on the LUID HIP reports for each ordinal.

    The exact key, so unlike the DirectX join this separates two cards of one
    model. Ordinals come from ``visible_ordinal``, not from this list's own
    positions, which are compacted when a device fails to probe.

    A linked-node adapter puts several ordinals behind one LUID, and the
    counters index its nodes as ``phys_N`` without saying which ordinal owns
    which. The node mask says how many nodes each ordinal holds, which is
    enough to tell "these counters are exactly these ordinals' nodes" from "one
    of them belongs to a node HIP is not showing us" -- but not enough to pair
    them, so several ordinals under one LUID report unknown per device and
    contribute only to the aggregate, which the pairing does not change.

    The third return is that LUID per device, but only where the device is the whole
    of what it names, which is what the engine counters need to filter safely.
    """
    ordinals = [int(meta["visible_ordinal"]) for meta in dev_meta]
    identities = _rocm_windows_hip_adapter_ids(ordinals, [str(meta["name"]) for meta in dev_meta])
    if identities is None:
        return None

    useds_by_luid: dict[int, list[float]] = {}
    physes_by_luid: dict[int, set[int]] = {}
    for instance, used in adapters:
        luid = _parse_adapter_luid(instance)
        phys = re.search(r"_phys_(\d+)", instance, re.IGNORECASE)
        if luid is None or phys is None:
            continue
        index = int(phys.group(1))
        if index in physes_by_luid.setdefault(luid, set()):
            # One physical node cannot report twice; summing would double-count.
            return None
        physes_by_luid[luid].add(index)
        useds_by_luid.setdefault(luid, []).append(used)

    positions_by_luid: dict[int, list[int]] = {}
    nodes_by_luid: dict[int, int] = {}
    for position, (luid, node_mask) in enumerate(identities):
        positions_by_luid.setdefault(luid, []).append(position)
        # A plain adapter reports no mask and is one node.
        nodes_by_luid[luid] = nodes_by_luid.get(luid, 0) + (bin(node_mask).count("1") or 1)

    assigned: list[Optional[float]] = [None] * len(dev_meta)
    whole_adapter: list[Optional[int]] = [None] * len(dev_meta)
    total_used = 0.0
    for luid, positions in positions_by_luid.items():
        useds = useds_by_luid.get(luid, [])
        # Fewer counters than nodes means a visible node has no reading; more
        # means the adapter has a node these ordinals do not own, whose usage is
        # not theirs to claim.
        if len(useds) != nodes_by_luid[luid]:
            return None
        used = float(sum(useds))
        # The carve-out, not the displayed total: on a unified APU the latter is
        # the whole driver pool, against which no reading is out of range.
        capacity = sum(_adapter_counter_capacity(dev_meta[position]) for position in positions)
        if used > capacity:
            return None
        total_used += used
        if len(positions) == 1:
            assigned[positions[0]] = used
            # Equal counts say how many nodes this ordinal owns, never which, and
            # a visible node whose sample is missing alongside a hidden node whose
            # sample is present counts the same. The mask names them (bit i is
            # node i), so where it does the observed phys_N set has to BE those
            # or the LUID covers a node this device does not own. A zero mask
            # names nothing and keeps the count check alone.
            named = {i for i in range(32) if identities[positions[0]][1] >> i & 1}
            if not named or named == physes_by_luid.get(luid, set()):
                whole_adapter[positions[0]] = luid
    return assigned, total_used, whole_adapter


_ADAPTER_NAME_NOISE = re.compile(r"\((?:tm|r)\)|[™®]", re.IGNORECASE)


def _normalize_adapter_name(name: str) -> str:
    """A GPU name in the one spelling both sides of the join can agree on.

    DirectX takes its Description from the driver INF and HIP fills props.name
    from the ASIC record, so the same card reaches the two sides with the
    trademark marks and the spacing around them differing -- "AMD Radeon(TM)
    780M Graphics" against "AMD Radeon 780M Graphics". Nothing here merges two
    different models, and a collision between two that did normalize alike is
    caught by the count check in _attribute_adapter_useds_by_key.
    """
    return " ".join(_ADAPTER_NAME_NOISE.sub(" ", name).split()).casefold()


def _parse_adapter_family_gfx(family: str) -> str:
    """The gfx target in a DirectX ``AdapterFamily``, or "" when it holds none.

    The AMD driver writes ``AMD_NAVI44:gfx1200``; torch reports the same target
    as ``props.gcnArchName``, which on Linux carries feature suffixes
    (``gfx1201:sramecc-:xnack-``) the comparison has to drop.
    """
    for token in str(family).split(":"):
        token = token.strip().lower()
        if re.fullmatch(r"gfx[0-9a-f]+", token):
            return token
    return ""


def _windows_amd_adapter_records_by_luid() -> dict[int, Dict[str, Any]]:
    """DirectX registry metadata for AMD adapters, keyed by LUID.

    ``gfx`` is absent when the driver wrote no ``AdapterFamily``.
    ``dedicated_memory_bytes`` is absent when neither dedicated-memory value is available.

    All or nothing: a record this cannot read makes the map incomplete, and an
    incomplete map is indistinguishable from a complete one at the join, which
    would then pair a visible card with a hidden same-named card's counter. So
    any failure past the point where a subkey is known to be an adapter returns
    ``{}``, which drops the caller back to capacity ranking. Same for off
    Windows or without the key.
    """
    if platform.system() != "Windows":
        return {}
    try:
        import winreg
    except ImportError:
        return {}
    by_luid: dict[int, Dict[str, Any]] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_DIRECTX_KEY) as dx_key:
            for index in range(winreg.QueryInfoKey(dx_key)[0]):
                subkey = winreg.EnumKey(dx_key, index)
                # Adapter records are GUID-named; ShaderCache and any future
                # named subkey are not adapters and are not ours to read.
                if not (subkey.startswith("{") and subkey.endswith("}")):
                    continue
                with winreg.OpenKey(dx_key, subkey) as adapter_key:
                    vendor_id, _ = winreg.QueryValueEx(adapter_key, "VendorId")
                    if int(vendor_id) != _AMD_PCI_VENDOR_ID:
                        continue
                    luid, _ = winreg.QueryValueEx(adapter_key, "AdapterLuid")
                    description, _ = winreg.QueryValueEx(adapter_key, "Description")
                    try:
                        family, _ = winreg.QueryValueEx(adapter_key, "AdapterFamily")
                    except OSError:
                        family = ""
                    dedicated_memory_bytes = 0
                    has_dedicated_memory = False
                    for value_name in ("DedicatedVideoMemory", "DedicatedSystemMemory"):
                        try:
                            value, _ = winreg.QueryValueEx(adapter_key, value_name)
                            value = int(value)
                            if value < 0:
                                raise ValueError(value)
                            dedicated_memory_bytes += value
                            has_dedicated_memory = True
                        except (OSError, TypeError, ValueError):
                            pass
                name = str(description).strip()
                if not name:
                    # An AMD adapter this cannot name: see the all-or-nothing note.
                    return {}
                record = {"name": name}
                gfx = _parse_adapter_family_gfx(str(family))
                if gfx:
                    record["gfx"] = gfx
                if has_dedicated_memory:
                    record["dedicated_memory_bytes"] = dedicated_memory_bytes
                by_luid[int(luid)] = record
    except Exception as e:
        logger.debug("DirectX adapter registry read declined: %s", e)
        return {}
    return by_luid


def _windows_rocm_shared_pool_host_gb_by_index(devices: list[Dict[str, Any]]) -> Dict[int, float]:
    """Map shared ROCm devices to the host-backed part of their Windows pool."""
    if platform.system() != "Windows":
        return {}
    records = list(_windows_amd_adapter_records_by_luid().values())
    shared_positions = [
        position for position, device in enumerate(devices) if device.get("shared_memory")
    ]
    gfx_available = (
        bool(records)
        and all(record.get("gfx") for record in records)
        and all(devices[position].get("_rocm_gfx") for position in shared_positions)
    )
    candidates_by_position: dict[int, set[int]] = {}
    for position in shared_positions:
        device = devices[position]
        device_name = _normalize_adapter_name(str(device.get("name", "")))
        device_gfx = _parse_adapter_family_gfx(str(device.get("_rocm_gfx", "")))
        candidates: set[int] = set()
        for record_position, record in enumerate(records):
            record_name = _normalize_adapter_name(str(record.get("name", "")))
            record_gfx = _parse_adapter_family_gfx(str(record.get("gfx", "")))
            name_matches = bool(device_name) and device_name == record_name
            if name_matches and device_gfx and record_gfx and device_gfx != record_gfx:
                name_matches = False
            gfx_matches = gfx_available and bool(device_gfx) and device_gfx == record_gfx
            if name_matches or gfx_matches:
                candidates.add(record_position)
        if candidates:
            candidates_by_position[position] = candidates

    dedicated_bytes_by_position: dict[int, int] = {}
    remaining_positions = set(candidates_by_position)
    while remaining_positions:
        first = remaining_positions.pop()
        component_positions = {first}
        component_records = set(candidates_by_position[first])
        while True:
            connected = {
                position
                for position in remaining_positions
                if candidates_by_position[position] & component_records
            }
            if not connected:
                break
            remaining_positions -= connected
            component_positions |= connected
            for position in connected:
                component_records |= candidates_by_position[position]
        if len(component_records) < len(component_positions):
            continue
        record_owner: dict[int, int] = {}

        def assign_record(position: int, seen: set[int]) -> bool:
            for record_position in candidates_by_position[position]:
                if record_position not in component_records or record_position in seen:
                    continue
                seen.add(record_position)
                owner = record_owner.get(record_position)
                if owner is None or assign_record(owner, seen):
                    record_owner[record_position] = position
                    return True
            return False

        if not all(assign_record(position, set()) for position in component_positions):
            continue
        try:
            dedicated_values = {
                int(records[record_position]["dedicated_memory_bytes"])
                for record_position in component_records
            }
        except (KeyError, TypeError, ValueError):
            continue
        if len(dedicated_values) != 1:
            continue
        dedicated_bytes = dedicated_values.pop()
        if dedicated_bytes < 0:
            continue
        for position in component_positions:
            dedicated_bytes_by_position[position] = dedicated_bytes

    host_gb_by_index: Dict[int, float] = {}
    for position, dedicated_bytes in dedicated_bytes_by_position.items():
        device = devices[position]
        index = device.get("index")
        total_gb = float(device.get("total_gb") or 0.0)
        dedicated_gb = float(dedicated_bytes) / (1024**3)
        if not isinstance(index, int) or total_gb <= 0 or not (0 <= dedicated_gb <= total_gb):
            continue
        host_gb_by_index[index] = round(total_gb - dedicated_gb, 2)
    return host_gb_by_index


def _adapter_counter_capacity(meta: Dict[str, Any]) -> float:
    """The capacity a ``Dedicated Usage`` counter for this device can actually fill.

    That counter measures the dedicated segment, so the carve-out is its ceiling.
    ``total_bytes`` is what the user is SHOWN, and on a unified APU may be the
    whole driver pool, against which a counter no visible card could hold still
    fits. Rank or bound a counter with this, never ``total_bytes``. The fallback
    keeps it right on a ``dev_meta`` built before ``dedicated_bytes`` existed.
    """
    return float(meta.get("dedicated_bytes", meta["total_bytes"]))


def _attribute_adapter_useds_by_key(
    useds_by_key: dict[str, list[float]],
    positions_by_key: dict[str, list[int]],
    dev_meta: list[Dict[str, Any]],
) -> Optional[tuple[list[Optional[float]], float]]:
    """Pair each key's counters with the devices carrying that key, or decline.

    Returns ``(per_device_used_bytes, aggregate_used_bytes)``, or None when any
    key's counters are not exactly its devices'. Several devices under one key
    (two cards of a model, two cards of an arch) leave per-device unknown --
    nothing says which counter is which ordinal -- while still contributing to
    the aggregate, which the pairing does not change.
    """
    assigned: list[Optional[float]] = [None] * len(dev_meta)
    total_used = 0.0
    for key, positions in positions_by_key.items():
        useds = useds_by_key.get(key, [])
        # Fewer counters than cards under this key means a visible card has no
        # reading; more means a same-keyed adapter HIP does not enumerate, or one
        # adapter emitting several _phys_N instances. Either way this key's
        # counters are not exactly its devices'.
        if len(useds) != len(positions):
            return None
        # Largest usage against the largest capacity: any other pairing of the
        # same multiset only makes the check below stricter, never truer.
        by_capacity = sorted(positions, key = lambda p: -_adapter_counter_capacity(dev_meta[p]))
        for used, position in zip(sorted(useds, reverse = True), by_capacity):
            # A usage above its own card's capacity: a record outliving its
            # hardware, so the key is not identifying what it appears to.
            if used > _adapter_counter_capacity(dev_meta[position]):
                return None
            total_used += used
        if len(positions) == 1:
            assigned[positions[0]] = useds[0]
    return assigned, total_used


def _match_adapter_used_by_luid(
    adapters: list[tuple[str, float]], dev_meta: list[Dict[str, Any]]
) -> Optional[tuple[list[Optional[float]], float]]:
    """Attribute per-adapter used bytes to torch devices on the adapter LUID.

    Joins each counter to a DirectX adapter record by the LUID in its instance
    name, then to a torch device by what the record and the device agree on.
    Identity, not capacity, so it resolves the single-GPU case that
    _match_adapter_used_to_devices can never force (nothing smaller exists to
    exceed) and stays right when a busy foreign adapter outweighs an idle card.

    The model name is tried first, because it separates two cards of one arch
    (a 9070 beside a 9070 XT) where the arch cannot. DirectX takes it from the
    driver INF and HIP from the ASIC record, so the two spellings CAN differ by
    more than normalizing fixes, and a declined name join then falls to the gfx
    target -- which is also what tells an iGPU from the dGPU beside it. The arch
    pass runs only when every AMD record has one, so a driver too old to write
    ``AdapterFamily`` cannot leave a hidden card's counter looking like the
    visible card's.

    Measured on a Windows gfx1151 (driver 32.0.21041.1000): DirectX said
    "AMD Radeon(TM) 8060S Graphics" and so did ``props.name``, an exact match, so
    the name pass carried it. That driver wrote NO ``AdapterFamily`` at all, so
    the gfx fallback was unavailable on the one machine this has been measured
    on. The two keys are therefore not the belt and braces this reads as: a
    driver that both spells the name differently and omits ``AdapterFamily``
    declines the join outright and drops back to capacity ranking. That is the
    safe direction, but it means the name pass is load bearing in practice.

    Returns ``(per_device_used_bytes, aggregate_used_bytes)``, or None when
    neither key establishes the join, so the caller falls back to capacity
    ranking.
    """
    records = _windows_amd_adapter_records_by_luid()
    if not records:
        return None

    useds_by_luid: dict[int, list[float]] = {}
    for instance, used in adapters:
        luid = _parse_adapter_luid(instance)
        if luid is not None and luid in records:
            useds_by_luid.setdefault(luid, []).append(used)

    best: Optional[tuple[list[Optional[float]], float]] = None
    best_resolved = -1

    for field, record_key, device_key in (
        (
            "name",
            lambda record: _normalize_adapter_name(record["name"]),
            lambda meta: _normalize_adapter_name(str(meta.get("name", ""))),
        ),
        (
            "gfx",
            lambda record: record["gfx"],
            lambda meta: _parse_adapter_family_gfx(str(meta.get("gfx", ""))),
        ),
    ):
        if not all(record.get(field) for record in records.values()):
            continue
        useds_by_key: dict[str, list[float]] = {}
        for luid, useds in useds_by_luid.items():
            useds_by_key.setdefault(record_key(records[luid]), []).extend(useds)
        positions_by_key: dict[str, list[int]] = {}
        for position, meta in enumerate(dev_meta):
            positions_by_key.setdefault(device_key(meta), []).append(position)
        if "" in positions_by_key:  # a visible device carrying no such key
            continue
        # AMD usage this key cannot place, so the key is not identifying
        # reliably: a card whose Description differs from its props.name leaves
        # its own counter under one key while a hidden card sits under the
        # device's, and the pairing would hand over the hidden card's bytes.
        # Declining drops a masked second AMD card back to capacity ranking,
        # which is where it already was.
        if set(useds_by_key) - set(positions_by_key):
            continue
        matched = _attribute_adapter_useds_by_key(useds_by_key, positions_by_key, dev_meta)
        if matched is not None:
            # Same-keyed cards leave those devices unknown and feed only the
            # aggregate (#7452). That is a result, not a reason to stop, and it
            # is not all-or-nothing: a host with one uniquely named card beside
            # two same-named ones of different arch resolves ONE device by name
            # and all three by gfx. So rank passes by how many devices they
            # actually place, keep the first on a tie, and stop early only on a
            # pass that leaves nothing for a later one to improve.
            resolved = sum(used is not None for used in matched[0])
            if resolved == len(dev_meta):
                return matched
            if resolved > best_resolved:
                best, best_resolved = matched, resolved
    return best


def _match_adapter_used_to_devices(
    adapter_useds: list[float], device_totals: list[float]
) -> list[Optional[float]]:
    """Attribute per-adapter used bytes to torch devices by capacity ranking.

    Windows shares no key between LUID counters and torch ordinals, so usages are
    ranked against device totals and each is trusted only when capacity *forces* it
    (it exceeds every smaller device); an ambiguous ranking reports unknown
    (``None``) rather than fabricate a per-index free.

    Extra counters mean a hidden/display adapter, and the noise filter may have
    dropped a real reading, so values are emitted only when the supra-threshold
    counters number EXACTLY the visible devices AND capacity forces the mapping;
    otherwise every device is unknown. Best-effort but correct for the common
    loaded-card case (#7072). Returns a list aligned to ``device_totals``.

    ``device_totals`` must be the DEDICATED capacity each counter can fill, never a
    unified-memory total. A widened total reranks its device, raises the threshold
    that forces a pairing, and lifts the ceiling below which a usage is judged
    impossible, which costs other devices their readings and admits counters that
    belong to no visible card.
    """
    n = len(device_totals)
    if n == 0:
        return []
    useds = sorted(adapter_useds, reverse = True)
    ranked_positions = sorted(range(n), key = lambda i: -device_totals[i])
    ranked_totals = [device_totals[pos] for pos in ranked_positions]
    assigned: list[Optional[float]]
    # More counters than devices -> a hidden/display adapter (check before noise filter).
    if len(useds) > n:
        non_trivial = [u for u in useds if u >= _ROCM_WIN_ADAPTER_MIN_BYTES]
        if len(non_trivial) != n:
            # Not a clean bijection (a masked GPU is busy or a visible card idle):
            # no counter maps to a specific card, so report unknown.
            return [None] * n
        # Exactly n supra-threshold counters: extras were placeholders, so a
        # capacity-ranked bijection is plausible.
        dropped = [u for u in useds if u < _ROCM_WIN_ADAPTER_MIN_BYTES]
        useds = non_trivial
        ranked_useds = [useds[rank] for rank in range(n)]
        # A usage above its ranked capacity is a hidden larger GPU; clamping onto the
        # smaller card would fabricate a fully-used reading.
        for rank in range(n):
            if ranked_useds[rank] > ranked_totals[rank]:
                return [None] * n
        # One visible device, one supra-threshold counter, and every other counter
        # at EXACTLY zero: attribute it. A Strix Halo host emits three instances,
        # two of them at 0 dedicated, and the capacity test below cannot decide
        # this shape because it needs a next-smaller device to compare against.
        #
        # Zero is the load-bearing part, not the cardinality. A merely sub-floor
        # counter (10 MiB) can be the visible card sitting idle, which would make
        # the survivor a hidden GPU's -- see the [6 GiB, 10 MiB] / [8 GiB] case,
        # which must stay unknown. An adapter at exactly zero has nothing
        # committed and cannot be the holder of the survivor's bytes.
        #
        # Residual risk, stated rather than hidden: the visible card could itself
        # be the zero while a hidden adapter holds the survivor, and then this
        # over-reports used. That errs toward understating free, which is the safe
        # direction here -- the consumer is training-method selection, where an
        # overstated free picks a method that OOMs. Clamped like every other
        # branch, so a hidden larger adapter cannot report a fully-used card.
        if n == 1 and all(u == 0 for u in dropped):
            return [min(ranked_useds[0], device_totals[0])]
        # Capacity forces the mapping only when the usage exceeds the next-smaller
        # capacity; the smallest card and merely-fitting usages stay unknown.
        # Keeps 40 GiB over 48/8 GiB -> [40, None].
        assigned = [None] * n
        for rank, pos in enumerate(ranked_positions):
            if rank + 1 < n and ranked_useds[rank] > ranked_totals[rank + 1]:
                assigned[pos] = min(ranked_useds[rank], device_totals[pos])
        return assigned
    # No hidden adapters: every counter is a visible card, so ranking is a permutation.
    ranked_useds = [useds[rank] if rank < len(useds) else 0.0 for rank in range(n)]
    # Ambiguous if a strictly larger usage also fits the next smaller card: the two
    # could be swapped without breaking capacity, so ranking can't tell them apart.
    for rank in range(n - 1):
        upper, lower = ranked_useds[rank], ranked_useds[rank + 1]
        if upper > lower and upper <= ranked_totals[rank + 1]:
            return [None] * n
    assigned = [None] * n
    for rank, pos in enumerate(ranked_positions):
        if rank < len(useds):
            assigned[pos] = min(useds[rank], device_totals[pos])
    return assigned


def _rocm_windows_aggregate_used_bytes(
    adapter_useds: list[float], device_totals: list[float]
) -> Optional[float]:
    """Total VRAM used across the visible devices, when the counters cover them 1:1.

    Per-device attribution needs capacity to FORCE a pairing, and on an asymmetric
    pair (45 GiB + 8 GiB) nothing at or below 8 GiB is forced, so idle and every
    small model report unknown (#7452). The SUM does not need the pairing -- over a
    bijection it is the same whichever way round the usages go -- so the System tab
    keeps a real figure where per-device honestly cannot. Emitted only when the
    counter list IS the visible set, established by cardinality alone. That rests on
    one ASSUMPTION, stated as such because it is not confirmed against Microsoft's
    counter documentation: that ``Get-Counter`` emits exactly one instance per WDDM
    adapter, so a visible card is always in the list and a list exactly as long as
    the visible set therefore holds those cards and nothing else. It fails closed if
    that is wrong: a second instance for one adapter makes the list longer than the
    visible set, which returns None rather than a total. Verify it before widening
    this, not before trusting it. ``device_totals`` carries the same requirement as
    _match_adapter_used_to_devices: dedicated capacity, never a unified total, or
    the check above stops rejecting usages no visible card could hold.

    Deliberately NOT the noise filter _match_adapter_used_to_devices uses. Dropping
    sub-threshold counters and summing the rest is safe for per-device attribution,
    which only ever emits a capacity-FORCED value, but not for a sum, which emits
    every counter it kept. The counters carry no vendor, LUID or PCI key, so a
    retained counter cannot be told apart from a foreign adapter: a card hidden by
    ``HIP_VISIBLE_DEVICES``, an iGPU, an NVIDIA card in the same box or a Basic
    Render Driver placeholder above the cutoff. Whenever such an adapter is busier
    than one visible card is idle, the filter drops the visible card and keeps the
    foreign one, and the sum silently gains bytes on no visible card at all. A host
    total that is confidently wrong is worse than Unknown, so an unexplained instance
    means ``None``. Extra instances are common, so this is narrow on purpose;
    widening it needs the counters joined to devices on LUID or PCI bus id rather
    than on capacity rank, the same key _match_adapter_used_to_devices lacks.
    """
    n = len(device_totals)
    if n == 0 or not adapter_useds:
        return None
    # More counters than devices: one is not ours, and no key says which. Fewer: a
    # visible card has no reading. Either way the sum is not the visible set's.
    if len(adapter_useds) != n:
        return None
    useds = sorted(adapter_useds, reverse = True)
    ranked_totals = sorted(device_totals, reverse = True)
    # A usage above its ranked capacity is on no visible card, so even at matching
    # length the list is not the visible set (a reading was dropped while parsing, or
    # a counter is not a dedicated-VRAM figure).
    for rank in range(n):
        if useds[rank] > ranked_totals[rank]:
            return None
    return float(sum(useds))


def _rocm_windows_unified_used_bytes(
    dedicated: Optional[list[tuple[str, float]]] = None,
) -> Optional[float]:
    """Used VRAM for a unified-memory ROCm APU on Windows, from the WDDM counters.

    Dedicated Usage alone saturates at the carve-out on an APU and is wrong past
    it. Measured on a gfx1151 Strix Halo host (89.47 GiB torch total), holding N
    GiB in another process, deltas over baseline:

        held    dedicated    shared      sum
         4       + 4.14      + 0.04    + 4.18
        16       +16.60      + 0.08    +16.68
        24       +24.58      + 0.04    +24.62
        40       +29.07      +11.29    +40.36
        48       +29.19      +19.02    +48.21

    Dedicated plateaus around 30.5 GiB and the overflow lands in Shared, so only
    the sum tracks the allocation. Discrete cards are unaffected: this is reached
    only when ``_rocm_props_total_is_carve_out`` says the part is unified.

    Adapter SELECTION still keys off Dedicated Usage alone, deliberately. Display
    and placeholder adapters report 0 dedicated while carrying gigabytes of shared
    (a Basic Render Driver instance on the same host holds 1.30 GiB shared), so
    filtering on the sum would stop telling them apart from the compute device and
    would silently add a foreign adapter's bytes. Returns ``None`` unless exactly
    one adapter clears the noise floor, matching the caution in
    ``_rocm_windows_aggregate_used_bytes``.

    ``dedicated`` lets a caller that already holds a Dedicated Usage snapshot
    hand it over: each query is an out-of-process PowerShell call, and re-sampling
    would also answer from a different instant than the caller's own attribution.
    """
    if dedicated is None:
        dedicated = _rocm_windows_perf_counter_vram_by_adapter()
    if not dedicated:
        return None
    candidates = [
        (instance, used) for instance, used in dedicated if used >= _ROCM_WIN_ADAPTER_MIN_BYTES
    ]
    # Two compute adapters means no key says which is the visible one, and zero
    # means the card is idle below the floor: neither is a figure we can stand on.
    if len(candidates) != 1:
        return None
    instance, dedicated_used = candidates[0]
    shared = _rocm_windows_perf_counter_vram_by_adapter("Shared Usage")
    # A failed Shared query is not zero shared usage. Past the carve-out the
    # overflow lives entirely in Shared, so defaulting to zero there reports the
    # measured 48 GiB case as 30.5 and overstates free by 19 GiB, which is the
    # direction that OOMs. Nothing here knows where the carve-out sits, so decline
    # rather than guess whether this reading was saturated. A query that SUCCEEDS
    # but omits the LUID is a real zero, and is kept.
    if shared is None:
        return None
    shared_used = next((used for name, used in shared if name == instance), 0.0)
    # A negative cooked counter is a broken reading, not low usage. The caller
    # clamps the upper bound only, and the payload derives free as total minus
    # used, so a negative sum publishes negative used and a free above total.
    # Decline for the same reason a failed Shared query declines: nothing here
    # can tell how much of the reading is wrong.
    if dedicated_used < 0.0 or shared_used < 0.0:
        return None
    return dedicated_used + shared_used


def _rocm_windows_per_device_vram(
    device_indices: list[int], adapters: Optional[list[tuple[str, float]]] = None
) -> tuple[list[Dict[str, Any]], Optional[float]]:
    """Per-GPU VRAM on Windows AMD/ROCm: total from torch properties, widened to
    the driver pool on a unified APU, used from the per-adapter Dedicated Usage
    counter.

    Returns ``([{index, visible_ordinal, name, used_gb, total_gb}], aggregate_gb)``
    per visible GPU, or ``([], None)`` when torch can't enumerate devices so callers
    fall through to the torch last resort. ``aggregate_gb`` is the visible set's
    total used VRAM, which survives a pairing no single device can claim (#7452),
    and ``None`` when even that is not established.

    ``used_gb`` is ``None`` when the counter is unavailable, when the pairing is not
    capacity-forced, or when this device's total was widened to the unified pool --
    that counter measures the dedicated segment, so it is not a numerator for a
    total spanning the shared one. The two totals are kept apart internally for the
    same reason: the counters are ranked against the carve-out, never the pool.
    """
    if platform.system() != "Windows":
        return [], None
    mod, _ = _torch_get_device_module()
    if mod is None:
        return [], None
    # Totals/names from torch properties (mem_get_info's free==total quirk zeroes used).
    dev_meta: list[Dict[str, Any]] = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            props = mod.get_device_properties(ordinal)
            total_bytes = int(props.total_memory)
            # What the Dedicated Usage counters are RANKED against, which is not
            # what is shown to the user: that counter measures the dedicated
            # segment, so the carve-out is its ceiling and the pool is not.
            # Ranking against a widened total puts a 2 GiB APU reading above a
            # 10 GiB discrete one and admits a counter no visible card can hold.
            dedicated_bytes = total_bytes
            # On a unified-memory APU props.total_memory CAN be the dedicated
            # carve-out rather than what torch can use; same correction, and the
            # same APU-only price for a context, as _torch_get_device_inventory.
            # Measured on a Windows gfx1151, driver 32.0.21041.1000 with torch
            # 2.11.0+rocm7.13.0, props.total_memory already spans the GTT pool
            # (89.47 GB against a 32 GB BIOS carve-out), so the split is a
            # property of some stacks and not of Windows. Hence the comparison
            # below rather than an unconditional adopt: this has to be inert
            # where the two agree. The free half of the reading is the untrusted
            # one either way, the total is fine.
            total_is_pool = False
            # Whether the driver CONFIRMED this total reaches the pool, as opposed
            # to it merely not having been widened. A probe that fails leaves a
            # carve-out-sized total standing, and a unified part is exactly where
            # that is possible, so the two are not the same question.
            pool_confirmed = False
            # POSITIVELY unified, not _rocm_props_total_is_carve_out, and this is
            # the gate on the probe rather than only on what the probe's answer is
            # used for. mem_get_info attaches a primary HIP context worth ~612 MiB
            # that is never released, main reaches this function without ever
            # calling it, and the carve-out classifier deliberately fails open for
            # an unclassified DISCRETE card. Gating on it would therefore have made
            # a telemetry poll take 612 MiB off every discrete GPU on the host, to
            # obtain a total that is then not used for anything: only a positively
            # unified part takes the widened total or the unified numerator.
            unified = False
            try:
                # Inside the try with the probe it gates: both are probes, and a
                # probe that throws must cost this device its correction, not its
                # place in the list.
                unified = _rocm_props_are_positively_unified(props)
                if unified and hasattr(mod, "mem_get_info"):
                    pool_bytes = int(mod.mem_get_info(ordinal)[1])
                    # >= not >, and the equal case is the ONLY one that occurs.
                    # hipMemGetInfo's total and hipDeviceProp_t::totalGlobalMem
                    # are both device->info().globalMemSize_ in clr (hip_memory.cpp,
                    # hip_device.cpp); only `free` is sourced separately. So these
                    # two numbers are the same variable read twice and can never
                    # differ on an AMD backend.
                    #
                    # That makes this look like a tautology, and taken alone it is.
                    # What carries it is the conjunction at pool_scoped below:
                    # on Windows the backend is PAL, and paldevice.cpp adds 50-75%
                    # of the WDDM shared heap to globalMemSize_ whenever
                    # settings().apuSystem_, which is Pal::GpuType::Integrated --
                    # the same flag that fills hipDeviceProp_t::integrated and so
                    # the same one _rocm_classify_unified_memory gates on. A part
                    # this classifier calls unified therefore HAS a pool-scoped
                    # total by construction, which is the proof the equality alone
                    # does not supply. Measured twice: 89.465 GB under a 32 GB
                    # carve-out here, and 107.87 GB under a 96 GB VGM in
                    # ROCm/TheRock#3032, matching the PAL formula exactly.
                    #
                    # Linux is the opposite regime -- globalMemSize_ is the KFD
                    # framebuffer heap and CAN be carve-out sized -- but this
                    # function returns ([], None) off Windows, so it never applies.
                    pool_confirmed = pool_bytes >= total_bytes
                    # Kept for the shrink guard, not because widening is reachable:
                    # per the above, pool_bytes > total_bytes cannot hold on AMD.
                    # The classifier says carve-out for a discrete card on an
                    # unsettled runtime too, and unlike the inventory this path
                    # carries a used: a shrunk total reports past 100% utilization
                    # and zero free.
                    if pool_bytes > total_bytes:
                        logger.debug(
                            "ROCm unified memory: ordinal %d total %.2f -> %.2f GB (driver pool)",
                            ordinal,
                            total_bytes / (1024**3),
                            pool_bytes / (1024**3),
                        )
                        total_bytes = pool_bytes
                        total_is_pool = True
            except Exception as e:
                logger.debug("ROCm APU driver total failed for ordinal %d: %s", ordinal, e)
            dev_meta.append(
                {
                    "index": phys_idx,
                    "visible_ordinal": ordinal,
                    "name": props.name,
                    "total_bytes": total_bytes,
                    "dedicated_bytes": dedicated_bytes,
                    # Second join key for _match_adapter_used_by_luid (#8863).
                    "gfx": str(getattr(props, "gcnArchName", "") or ""),
                    "total_is_pool": total_is_pool,
                    # Whether Shared Usage belongs in this device's numerator is
                    # a question about the PART, not about whether this call
                    # happened to widen anything. On the measured gfx1151 the two
                    # totals already agree, so total_is_pool stays false while the
                    # total is pool-scoped all the same, and Dedicated alone would
                    # be paired with it. It is also the stricter question:
                    # _rocm_props_total_is_carve_out fails open, so a discrete
                    # card can widen, and shared bytes are host memory its
                    # props.total_memory never counted. Same test
                    # get_gpu_memory_info applies before summing the two counters.
                    "positively_unified": unified,
                    "pool_confirmed": pool_confirmed,
                }
            )
        except Exception as e:
            logger.debug("torch property probe failed for ordinal %d: %s", ordinal, e)
    if not dev_meta:
        return [], None

    # Re-sampling here would leave a caller's cardinality check applied to a
    # DIFFERENT sample than the one attribution runs on, and costs a second
    # ~1.3 s PowerShell call. A validated snapshot is passed in instead.
    if adapters is None:
        adapters = _rocm_windows_perf_counter_vram_by_adapter()
    aggregate_gb: Optional[float] = None
    whole_adapter: list[Optional[int]] = [None] * len(dev_meta)
    if adapters:
        # Identity first, in the order the keys are exact: HIP's own LUID, then
        # the DirectX record reached by name or arch. Either answers where
        # capacity ranking declines unless the sizes force a pairing, which one
        # visible GPU never does.
        by_hip = _match_adapter_used_by_hip_luid(adapters, dev_meta)
        by_identity: Optional[tuple[list[Optional[float]], float]] = None
        if by_hip is not None:
            assigned, aggregate_bytes, whole_adapter = by_hip
            by_identity = (assigned, aggregate_bytes)
        else:
            by_identity = _match_adapter_used_by_luid(adapters, dev_meta)
        if by_identity is not None:
            assigned, aggregate_bytes = by_identity
        else:
            adapter_useds = [used for _, used in adapters]
            # Capacities the counters can fill, not the displayed totals.
            totals = [_adapter_counter_capacity(d) for d in dev_meta]
            assigned = _match_adapter_used_to_devices(adapter_useds, totals)
            aggregate_bytes = _rocm_windows_aggregate_used_bytes(adapter_useds, totals)
        if aggregate_bytes is not None:
            aggregate_gb = round(aggregate_bytes / (1024**3), 2)
    else:
        # Counter unavailable: show every GPU with a correct total, used unknown.
        assigned = [None] * len(dev_meta)

    # A pool-scoped total needs a numerator that spans the same ground. Dedicated
    # Usage does not: it saturates at the carve-out and is wrong past it, so
    # pairing it with the pool total would report a loaded card as mostly free,
    # and the caller derives free as total minus used.
    # _rocm_windows_unified_used_bytes sums Dedicated and Shared for exactly this
    # (#9362 measured the plateau at ~30.5 GiB on a gfx1151 and the overflow
    # landing in Shared). A decline still has to null the reading rather than
    # leave the carve-out figure standing under a pool-sized total.
    #
    # Pool-scoped covers both ways to get there: a total this call widened, and a
    # confirmed APU whose props.total_memory already spanned the pool, which is
    # what the measured gfx1151 does. Keying on the widening alone would leave
    # that host, the one this PR is about, on the plateaued counter. Being a UMA
    # part is not enough by itself though: it says what the device is, not what
    # scope the retained total has, and an APU whose driver probe failed still
    # carries a carve-out-sized one that Dedicated Usage is the right numerator
    # for. Hence pool_confirmed.
    pool_scoped = [
        m["total_is_pool"] or (m["positively_unified"] and m["pool_confirmed"]) for m in dev_meta
    ]
    if any(pool_scoped):
        only = dev_meta[0] if len(dev_meta) == 1 else None
        # Two conditions beyond "this device wants the sum", both of which the
        # matcher applies to its own readings and the unified helper does not:
        #
        #   adapters falsy   the Dedicated query already failed. Passing None
        #                    reads as "no snapshot supplied" and pays for the
        #                    same failing PowerShell call a second time, on a
        #                    path that polls, only to return unknown anyway.
        #   a NON-ZERO sub-threshold row
        #                    the helper picks the lone counter above the noise
        #                    floor, which on a masked host can be a hidden GPU
        #                    while the visible APU is the small one; its usage
        #                    would then be published as the APU's. The matcher
        #                    declines exactly this shape, requiring every dropped
        #                    counter to be an EXACT zero, so the same rule holds
        #                    here. An adapter at zero has nothing committed and
        #                    cannot be the holder of the survivor's bytes.
        wants_sum = only is not None and pool_scoped[0] and only["positively_unified"]
        placeholders_only = bool(adapters) and all(
            used == 0 for _, used in adapters if used < _ROCM_WIN_ADAPTER_MIN_BYTES
        )
        unified_used = (
            _rocm_windows_unified_used_bytes(adapters) if wants_sum and placeholders_only else None
        )
        if unified_used is not None:
            # Every other reading is clamped on its way through the matcher.
            # This one bypasses it, and the payload derives free as total minus
            # used, so an unclamped sum publishes negative free. The lower bound
            # is belt and braces: the helper already declines a negative reading
            # at source, and this keeps the guarantee local to the consumer that
            # publishes the number.
            unified_used = max(0.0, min(unified_used, float(only["total_bytes"])))
        assigned = [
            (unified_used if scoped else used) for scoped, used in zip(pool_scoped, assigned)
        ]
        # The aggregate is the visible set's exact total, so it survives only
        # when every pool-scoped member got a figure.
        aggregate_gb = (
            round(unified_used / (1024**3), 2)
            if unified_used is not None and len(dev_meta) == 1
            else None
        )

    devices: list[Dict[str, Any]] = []
    for meta, used_bytes, luid in zip(dev_meta, assigned, whole_adapter):
        total_gb = round(meta["total_bytes"] / (1024**3), 2)
        used_gb = round(used_bytes / (1024**3), 2) if used_bytes is not None else None
        devices.append(
            {
                "index": meta["index"],
                "visible_ordinal": meta["visible_ordinal"],
                "name": meta["name"],
                "used_gb": used_gb,
                "total_gb": total_gb,
                # Internal: filters the engine counters. Never served.
                "luid": luid,
            }
        )
    return devices, aggregate_gb


def _rocm_windows_device_payload_entry(
    device: DeviceType, dev: Dict[str, Any], gpu_util_pct: Optional[float]
) -> Dict[str, Any]:
    """Build a ``get_gpu_utilization`` device entry from a per-device VRAM dict."""
    total_gb = dev["total_gb"]
    used_gb = dev["used_gb"]
    return {
        "available": True,
        "backend": _backend_label(device),
        "index": dev["index"],
        "visible_ordinal": dev["visible_ordinal"],
        "name": dev.get("name", "Unknown"),
        "gpu_utilization_pct": gpu_util_pct,
        "temperature_c": None,
        "vram_used_gb": used_gb,
        "vram_total_gb": total_gb,
        "vram_utilization_pct": round((used_gb / total_gb) * 100, 1)
        if total_gb and total_gb > 0 and used_gb is not None
        else None,
        "power_draw_w": None,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


def _gpu_utilization_payload(
    device: DeviceType, devices: list[Dict[str, Any]], **metadata: Any
) -> Dict[str, Any]:
    """Keep the legacy primary-GPU shape and append all visible devices."""
    backend = _backend_label(device)
    normalized = []
    for ordinal, raw in enumerate(devices):
        dev = dict(raw)
        dev.setdefault("available", True)
        dev.setdefault("backend", backend)
        if dev.get("visible_ordinal") is None:
            dev["visible_ordinal"] = ordinal
        normalized.append(dev)

    normalized.sort(key = lambda dev: dev.get("visible_ordinal", dev.get("index", 0)))
    payload: Dict[str, Any] = {
        "available": bool(normalized),
        "backend": backend,
        "devices": normalized,
    }
    payload.update(metadata)
    if normalized:
        payload.update(normalized[0])
        payload["available"] = True
        payload["backend"] = normalized[0].get("backend", backend)
        payload["devices"] = normalized
    return payload


def get_gpu_utilization() -> Dict[str, Any]:
    """Live utilization snapshot for the primary GPU plus all visible GPUs."""
    device = get_device()

    if device == DeviceType.XPU:
        result = get_visible_gpu_utilization()
        return _gpu_utilization_payload(
            device,
            result.get("devices", []),
            parent_visible_gpu_ids = result.get("parent_visible_gpu_ids", []),
            index_kind = result.get("index_kind"),
        )

    if device == DeviceType.CUDA:
        parent_visible_spec = _get_parent_visible_gpu_spec()
        result = _smi_query(
            "get_visible_gpu_utilization",
            parent_visible_spec["numeric_ids"],
            parent_cuda_visible_devices = parent_visible_spec["raw"],
        )
        if result is not None and "devices" in result:
            devices = result["devices"]
            numeric_ids = parent_visible_spec.get("numeric_ids")
            if IS_ROCM and numeric_ids is not None:
                _reconcile_rocm_unified_memory(result, numeric_ids)

            return _gpu_utilization_payload(
                device,
                devices,
                backend_cuda_visible_devices = result.get("backend_cuda_visible_devices"),
                parent_visible_gpu_ids = result.get("parent_visible_gpu_ids", []),
                index_kind = result.get("index_kind"),
            )

        # Fallback Windows ROCm: per-adapter VRAM attribution (issue #7072), so
        # every visible GPU is shown instead of a sum collapsed onto one device.
        if IS_ROCM and platform.system() == "Windows":
            _win_ids = _get_parent_visible_gpu_spec().get("numeric_ids")
            if not _win_ids:
                _win_ids = list(range(_torch_get_physical_gpu_count() or 0))
            _win_devices, _win_aggregate = _rocm_windows_per_device_vram(_win_ids)
            if _win_devices:
                # Across several GPUs the engine sum isn't per-device, so leave it unset.
                _win_util = (
                    _rocm_windows_perf_counter_gpu_util_pct(_win_devices[0].get("luid"))
                    if len(_win_devices) == 1
                    else None
                )
                return _gpu_utilization_payload(
                    device,
                    [
                        _rocm_windows_device_payload_entry(device, _wd, _win_util)
                        for _wd in _win_devices
                    ],
                    vram_used_gb_aggregate = _win_aggregate,
                )

        # Fallback Linux ROCm
        if IS_ROCM and platform.system() == "Linux":
            _linux_used, _linux_total = _rocm_linux_sysfs_vram_gb()
            if _linux_used is not None and _linux_total is not None:
                _linux_util = _rocm_linux_sysfs_gpu_busy_pct()
                _linux_temp = _rocm_linux_sysfs_temp_c()
                _linux_power = _rocm_linux_sysfs_power_w()
                return _gpu_utilization_payload(
                    device,
                    [
                        {
                            "available": True,
                            "backend": _backend_label(device),
                            "index": 0,
                            "visible_ordinal": 0,
                            "gpu_utilization_pct": _linux_util,
                            "temperature_c": _linux_temp,
                            "vram_used_gb": _linux_used,
                            "vram_total_gb": _linux_total,
                            "vram_utilization_pct": round((_linux_used / _linux_total) * 100, 1)
                            if _linux_total > 0
                            else None,
                            "power_draw_w": _linux_power,
                            "power_limit_w": None,
                            "power_utilization_pct": None,
                        }
                    ],
                )

        # Last resort: torch mem_get_info (process-local) for all visible GPUs
        _visible_spec = _get_parent_visible_gpu_spec()
        _numeric_ids = _visible_spec.get("numeric_ids") or []
        if not _numeric_ids:
            visible_count = _torch_get_physical_gpu_count() or 0
            _numeric_ids = list(range(visible_count))

        _torch_devices = _torch_get_per_device_info(_numeric_ids)
        if _torch_devices:
            gpu_array = []
            for _td in _torch_devices:
                _total = _td["total_gb"]
                _used = _td["used_gb"]
                gpu_array.append(
                    {
                        "available": True,
                        "backend": _backend_label(device),
                        "index": _td["index"],
                        "name": _td.get("name", "Unknown"),
                        "gpu_utilization_pct": None,
                        "temperature_c": None,
                        "vram_used_gb": _used,
                        "vram_total_gb": _total,
                        "vram_utilization_pct": round((_used / _total) * 100, 1)
                        if _total > 0 and _used is not None
                        else None,
                        "power_draw_w": None,
                        "power_limit_w": None,
                        "power_utilization_pct": None,
                    }
                )
            return _gpu_utilization_payload(device, gpu_array)

    # MLX
    if device == DeviceType.MLX:
        try:
            import psutil
            agx = _read_apple_gpu_stats()
            total_bytes = psutil.virtual_memory().total
        except Exception as e:
            logger.error(f"Error getting MLX GPU utilization: {e}")
            return {"available": False, "backend": device.value, "devices": [], "error": str(e)}

        allocated_bytes = agx.get("vram_used_bytes", 0) or 0
        vram_used_gb = allocated_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        try:
            from core.training import get_training_backend

            tb = get_training_backend()
            tb_progress = getattr(tb, "_progress", None)
            if tb_progress is not None and getattr(tb_progress, "is_training", False):
                tb_peak = getattr(tb_progress, "peak_memory_gb", None)
                if tb_peak is not None and tb_peak > 0:
                    vram_used_gb = float(tb_peak)
        except Exception:
            pass

        from . import apple

        return _gpu_utilization_payload(
            device,
            [
                {
                    "available": True,
                    "backend": device.value,
                    "index": 0,
                    "visible_ordinal": 0,
                    "gpu_utilization_pct": agx.get("utilization_pct") if agx else None,
                    "temperature_c": apple.read_gpu_temperature_c(),
                    "vram_used_gb": round(vram_used_gb, 2),
                    "vram_total_gb": round(total_gb, 2),
                    "vram_utilization_pct": round((vram_used_gb / total_gb) * 100, 1)
                    if total_gb > 0
                    else None,
                    "power_draw_w": apple.read_gpu_power_w(),
                    "power_limit_w": None,
                    "power_utilization_pct": None,
                }
            ],
        )

    mem = get_gpu_memory_info()
    if device != DeviceType.CPU and mem.get("available"):
        return _gpu_utilization_payload(
            device,
            [
                {
                    "available": True,
                    "backend": _backend_label(device),
                    "index": mem.get("device", 0),
                    "visible_ordinal": 0,
                    "gpu_utilization_pct": None,
                    "temperature_c": None,
                    "vram_used_gb": round(mem.get("allocated_gb", 0), 2),
                    "vram_total_gb": round(mem.get("total_gb", 0), 2),
                    "vram_utilization_pct": round(mem.get("utilization_pct", 0), 1),
                    "power_draw_w": None,
                    "power_limit_w": None,
                    "power_utilization_pct": None,
                }
            ],
        )

    return {"available": False, "backend": _backend_label(device), "devices": []}


def _apply_unified_memory_correction(
    device_metrics: Dict[str, Any], torch_info: Dict[str, Any]
) -> None:
    """Per-device reconciliation: when torch reports a larger memory total
    than amd-smi, overwrite the smi VRAM fields in place.

    Used by both the multi-device and primary-device reconcilers so the two
    endpoints stay in sync on AMD iGPUs with unified memory.
    """
    torch_total_gb = torch_info["total_gb"]
    torch_used_gb = torch_info.get("used_gb")
    smi_total_gb = device_metrics.get("vram_total_gb") or 0.0
    # torch sees the full unified (GTT) pool; amd-smi only the dedicated carve-out.
    # Adopt torch's larger total regardless of used: on Windows ROCm torch_used is
    # None (free==total sentinel) but its total stays authoritative. Overwrite used
    # only when torch's is known, then recompute utilization against whatever remains.
    if torch_total_gb > smi_total_gb:
        device_metrics["vram_total_gb"] = torch_total_gb
        if torch_used_gb is not None:
            device_metrics["vram_used_gb"] = torch_used_gb
        _used_for_pct = device_metrics.get("vram_used_gb")
        device_metrics["vram_utilization_pct"] = (
            round((_used_for_pct / torch_total_gb) * 100, 1)
            if torch_total_gb > 0 and _used_for_pct is not None
            else None
        )
        logger.debug(
            "ROCm unified memory: adopted torch mem_get_info total (%.2f GB) over "
            "amd-smi (%.2f GB) for device %s",
            torch_total_gb,
            smi_total_gb,
            torch_info.get("index"),
        )


def _reconcile_rocm_unified_memory(utilization: Dict[str, Any], device_indices: list[int]) -> None:
    """Fix amd-smi VRAM for ROCm unified-memory GPUs (e.g. Strix Halo).

    amd-smi reports only the dedicated slice; torch sees the full GTT pool. When
    torch total > smi total, overwrite per-device VRAM fields with the real value.
    """
    torch_devices = _torch_get_per_device_info(device_indices)
    if not torch_devices:
        return
    torch_by_index = {td["index"]: td for td in torch_devices}
    for dev in utilization.get("devices", []):
        td = torch_by_index.get(dev.get("index"))
        if td is None:
            continue
        _apply_unified_memory_correction(dev, td)


def _reconcile_primary_rocm_unified_memory(
    utilization: Dict[str, Any], parent_visible_spec: Dict[str, Any]
) -> None:
    """Same fix as _reconcile_rocm_unified_memory for the flat primary-GPU dict."""
    numeric_ids = parent_visible_spec.get("numeric_ids")
    if numeric_ids is None:
        # No visibility env var set: torch ordinal 0 is the primary device.
        primary_idx = [0]
    elif len(numeric_ids) == 0:
        # Empty mask: no GPU visible. Querying torch device 0 would raise or
        # return stale data, so bail rather than write bad values.
        return
    else:
        primary_idx = [int(numeric_ids[0])]
    torch_devices = _torch_get_per_device_info(primary_idx)
    if not torch_devices:
        return
    _apply_unified_memory_correction(utilization, torch_devices[0])


def _rocm_visibility_mask_active() -> bool:
    """True when any ROCm/CUDA visibility variable filters the device set."""
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        value = os.environ.get(var)
        if value and value.strip():
            return True
    return False


def _rocm_single_numeric_mask_matches(devices: list[Dict[str, Any]]) -> bool:
    if _rocm_device_ordinal_active() or _rocm_visibility_masks_are_stacked():
        return False

    selected_mask = None
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if var in os.environ:
            selected_mask = os.environ[var]
            break
    if selected_mask is None:
        return False
    tokens = [token.strip() for token in selected_mask.split(",") if token.strip()]
    try:
        numeric_ids = [int(token) for token in tokens]
    except ValueError:
        return False
    return len(set(numeric_ids)) == len(numeric_ids) and numeric_ids == [
        dev.get("index") for dev in devices
    ]


def _rocm_linux_sysfs_vram_by_index(
    devices: list[Dict[str, Any]], *, allow_numeric_mask: bool = False
) -> Dict[int, tuple[float, float]]:
    """Map safe physical ROCm indices to their raw Linux sysfs VRAM readings."""
    if not devices or platform.system() != "Linux":
        return {}
    pci_by_ordinal = _rocm_kfd_gpu_pci_ids()
    if not pci_by_ordinal:
        return {}
    if len(devices) != len(pci_by_ordinal):
        return {}
    if _rocm_visibility_mask_active():
        unambiguous_single = (
            allow_numeric_mask
            and len(devices) == 1
            and len(pci_by_ordinal) == 1
            and not _rocm_device_ordinal_active()
            and not _rocm_visibility_masks_are_stacked()
        )
        if not unambiguous_single and (
            not allow_numeric_mask or not _rocm_single_numeric_mask_matches(devices)
        ):
            return {}
    vram_by_pci = _rocm_linux_sysfs_vram_by_pci_gb()
    resolved: Dict[int, tuple[float, float]] = {}
    for dev in devices:
        index = dev.get("index")
        if not isinstance(index, int) or not (0 <= index < len(pci_by_ordinal)):
            continue
        entry = vram_by_pci.get(pci_by_ordinal[index].lower())
        if entry is None:
            continue
        resolved[index] = entry
    return resolved


def _rocm_system_wide_vram_by_index(
    devices: list[Dict[str, Any]],
) -> Dict[int, tuple[float, float]]:
    """Decide the system-wide overlay without applying it.

    Returns ``{physical index: (used_gb, total_gb)}`` for every device sysfs can
    speak for, and omits the ones it cannot. Split out from
    _overlay_system_wide_vram so a caller can ask whether sysfs covers the whole
    visible set BEFORE paying torch for occupancy it would then discard. Reads
    only ``vram_total_gb`` off ``devices``; it never mutates them.
    """
    raw = _rocm_linux_sysfs_vram_by_index(devices)
    resolved: Dict[int, tuple[float, float]] = {}
    for dev in devices:
        index = dev.get("index")
        entry = raw.get(index)
        if entry is None:
            continue
        used, total = entry
        dev_total = dev.get("vram_total_gb") or 0.0
        # Overlay only a device that maps 1:1 to the whole card: torch total must
        # match sysfs total within ~10%. A mismatch either way means a different
        # memory scope -- a unified-memory APU (sysfs sees only the dedicated
        # slice, torch the GTT pool) or a partitioned MI300 (sysfs reports the
        # whole card, dwarfing a partition) -- and overlaying would misstate free
        # VRAM (a partition would look like it has the whole card free).
        if dev_total <= 0 or abs(total - dev_total) > 0.1 * dev_total:
            continue
        resolved[index] = (used, total)
    return resolved


def _rocm_linux_shared_pool_host_gb_by_index(devices: list[Dict[str, Any]]) -> Dict[int, float]:
    """Map known APUs to the host-backed part above the reserved sysfs heap."""
    raw = _rocm_linux_sysfs_vram_by_index(devices, allow_numeric_mask = True)
    shared: Dict[int, float] = {}
    for dev in devices:
        if not dev.get("_rocm_known_unified"):
            continue
        index = dev.get("index")
        entry = raw.get(index)
        if entry is None:
            continue
        _used, sysfs_total = entry
        torch_total = dev.get("total_gb") or 0.0
        if sysfs_total > 0 and torch_total - sysfs_total > 0.1 * torch_total:
            shared[index] = round(torch_total - sysfs_total, 2)
    return shared


def _apply_system_wide_vram(
    devices: list[Dict[str, Any]], resolved: Dict[int, tuple[float, float]]
) -> None:
    """Write a _rocm_system_wide_vram_by_index result onto the device dicts."""
    for dev in devices:
        entry = resolved.get(dev.get("index"))
        if entry is None:
            continue
        used, total = entry
        dev["vram_used_gb"] = used
        dev["vram_total_gb"] = total
        dev["vram_utilization_pct"] = round((used / total) * 100, 1) if total > 0 else None


def _overlay_system_wide_vram(devices: list[Dict[str, Any]]) -> None:
    """Replace process-local torch VRAM with system-wide Linux ROCm figures.

    The torch fallback is process-local, so a model served by the separate
    llama-server process reads as ~0 used even with the GPU full (#7072). DRM
    sysfs gives per-card figures the kernel updates across all processes. Sources
    are matched by the device's PHYSICAL index (never list position), and only
    when NO visibility mask is active and the device count equals the host GPU
    count; under any mask the index is not a verifiable host ordinal, so torch's
    figures are kept. Best-effort, in place: a device with no matching card, or a
    unified-memory APU whose sysfs total is below torch's GTT-backed total, keeps
    torch's (mirrors _apply_unified_memory_correction).

    Windows is intentionally not overlaid: its per-adapter perf counters cannot be
    mapped to ROCm ordinals and miss WDDM shared memory, so the multi-GPU view
    keeps torch there rather than risk misattributing another adapter's usage.
    """
    _apply_system_wide_vram(devices, _rocm_system_wide_vram_by_index(devices))


def get_visible_gpu_utilization() -> Dict[str, Any]:
    device = get_device()

    if device == DeviceType.CUDA:
        parent_visible_spec = _get_parent_visible_gpu_spec()
        result = _smi_query(
            "get_visible_gpu_utilization",
            parent_visible_spec["numeric_ids"],
            parent_cuda_visible_devices = parent_visible_spec["raw"],
        )
        if result is not None:
            result["backend"] = _backend_label(device)
            numeric_ids = parent_visible_spec.get("numeric_ids")
            if IS_ROCM and numeric_ids is not None:
                # Fix unified-memory VRAM on AMD iGPUs (Strix Halo etc.).
                _reconcile_rocm_unified_memory(result, numeric_ids)
            return result

        # Windows AMD/ROCm (issue #7072): the System tab's VRAM source. The torch
        # fallback below would report used==0 (free==total), so read per-adapter
        # Dedicated Usage instead; total from torch properties.
        if IS_ROCM and platform.system() == "Windows":
            win_numeric_ids = parent_visible_spec.get("numeric_ids")
            if win_numeric_ids:
                win_ids = win_numeric_ids
                win_index_kind = "physical"
            else:
                win_ids = list(range(_torch_get_physical_gpu_count() or 0))
                win_index_kind = "relative"
            win_devices, win_aggregate = _rocm_windows_per_device_vram(win_ids)
            if win_devices:
                devices = []
                for wd in win_devices:
                    total = wd["total_gb"]
                    used = wd["used_gb"]
                    devices.append(
                        {
                            "index": wd["index"],
                            "index_kind": win_index_kind,
                            "visible_ordinal": wd["visible_ordinal"],
                            "name": wd.get("name"),
                            "gpu_utilization_pct": None,
                            "temperature_c": None,
                            "vram_used_gb": used,
                            "vram_total_gb": total,
                            "vram_utilization_pct": round((used / total) * 100, 1)
                            if total and total > 0 and used is not None
                            else None,
                            "power_draw_w": None,
                            "power_limit_w": None,
                            "power_utilization_pct": None,
                        }
                    )
                return {
                    "available": True,
                    "backend": _backend_label(device),
                    "parent_visible_gpu_ids": win_numeric_ids or [],
                    "devices": devices,
                    "index_kind": win_index_kind,
                    # Host total for the System tab tile: known even when no single
                    # device's usage is attributable (#7452).
                    "vram_used_gb_aggregate": win_aggregate,
                }

    # Torch-based fallback for CUDA (nvidia-smi unavailable, AMD ROCm) and XPU (Intel)
    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_ids = get_parent_visible_gpu_ids()
        # Empty parent_ids (UUID/MIG mask or no CVD): enumerate torch ordinals.
        if parent_ids:
            torch_indices = parent_ids
            index_kind = "physical"
        else:
            visible_count = _torch_get_physical_gpu_count() or 0
            torch_indices = list(range(visible_count))
            index_kind = "relative"

        # Linux ROCm: sysfs first. The overlay below replaces torch's process-local
        # figures with these same numbers anyway, so asking torch first buys a
        # permanent context and throws the answer away. Totals come from device
        # properties, which cost nothing and are what the overlay's check compares
        # against. Only when sysfs covers EVERY visible device; otherwise fall
        # through to torch, so a device the overlay declines still gets a number.
        if IS_ROCM and index_kind == "physical" and platform.system() == "Linux":
            inventory = _torch_get_device_inventory(torch_indices)
            probe = [{"index": inv["index"], "vram_total_gb": inv["total_gb"]} for inv in inventory]
            resolved = _rocm_system_wide_vram_by_index(probe)
            if inventory and all(inv["index"] in resolved for inv in inventory):
                devices = [
                    {
                        "index": inv["index"],
                        "index_kind": index_kind,
                        "visible_ordinal": inv["visible_ordinal"],
                        "gpu_utilization_pct": None,
                        "temperature_c": None,
                        "vram_used_gb": None,
                        "vram_total_gb": inv["total_gb"],
                        "vram_utilization_pct": None,
                        "power_draw_w": None,
                        "power_limit_w": None,
                        "power_utilization_pct": None,
                    }
                    for inv in inventory
                ]
                _apply_system_wide_vram(devices, resolved)
                return {
                    "available": True,
                    "backend": _backend_label(device),
                    "parent_visible_gpu_ids": parent_ids,
                    "devices": devices,
                    "index_kind": index_kind,
                }

        torch_devices = _torch_get_per_device_info(torch_indices)
        if torch_devices:
            devices = []
            for td in torch_devices:
                total = td["total_gb"]
                used = td["used_gb"]
                # used=None is a deliberate "telemetry unavailable" signal
                # from _torch_get_per_device_info (e.g. XPU without
                # mem_get_info); propagate None instead of dividing by it. On
                # CUDA/ROCm used is always an int, so this stays byte-identical.
                vram_pct = (
                    round((used / total) * 100, 1) if used is not None and total > 0 else None
                )
                devices.append(
                    {
                        "index": td["index"],
                        "index_kind": index_kind,
                        "visible_ordinal": td["visible_ordinal"],
                        "gpu_utilization_pct": None,
                        "temperature_c": None,
                        "vram_used_gb": used,
                        "vram_total_gb": total,
                        "vram_utilization_pct": vram_pct,
                        "power_draw_w": None,
                        "power_limit_w": None,
                        "power_utilization_pct": None,
                    }
                )
            if IS_ROCM and index_kind == "physical":
                # Swap process-local torch VRAM for system-wide sysfs so a model
                # held by the separate llama-server process shows up (#7072).
                # Physical-index only: a relative index (UUID/MIG mask) is not a
                # host GPU id. The overlay verifies the rest itself.
                _overlay_system_wide_vram(devices)
            return {
                "available": True,
                "backend": _backend_label(device),
                "parent_visible_gpu_ids": parent_ids,
                "devices": devices,
                "index_kind": index_kind,
            }

    if device == DeviceType.MLX:
        mem = get_gpu_memory_info()
        if not mem.get("available"):
            return {
                "available": False,
                "backend": _backend_label(device),
                "parent_visible_gpu_ids": [],
                "devices": [],
                "index_kind": "relative",
            }
        return {
            "available": True,
            "backend": _backend_label(device),
            "parent_visible_gpu_ids": [0],
            "devices": [
                {
                    "index": 0,
                    "index_kind": "relative",
                    "visible_ordinal": 0,
                    "gpu_utilization_pct": None,
                    "temperature_c": None,
                    "vram_used_gb": round(mem.get("allocated_gb", 0), 2),
                    "vram_total_gb": round(mem.get("total_gb", 0), 2),
                    "vram_utilization_pct": round(mem.get("utilization_pct", 0), 1),
                    "power_draw_w": None,
                    "power_limit_w": None,
                    "power_utilization_pct": None,
                }
            ],
            "index_kind": "relative",
        }

    return {
        "available": False,
        "backend": _backend_label(device),
        "parent_visible_gpu_ids": [],
        "devices": [],
        "index_kind": "vulkan",
    }


# ========== Multi-GPU Detection & Safe num_proc ==========

_physical_gpu_count: Optional[int] = None
# Whether the cached count came from the SMI (physical) or the torch fallback
# (visibility-filtered). Only the former can answer "is this host single-GPU".
_physical_gpu_count_from_smi: bool = False
_visible_gpu_count: Optional[int] = None


def _get_parent_visible_gpu_spec() -> Dict[str, Any]:
    # On Intel XPU, visibility is controlled by ZE_AFFINITY_MASK (Level Zero),
    # not CUDA_VISIBLE_DEVICES.
    if get_device() == DeviceType.XPU:
        xpu_mask_raw = os.environ.get("ZE_AFFINITY_MASK")
        composite = _xpu_hierarchy_is_composite()

        if xpu_mask_raw is None:
            # COMPOSITE: root GPU IDs are stable physical IDs.
            if composite:
                return {
                    "raw": None,
                    "numeric_ids": list(range(get_physical_gpu_count())),
                    "supports_explicit_gpu_ids": True,
                }
            # FLAT (oneAPI default): ordinals are tile/device handles, not
            # physical GPU IDs. numeric_ids=None so telemetry uses relative
            # ordinals; explicit selection needs ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE.
            return {
                "raw": None,
                "numeric_ids": None,
                "supports_explicit_gpu_ids": False,
            }

        xpu_mask = xpu_mask_raw.strip()
        if xpu_mask == "":
            return {
                "raw": xpu_mask,
                "numeric_ids": [],
                "supports_explicit_gpu_ids": True,
            }

        # Subdevice syntax ("N.M") expands one root into multiple
        # logical devices -- not addressable by explicit root-ID selection.
        has_subdevice = any("." in token.strip() for token in xpu_mask.split(",") if token.strip())
        if has_subdevice:
            return {
                "raw": xpu_mask,
                "numeric_ids": None,
                "supports_explicit_gpu_ids": False,
            }

        # FLAT numeric entries are tile handles, not physical GPU IDs. Keep
        # numeric_ids unresolved so every telemetry and picker consumer uses
        # relative torch ordinals and cannot advertise them as pinnable roots.
        if not composite:
            tokens = [token.strip() for token in xpu_mask.split(",") if token.strip()]
            if tokens and all(token.isdecimal() for token in tokens):
                return {
                    "raw": xpu_mask,
                    "numeric_ids": None,
                    "supports_explicit_gpu_ids": False,
                }
            return {
                "raw": xpu_mask,
                "numeric_ids": None,
                "supports_explicit_gpu_ids": False,
            }

        # COMPOSITE + pure numeric (subdevice handled above). _parse_ze_mask_roots
        # maps to root GPU IDs, dropping non-decimal tokens so "*"/"GPU-uuid" -> [].
        roots_with_dupes = _parse_ze_mask_roots(xpu_mask)
        if not roots_with_dupes:
            # Unparseable mask (e.g. "*", "GPU-uuid") -- cannot map to
            # physical root IDs.
            return {
                "raw": xpu_mask,
                "numeric_ids": None,
                "supports_explicit_gpu_ids": False,
            }

        return {
            "raw": xpu_mask,
            "numeric_ids": roots_with_dupes,
            "supports_explicit_gpu_ids": True,
        }

    # ROCm uses HIP/ROCR_VISIBLE_DEVICES on top of CUDA_VISIBLE_DEVICES; check
    # them first. Explicit None checks (not `or`) so "" reads as "no visible GPUs".
    cuda_visible = None
    # Prefer ROCm masks only on a ROCm host or when no CUDA mask is set, so a
    # stale HIP_VISIBLE_DEVICES on NVIDIA can't override CUDA_VISIBLE_DEVICES.
    _is_rocm_spec = IS_ROCM or (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        and ("HIP_VISIBLE_DEVICES" in os.environ or "ROCR_VISIBLE_DEVICES" in os.environ)
    )
    if _is_rocm_spec:
        hip_vis = os.environ.get("HIP_VISIBLE_DEVICES")
        # ROCR_VISIBLE_DEVICES is Linux-only: Windows HIP has no ROCr layer, so a
        # stray ROCR var there masks nothing and must not be read as the
        # ordinal->physical mapping (mirrors the llama.cpp backend).
        rocr_vis = None if sys.platform == "win32" else os.environ.get("ROCR_VISIBLE_DEVICES")
        if hip_vis is not None:
            cuda_visible = hip_vis
        elif rocr_vis is not None:
            cuda_visible = rocr_vis
    if cuda_visible is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible is None:
        return {
            "raw": None,
            "numeric_ids": list(range(get_physical_gpu_count())),
            "supports_explicit_gpu_ids": True,
        }

    cuda_visible = cuda_visible.strip()
    if cuda_visible == "" or cuda_visible == "-1":
        return {
            "raw": cuda_visible,
            "numeric_ids": [],
            "supports_explicit_gpu_ids": True,
        }

    tokens = [value.strip() for value in cuda_visible.split(",") if value.strip()]
    try:
        numeric_ids = [int(value) for value in tokens]
    except ValueError:
        return {
            "raw": cuda_visible,
            "numeric_ids": None,
            "supports_explicit_gpu_ids": False,
        }

    return {
        "raw": cuda_visible,
        "numeric_ids": numeric_ids,
        "supports_explicit_gpu_ids": True,
    }


def get_parent_visible_gpu_ids() -> list[int]:
    parent_visible_ids = _get_parent_visible_gpu_spec()["numeric_ids"]
    return list(parent_visible_ids) if parent_visible_ids is not None else []


def resolve_requested_gpu_ids(
    gpu_ids: Optional[list[int]], *, is_vulkan: bool = False
) -> list[int]:
    parent_visible_spec = _get_parent_visible_gpu_spec()
    parent_visible_ids = get_parent_visible_gpu_ids()
    physical_gpu_count = get_physical_gpu_count()

    if gpu_ids is None:
        return [] if is_vulkan else parent_visible_ids

    requested_ids = list(gpu_ids)
    if len(requested_ids) == 0:
        return [] if is_vulkan else parent_visible_ids

    if is_vulkan:
        # A Vulkan build selects by ggml Vulkan ordinal (--device VulkanN), a separate
        # index space from CUDA/ROCm ids that may be empty under CPU-only torch. The
        # CUDA parent-visible / physical-count checks below do not apply; only reject
        # malformed ordinals (issue #7239).
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError(f"Invalid gpu_ids {requested_ids}: duplicate GPU IDs are not allowed.")
        negative_ids = [gpu_id for gpu_id in requested_ids if gpu_id < 0]
        if negative_ids:
            raise ValueError(
                f"Invalid gpu_ids {requested_ids}: GPU IDs must be non-negative. "
                f"Rejected IDs: {negative_ids}."
            )
        return requested_ids

    if not parent_visible_spec["supports_explicit_gpu_ids"]:
        env_var_name = (
            "ZE_AFFINITY_MASK" if get_device() == DeviceType.XPU else "CUDA_VISIBLE_DEVICES"
        )
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: explicit physical GPU IDs are "
            f"unsupported when {env_var_name} uses non-numeric or subdevice "
            f"entries ({parent_visible_spec['raw']!r}). Omit gpu_ids to use "
            "the parent-visible devices."
        )

    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: duplicate GPU IDs are not allowed. "
            f"Parent-visible GPUs: {parent_visible_ids}"
        )

    # Reject negative IDs.
    negative_ids = [gpu_id for gpu_id in requested_ids if gpu_id < 0]
    if negative_ids:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: GPU IDs must be non-negative. "
            f"Rejected IDs: {negative_ids}. Parent-visible GPUs: {parent_visible_ids}"
        )

    # Only enforce the physical upper bound when the count is reliable (nvidia-smi).
    # A torch count reflects only visible devices, so it could falsely reject valid
    # physical indices. The parent-visible check below is always authoritative.
    if physical_gpu_count > 0 and parent_visible_ids:
        max_parent_id = max(parent_visible_ids)
        if physical_gpu_count > max_parent_id:
            # Count is plausibly physical, so enforce it.
            out_of_range = [gpu_id for gpu_id in requested_ids if gpu_id >= physical_gpu_count]
            if out_of_range:
                raise ValueError(
                    f"Invalid gpu_ids {requested_ids}: IDs must be physical GPU IDs "
                    f"between 0 and {physical_gpu_count - 1}. "
                    f"Rejected IDs: {out_of_range}. Parent-visible GPUs: {parent_visible_ids}"
                )

    disallowed_ids = [gpu_id for gpu_id in requested_ids if gpu_id not in parent_visible_ids]
    if disallowed_ids:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: requested GPUs {disallowed_ids} are "
            f"outside the parent-visible set {parent_visible_ids}"
        )

    return requested_ids


def _resolve_model_identifier_for_gpu_estimate(
    model_name: str, hf_token: Optional[str] = None
) -> str:
    try:
        from utils.models.model_config import ModelConfig

        config = ModelConfig.from_identifier(model_name, hf_token = hf_token)
        if config and config.is_lora and config.base_model:
            return config.base_model
        return config.identifier if config else model_name
    except Exception as e:
        logger.debug("Could not resolve base model for GPU estimate '%s': %s", model_name, e)
        return model_name


def _get_local_weight_size_bytes(model_name: str) -> Optional[int]:
    model_path = Path(model_name)
    if not model_path.exists():
        return None

    weight_exts = (".safetensors", ".bin", ".pt", ".pth")
    # Skip intermediate training checkpoints: a run dir can hold several
    # checkpoint-*/global_step* snapshots, but export loads only the model at
    # the root, so counting them would multiply the estimate.
    skip_prefixes = ("checkpoint-", "global_step")
    total = 0
    for file in model_path.rglob("*"):
        if not file.is_file() or file.suffix not in weight_exts:
            continue
        rel = file.relative_to(model_path)
        if any(part.startswith(skip_prefixes) for part in rel.parts):
            continue
        total += file.stat().st_size
    return total if total > 0 else None


def _get_hf_safetensors_total_params(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[int]:
    try:
        from utils.utils import hf_env_offline

        if hf_env_offline():
            return None

        from huggingface_hub import model_info as hf_model_info

        info = hf_model_info(model_name, token = hf_token)
        safetensors = getattr(info, "safetensors", None)
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if total:
                return int(total)
    except Exception as e:
        logger.warning("Could not get safetensors metadata for '%s': %s", model_name, e)
    return None


def _load_config_for_gpu_estimate(model_name: str, hf_token: Optional[str] = None):
    # Estimation needs only declarative config.json fields, and this probe runs
    # on model selection, so read raw config.json (never run auto_map Python) and
    # expose it as an attribute namespace for downstream getattr access.
    try:
        from utils.transformers_version import _load_config_json

        cfg = _load_config_json(model_name, hf_token = hf_token)
        if cfg is None:
            return None

        def _to_ns(d):
            if isinstance(d, dict):
                return types.SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
            return d

        return _to_ns(cfg)
    except Exception as e:
        # A 5.x-only config can't be parsed by the default transformers; that is
        # expected (the worker reloads under the sidecar), so only warn for default tier.
        tier = "default"
        try:
            from utils.transformers_version import get_transformers_tier
            tier = get_transformers_tier(model_name)
        except Exception:
            pass
        if tier != "default":
            _tier_version = {"510": "5.10.x", "530": "5.3.0", "550": "5.5.0"}.get(tier, "5.x")
            logger.info(
                "Config for '%s' not parseable by the default transformers; "
                "needs transformers %s and will be loaded with that sidecar in the worker",
                model_name,
                _tier_version,
            )
        else:
            logger.warning("Could not load config for '%s': %s", model_name, e)
        return None


def _determine_attention_impl_for_gpu_estimate(config) -> str:
    # torch.distributed is incomplete on Windows ROCm (torch._C._distributed_c10d
    # can't be imported). Inject stubs into sys.modules before importing
    # torch.distributed, then patch the missing process-group helpers.
    if sys.platform == "win32" and IS_ROCM:
        # Dummy for any name torch.distributed imports from these stubs.
        class _Dummy:
            pass

        for _c10d_name in (
            "torch._C._distributed_c10d",
            "torch._C._distributed_autograd",
            "torch._C._distributed_rpc",
        ):
            if _c10d_name not in sys.modules:
                _stub = types.ModuleType(_c10d_name)
                # No-op dummies for names torch.distributed imports from _distributed_c10d.
                for _sym in (
                    "FakeProcessGroup",
                    "ProcessGroup",
                    "Work",
                    "Store",
                    "PrefixStore",
                    "FileStore",
                    "TCPStore",
                    "HashStore",
                    "Reducer",
                    "Logger",
                    "DistributedDebugLevel",
                    "GradBucket",
                    "BuiltinCommHookType",
                ):
                    setattr(_stub, _sym, _Dummy)
                sys.modules[_c10d_name] = _stub

    try:
        import torch.distributed as _td
        for _attr, _stub in (
            ("is_initialized", lambda: False),
            ("is_available", lambda: False),
            ("get_rank", lambda: 0),
            ("get_world_size", lambda: 1),
            ("is_torchelastic_launched", lambda: False),
        ):
            if not hasattr(_td, _attr):
                setattr(_td, _attr, _stub)
    except ImportError:
        pass

    from unsloth.models._utils import resolve_attention_implementation
    from transformers import AutoModel, AutoModelForCausalLM

    # why: resolve_attention_implementation writes _attn_implementation onto the
    # config and propagates to nested sub-configs; a shallow copy would still
    # mutate the cached config's shared inner objects. Deepcopy isolates them.
    config_copy = copy.deepcopy(config)

    model_class = None
    for auto_model in (AutoModelForCausalLM, AutoModel):
        mapping = getattr(auto_model, "_model_mapping", None)
        if mapping is None:
            continue
        try:
            if config_copy.__class__ in mapping:
                model_class = mapping[config_copy.__class__]
                break
        except Exception:
            continue

    return resolve_attention_implementation(model_class, config_copy)


def _estimate_fp16_model_size_bytes_from_config(config) -> Optional[int]:
    from .vram_estimation import extract_arch_config, compute_total_params

    arch = extract_arch_config(config)
    if arch is None:
        return None
    return compute_total_params(arch) * 2


def _estimate_fp16_model_size_bytes_from_vllm_utils(config) -> Optional[int]:
    if config is None:
        return None

    previous_unsloth_present = os.environ.get("UNSLOTH_IS_PRESENT")
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    try:
        from unsloth_zoo import vllm_utils as _vllm_utils

        synthetic_total_bytes = 1024 * (1024**3)
        original_get_mem_info = _vllm_utils.get_mem_info
        try:
            _vllm_utils.get_mem_info = lambda: (
                synthetic_total_bytes,
                synthetic_total_bytes,
            )
            _, _, _, memory_left_for_kv_cache_gb = _vllm_utils.approximate_vllm_memory_usage(
                config,
                load_in_4bit = False,
                load_in_8bit = False,
                max_seq_length = 1,
                gpu_memory_utilization = 1.0,
                enable_lora = False,
                account_for_gradients = False,
                cuda_graph_overhead = False,
            )
        finally:
            _vllm_utils.get_mem_info = original_get_mem_info
    except Exception as e:
        logger.debug("Could not estimate model size via vllm_utils: %s", e)
        return None
    finally:
        if previous_unsloth_present is None:
            os.environ.pop("UNSLOTH_IS_PRESENT", None)
        else:
            os.environ["UNSLOTH_IS_PRESENT"] = previous_unsloth_present

    model_size_gb = 1024.0 - memory_left_for_kv_cache_gb
    if model_size_gb <= 0:
        return None
    return int(round(model_size_gb * (1024**3)))


def estimate_fp16_model_size_bytes(
    model_name: str, hf_token: Optional[str] = None
) -> tuple[Optional[int], str]:
    estimate_model = _resolve_model_identifier_for_gpu_estimate(model_name, hf_token = hf_token)

    total_params = None
    if "/" in estimate_model and not Path(estimate_model).exists():
        total_params = _get_hf_safetensors_total_params(estimate_model, hf_token = hf_token)
    if total_params:
        return int(total_params * 2), "safetensors"

    config = _load_config_for_gpu_estimate(estimate_model, hf_token = hf_token)
    config_bytes: Optional[int] = None
    if config is not None:
        config_bytes = _estimate_fp16_model_size_bytes_from_config(config)

    local_bytes = _get_local_weight_size_bytes(estimate_model)

    # why: config-derived bytes cover only the text tower; local safetensors
    # include vision/audio towers. Take the larger so the multimodal
    # extra_bytes correction can fire.
    if config_bytes is not None and local_bytes is not None:
        if local_bytes > config_bytes:
            return local_bytes, "weight_bytes"
        return config_bytes, "config"
    if config_bytes is not None:
        return config_bytes, "config"
    if local_bytes is not None:
        return local_bytes, "weight_bytes"

    vllm_bytes = _estimate_fp16_model_size_bytes_from_vllm_utils(config)
    if vllm_bytes is not None:
        return vllm_bytes, "vllm_utils"

    return None, "unavailable"


def estimate_required_model_memory_gb(
    model_name: str,
    *,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    target_modules: Optional[list] = None,
    gradient_checkpointing: str = "unsloth",
    optimizer: str = "adamw_8bit",
) -> tuple[Optional[float], Dict[str, Any]]:
    from .vram_estimation import (
        TrainingVramConfig,
        extract_arch_config,
        estimate_training_vram,
        compute_total_params,
        compute_optimizer_bytes,
        compute_gradient_bytes,
        CUDA_OVERHEAD_BYTES,
        QUANT_4BIT_FACTOR,
        DEFAULT_TARGET_MODULES,
    )

    model_size_bytes, source = estimate_fp16_model_size_bytes(model_name, hf_token = hf_token)
    metadata: Dict[str, Any] = {
        "mode": "inference" if training_type is None else "training",
        "model_size_source": source,
    }
    if model_size_bytes is None:
        metadata["required_gb"] = None
        return None, metadata

    model_size_gb = model_size_bytes / (1024**3)
    metadata["model_size_gb"] = round(model_size_gb, 3)
    min_buffer_gb = 2.0

    if training_type is None:
        if load_in_4bit:
            base_4bit_gb = model_size_gb / QUANT_4BIT_FACTOR
            required_gb = base_4bit_gb + max(base_4bit_gb * 0.3, min_buffer_gb)
        else:
            required_gb = model_size_gb * 1.3
        metadata["required_gb"] = round(required_gb, 3)
        return required_gb, metadata

    training_method = (
        "full" if training_type == "Full Finetuning" else ("qlora" if load_in_4bit else "lora")
    )
    vram_config = TrainingVramConfig(
        training_method = training_method,
        batch_size = batch_size,
        max_seq_length = max_seq_length,
        lora_rank = lora_rank,
        target_modules = target_modules or list(DEFAULT_TARGET_MODULES),
        gradient_checkpointing = gradient_checkpointing,
        optimizer = optimizer,
        load_in_4bit = load_in_4bit,
    )

    estimate_model = _resolve_model_identifier_for_gpu_estimate(model_name, hf_token = hf_token)
    config = _load_config_for_gpu_estimate(estimate_model, hf_token = hf_token)
    if config is not None:
        try:
            vram_config.attention_implementation = _determine_attention_impl_for_gpu_estimate(
                config
            )
        except Exception as e:
            # Debug-level: fires every estimate on Windows ROCm (stub lacks Store);
            # expected and non-actionable -- eager is the safe fallback.
            logger.debug(
                "Could not resolve attention implementation for '%s': %s",
                estimate_model,
                e,
            )
            # why: charge the quadratic non-flash activation path so GPU
            # selection stays conservative when flash attn isn't proven usable.
            vram_config.attention_implementation = "eager"
    arch = extract_arch_config(config) if config is not None else None

    if arch is not None:
        breakdown = estimate_training_vram(arch, vram_config)
        # why: extract_arch_config only sees text_config; add the vision/audio
        # tower bytes that the text-arch fp16 total misses.
        arch_fp16_bytes = compute_total_params(arch) * 2
        extra_bytes = max(0, int(model_size_bytes) - arch_fp16_bytes)
        if extra_bytes > 0:
            breakdown.model_weights += extra_bytes
            if training_method == "full":
                # why: full fine-tuning makes extra params trainable; optimizer +
                # gradient bytes scale with them.
                extra_params = extra_bytes // 2
                breakdown.optimizer_states += compute_optimizer_bytes(
                    extra_params,
                    vram_config.optimizer,
                )
                breakdown.gradients += compute_gradient_bytes(extra_params)
        required_gb = breakdown.total / (1024**3)
        metadata["required_gb"] = round(required_gb, 3)
        metadata["estimation_mode"] = "detailed"
        metadata["attention_implementation"] = vram_config.attention_implementation
        metadata["vram_breakdown"] = breakdown.to_gb_dict()
        max_gpus = max(1, get_visible_gpu_count())
        for n_gpus in range(1, max_gpus + 1):
            metadata["vram_breakdown"][f"min_per_gpu_{n_gpus}"] = round(
                breakdown.min_gpu_vram(n_gpus) / (1024**3), 3
            )
        return required_gb, metadata

    # Fallback when model config is unavailable.
    overhead_gb = CUDA_OVERHEAD_BYTES / (1024**3)
    if training_method == "full":
        required_gb = model_size_gb * 3.5 + overhead_gb
    elif training_method == "qlora":
        base_4bit_gb = model_size_gb / QUANT_4BIT_FACTOR
        lora_overhead_gb = model_size_gb * 0.04
        act_gb = model_size_gb * 0.15 * (batch_size / 4) * (max_seq_length / 2048)
        required_gb = base_4bit_gb + lora_overhead_gb + act_gb + overhead_gb
    else:
        lora_overhead_gb = model_size_gb * 0.04
        act_gb = model_size_gb * 0.15 * (batch_size / 4) * (max_seq_length / 2048)
        required_gb = model_size_gb + lora_overhead_gb + act_gb + overhead_gb

    metadata["required_gb"] = round(required_gb, 3)
    metadata["estimation_mode"] = "fallback"
    return required_gb, metadata


def auto_select_gpu_ids(
    model_name: str,
    *,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    target_modules: Optional[list] = None,
    gradient_checkpointing: str = "unsloth",
    optimizer: str = "adamw_8bit",
) -> tuple[Optional[list[int]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {"selection_mode": "auto"}

    # Auto-selection needs per-device free-VRAM telemetry, available on CUDA
    # (nvidia-smi) and XPU (torch.xpu) but not MLX/CPU, which fall
    # through to inheriting parent visibility.
    if get_device() not in (DeviceType.CUDA, DeviceType.XPU):
        metadata["selection_mode"] = "non_accelerator"
        return None, metadata

    required_gb, estimate_metadata = estimate_required_model_memory_gb(
        model_name,
        hf_token = hf_token,
        training_type = training_type,
        load_in_4bit = load_in_4bit,
        batch_size = batch_size,
        max_seq_length = max_seq_length,
        lora_rank = lora_rank,
        target_modules = target_modules,
        gradient_checkpointing = gradient_checkpointing,
        optimizer = optimizer,
    )
    metadata.update(estimate_metadata)
    parent_visible_spec = _get_parent_visible_gpu_spec()
    metadata["parent_cuda_visible_devices"] = parent_visible_spec["raw"]

    if not parent_visible_spec["supports_explicit_gpu_ids"]:
        metadata["selection_mode"] = "inherit_parent_visible"
        metadata["selected_gpu_ids"] = None
        return None, metadata

    if required_gb is None:
        # Can't estimate size -- use all visible GPUs rather than risk one too small.
        parent_ids = get_parent_visible_gpu_ids()
        metadata["selection_mode"] = "fallback_all"
        metadata["selected_gpu_ids"] = parent_ids
        return parent_ids, metadata

    utilization = get_visible_gpu_utilization()
    devices = utilization.get("devices", [])
    parent_ids = get_parent_visible_gpu_ids()

    if not devices:
        metadata["selection_mode"] = "fallback_all"
        metadata["selected_gpu_ids"] = parent_ids
        return parent_ids, metadata

    gpu_candidates = []
    for device in devices:
        total_gb = device.get("vram_total_gb")
        used_gb = device.get("vram_used_gb")
        if total_gb is None or used_gb is None:
            continue
        free_gb = max(total_gb - used_gb, 0.0)
        gpu_candidates.append(
            {
                "index": device["index"],
                "free_gb": free_gb,
            }
        )

    if not gpu_candidates:
        metadata["selection_mode"] = "fallback_all"
        metadata["selected_gpu_ids"] = parent_ids
        return parent_ids, metadata

    ranked = sorted(gpu_candidates, key = lambda item: (-item["free_gb"], item["index"]))
    free_by_index = {item["index"]: item["free_gb"] for item in ranked}
    selected: list[int] = []
    usable_gb = 0.0
    # Sharding has inter-GPU overhead, so each extra GPU contributes less than
    # its raw free memory (first GPU keeps full capacity). 0.85 is empirical on
    # 2-8 GPU setups: covers NCCL buffers, pipeline bubbles, fragmentation.
    multi_gpu_overhead = 0.85

    # Per-GPU check: activations don't shard, so each GPU needs its weight shard
    # + full activation cost. Uses precomputed min_per_gpu_N values.
    vram_breakdown = estimate_metadata.get("vram_breakdown", {})

    for candidate in ranked:
        selected.append(candidate["index"])
        if len(selected) == 1:
            usable_gb = candidate["free_gb"]
        else:
            first_gpu_id = selected[0]
            usable_gb = free_by_index[first_gpu_id] + sum(
                free_by_index[gpu_id] * multi_gpu_overhead for gpu_id in selected[1:]
            )

        total_fits = usable_gb >= required_gb

        per_gpu_fits = True
        if total_fits and len(selected) > 1:
            min_key = f"min_per_gpu_{len(selected)}"
            min_per_gpu_gb = vram_breakdown.get(min_key)
            if min_per_gpu_gb is not None:
                smallest_free = min(free_by_index[gpu_id] for gpu_id in selected)
                per_gpu_fits = smallest_free >= min_per_gpu_gb

        if total_fits and per_gpu_fits:
            metadata["usable_gb"] = round(usable_gb, 3)
            metadata["selection_mode"] = "auto"
            metadata["selected_gpu_ids"] = selected
            logger.debug(
                "Selected GPUs automatically: model=%s selected=%s usable_gb=%s "
                "required_gb=%s multi_gpu_overhead=%s",
                model_name,
                selected,
                metadata["usable_gb"],
                metadata.get("required_gb"),
                multi_gpu_overhead,
            )
            return selected, metadata

    # Use only GPUs with verified VRAM data.
    fallback_all = [c["index"] for c in gpu_candidates] if gpu_candidates else parent_ids
    metadata["selection_mode"] = "fallback_all"
    if ranked:
        fallback_usable = ranked[0]["free_gb"] + sum(
            c["free_gb"] * multi_gpu_overhead for c in ranked[1:]
        )
    else:
        fallback_usable = 0.0
    metadata["usable_gb"] = round(fallback_usable, 3)
    metadata["selected_gpu_ids"] = fallback_all
    logger.warning(
        "Falling back to all visible GPUs; model may not fit: model=%s "
        "selected=%s usable_gb=%s required_gb=%s multi_gpu_overhead=%s",
        model_name,
        fallback_all,
        metadata["usable_gb"],
        metadata.get("required_gb"),
        multi_gpu_overhead,
    )
    return fallback_all, metadata


def prepare_gpu_selection(
    gpu_ids: Optional[list[int]],
    *,
    model_name: str,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
    batch_size: int = 4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    target_modules: Optional[list] = None,
    gradient_checkpointing: str = "unsloth",
    optimizer: str = "adamw_8bit",
) -> tuple[Optional[list[int]], Dict[str, Any]]:
    """Resolve which physical GPUs to use for a model load.

    GPU selection modes:
      - **Explicit** (``gpu_ids=[5, 6, 7]``): caller chooses exact GPUs.
        All listed GPUs are used and the model is sharded via
        ``device_map="balanced"``, even if it would fit on fewer. IDs are
        validated against the parent-visible set.
      - **Auto** (``gpu_ids=None`` or ``[]``): ``auto_select_gpu_ids``
        estimates VRAM needs and picks the *minimum* GPUs needed,
        preferring those with the most free memory.

    The returned ``gpu_ids`` is later passed to ``get_device_map()`` (maps it
    to a Hugging Face ``device_map`` string) and to ``apply_gpu_ids()`` in the
    worker subprocess (narrows ``CUDA_VISIBLE_DEVICES`` before torch/CUDA init).
    """
    if gpu_ids and get_device() not in (DeviceType.CUDA, DeviceType.XPU):
        raise ValueError(
            f"gpu_ids {list(gpu_ids)} is only supported on CUDA and Intel XPU "
            f"devices, but the current backend is '{get_device().value}'."
        )

    if gpu_ids:
        resolved = resolve_requested_gpu_ids(gpu_ids)
        metadata = {
            "selection_mode": "explicit",
            "selected_gpu_ids": resolved,
        }
        return resolved, metadata

    selected_gpu_ids, metadata = auto_select_gpu_ids(
        model_name,
        hf_token = hf_token,
        training_type = training_type,
        load_in_4bit = load_in_4bit,
        batch_size = batch_size,
        max_seq_length = max_seq_length,
        lora_rank = lora_rank,
        target_modules = target_modules,
        gradient_checkpointing = gradient_checkpointing,
        optimizer = optimizer,
    )
    return selected_gpu_ids, metadata


def get_physical_gpu_count() -> int:
    """
    Return the number of physical GPUs on the machine.

    Uses ``nvidia-smi -L`` on NVIDIA (unaffected by CUDA_VISIBLE_DEVICES),
    with a torch fallback for AMD ROCm and Intel XPU. Cached after first call.
    """
    global _physical_gpu_count, _physical_gpu_count_from_smi
    if _physical_gpu_count is not None:
        return _physical_gpu_count

    device = get_device()

    if device == DeviceType.CUDA:
        try:
            if IS_ROCM:
                from . import amd as _smi_mod
            else:
                from . import nvidia as _smi_mod
            count = _smi_mod.get_physical_gpu_count()
            if count is not None:
                _physical_gpu_count = count
                _physical_gpu_count_from_smi = True
                return _physical_gpu_count
        except Exception:
            pass
        # SMI unavailable -- fall back to torch.
        count = _torch_get_physical_gpu_count()
        _physical_gpu_count = count if count is not None else 1
        return _physical_gpu_count

    if device == DeviceType.XPU:
        count = _torch_get_physical_gpu_count()
        _physical_gpu_count = count if count is not None else 1
        return _physical_gpu_count

    if device == DeviceType.MLX:
        _physical_gpu_count = 1
        return _physical_gpu_count

    _physical_gpu_count = 0

    return _physical_gpu_count


def _backend_visible_devices_env() -> Optional[str]:
    """Return the raw visibility env string that applies to this backend.

    On XPU the control is ``ZE_AFFINITY_MASK`` (not ``CUDA_VISIBLE_DEVICES``);
    on ROCm, HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES take precedence over
    CUDA_VISIBLE_DEVICES. Mirrors ``_get_parent_visible_gpu_spec`` so
    ``backend_cuda_visible_devices`` reports the value actually narrowing the
    visible device set on the current backend.
    """
    if get_device() == DeviceType.XPU:
        return os.environ.get("ZE_AFFINITY_MASK")
    if IS_ROCM:
        return _get_parent_visible_gpu_spec().get("raw")
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def get_vulkan_inference_gpu_info() -> Optional[Dict[str, Any]]:
    """Return llama.cpp Vulkan devices, or None when Vulkan is not installed."""
    # Vulkan is a llama.cpp inference backend, not a PyTorch training device, so
    # keep it separate from the PyTorch/MLX training-device report.
    try:
        from core.inference.llama_cpp import (
            LlamaCppBackend,
            _apply_igpu_host_reserve_mib,
        )
    except Exception as e:
        logger.debug("Could not inspect the llama.cpp Vulkan backend: %s", e)
        return None

    try:
        if not LlamaCppBackend._is_vulkan_backend():
            return None
    except Exception as e:
        logger.debug("Could not identify the llama.cpp Vulkan backend: %s", e)
        return None

    result = {
        "available": False,
        "backend": "vulkan",
        "backend_cuda_visible_devices": None,
        "parent_visible_gpu_ids": [],
        "devices": [],
        "index_kind": "vulkan",
    }
    try:
        for row in LlamaCppBackend.vulkan_device_inventory():
            ordinal = row["index"]
            shared_memory = bool(row["is_igpu"])
            free_mib = _apply_igpu_host_reserve_mib(row["free_mib"], shared_memory)
            total_mib = 0 if shared_memory else row["total_mib"]
            budget_mib = total_mib or free_mib
            used_mib = max(0, total_mib - free_mib) if total_mib else None
            result["devices"].append(
                {
                    "index": ordinal,
                    # ggml Vulkan ordinals are the space `--device Vulkan<i>` pins,
                    # so unlike a torch-xpu relative ordinal these are selectable.
                    "index_kind": "vulkan",
                    "visible_ordinal": ordinal,
                    "name": row["name"],
                    "memory_total_gb": round(budget_mib / 1024, 2),
                    "vram_used_gb": round(used_mib / 1024, 2) if used_mib is not None else None,
                    "vram_free_gb": round(free_mib / 1024, 2),
                    "vram_utilization_pct": round((used_mib / total_mib) * 100, 1)
                    if used_mib is not None and total_mib > 0
                    else None,
                    "shared_memory": shared_memory,
                }
            )
    except Exception as e:
        logger.debug("Vulkan GPU visibility query failed: %s", e)
        return result

    result["available"] = bool(result["devices"])
    return result


def get_backend_visible_gpu_info() -> Dict[str, Any]:
    device = get_device()

    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_visible_ids = get_parent_visible_gpu_ids()
        # Try native SMI first (nvidia-smi; skipped for ROCm).
        if device == DeviceType.CUDA and not IS_ROCM:
            try:
                from . import nvidia

                parent_visible_spec = _get_parent_visible_gpu_spec()
                result = nvidia.get_backend_visible_gpu_info(
                    parent_visible_spec["numeric_ids"],
                    parent_visible_spec["raw"],
                )
                if result.get("available"):
                    result["backend"] = _backend_label(device)
                    return result
            except Exception as e:
                logger.warning("Backend GPU visibility query failed: %s", e)

        # Torch fallback (ROCm, XPU, nvidia-smi missing). Empty parent_visible_ids
        # (UUID/MIG mask) -> enumerate by torch ordinal so the UI shows devices.
        if parent_visible_ids:
            torch_indices = parent_visible_ids
            index_kind = "physical"
        else:
            visible_count = _torch_get_physical_gpu_count() or 0
            torch_indices = list(range(visible_count))
            index_kind = "relative"
        # Inventory only: this endpoint reads name and total and discards used, so
        # there is nothing here worth a permanent driver context.
        torch_devices = _torch_get_device_inventory(torch_indices)
        if torch_devices:
            if IS_ROCM and platform.system() == "Linux":
                shared_host_gb = _rocm_linux_shared_pool_host_gb_by_index(torch_devices)
                for td in torch_devices:
                    if td["index"] in shared_host_gb:
                        td["shared_memory"] = True
                        td["shared_memory_host_backed_gb"] = shared_host_gb[td["index"]]
            elif IS_ROCM and platform.system() == "Windows":
                shared_host_gb = _windows_rocm_shared_pool_host_gb_by_index(torch_devices)
                for td in torch_devices:
                    if td["index"] in shared_host_gb:
                        td["shared_memory_host_backed_gb"] = shared_host_gb[td["index"]]
            devices = [
                {
                    "index": td["index"],
                    "index_kind": index_kind,
                    "visible_ordinal": td["visible_ordinal"],
                    "name": td["name"],
                    "memory_total_gb": td["total_gb"],
                    "shared_memory": bool(td.get("shared_memory")),
                    "shared_memory_host_backed_gb": td.get("shared_memory_host_backed_gb"),
                    # Surfaced from the inventory's own `_rocm_known_unified`
                    # rather than re-derived: a ROCm APU's total is the GTT pool,
                    # which moves with host usage and is not a VRAM ceiling a fit
                    # verdict can be measured against. Distinct from
                    # `shared_memory`, which is that flag AND Windows, so a Linux
                    # APU reads as not-shared while still having no such ceiling.
                    "unified_memory": bool(td.get("_rocm_known_unified")),
                }
                for td in torch_devices
            ]
            return {
                "available": True,
                "backend": _backend_label(device),
                "backend_cuda_visible_devices": _backend_visible_devices_env(),
                "parent_visible_gpu_ids": parent_visible_ids,
                "devices": devices,
                "index_kind": index_kind,
            }

        return {
            "available": False,
            "backend": _backend_label(device),
            "backend_cuda_visible_devices": _backend_visible_devices_env(),
            "parent_visible_gpu_ids": parent_visible_ids,
            "devices": [],
            "index_kind": "physical",
        }

    if device == DeviceType.MLX:
        mem = get_gpu_memory_info()
        if not mem.get("available"):
            return {
                "available": False,
                "backend": _backend_label(device),
                "backend_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "parent_visible_gpu_ids": [],
                "devices": [],
                "index_kind": "relative",
            }
        memory_total_gb = round(mem.get("total_gb", 0), 2)
        return {
            "available": True,
            "backend": _backend_label(device),
            "backend_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "parent_visible_gpu_ids": [0],
            "devices": [
                {
                    "index": 0,
                    "index_kind": "relative",
                    "visible_ordinal": 0,
                    "name": mem.get("device_name", "MLX"),
                    "memory_total_gb": memory_total_gb,
                    "shared_memory": True,
                    "shared_memory_host_backed_gb": memory_total_gb,
                }
            ],
            "index_kind": "relative",
        }

    return {
        "available": False,
        "backend": _backend_label(device),
        "backend_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parent_visible_gpu_ids": [],
        "devices": [],
        "index_kind": "vulkan",
    }


def get_visible_gpu_count() -> int:
    """
    Return the number of GPUs visible to this process.

    Respects ``CUDA_VISIBLE_DEVICES`` -- if set, only those GPUs count.
    Falls back to physical count if unset or torch is unavailable.
    Cached after the first call.
    """
    global _visible_gpu_count
    if _visible_gpu_count is not None:
        return _visible_gpu_count

    # Prefer torch.xpu.device_count() on Intel XPU: the Level Zero runtime
    # correctly interprets ZE_AFFINITY_MASK semantics (e.g. subdevice syntax
    # "0.0,0.1" collapses onto one root GPU). Supersedes the torch fallback below.
    if get_device() == DeviceType.XPU:
        xpu_mask_raw = os.environ.get("ZE_AFFINITY_MASK")
        xpu_mask_set = xpu_mask_raw is not None
        xpu_visible = (xpu_mask_raw or "").strip()
        if xpu_mask_set and xpu_visible == "":
            _visible_gpu_count = 0
            return _visible_gpu_count

        try:
            import torch
            _visible_gpu_count = torch.xpu.device_count()
        except Exception as e:
            logger.debug(
                "torch.xpu.device_count() failed, falling back to mask parsing: %s",
                e,
            )
            if xpu_visible:
                # Fallback: count unique root device IDs from the mask.
                # "device.subdevice" notation means "0.0,0.1" is 1 root, not 2.
                # Without torch the hierarchy mode is unknown, so root-device
                # counting is the conservative choice.
                if xpu_visible == "*":
                    # Documented wildcard: all physical XPUs visible.
                    _visible_gpu_count = get_physical_gpu_count()
                else:
                    roots = _parse_ze_mask_roots(xpu_visible)
                    # Non-parseable masks (",,,", "GPU-abc") yield an empty
                    # roots list, treated as 0 visible devices, not "all
                    # visible" -- no evidence the whole fleet was intended.
                    _visible_gpu_count = len(set(roots))
            else:
                _visible_gpu_count = get_physical_gpu_count()
        return _visible_gpu_count

    # _get_parent_visible_gpu_spec() already handles HIP_VISIBLE_DEVICES /
    # ROCR_VISIBLE_DEVICES on ROCm.
    visible_spec = _get_parent_visible_gpu_spec()
    if visible_spec["raw"] is not None:
        raw = visible_spec["raw"].strip()
        if raw == "" or raw == "-1":
            _visible_gpu_count = 0
        elif visible_spec["numeric_ids"] is not None:
            _visible_gpu_count = len(visible_spec["numeric_ids"])
        else:
            _visible_gpu_count = len([x for x in raw.split(",") if x.strip()])
        return _visible_gpu_count

    # No visibility env var set -- try torch, else physical count. XPU is
    # handled by the early return above, so only torch.cuda is needed here.
    try:
        import torch
        _visible_gpu_count = torch.cuda.device_count()
    except Exception:
        _visible_gpu_count = get_physical_gpu_count()

    return _visible_gpu_count


def apply_gpu_ids(gpu_ids, backend: Optional[str] = None) -> None:
    if gpu_ids is None:
        return

    # Empty list -> treat like None (inherit parent); setting CUDA_VISIBLE_DEVICES=""
    # disables CUDA entirely and crashes downstream torch calls.
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) == 0:
        return

    global _visible_gpu_count

    if isinstance(gpu_ids, (list, tuple)):
        value = ",".join(str(g) for g in gpu_ids)
    else:
        value = str(gpu_ids)

    # Intel XPU honors ZE_AFFINITY_MASK, not CUDA_VISIBLE_DEVICES; route XPU
    # pinning through it so worker subprocesses are restricted to the intended GPU.
    # Decide WITHOUT get_device(): workers call this before detect_hardware(),
    # and a lazy detect would probe torch.cuda against the unmasked parent env,
    # latching device enumeration before the mask below is written. Pre-detect,
    # use env + torch BUILD attributes only (no runtime init, like the ROCm
    # mirror below).
    _is_xpu = DEVICE == DeviceType.XPU
    if backend is not None:
        # The spawning parent's detected backend (config["device_backend"]):
        # exact and probe-free, so the mask target always matches what
        # detect_hardware() decided in the parent, including its XPU
        # availability check and CUDA fallback.
        _is_xpu = backend == DeviceType.XPU.value
    elif DEVICE is None:
        # No parent backend passed (direct caller). version.xpu can be None
        # on a working XPU build, so also accept torch.xpu._is_compiled()
        # (a pure symbol-presence check, no runtime init). UNSLOTH_FORCE_XPU
        # counts only on an XPU-capable build: detect_hardware() falls back
        # to CUDA when XPU is missing, and the mask target must follow.
        try:
            import torch as _torch

            _ver = _torch.version
            _is_comp = getattr(getattr(_torch, "xpu", None), "_is_compiled", None)
            _xpu_build = (callable(_is_comp) and bool(_is_comp())) or (
                getattr(_ver, "xpu", None) is not None
            )
            if os.environ.get("UNSLOTH_FORCE_XPU") == "1":
                _is_xpu = _xpu_build
            else:
                # Mirror detect_hardware: hidden CUDA prefers XPU on an
                # XPU-capable build (with or without a ZE mask -- detection
                # falls through to XPU either way), where writing these ids
                # to CUDA_VISIBLE_DEVICES would re-expose the deliberately
                # hidden CUDA.
                _cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                _cuda_hidden = _cvd is not None and _cvd.strip() in ("", "-1")
                _is_xpu = _xpu_build and (
                    _cuda_hidden
                    or (getattr(_ver, "cuda", None) is None and getattr(_ver, "hip", None) is None)
                )
        except Exception as e:
            logger.debug(
                "apply_gpu_ids: torch XPU probe skipped (%s: %s)",
                type(e).__name__,
                e,
            )
    if _is_xpu:
        os.environ["ZE_AFFINITY_MASK"] = value
        # Leave inherited CUDA_VISIBLE_DEVICES alone -- clearing it could let
        # the worker flip back to CUDA on hybrid hosts.
        _visible_gpu_count = None
        logger.info("Applied gpu_ids: ZE_AFFINITY_MASK='%s'", value)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = value
    # Keep ROCm visibility env vars in sync. Workers may call apply_gpu_ids()
    # before detect_hardware() (IS_ROCM still False), so also mirror when the
    # parent set a ROCm visibility var, with a torch.version.hip probe fallback.
    _inherits_rocm_visibility = (
        "HIP_VISIBLE_DEVICES" in os.environ or "ROCR_VISIBLE_DEVICES" in os.environ
    )
    _is_rocm = IS_ROCM or _inherits_rocm_visibility
    if not _is_rocm:
        # torch.version.hip is set on ROCm, None on CUDA; AMD SDK wheels may leave
        # it unset but encode "rocm" in __version__. Broad except: never crash a worker.
        try:
            import torch as _torch
            _is_rocm = (
                getattr(_torch.version, "hip", None) is not None
                or "rocm" in getattr(_torch, "__version__", "").lower()
            )
        except Exception as e:
            logger.debug(
                "apply_gpu_ids: torch ROCm probe skipped (%s: %s)",
                type(e).__name__,
                e,
            )
    if _is_rocm:
        os.environ["HIP_VISIBLE_DEVICES"] = value
        # ROCR_VISIBLE_DEVICES operates at the HSA agent level and uses
        # different indexing semantics to HIP_VISIBLE_DEVICES. Setting it
        # to a physical GPU index breaks multi-GPU ROCm systems where the
        # parent already set ROCR_VISIBLE_DEVICES (e.g. "0,1"): narrowing
        # to "1" causes torch.cuda.is_available() to return False in the
        # worker subprocess. HIP_VISIBLE_DEVICES is sufficient for GPU
        # selection on ROCm -- leave ROCR_VISIBLE_DEVICES inherited.
    _visible_gpu_count = None
    if _is_rocm:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s' (rocm)", value)
    else:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s'", value)


def get_device_map(gpu_ids: Optional[list[int]] = None) -> str:
    """Return the Hugging Face ``device_map`` string for model loading.

    Returns ``"balanced"`` (shard evenly across GPUs) when:
      - ``gpu_ids`` explicitly lists >1 GPU, **or**
      - ``CUDA_VISIBLE_DEVICES``/``ZE_AFFINITY_MASK`` uses non-numeric
        identifiers (UUID/MIG/wildcard) and >1 GPU is visible (fallback:
        numeric IDs unresolvable, so assume multi-GPU is intended).

    Returns ``"sequential"`` (single device) otherwise, including CPU/MLX
    backends.

    Use ``prepare_gpu_selection()`` upstream to determine ``gpu_ids`` -- it
    handles auto-selecting the minimum GPUs needed for a model.
    """
    device = get_device()
    if device in (DeviceType.CUDA, DeviceType.XPU):
        multi_gpu = gpu_ids is not None and len(gpu_ids) > 1

        if not multi_gpu:
            parent_visible_spec = _get_parent_visible_gpu_spec()
            if device == DeviceType.CUDA:
                # UUID/MIG masks can't be split into numeric IDs; >1 visible GPU
                # means multi-GPU sharding is intended.
                if parent_visible_spec["numeric_ids"] is None and get_visible_gpu_count() > 1:
                    multi_gpu = True
            elif device == DeviceType.XPU and gpu_ids is None:
                # Shard across visible XPU ordinals via HF (no mask rewrite),
                # only when no gpu_ids were passed -- an explicit gpu_ids=[0]
                # means "use exactly device 0" and must stay sequential.
                supports_physical = parent_visible_spec["supports_explicit_gpu_ids"]
                has_multiple_numeric = (
                    parent_visible_spec["numeric_ids"] is not None
                    and len(parent_visible_spec["numeric_ids"]) > 1
                )
                has_multiple_unresolved = (
                    parent_visible_spec["numeric_ids"] is None and get_visible_gpu_count() > 1
                )
                if has_multiple_unresolved or (not supports_physical and has_multiple_numeric):
                    multi_gpu = True

        if multi_gpu:
            return "balanced"

    return "sequential"


def get_offloaded_device_map_entries(model) -> dict[str, str]:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict):
        return {}
    return {
        module_name: placement
        for module_name, placement in hf_device_map.items()
        if placement in ("cpu", "disk")
    }


def raise_if_offloaded(
    model,
    device_map: str,
    context: str = "Loading",
) -> None:
    """Raise ``ValueError`` if *model* has modules offloaded to CPU or disk."""
    offloaded = get_offloaded_device_map_entries(model)
    if not offloaded:
        return
    example = ", ".join(f"{name}={placement}" for name, placement in list(offloaded.items())[:5])
    raise ValueError(
        f"{context} does not support models loaded with CPU or disk offload. "
        f"device_map='{device_map}' produced offloaded modules: {example}"
    )


def get_torch_device_str() -> str:
    """
    Return the torch device string for the detected hardware.
    E.g. "cuda", "xpu", or "cpu".
    """
    device = get_device()
    if device == DeviceType.CUDA:
        return "cuda"
    elif device == DeviceType.XPU:
        return "xpu"
    return "cpu"


# Mirrors AUTO_NUM_PROC_CAP in unsloth_zoo.dataset_num_proc; copied rather than
# imported to keep hardware detection free of the training package. A canary in
# tests/utils/test_dataset_num_proc.py fails if the two drift.
_STUDIO_NUM_PROC_CAP = 8


def safe_num_proc(desired: Optional[int] = None) -> int:
    """
    Return a safe ``num_proc`` for ``dataset.map()`` calls.

    On Windows always returns 1: Python uses ``spawn`` not ``fork``, so
    re-importing torch/transformers/unsloth per worker is typically slower
    than single-process for normal dataset sizes.

    On multi-GPU machines (multiple GPUs *visible* to this process) the
    NVIDIA driver spawns extra background threads, making ``os.fork()``
    deadlock-prone with many workers, so this caps ``num_proc`` to 4.
    The cap does not apply when ``CUDA_VISIBLE_DEVICES`` restricts to one GPU.

    Args:
        desired: The num_proc you *want*. If None, auto-computes from
                 ``os.cpu_count()``.

    Returns:
        A safe integer ≥ 1.
    """
    # Windows/macOS use 'spawn'; re-importing torch/transformers/unsloth per
    # worker is typically slower than single-process.
    if sys.platform in ("win32", "darwin"):
        return 1

    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)

    # Every number reaching here is a backend heuristic (cpu_count // 3 above,
    # cpu_count // 4 from trainer.py), but downstream it looks like user intent
    # and is only clamped by free memory, so a 64-core host sends 16 workers to
    # Dataset.map -- the shape of issue #2693, and slower besides (32 workers
    # measured 14.2s against 6.3s in-process, ~1GB each).
    if desired > _STUDIO_NUM_PROC_CAP:
        # No mention of UNSLOTH_DATASET_NUM_PROC here: this function returns an
        # int >= 1 and cannot express the in-process the hatch promises. The
        # hatch is read in dataset_map_num_proc, which is the path whose value
        # reaches Dataset.map.
        logger.info(
            f"num_proc {desired} -> {_STUDIO_NUM_PROC_CAP}: tokenization stops "
            f"scaling well before this and each worker holds its own tokenizer copy."
        )
        desired = _STUDIO_NUM_PROC_CAP

    visible = get_visible_gpu_count()
    if visible > 1:
        capped = max(1, min(4, desired))
        logger.info(
            f"Multi-GPU detected ({visible} visible GPUs) -- "
            f"capping num_proc {desired} -> {capped} to avoid fork deadlocks"
        )
        return capped

    return max(1, desired)


def safe_thread_num_proc(desired: Optional[int] = None) -> int:
    """
    Return a safe worker count for ``ThreadPoolExecutor`` calls.

    Unlike ``safe_num_proc()``, does NOT cap to 1 on macOS/Windows: threads
    share the parent address space, unaffected by ``spawn`` vs ``fork``.

    Args:
        desired: The thread count you *want*. If None, auto-computes
                 from ``os.cpu_count()``.

    Returns:
        A safe integer >= 1.
    """
    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)

    return max(1, desired)


def dataset_map_num_proc(
    desired: Optional[int] = None, *, serial_as_none: bool = True
) -> Optional[int]:
    """
    Return a safe ``num_proc`` for ``Dataset.map()`` and ``Dataset.filter()``.

    Returns ``None`` on spawn platforms (Windows, macOS). ``None`` -- not ``1``
    -- is the disable sentinel: ``datasets`` >= 4.1 (Unsloth pins 4.3.0) takes
    the pool branch for any ``num_proc >= 1``, so ``1`` still builds a
    ``Pool(1)``.

    Also returns ``None`` on XPU once its runtime is initialized in this
    process: ``os.fork()`` corrupts the Level-Zero context, making Triton
    kernels fail with "Pointer argument doesn't reference XPU device memory".
    Pre-init XPU hosts can still parallelize CPU-side preprocessing.

    There is deliberately no CUDA equivalent: the child only runs the tokenizer,
    and 300 forced-fork map() runs on an initialized CUDA context produced no
    failures. Since ``detect_hardware()`` always initializes CUDA, such a guard
    would serialize every CUDA run for nothing. The worker-count bound in
    ``unsloth_zoo.dataset_num_proc`` is what addresses issue #2693.

    ``serial_as_none`` says how to spell "run in-process" for the layer that
    reads the value back, exactly as in ``unsloth_zoo.dataset_num_proc``. Leave
    it True at a ``map()`` call site, where ``None`` is the only value that
    builds no pool. Pass **False when the result is written into a config**
    (``SFTConfig.dataset_num_proc``): a config ``None`` means "auto-size me" to
    every downstream reader, so a serial request stored as ``None`` comes back
    out as a full worker set. Only ``1`` survives that round trip, and the SFT
    map site turns it back into ``None``.
    """
    if sys.platform in ("win32", "darwin"):
        # ``UNSLOTH_DATASET_NUM_PROC`` is an unvetoed escape hatch in the shared
        # policy, so a user who has read the dead-worker message and accepted
        # spawn workers must not be overruled here without a word. Only the
        # hatch can produce a count on this platform; everything else falls
        # through to the veto below.
        if _num_proc_override_is_set():
            return _bounded_by_the_shared_policy(desired, serial_as_none)
        # ``None`` is safe at either layer here: workers are unusable on a spawn
        # platform, so every auto-sizer that reads a config ``None`` vetoes too
        # and nothing can inflate it. A ``1`` would be worse -- only the SFT map
        # site is rewritten, so DPO/KTO/CPO/ORPO/Reward/PRM would hand that 1
        # straight to Dataset.map and get a Pool(1) whose child re-imports the
        # user's __main__ (#3211 / #3397).
        return None

    if get_device() == DeviceType.XPU:
        try:
            import torch
        except Exception:
            # No torch means no active XPU runtime, so CPU-side dataset
            # parallelism is still safe -- but it is still bounded, or a
            # torch-less container would be the one path that ignores the
            # memory ceiling and the escape hatch.
            return _bounded_by_the_shared_policy(desired, serial_as_none)

        xpu = getattr(torch, "xpu", None)
        is_initialized = getattr(xpu, "is_initialized", None)
        if callable(is_initialized):
            try:
                if is_initialized():
                    # Same reading as the spawn branch above: the hatch is
                    # unvetoed by contract, so someone who has accepted the
                    # risk on XPU is not overruled here without a word.
                    if _num_proc_override_is_set():
                        return _bounded_by_the_shared_policy(desired, serial_as_none)
                    # Unlike the spawn platforms above, forking is still
                    # available here, so a config ``None`` WOULD be auto-sized
                    # back up and fork the corrupted Level-Zero context this
                    # guard exists to protect. Encode serial for the layer.
                    return None if serial_as_none else 1
            except Exception as e:
                # Treat a failing probe as "runtime not touched yet" so
                # pre-init CPU preprocessing can still parallelize.
                logger.debug("torch.xpu.is_initialized() probe failed: %s", e)

    return _bounded_by_the_shared_policy(desired, serial_as_none)


# sys.modules key for the copy loaded off disk below. Not "unsloth.dataset_num_proc":
# that name belongs to the package, and claiming it would make a later real import
# of unsloth return this module instead.
_LOCAL_POLICY_MODULE = "unsloth_studio_local_dataset_num_proc"


def _shared_policy():
    """The shared num_proc policy module, or None on an installation without it.

    The Zoo owns it. ``unsloth.dataset_num_proc`` is a byte-identical fallback
    for a Zoo that predates the module. ``import unsloth.dataset_num_proc``
    would run the package __init__, which patches torch and loads the model
    stack -- unacceptable from inside hardware detection -- so that form is used
    only when the package is already imported. Otherwise the file is loaded
    straight off disk, which is safe because the module is stdlib-only by
    design: the API process reaches format conversion without importing unsloth
    at all, and leaving it with no policy there is what the 2GB container with
    eight cores used to hit.
    """
    try:
        import unsloth_zoo.dataset_num_proc as policy
        return policy
    except Exception:
        pass
    if "unsloth" in sys.modules:
        try:
            import unsloth.dataset_num_proc as policy
            return policy
        except Exception as e:
            logger.debug("local dataset_num_proc fallback unavailable: %s", e)
            return None
    # Memoised through sys.modules: the policy warns once per process about a
    # vetoed count, and re-executing the file on every map() call would reset
    # that and re-read the cgroup tree each time.
    cached = sys.modules.get(_LOCAL_POLICY_MODULE)
    if cached is not None:
        return cached
    try:
        import importlib.util

        # find_spec does not execute a top-level package, so this locates
        # unsloth/ without importing it.
        package = importlib.util.find_spec("unsloth")
        if package is None or not package.submodule_search_locations:
            return None
        path = Path(list(package.submodule_search_locations)[0]) / "dataset_num_proc.py"
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(_LOCAL_POLICY_MODULE, path)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        sys.modules[_LOCAL_POLICY_MODULE] = policy
        return policy
    except Exception as e:
        logger.debug("local dataset_num_proc fallback unavailable: %s", e)
    return None


def _num_proc_override_is_set() -> bool:
    """Whether the escape hatch decided the count, not merely whether it is set.

    The policy ignores an unparseable or negative value with a warning, so
    reading the variable directly would let ``UNSLOTH_DATASET_NUM_PROC=-1``
    skip the multi-GPU cap while contributing nothing.
    """
    policy = _shared_policy()
    if policy is None:
        return False
    parsed = getattr(policy, "environment_override", None)
    if parsed is None:
        # A policy that predates the public reader: presence is the best
        # available answer, and it errs toward honouring the hatch.
        return bool(os.environ.get(policy.NUM_PROC_ENV_VAR, "").strip())
    try:
        was_set, _value = parsed()
    except Exception as e:
        logger.debug("dataset_num_proc override unreadable: %s", e)
        return False
    return bool(was_set)


def _bounded_by_the_shared_policy(
    desired: Optional[int], serial_as_none: bool = True
) -> Optional[int]:
    """Apply the training-side num_proc policy to an Unsloth request.

    ``format_conversion.py`` and ``chat_templates.py`` hand this straight to
    ``Dataset.map``, so without it a container with 2GB and eight cores still got
    eight tokenizer workers -- the OOM this policy exists to stop -- and
    ``UNSLOTH_DATASET_NUM_PROC`` did nothing on those paths.

    ``desired`` is passed through as the caller wrote it. Materializing an auto
    request with ``safe_num_proc`` first would hide it from the policy, whose
    auto path reads this process's CPU affinity and cgroup quota while
    ``safe_num_proc`` reads the host's ``os.cpu_count()``: a 2-core container on
    a 64-core box asked for 21 workers and got them bounded only by memory.
    Unsloth's own caps are then applied to whatever the policy chose, since the
    multi-GPU fork-deadlock cap is knowledge the policy does not have -- except
    over the escape hatch, which is uncapped by contract.
    """
    policy = _shared_policy()
    if policy is None:
        return safe_num_proc(desired)  # the behaviour before the shared policy

    try:
        bounded = policy.get_dataset_num_proc(desired, serial_as_none = serial_as_none)
    except Exception as e:
        logger.debug("dataset_num_proc policy unavailable: %s", e)
        return safe_num_proc(desired)

    if isinstance(bounded, int) and bounded > 1 and not _num_proc_override_is_set():
        bounded = safe_num_proc(bounded)
    return bounded
