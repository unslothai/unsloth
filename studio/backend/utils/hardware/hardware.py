# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hardware detection: call detect_hardware() once at FastAPI lifespan startup, then read DEVICE / DeviceType / is_apple_silicon anywhere."""

import ast
import copy
import gc
import glob
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
import structlog
from loggers import get_logger
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

logger = get_logger(__name__)


# CUDA orders FASTEST_FIRST but nvidia-smi by PCI bus id; pin one index space before torch.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Workers can import MLX without unsloth, so mirror the package bootstrap here.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")


class DeviceType(str, Enum):
    """Supported compute backends. str subclass for clean JSON serialization."""

    CUDA = "cuda"
    XPU = "xpu"
    MLX = "mlx"
    CPU = "cpu"


DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True
# Why CHAT_ONLY is True, None when training is enabled. torch_cpu_build and
# torch_cuda_unavailable mean GPUs exist but this PyTorch cannot use them.
CHAT_ONLY_REASON: Optional[str] = None
# Only mlx_unavailable sets it, naming the offending MLX package.
CHAT_ONLY_DETAIL: Optional[str] = None
# Vendors behind a mismatch reason, so a frozen verdict can tell when they disappear.
CHAT_ONLY_MISMATCH_VENDORS: frozenset = frozenset()
IS_ROCM: bool = False

# Detection has concurrent callers; re-entrant because get_device() nests.
_DETECT_LOCK = threading.RLock()

# Bumped by shutdown so an in-flight detection cannot publish over the reset.
_EPOCH_LOCK = threading.Lock()
DETECTION_EPOCH = 0


def invalidate_detection() -> int:
    """Retire any detection in flight. Returns the new epoch."""
    global DETECTION_EPOCH
    from . import gpu_query

    gpu_query.invalidate_static("hardware re-detection")
    with _EPOCH_LOCK:
        DETECTION_EPOCH += 1
        return DETECTION_EPOCH


def current_detection_epoch() -> int:
    with _EPOCH_LOCK:
        return DETECTION_EPOCH


# Thread-local epoch for nested detections, since get_device() takes none.
_OWNING_EPOCH = threading.local()


@contextmanager
def owning_detection_epoch(epoch: Optional[int]):
    """Bind epoch-less detections on this thread to ``epoch`` for the block. Nested, not assigned: restoring the previous value keeps concurrent scopes on other threads apart."""
    previous = getattr(_OWNING_EPOCH, "value", None)
    _OWNING_EPOCH.value = epoch
    try:
        yield
    finally:
        _OWNING_EPOCH.value = previous


def _discard_detection_locked() -> None:
    """Drop a verdict produced for an epoch that has been retired."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM
    global CHAT_ONLY_MISMATCH_VENDORS
    DEVICE = None
    CHAT_ONLY = True
    CHAT_ONLY_REASON = None
    CHAT_ONLY_DETAIL = None
    CHAT_ONLY_MISMATCH_VENDORS = frozenset()
    IS_ROCM = False
    DETECTION_COMPLETE.clear()


# Set once detection has settled; poll this, not DEVICE, which can be revised.
DETECTION_COMPLETE = threading.Event()
# Bumped on each settle: detection can rerun (MLX self-heal), so snapshots can go stale.
DETECTION_GENERATION = 0

# Separate from _DETECT_LOCK: never held across the import.
_DETECT_KICK_LOCK = threading.Lock()
_DETECT_THREAD: Optional[threading.Thread] = None


def start_background_detection() -> None:
    """Run detection on a daemon thread if nothing is running it yet, for callers on a deadline that cannot await ensure_hardware_detected() (e.g. /api/health under the launcher's 2s timeout). At most one thread, and none once DEVICE is set, so a polling route cannot pile them up. Not the asyncio executor: a to_thread outliving its awaiter holds a slot and a polled endpoint would exhaust the pool during a slow import."""
    global _DETECT_THREAD
    if DEVICE is not None:
        return
    with _DETECT_KICK_LOCK:
        if DEVICE is not None:
            return
        if _DETECT_THREAD is not None and _DETECT_THREAD.is_alive():
            return
        # Epoch read before start() so a shutdown in between wins.
        _DETECT_THREAD = threading.Thread(
            target = ensure_hardware_detected,
            args = (current_detection_epoch(),),
            daemon = True,
            name = "hardware-detect",
        )
        _DETECT_THREAD.start()


def _backend_label(device: DeviceType) -> str:
    """The user-facing backend name. ROCm hosts stay DeviceType.CUDA internally (ROCm reuses torch.cuda.*), but "cuda" is misleading in JSON, so swap to "rocm" when IS_ROCM is set."""
    if IS_ROCM and device == DeviceType.CUDA:
        return "rocm"
    return device.value


def is_apple_silicon() -> bool:
    """True on Apple Silicon (pure platform check, no ML imports)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# torch installed but its import failed; reported as a detection failure, not no GPU.
TORCH_IMPORT_ERROR: Optional[str] = None


def _has_torch() -> bool:
    """True if PyTorch is importable. Any failure counts as "no torch": ensure_hardware_detected() re-runs while DEVICE is None, so an escaping OSError would make every request retry the import. The error is recorded, or the host is told to install the torch it already has."""
    global TORCH_IMPORT_ERROR
    try:
        import torch
        TORCH_IMPORT_ERROR = None
        return True
    except Exception as exc:
        # Only ModuleNotFoundError for torch itself means absent; other ImportErrors are broken.
        absent = isinstance(exc, ModuleNotFoundError) and exc.name == "torch"
        TORCH_IMPORT_ERROR = None if absent else repr(exc)
        if TORCH_IMPORT_ERROR is not None:
            logger.error("torch is installed but failed to import: %r", exc)
        # A part-way failure leaves stale submodules cached; purge_partial_import() clears them.
        try:
            from utils.torch_warmup import purge_partial_import
        except Exception:
            # Also exec'd standalone (tests/python/test_e2e_no_torch_sandbox.py); nothing to purge.
            pass
        else:
            purge_partial_import("torch")
        return False


def _torch_mps_available() -> bool:
    """True when torch exposes a usable Metal (MPS) device. Apple Silicon alone is not enough: a torch built without MPS leaves the pipelines nowhere to run. Never raises; a failed probe reads as no MPS."""
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


# Cached so the CPU fallback can name a blocker without re-running mlx imports that can hang.
_MLX_BLOCKERS_MEASURED: Optional[list[str]] = None
# Epoch whose post-warm probe found the --no-torch MLX stack unusable.
_NO_TORCH_SETTLED_EPOCH: Optional[int] = None


def _has_usable_mlx_stack() -> bool:
    """True only when the FULL MLX training/export stack is usable (mlx + mlx-lm + mlx-vlm at the versions unsloth-zoo requires), not just `import mlx.core`: a backtracked mlx-vlm still imports but breaks VLM Train/Export, so this must match the self-heal's own criterion (utils.mlx_repair) or Train is enabled on exactly the stack being repaired. Asked as "no blockers" rather than via mlx_stack_available(), which is the same question, so the answer can be explained without measuring it again."""
    global _MLX_BLOCKERS_MEASURED
    _MLX_BLOCKERS_MEASURED = None
    try:
        from utils.mlx_repair import mlx_stack_blockers
        blockers = mlx_stack_blockers()
    except Exception as exc:
        # Fall back to the bare import check rather than forcing a working host to chat-only.
        logger.debug("MLX stack availability check failed, using bare import: %s", exc)
        return _has_mlx()
    _MLX_BLOCKERS_MEASURED = blockers
    return not blockers


def _mlx_stack_detail() -> Optional[str]:
    """One line naming what the MLX gate is unhappy about, or None. Never raises and never re-runs the gate's verdict: this only describes one already reached, so a failure costs a sentence, not Train. Measuring again is the fallback for a caller that arrived without a measurement."""
    global _MLX_BLOCKERS_MEASURED
    blockers = _MLX_BLOCKERS_MEASURED
    _MLX_BLOCKERS_MEASURED = None
    if blockers is None:
        try:
            from utils.mlx_repair import mlx_stack_blockers
            blockers = mlx_stack_blockers()
        except Exception as exc:
            logger.debug("MLX blocker detail unavailable: %s", exc)
            return None
    if not blockers:
        return None
    return "; ".join(blockers[:3])


# Asks the OS rather than torch; display-only, must not feed runtime device lists.
_PHYSICAL_GPU_INVENTORY_TTL_SECONDS = 60.0
_physical_gpu_inventory_lock = threading.Lock()
_physical_gpu_inventory_refresh_lock = threading.Lock()
_physical_gpu_inventory_refreshing = False
# One re-detection request per recovery, or frequent polls starve detection.
_REDETECTION_REQUESTED = False
_physical_gpu_inventory_cache: Optional[tuple[float, Dict[str, Any]]] = None
# Cached: availability probes can block on a wedged driver and health reads this.
_TORCH_BUILD_SNAPSHOT_TTL_SECONDS = 60.0
_torch_build_snapshot_lock = threading.Lock()
_torch_build_snapshot_refresh_lock = threading.Lock()
_torch_build_snapshot_refreshing = False
_torch_build_snapshot_cache: Optional[tuple[float, Dict[str, Any]]] = None
# "No measurement yet", distinct from a measured reason None.
_UNKNOWN_TORCH_BUILD_SNAPSHOT: Dict[str, Any] = {
    "reason": None,
    "usable": False,
    "unknown": True,
}


def _probe_physical_gpu_inventory() -> Dict[str, Any]:
    """One uncached pass over the vendor probes. Never raises. ``index`` is the probe's own row number and is vendor-local, so it identifies a device only together with ``vendor``. It is not a pin."""
    devices: list[Dict[str, Any]] = []
    sources: list[str] = []
    # "No devices" differs from "no probe answered", tracked per vendor.
    unanswered: set = set()

    try:
        from . import nvidia
        result = nvidia.get_physical_gpu_inventory()
    except Exception as e:
        logger.debug("NVIDIA physical inventory probe failed: %s", e)
        unanswered.add("nvidia")
    else:
        # An absent nvidia-smi is an answer: the normal state of non-NVIDIA hosts.
        if result.get("error") and not result.get("absent"):
            unanswered.add("nvidia")
        nvidia_devices = result.get("devices") or []
        if nvidia_devices:
            devices.extend(nvidia_devices)
            sources.append(result.get("source") or "nvidia-smi")

    # Windows AMD has no reliable vendor CLI; the DirectX registry lists every adapter.
    if platform.system() == "Windows":
        # The registry outlives hardware, so records only relabel live WMI adapters (as setup.ps1).
        _live = _windows_live_adapter_names()
        for _vendor, _vendor_id in (
            ("amd", _AMD_PCI_VENDOR_ID),
            ("intel", _INTEL_PCI_VENDOR_ID),
        ):
            try:
                records = _windows_amd_adapter_records_by_luid(_vendor_id, distinguish_failure = True)
            except Exception as e:
                logger.debug("Windows %s adapter inventory probe failed: %s", _vendor, e)
                records = None
            if records is None:
                # An unreadable key is not "no adapters"; treating it so would collapse to no_gpu.
                unanswered.add(_vendor)
                continue
            if not records:
                continue
            if _live is None:
                unanswered.add(_vendor)
                continue
            # Consumed one-to-one so one live adapter cannot match two records.
            _unclaimed = list(_live)
            _corroborated = []
            for luid in sorted(records):
                _match = _claim_live_adapter(records[luid].get("name"), _unclaimed)
                if _match is not None:
                    _unclaimed.pop(_match)
                    _corroborated.append(luid)
            for ordinal, luid in enumerate(_corroborated):
                record = records[luid]
                dedicated = record.get("dedicated_memory_bytes")
                devices.append(
                    {
                        "vendor": _vendor,
                        "index": ordinal,
                        "name": record.get("name"),
                        "memory_total_gb": (round(dedicated / 1024**3, 2) if dedicated else None),
                        **({"gfx": record["gfx"]} if record.get("gfx") else {}),
                        "source": "directx-registry",
                    }
                )
            if _corroborated and "directx-registry" not in sources:
                sources.append("directx-registry")

    # sysfs, not amd-smi: amdgpu publishes these with no ROCm userspace needed.
    if platform.system() == "Linux":
        try:
            sysfs_devices = _linux_drm_sysfs_records(distinguish_failure = True)
        except Exception as e:
            logger.debug("Linux DRM sysfs inventory probe failed: %s", e)
            sysfs_devices = None
        if sysfs_devices is None:
            sysfs_devices = []
            unanswered.update(("amd", "intel"))
        if sysfs_devices:
            if any(device.get("vendor") == "amd" for device in sysfs_devices):
                # Installers ship ROCm only for _ROCM_SUPPORTED_GFX; attached to every AMD record
                # since the orderings need not agree.
                _gfx = _linux_amd_gfx_candidates()
                if _gfx:
                    for device in sysfs_devices:
                        if device.get("vendor") == "amd":
                            device["gfx_candidates"] = _gfx
            devices.extend(sysfs_devices)
            sources.append("sysfs-drm")

    _unanswered = sorted(unanswered - {device.get("vendor") for device in devices})
    return {
        "available": bool(devices),
        "devices": devices,
        "sources": sources,
        "unknown": bool(_unanswered),
        "unanswered": _unanswered,
    }


def _linux_amd_gfx_candidates() -> list[str]:
    """gfx targets the ROCm userspace reports for this host, sorted, or []. rocminfo is what studio/setup.sh reads to pick an arch, with amd-smi as its fallback; neither is required to exist, and a host with no ROCm userspace reports nothing, so the caller keeps the card rather than guessing an arch. Bounded and never raises."""
    for command, pattern in (
        (["rocminfo"], r"\bgfx[0-9a-f]+\b"),
        (["amd-smi", "static", "--asic"], r"\bgfx[0-9a-f]+\b"),
    ):
        try:
            result = subprocess.run(
                command,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                check = False,
            )
        except FileNotFoundError:
            continue
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.debug("%s could not report a gfx target: %s", command[0], e)
            continue
        found = {match.group(0).lower() for match in re.finditer(pattern, result.stdout or "")}
        if found:
            return sorted(found)
    return []


def _hidden_console_kwargs() -> Dict[str, Any]:
    """subprocess kwargs that keep a console window from flashing on Windows. Imported here rather than at module scope because test_e2e_no_torch_sandbox.py executes this module against a stub tree with no utils.subprocess_compat, and a top-level import would make it unloadable there."""
    try:
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs
    except Exception:
        return {}
    return windows_hidden_subprocess_kwargs()


def _windows_live_adapter_names() -> Optional[list[str]]:
    """Display adapter names Windows reports as PRESENT, or None when it could not say. Win32_VideoController is the live source setup.ps1 scans; None is the important third answer, since "the scan failed" must not read as "no adapters" and drop every real card on a host with unavailable WMI."""
    if platform.system() != "Windows":
        return None
    try:
        ps = (
            # -ErrorAction Stop: SilentlyContinue exits 0 with empty output on broken WMI.
            "$ErrorActionPreference='Stop';"
            "(Get-CimInstance Win32_VideoController"
            ' | Select-Object -ExpandProperty Name) -join "`n"'
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            **_hidden_console_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("Live Windows adapter scan failed: %s", e)
        return None
    if r.returncode != 0:
        return None
    return [line.strip() for line in (r.stdout or "").splitlines() if line.strip()]


def _claim_live_adapter(name: Optional[str], live_names: list[str]) -> Optional[int]:
    """Index of the first unclaimed live adapter this record matches, or None. Returned rather than a bool so the caller can consume the entry."""
    candidate = (name or "").strip().lower()
    if not candidate:
        return None
    # Exact match first, so "RX 7900 XT" cannot claim "RX 7900 XTX".
    for index, live in enumerate(live_names):
        if live.strip().lower() == candidate:
            return index
    for index, live in enumerate(live_names):
        if _adapter_name_is_live(name, [live]):
            return index
    return None


def _adapter_name_is_live(name: Optional[str], live_names: list[str]) -> bool:
    """Whether a registry adapter name corresponds to one the live scan returned. Substring in either direction, because the registry carries the driver's description and WMI the display name, and one is often a prefix of the other. setup.ps1 joins them the same way."""
    candidate = (name or "").strip().lower()
    if not candidate:
        return False
    return any(
        candidate in live.lower() or live.lower() in candidate
        for live in live_names
        if live.strip()
    )


# XPU-capable Intel PCI IDs (DG2/ATS-M, PVC, BMG); DG1 is excluded as unsupported.
_INTEL_XPU_PCI_ID_RANGES = ((0x5690, 0x56C2), (0x0B69, 0x0BE5), (0xE200, 0xE2FF))


def _intel_pci_device_is_xpu_class(device_dir: str) -> Optional[bool]:
    """Whether the card's PCI device ID is an XPU-capable family; None if unreadable."""
    try:
        with open(os.path.join(device_dir, "device"), encoding = "utf-8") as fh:
            device_id = int(fh.read().strip(), 16)
    except (OSError, ValueError):
        return None
    return any(lo <= device_id <= hi for lo, hi in _INTEL_XPU_PCI_ID_RANGES)


def _linux_drm_sysfs_records(*, distinguish_failure: bool = False) -> "list[Dict[str, Any]] | None":
    """Every AMD (0x1002) or Intel (0x8086) card the DRM drivers have bound, from /sys/class/drm. amdgpu's mem_info_vram_total is a byte count; Intel publishes no equivalent on the discrete path, so an Arc card is reported with unknown capacity rather than left out. Only cardN is walked, since connector entries and render nodes would double-count. No name is reported: the kernel publishes none, and borrowing an amd-smi row whose ordering is not guaranteed would attach the wrong one. Never raises, and returns None when the walk could not be trusted, so the inventory marks the vendors unanswered rather than publishing "no cards" for a TTL."""
    root = "/sys/class/drm"
    unreadable = False
    try:
        entries = sorted(os.listdir(root))
    except FileNotFoundError:
        return []
    except OSError:
        return None if distinguish_failure else []
    records: list[Dict[str, Any]] = []
    for entry in entries:
        if not re.fullmatch(r"card\d+", entry):
            continue
        device = os.path.join(root, entry, "device")
        try:
            with open(os.path.join(device, "vendor"), encoding = "utf-8") as fh:
                vendor = fh.read().strip().lower()
        except FileNotFoundError:
            # No PCI vendor means a virtual or platform device.
            continue
        except OSError:
            unreadable = True
            continue
        vendors = {"0x1002": "amd", "0x8086": "intel"}
        if vendor not in vendors:
            continue
        total_gb = None
        try:
            with open(os.path.join(device, "mem_info_vram_total"), encoding = "utf-8") as fh:
                total_bytes = int(fh.read().strip())
            total_gb = round(total_bytes / 1024**3, 2) if total_bytes > 0 else None
        except (OSError, ValueError):
            pass
        record = {
            "vendor": vendors[vendor],
            "index": len(records),
            "name": None,
            "memory_total_gb": total_gb,
            "source": "sysfs-drm",
        }
        if vendors[vendor] == "intel":
            record["xpu_class"] = _intel_pci_device_is_xpu_class(device)
        records.append(record)
    if unreadable and distinguish_failure:
        # One unreadable card makes the walk partial; never publish it as complete.
        return None
    return records


_UNKNOWN_PHYSICAL_GPU_INVENTORY: Dict[str, Any] = {
    "available": False,
    "devices": [],
    "sources": [],
    "unknown": True,
    "unanswered": [],
}


def _carry_unanswered_vendors_forward(
    inventory: Dict[str, Any], previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Re-add the devices only a vendor this pass could not ask had reported: a refresh that cannot reach nvidia-smi is not news that the cards left, but cached as the new inventory it says exactly that for a TTL. Only for vendors in ``unanswered``, since a card that really was removed has to be allowed to disappear."""
    stale_vendors = set(inventory.get("unanswered") or ())
    if not stale_vendors or not previous:
        return inventory
    carried = [
        device
        for device in (previous.get("devices") or [])
        if device.get("vendor") in stale_vendors
    ]
    if not carried:
        return inventory
    merged = dict(inventory)
    merged["devices"] = list(inventory.get("devices") or []) + carried
    merged["available"] = True
    merged["sources"] = list(inventory.get("sources") or []) + [
        source
        for source in (previous.get("sources") or [])
        if source not in (inventory.get("sources") or [])
    ]
    return merged


def _run_physical_gpu_inventory_probe() -> Dict[str, Any]:
    """One probe pass, stored in the cache. Never raises."""
    global _physical_gpu_inventory_cache
    previous = _physical_gpu_inventory_cache
    try:
        inventory = _probe_physical_gpu_inventory()
    except Exception as e:
        logger.debug("Physical GPU inventory probe failed: %s", e)
        inventory = dict(_UNKNOWN_PHYSICAL_GPU_INVENTORY)
        inventory["unanswered"] = ["amd", "intel", "nvidia"]
    inventory = _carry_unanswered_vendors_forward(
        inventory, previous[1] if previous is not None else None
    )
    _physical_gpu_inventory_cache = (time.monotonic(), inventory)
    return inventory


def get_physical_gpu_inventory(*, block: bool = True) -> Dict[str, Any]:
    """GPUs the OS enumerates, whether or not PyTorch can use them: ``{available, devices, sources, unknown}``. Runs regardless of DEVICE and never raises. ``block=False`` for anything on a request path -- the NVIDIA half shells out with a 10s timeout behind a lock, so a hung driver would stall the event loop every TTL; a non-blocking caller takes the last answer, kicks a daemon-thread refresh, and gets the new value later."""
    now = time.monotonic()
    cached_entry = _physical_gpu_inventory_cache
    if cached_entry is not None and now - cached_entry[0] < _PHYSICAL_GPU_INVENTORY_TTL_SECONDS:
        return cached_entry[1]
    if not block:
        _schedule_physical_gpu_inventory_refresh()
        return (
            cached_entry[1] if cached_entry is not None else dict(_UNKNOWN_PHYSICAL_GPU_INVENTORY)
        )
    with _physical_gpu_inventory_lock:
        cached_entry = _physical_gpu_inventory_cache
        if (
            cached_entry is not None
            and time.monotonic() - cached_entry[0] < _PHYSICAL_GPU_INVENTORY_TTL_SECONDS
        ):
            return cached_entry[1]
        return _run_physical_gpu_inventory_probe()


def _schedule_single_flight_refresh(
    flag: str, refresh_lock, work_lock, run, thread_name: str, what: str
) -> None:
    """Run ``run`` off the caller's thread, one pass at a time. Single-flight through ``refresh_lock``, tried without waiting, since a refresh already running is what a second caller wants; ``work_lock`` is the blocking path's lock, so the background pass cannot publish underneath one. ``flag`` names the module global holding the in-flight bit, kept a real global so a test can reset it."""
    with refresh_lock:
        if globals()[flag]:
            return
        globals()[flag] = True

    def _refresh() -> None:
        try:
            with work_lock:
                run()
        except Exception as e:
            logger.debug("Background %s refresh failed: %s", what, e)
        finally:
            with refresh_lock:
                globals()[flag] = False

    try:
        threading.Thread(target = _refresh, name = thread_name, daemon = True).start()
    except Exception as e:
        logger.debug("Could not start the %s refresh thread: %s", what, e)
        with refresh_lock:
            globals()[flag] = False


def _schedule_physical_gpu_inventory_refresh() -> None:
    """Refresh the inventory off the caller's thread, one pass at a time."""
    _schedule_single_flight_refresh(
        "_physical_gpu_inventory_refreshing",
        _physical_gpu_inventory_refresh_lock,
        _physical_gpu_inventory_lock,
        _run_physical_gpu_inventory_probe,
        "gpu-inventory-refresh",
        "inventory",
    )


def _reported_torch_label(published: Optional[str] = None) -> Optional[str]:
    """The version to show for this torch, without ever retrying a failed import: on a host whose native runtime will not load, that import is the thing that fails and _has_torch() purges the partial module so the next attempt genuinely re-runs it. Whatever detection published wins, then the label the wheel carries on disk."""
    if TORCH_IMPORT_ERROR is not None:
        return published or _installed_torch_label_on_disk() or None
    return _torch_version_label() or _installed_torch_label_on_disk() or None


def _torch_version_label() -> Optional[str]:
    """``torch.__version__`` when it can be read, else None. Never raises."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return None


_VISIBILITY_MASK_VENDORS: Dict[str, frozenset] = {
    # HIP honours CUDA_VISIBLE_DEVICES too.
    "CUDA_VISIBLE_DEVICES": frozenset({"nvidia", "amd"}),
    "HIP_VISIBLE_DEVICES": frozenset({"amd"}),
    "ROCR_VISIBLE_DEVICES": frozenset({"amd"}),
    "ZE_AFFINITY_MASK": frozenset({"intel"}),
}


def _mask_is_emptied(var: str) -> bool:
    """True when ``var`` is set to a value that hides every device it addresses: set-but-empty and "-1". A mask NAMING devices is not this, since that host expects those devices to work."""
    value = os.environ.get(var)
    return value is not None and value.strip() in ("", "-1")


def _masks_hide_every_accelerator(*, block_inventory: bool = False) -> bool:
    """True when the masks account for every accelerator this host has, so torch reporting none is the configuration working. A mask covering only SOME cards is not this, or a CPU-only wheel would go unreported for a card the user never masked; those cards are dropped from the mismatch inventory instead. An inventory that found nothing, or could not answer, stays conservative."""
    masked = _vendors_masked_off(block_inventory = block_inventory)
    if not masked:
        return False
    try:
        inventory = get_physical_gpu_inventory(block = block_inventory)
    except Exception:
        inventory = dict(_UNKNOWN_PHYSICAL_GPU_INVENTORY)
    devices = inventory.get("devices") or []
    if devices:
        return all(device.get("vendor") in masked for device in devices)
    # An inventory that could not answer stays conservative; caching callers block.
    return True


def _emptied_cuda_mask_hides_amd_on_a_mixed_host() -> bool:
    """Emptied CUDA_VISIBLE_DEVICES hides the AMD card from ROCm torch on an NVIDIA + AMD host. A log line, not a mismatch: the mask is deliberate on AMD-only hosts. Blocks on the inventory."""
    if not _mask_is_emptied("CUDA_VISIBLE_DEVICES") or "HIP_VISIBLE_DEVICES" in os.environ:
        return False
    if not _torch_reports_a_hip_runtime():
        return False
    try:
        devices = get_physical_gpu_inventory().get("devices") or []
    except Exception:
        return False
    return {"nvidia", "amd"} <= {device.get("vendor") for device in devices}


def _vendors_masked_off(*, block_inventory: bool = False) -> set:
    """Vendors whose devices are all hidden by a mask that can take effect here."""
    relevant = _relevant_visibility_masks(block_inventory = block_inventory)
    masked: set = set()
    for var in relevant:
        if _mask_is_emptied(var):
            masked |= _VISIBILITY_MASK_VENDORS.get(var, frozenset())
    # HIP reads only the first SET of these, as amd._first_visible_amd_gpu_id does.
    amd_mask = next(
        (
            var
            for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            if var in relevant and os.environ.get(var) is not None
        ),
        None,
    )
    if amd_mask is not None and _mask_is_emptied(amd_mask):
        masked.add("amd")
    else:
        masked.discard("amd")
    return masked


def _relevant_visibility_masks(*, block_inventory: bool = False) -> tuple[str, ...]:
    """The visibility variables that can actually hide a GPU on THIS host, since a mask that cannot take effect must not silence the mismatch: Windows HIP has no ROCr layer, so a stray empty ROCR_VISIBLE_DEVICES would otherwise restore "no GPU" on a Windows NVIDIA host with a CPU wheel, and the HIP-layer variables are consulted only when an AMD card is present. An inventory that found nothing, or could not answer, keeps every variable."""
    masks = ["CUDA_VISIBLE_DEVICES"]
    hip_masks = ["HIP_VISIBLE_DEVICES"]
    if sys.platform != "win32":
        hip_masks.append("ROCR_VISIBLE_DEVICES")
    try:
        # block=False: /api/health and /api/liveness reach this via the verdict.
        devices = get_physical_gpu_inventory(block = block_inventory).get("devices") or []
    except Exception:
        devices = []
    if not devices or any(d.get("vendor") == "amd" for d in devices):
        masks.extend(hip_masks)
    # ZE_AFFINITY_MASK is the XPU equivalent.
    if not devices or any(d.get("vendor") == "intel" for d in devices):
        masks.append("ZE_AFFINITY_MASK")
    return tuple(masks)


def _torch_index_leaf(url: str) -> str:
    """Final path segment of a torch index URL, lowercased, query and fragment removed: a token-authenticated pin (.../whl/cpu?token=...) would otherwise read as "cpu?token=..." and a deliberate CPU install be reported broken. Trailing slashes come off the PATH only, or a token ending in "/" is corrupted. Mirrors the installer's _torch_index_leaf."""
    value = str(url).strip().lower()
    if not value:
        return ""
    path = re.fullmatch(r"([^?#]*)([?#].*)?", value)
    path = path.group(1) if path else value
    return path.rstrip("/").rsplit("/", 1)[-1]


def _stated_torch_index_source() -> str:
    """The torch index this install was told to use, or "". URL first, and the family ONLY when no URL is set: install.sh returns on UNSLOTH_TORCH_INDEX_URL without reading _FAMILY, so a family that disagrees is dead, not a second opinion."""
    url = (os.environ.get("UNSLOTH_TORCH_INDEX_URL") or "").strip()
    if url:
        return url
    return (os.environ.get("UNSLOTH_TORCH_INDEX_FAMILY") or "").strip()


def _installed_without_torch() -> bool:
    try:
        from utils.mlx_repair import _installed_without_torch as recorded
        return recorded()
    except Exception:
        return False


def _mlx_distribution_installed() -> bool:
    try:
        pkg_version("mlx")
    except PackageNotFoundError:
        return False
    except Exception:
        return True
    return True


def _recorded_install_flavor() -> "tuple[str, bool]":
    """``(expected_torch_tag, expected_torch_tag_pinned)`` from the venv's manifest, ``("", False)`` when there is none or it cannot be read: nothing recorded is not a choice. Read straight off disk, since install_manifest lives outside the backend package. Never raises."""
    try:
        path = os.path.join(sys.prefix, "unsloth_install_manifest.json")
        with open(path, encoding = "utf-8") as fh:
            manifest = json.load(fh)
        recorded = manifest.get("expected_torch_tag")
        pinned = manifest.get("expected_torch_tag_pinned")
    except (OSError, ValueError, AttributeError):
        return "", False
    if not isinstance(recorded, str):
        return "", False
    # `is True`, not bool(): the string "false" would read as a pin.
    return recorded.strip().lower(), pinned is True


def _expected_cpu_flavor_was_chosen() -> bool:
    """Whether THIS install deliberately selected a CPU wheel: an explicit index pin, or the flavor the last completed install recorded in the venv's manifest. Only "cpu" is acted on, since an absent record means nothing was recorded and must not read as a choice."""
    if _torch_index_leaf(_stated_torch_index_source()) == "cpu":
        return True
    # setup.ps1 records an auto-selected cpu exactly like a pinned one.
    recorded, pinned = _recorded_install_flavor()
    return recorded == "cpu" and pinned


def classify_torch_build(*, block_inventory: bool = False) -> Optional[str]:
    """Why this PyTorch exposes no accelerator, when the build is the reason. "torch_cpu_build": a CPU-only wheel, or an untagged build with neither torch.version.cuda nor .hip -- only reinstalling from the right index fixes it. "torch_cuda_unavailable": an accelerator wheel whose runtime refuses to initialise (old driver, no permission on the device nodes, a cudart that will not load) -- the wheel is right, the environment is not. None: torch is missing, unimportable, or healthy. Two reasons rather than one flag, because telling someone with a healthy cu124 wheel to reinstall torch sends them the wrong way."""
    # An emptied visibility mask looks like a broken install without being one.
    if _masks_hide_every_accelerator(block_inventory = block_inventory):
        return None
    if _expected_cpu_flavor_was_chosen():
        return None
    if not _has_torch():
        # _has_torch() collapses absent and unimportable to False.
        return _classification_from_disk_label()
    try:
        import torch

        # XPU as well, or a recovered Intel host stays torch_cuda_unavailable.
        for _available in (
            getattr(getattr(torch, "cuda", None), "is_available", None),
            getattr(getattr(torch, "xpu", None), "is_available", None),
        ):
            try:
                if callable(_available) and _available():
                    return None
            except Exception:
                continue
        version = str(getattr(torch, "__version__", ""))
        local = version.partition("+")[2].strip().lower()
        cuda_tag = getattr(getattr(torch, "version", None), "cuda", None)
        hip_tag = getattr(getattr(torch, "version", None), "hip", None)
        xpu_tag = getattr(getattr(torch, "version", None), "xpu", None)
        if local == "cpu" or local.startswith("cpu."):
            # Extended CPU local tags exist, e.g. "2.8.0+cpu.cxx11.abi".
            return "torch_cpu_build"
        if not local and cuda_tag is None and hip_tag is None and xpu_tag is None:
            # Untagged with no GPU runtime: the PyPI macOS/CPU wheel shape.
            return "torch_cpu_build"
        return "torch_cuda_unavailable"
    except Exception as e:
        logger.debug("torch build classification fell back to the on-disk label: %s", e)
        return _classification_from_disk_label()


def _classification_from_disk_label() -> Optional[str]:
    """Classify from the wheel's own version label, with no interpreter started. None when nothing is installed to read: an absent torch is not a mismatch."""
    label = _installed_torch_label_on_disk()
    markers = _installed_torch_markers_on_disk()
    if not label and not any(markers.values()):
        return None
    if "+cu" in label or "+rocm" in label or "+xpu" in label:
        return "torch_cuda_unavailable"
    if any(markers.values()):
        return "torch_cuda_unavailable"
    return "torch_cpu_build"


# setup.ps1's rule: only Arc and Data Center GPUs autodetect as XPU.
_XPU_ADAPTER_NAME_RE = re.compile(r"intel.*(arc|data center gpu)", re.IGNORECASE)


def _devices_that_can_establish_a_mismatch(devices: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """The subset of the inventory whose presence means PyTorch OUGHT to have a GPU. NVIDIA and AMD qualify outright; Intel does not by itself, since setup.sh does not autodetect Linux XPU and setup.ps1 limits it to Arc and Data Center GPU by name, so counting a UHD iGPU would offer a repair that reinstalls the CPU build it just replaced. An Intel card still counts when its name matches setup.ps1's rule, or an XPU expectation is recorded, or torch carries an XPU runtime -- the Linux sysfs walk publishes no name, which is why the latter two are consulted."""
    xpu_expected = _expected_xpu_flavor_was_chosen() or _torch_reports_an_xpu_runtime()
    # Per vendor: an emptied ZE_AFFINITY_MASK hides only Intel devices.
    masked_off = _vendors_masked_off()
    rocm_expected = _expected_rocm_flavor_was_chosen() or _torch_reports_a_hip_runtime()
    keep: list[Dict[str, Any]] = []
    for device in devices:
        if device.get("vendor") in masked_off:
            continue
        if device.get("vendor") == "amd":
            if rocm_expected or _amd_device_can_establish_a_mismatch(device):
                keep.append(device)
            continue
        if device.get("vendor") != "intel":
            keep.append(device)
            continue
        if (
            xpu_expected
            or _XPU_ADAPTER_NAME_RE.search(str(device.get("name") or ""))
            or device.get("xpu_class") is True
        ):
            keep.append(device)
    return keep


# Keep in sync with install.sh's _amd_arch_index_family_for_gfx (plus gfx906); other cards
# stay on CPU torch on purpose.
_ROCM_SUPPORTED_GFX = frozenset(
    {
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1034",
        "gfx1035",
        "gfx1036",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1200",
        "gfx1201",
    }
)


def _linux_kfd_reports_an_amd_gpu() -> bool:
    """Whether the KFD topology enumerates an AMD GPU node. Never raises. The same probe and vendor guard as install_python_stack._has_rocm_gpu(): gpu_id 0 is a CPU node, and NVIDIA's open kernel module registers KFD nodes with vendor_id 4318, so AMD ownership is confirmed rather than assumed."""
    if platform.system() != "Linux":
        return False
    nodes = "/sys/class/kfd/kfd/topology/nodes"
    try:
        entries = os.listdir(nodes)
    except OSError:
        return False
    for entry in entries:
        try:
            with open(os.path.join(nodes, entry, "gpu_id"), encoding = "utf-8") as fh:
                gpu_id = fh.read().strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not gpu_id or gpu_id == "0":
            continue
        try:
            with open(os.path.join(nodes, entry, "properties"), encoding = "utf-8") as fh:
                properties = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(r"\bvendor_id\s+4098\b", properties):
            return True
    return False


def _is_pip_rocm_family_leaf(leaf: str) -> bool:
    """True when a lowercased leaf names a pip ROCm family: EXACTLY rocm<digits>[.<digits>] or gfx<digit>, kept in step with install_python_stack._is_pip_rocm_family_leaf. A suffixed leaf is a custom pin the installer routes verbatim, and reading one as ROCm waives the supported-architecture filter, so a gfx803 host deliberately on CPU torch would be told its install is broken."""
    return bool(re.fullmatch(r"rocm\d+(?:\.\d+)?", leaf)) or bool(re.match(r"gfx\d", leaf))


def _expected_rocm_flavor_was_chosen() -> bool:
    """Whether this install selected a ROCm wheel, by pin or by recorded flavor."""
    if _is_pip_rocm_family_leaf(_torch_index_leaf(_stated_torch_index_source())):
        return True
    return _recorded_install_flavor()[0].startswith("rocm")


def _torch_reports_a_hip_runtime() -> bool:
    """Whether the installed torch is a ROCm build, however unusable it currently is."""
    if TORCH_IMPORT_ERROR is not None:
        return "+rocm" in _installed_torch_label_on_disk() or bool(
            _installed_torch_markers_on_disk()["hip"]
        )
    try:
        import torch
        if "+rocm" in str(getattr(torch, "__version__", "")).lower():
            return True
        return getattr(getattr(torch, "version", None), "hip", None) is not None
    except Exception:
        return False


def _torch_reports_another_vendors_runtime() -> bool:
    """Whether the installed torch is a CUDA or XPU build, whatever its label says.

    The mirror of _torch_reports_a_hip_runtime, and needed for the same reason in reverse.
    A conda or locally built CUDA wheel carries no +cu tag, so the label names no vendor,
    and the intent fallback then reads a stale recorded ROCm flavor as "this wheel targets
    AMD" -- on a host whose real repair is reinstalling ROCm torch. torch.version.cuda is
    written by the build itself and settles it.

    An import failure is answered from disk, as _torch_reports_a_hip_runtime and
    _torch_reports_an_xpu_runtime already answer it: torch/version.py records the runtime
    whether or not the package imports. Returning False there made the clearing above inert
    on the path it exists for, letting a stale ROCm flavor speak for a CUDA or XPU wheel.
    """
    if TORCH_IMPORT_ERROR is not None:
        if _torch_reports_a_hip_runtime():
            return False
        _markers = _installed_torch_markers_on_disk()
        return bool(_markers["cuda"]) or bool(_markers["xpu"])
    try:
        import torch

        _version = getattr(torch, "version", None)
        # A ROCm build sets hip and can carry cuda besides, so hip is read first.
        if getattr(_version, "hip", None) is not None:
            return False
        return bool(getattr(_version, "cuda", None)) or bool(getattr(_version, "xpu", None))
    except Exception:
        return False


# Mirrors setup.ps1 $nameArchTable and _WIN_GPU_NAME_ARCH_TABLE; most specific first.
_GPU_NAME_GFX_TABLE: "list[tuple[str, str]]" = [
    (r"9070|9080|R9700", "gfx1201"),
    (r"9060", "gfx1200"),
    (r"8065S|8060S|8050S|8040S|Strix Halo|Ryzen AI Max|AI Max", "gfx1151"),
    (r"890M|880M|Strix Point|HX 37[05]|AI 9 HX|AI 9 36[05]", "gfx1150"),
    (r"860M|840M|Krackan|AI 7 35[05]|AI 5 34[05]|AI 7 PRO 35|AI 5 33", "gfx1152"),
    (r"RX 7900|PRO W7900|PRO W7800", "gfx1100"),
    (r"RX 7800|RX 7700(?!S)|PRO W7700|PRO V710", "gfx1101"),
    (r"RX 7600|RX 7700S|RX 7650|PRO W7600|PRO W7500", "gfx1102"),
    (r"780M|760M|740M|Phoenix|Hawk Point|Z1 Extreme|Z2 Extreme", "gfx1103"),
    (r"RX 6950|RX 6900|RX 6850|RX 6800|RX 6750|RX 6700|PRO W6800|PRO W6900", "gfx1030"),
    (r"RX 6650|RX 6600|PRO W6600|PRO W6650", "gfx1032"),
    (r"RX 6550|RX 6500|RX 6450|RX 6400|RX 6300|PRO W6400|PRO W6500|PRO W6300", "gfx1034"),
    (r"Radeon Pro V520|Radeon Pro 5600M", "gfx1011"),
    (r"RX 5700|RX 5600|Radeon Pro 5600 XT|Radeon Pro 5700|Radeon Pro W5700", "gfx1010"),
    (r"RX 5500|RX 5300|Radeon Pro W5500|Radeon Pro W5300", "gfx1012"),
]


# Only the Windows multi-arch route ships these; Linux installers decline them.
_ROCM_SUPPORTED_GFX_WINDOWS_ONLY = frozenset({"gfx1010", "gfx1011", "gfx1012", "gfx1153"})


def _rocm_supported_gfx_here() -> "frozenset[str]":
    """The arches the installers ship a ROCm wheel for on THIS platform."""
    if platform.system() == "Windows":
        return _ROCM_SUPPORTED_GFX | _ROCM_SUPPORTED_GFX_WINDOWS_ONLY
    return _ROCM_SUPPORTED_GFX


def _rocm_supported_gfx_from_gpu_name(name: str) -> Optional[str]:
    """The gfx arch this marketing name maps to, when the ROCm wheels cover it."""
    if not name:
        return None
    _supported = _rocm_supported_gfx_here()
    for pattern, arch in _GPU_NAME_GFX_TABLE:
        if re.search(pattern, name, re.IGNORECASE) and arch in _supported:
            return arch
    return None


def _amd_device_can_establish_a_mismatch(device: Dict[str, Any]) -> bool:
    """Whether this AMD adapter is one the installers would have given a ROCm wheel: a stack that deliberately declines a card cannot then call the CPU wheel beside it a fault. The arch is consulted only when something can name it, so a host with no ROCm userspace reports no candidates and stays counted, which is the conservative direction."""
    candidates = [
        str(gfx).lower()
        for gfx in (
            device.get("gfx_candidates") or ([device.get("gfx")] if device.get("gfx") else [])
        )
        if gfx
    ]
    if not candidates:
        # The registry may omit AdapterFamily, so fall back to the marketing name.
        _named = _rocm_supported_gfx_from_gpu_name(device.get("name") or "")
        if _named:
            return True
        # Same KFD topology fallback the installer's _has_rocm_gpu() uses.
        return _linux_kfd_reports_an_amd_gpu()
    return any(gfx in _rocm_supported_gfx_here() for gfx in candidates)


def _expected_xpu_flavor_was_chosen() -> bool:
    """Whether this install selected an XPU wheel, by pin or by recorded flavor."""
    if _torch_index_leaf(_stated_torch_index_source()) == "xpu":
        return True
    return _recorded_install_flavor()[0] == "xpu"


def _torch_reports_an_xpu_runtime() -> bool:
    """Whether the installed torch is an XPU build, however unusable it currently is."""
    if TORCH_IMPORT_ERROR is not None:
        return "+xpu" in _installed_torch_label_on_disk() or bool(
            _installed_torch_markers_on_disk()["xpu"]
        )
    try:
        import torch
        if "+xpu" in str(getattr(torch, "__version__", "")).lower():
            return True
        return getattr(getattr(torch, "version", None), "xpu", None) is not None
    except Exception:
        return False


def _installed_torch_label_on_disk() -> str:
    """``torch.__version__`` read out of the installed torch/version.py, or "". No interpreter is started, which is the point: this is reached when importing torch is the thing that fails. Never raises."""
    try:
        spec = importlib.util.find_spec("torch")
    except Exception:
        return ""
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    for location in locations:
        try:
            with open(os.path.join(location, "version.py"), encoding = "utf-8") as fh:
                for line in fh:
                    if line.startswith("__version__"):
                        return line.partition("=")[2].strip().strip("\"'").lower()
        except OSError:
            continue
    return ""


def _installed_torch_markers_on_disk() -> Dict[str, Optional[str]]:
    """``{cuda, hip, xpu}`` as recorded in the installed torch/version.py. The version LABEL is not the whole story -- a conda or source CUDA build is untagged and records its runtime here -- and without these the failure path told the user to reinstall a GPU wheel it already has rather than fix the driver. Parsed, not executed. Never raises."""
    markers: Dict[str, Optional[str]] = {"cuda": None, "hip": None, "xpu": None}
    try:
        spec = importlib.util.find_spec("torch")
    except Exception:
        return markers
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    for location in locations:
        try:
            with open(os.path.join(location, "version.py"), encoding = "utf-8") as fh:
                source = fh.read()
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            # torch has shipped both `cuda = '12.8'` and `cuda: Optional[str] = '12.8'`.
            if isinstance(node, ast.Assign):
                names = [n.id for n in node.targets if isinstance(n, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            value = getattr(node, "value", None)
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for name in names:
                if name in markers:
                    markers[name] = value.value
        return markers
    return markers


def _run_torch_build_snapshot() -> Dict[str, Any]:
    """One uncached pass over torch, cached on the way out. Never raises."""
    global _torch_build_snapshot_cache
    snapshot = {
        "reason": classify_torch_build(block_inventory = True),
        "usable": _torch_reports_a_usable_accelerator(),
        "unknown": False,
    }
    _torch_build_snapshot_cache = (time.monotonic(), snapshot)
    return snapshot


def torch_build_snapshot(*, block: bool = True) -> Dict[str, Any]:
    """``{reason, usable, unknown}`` for this venv's torch, cached with a TTL. ``block=False`` on a request path: both probes import torch and ask the CUDA and XPU runtimes whether they are available, which a wedged driver -- the very state torch_cuda_unavailable names -- can hold for as long as it likes. With nothing measured yet a caller gets ``unknown``, never a guess: detect_hardware() takes the blocking path, so the cache is warm before any request consults it."""
    now = time.monotonic()
    cached_entry = _torch_build_snapshot_cache
    if cached_entry is not None and now - cached_entry[0] < _TORCH_BUILD_SNAPSHOT_TTL_SECONDS:
        return cached_entry[1]
    if not block:
        _schedule_torch_build_snapshot_refresh()
        return cached_entry[1] if cached_entry is not None else dict(_UNKNOWN_TORCH_BUILD_SNAPSHOT)
    with _torch_build_snapshot_lock:
        cached_entry = _torch_build_snapshot_cache
        if (
            cached_entry is not None
            and time.monotonic() - cached_entry[0] < _TORCH_BUILD_SNAPSHOT_TTL_SECONDS
        ):
            return cached_entry[1]
        return _run_torch_build_snapshot()


def _schedule_torch_build_snapshot_refresh() -> None:
    """Refresh the torch snapshot off the caller's thread, one pass at a time."""
    _schedule_single_flight_refresh(
        "_torch_build_snapshot_refreshing",
        _torch_build_snapshot_refresh_lock,
        _torch_build_snapshot_lock,
        _run_torch_build_snapshot,
        "torch-build-refresh",
        "torch build",
    )


def _seed_torch_build_snapshot(reason: Optional[str]) -> None:
    """Record a classification reached without probing torch: the broken-runtime host is classified from the wheel on disk, and letting the request paths re-measure would put the import that already failed back on the health thread."""
    global _torch_build_snapshot_cache
    _torch_build_snapshot_cache = (
        time.monotonic(),
        {"reason": reason, "usable": False, "unknown": False},
    )


def invalidate_torch_build_snapshot() -> None:
    """Drop the cached measurement so the next blocking caller re-probes."""
    global _torch_build_snapshot_cache
    _torch_build_snapshot_cache = None


def _mismatch_verdict_for_this_host(
    reason: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """``(reason, detail)`` when this host's GPUs are real but PyTorch cannot use them, else (None, None). Blocking, and only detection calls it: that pass runs off the request path and warms the caches /api/health then reads."""
    if reason is None:
        reason = torch_build_snapshot()["reason"]
    if reason is None:
        return None, None
    establishing = _devices_that_can_establish_a_mismatch(
        get_physical_gpu_inventory().get("devices") or []
    )
    if not establishing:
        return None, None
    _remember_the_vendors_behind_the_mismatch(establishing)
    detail = _reported_torch_label()
    logger.warning(
        "GPUs are present on this host but PyTorch cannot use them (%s%s); "
        "Train/Export disabled (chat-only). Repair the installation to restore GPU "
        "support.",
        reason,
        f", installed {detail}" if detail else "",
    )
    return reason, detail


def _torch_gpu_mismatch_report() -> Dict[str, Any]:
    """``physical_devices`` + ``mismatch`` for a host whose GPUs PyTorch cannot use; ``{}`` when there is nothing to report. Both keys sit BESIDE ``devices`` in the visibility payload and never inside it: ``devices`` is the runtime-usable list that model fit budgets against and the training device picker pins from."""
    # block=False: request path, and torch probes can hang on a wedged driver.
    reason = torch_build_snapshot(block = False)["reason"]
    if reason is None:
        return {}
    # block=False: GET /api/system holds _system_gpu_cache_lock for the whole call.
    inventory = get_physical_gpu_inventory(block = False)
    physical = _devices_that_can_establish_a_mismatch(inventory.get("devices") or [])
    if not physical:
        return {}
    return {
        "physical_devices": physical,
        "mismatch": {
            "reason": reason,
            "torch_version": _reported_torch_label(CHAT_ONLY_DETAIL),
            "physical_count": len(physical),
            "sources": inventory.get("sources") or [],
        },
    }


def verdict_pending_mlx_repair(chat_only: bool, reason: Optional[str]) -> bool:
    """True when this settled verdict is one the MLX self-heal is about to overturn: detection answers before utils.mlx_repair gets its turn, so an Apple Silicon host with a broken MLX stack settles chat-only and flips only once the background reinstall lands. Published as final, that greys Train behind a tooltip the repair makes wrong a minute later. Callers report it as still-detecting instead. Takes the verdict as arguments, so a caller holding a snapshot does not re-read the globals mid-pass."""
    if not chat_only or reason != "mlx_unavailable":
        return False
    if not is_apple_silicon():
        return False
    try:
        from utils.mlx_repair import mlx_repair_in_flight
        return mlx_repair_in_flight()
    except Exception as exc:
        # An unanswerable self-heal cannot be relied on, so let the verdict settle.
        logger.debug("MLX repair progress check failed, treating the verdict as final: %s", exc)
        return False


def verdict_blames_the_mlx_stack() -> bool:
    """Unlocked deliberately: _DETECT_LOCK spans a whole detection pass, imports included, so taking it would park the post-warm worker behind an early request's first import. The overturn re-reads under the lock, so a straddling read costs one needless measurement."""
    return bool(CHAT_ONLY) and CHAT_ONLY_REASON == "mlx_unavailable"


def settle_the_no_torch_verdict(epoch: int) -> bool:
    """For the post-warm probe that measured a --no-torch host's stack unusable: nothing will overturn mlx_unavailable now, so publish no_torch and let the sidebar stop polling. ``epoch`` predates the measurement, as for overturn_the_mlx_verdict: a shutdown since retired that probe, and the next lifespan measures for itself."""
    global CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, _NO_TORCH_SETTLED_EPOCH
    with _DETECT_LOCK:
        if epoch != current_detection_epoch():
            return False
        if not CHAT_ONLY or CHAT_ONLY_REASON != "mlx_unavailable":
            return False
        # Recorded only after confirming this probe's verdict is still live.
        _NO_TORCH_SETTLED_EPOCH = epoch
        CHAT_ONLY_REASON, CHAT_ONLY_DETAIL = "no_torch", None
        return True


def overturn_the_mlx_verdict(epoch: Optional[int] = None) -> bool:
    """For a caller that has just measured the stack as usable. Read and re-detect share one locked section, or a forced pass landing between them loses its answer to this one. ``epoch`` predates the measurement, so a shutdown since discards the pass instead of republishing for a dead lifespan."""
    with _DETECT_LOCK:
        if not CHAT_ONLY or CHAT_ONLY_REASON != "mlx_unavailable":
            return False
        with owning_detection_epoch(epoch):
            detect_hardware()
        return DEVICE is not None and DETECTION_COMPLETE.is_set() and not CHAT_ONLY


def _print_cuda_device_list(is_rocm: bool) -> None:
    """List every visible CUDA/ROCm GPU with its index at startup: the "Hardware detected" banner names only device 0, hiding the other cards. Indices are CUDA ordinals, matching `nvidia-smi -L` when no mask is set. CUDA_DEVICE_ORDER governs only CUDA, so it is shown for CUDA but not ROCm. Purely informational and never raises."""
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
        return


def detect_hardware() -> DeviceType:
    """Detect the best compute device and set the module-level DEVICE global; call once at FastAPI lifespan startup, idempotent. Order: XPU only on an unambiguous "prefer XPU" signal (CUDA hidden or unavailable, or UNSLOTH_FORCE_XPU=1) AND a non-empty ZE_AFFINITY_MASK AND torch.xpu reporting a device, since a stray inherited mask must not beat CUDA on hybrid hosts; then CUDA, XPU, MLX, CPU."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM, DETECTION_GENERATION
    with _DETECT_LOCK:
        # Clear the event during a forced pass, or health serves a half-written verdict.
        was_complete = DETECTION_COMPLETE.is_set()
        # Snapshot the whole verdict so a mid-pass raise can restore it.
        published = (DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM)
        # Owning epoch first: the MLX self-heal can outlast the lifespan.
        epoch = getattr(_OWNING_EPOCH, "value", None)
        if epoch is None:
            epoch = current_detection_epoch()
        elif current_detection_epoch() != epoch:
            # Retired before this pass began: leave the running lifespan alone.
            return DEVICE
        DETECTION_COMPLETE.clear()
        try:
            device = _detect_hardware_locked()
        except BaseException:
            if current_detection_epoch() != epoch:
                _discard_detection_locked()
                raise
            DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM = published
            # Restore it: start_background_detection() declines once DEVICE is set.
            if was_complete:
                DETECTION_COMPLETE.set()
            raise
        if current_detection_epoch() != epoch:
            _discard_detection_locked()
            return device
        DETECTION_GENERATION += 1
        DETECTION_COMPLETE.set()
        global _REDETECTION_REQUESTED
        _REDETECTION_REQUESTED = False
        return device


def ensure_hardware_detected(epoch: Optional[int] = None) -> DeviceType:
    """Detect once, from any thread; prefer this to detect_hardware() unless you want a forced re-detect, since it collapses the warm thread and an early request into one pass. Never raises: a raise on the warm thread would leave DEVICE None, so every later request retries the failing import and /api/health 500s; record CPU + chat-only with a reason instead. ``epoch`` is the epoch this pass belongs to, passed by a spawner because the thread can be scheduled after a shutdown retired it."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, DETECTION_GENERATION
    global _REDETECTION_REQUESTED
    with _DETECT_LOCK:
        if epoch is None:
            epoch = getattr(_OWNING_EPOCH, "value", None)
        if epoch is None:
            epoch = current_detection_epoch()
        elif DEVICE is None and current_detection_epoch() != epoch:
            return DEVICE
        produced_here = DEVICE is None
        if produced_here:
            # DEVICE is assigned before probes that can still fall back, so clear the stale event.
            DETECTION_COMPLETE.clear()
            try:
                _detect_hardware_locked()
            except BaseException as exc:  # noqa: BLE001 - degrade, never 500 the health check
                logger.error("Hardware detection failed; falling back to CPU: %r", exc)
                DEVICE = DeviceType.CPU
                CHAT_ONLY = True
                CHAT_ONLY_REASON = "detection_failed"
                CHAT_ONLY_DETAIL = None
            # Bump only here: the orchestrator rebuilds defaults whenever this moves.
            DETECTION_GENERATION += 1
        if produced_here and current_detection_epoch() != epoch:
            _discard_detection_locked()
            return DEVICE
        # Set only once the value is final; a non-None DEVICE may still be a candidate.
        DETECTION_COMPLETE.set()
        if produced_here:
            # After the epoch check, so a retired pass does not release the guard.
            _REDETECTION_REQUESTED = False
        return DEVICE


def _xpu_device_name_or_placeholder(torch) -> str:
    """A failing name probe must not demote an already-found XPU to CPU + detection_failed."""
    try:
        return torch.xpu.get_device_name(0)
    except Exception as e:
        logger.debug("XPU device 0 name probe failed: %s", e)
        return "<unavailable>"


# RDNA 3/3.5/4 only; gfx1250 is Instinct, so avoid a broad gfx12 prefix.
_MIOPEN_SEARCH_CUTOFF_ARCH_PREFIXES = ("gfx110", "gfx115", "gfx120")


def _configure_rocm_miopen(torch) -> None:
    if "MIOPEN_SEARCH_CUTOFF" in os.environ:
        return
    try:
        count = torch.cuda.device_count()
        # MIOpen's cutoff is process-wide, so every visible GPU must qualify.
        if not count or not all(
            _props_gfx_arch(torch.cuda.get_device_properties(i)).startswith(
                _MIOPEN_SEARCH_CUTOFF_ARCH_PREFIXES
            )
            for i in range(count)
        ):
            return
    except Exception as exc:
        logger.debug("MIOpen search cutoff device probe failed: %s", exc)
        return
    os.environ.setdefault("MIOPEN_SEARCH_CUTOFF", "1")
    logger.info("ROCm RDNA: enabled MIOpen search cutoff (MIOPEN_SEARCH_CUTOFF=1)")


def _detect_hardware_locked() -> DeviceType:
    """detect_hardware() body. Call only with _DETECT_LOCK held."""
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, CHAT_ONLY_DETAIL, IS_ROCM
    global _MLX_BLOCKERS_MEASURED, CHAT_ONLY_MISMATCH_VENDORS
    CHAT_ONLY = True
    CHAT_ONLY_REASON = None
    CHAT_ONLY_DETAIL = None
    CHAT_ONLY_MISMATCH_VENDORS = frozenset()
    _MLX_BLOCKERS_MEASURED = None
    IS_ROCM = False

    # Probe torch once per pass: a failed probe is expensive and a second can disagree.
    torch_ok = _has_torch()

    if torch_ok:
        import torch

        # A bare ZE_AFFINITY_MASK can leak from Intel tooling; torch.xpu must report a device.
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
                # Hide CUDA when forcing XPU, or spawned workers would silently train on CUDA.
                if force_xpu and not cuda_hidden and not cuda_unavailable:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                DEVICE = DeviceType.XPU
                CHAT_ONLY = False
                CHAT_ONLY_REASON = None
                device_name = _xpu_device_name_or_placeholder(torch)
                if force_xpu and not ze_mask:
                    reason = "UNSLOTH_FORCE_XPU=1"
                elif force_xpu:
                    reason = "UNSLOTH_FORCE_XPU=1 + ZE_AFFINITY_MASK"
                else:
                    reason = "ZE_AFFINITY_MASK hint honoured"
                print(f"Hardware detected: XPU -- {device_name} ({reason})")
                return DEVICE

        # Reuse the guarded answer: a raising second is_available() would skip the XPU branch.
        if not cuda_unavailable:
            DEVICE = DeviceType.CUDA
            CHAT_ONLY = False
            try:
                device_name = torch.cuda.get_device_properties(0).name
            except Exception as e:
                logger.debug("CUDA device 0 property probe failed: %s", e)
                device_name = "<unavailable>"

            # Display only; AMD SDK wheels do not set torch.version.hip, so check __version__.
            _hip_ver = getattr(torch.version, "hip", None)
            if _hip_ver is not None or "rocm" in torch.__version__.lower():
                IS_ROCM = True
                _configure_rocm_miopen(torch)
                _hip_label = _hip_ver or torch.__version__
                print(f"Hardware detected: ROCm (HIP {_hip_label}) -- {device_name}")
            else:
                print(f"Hardware detected: CUDA -- {device_name}")
            _print_cuda_device_list(IS_ROCM)
            return DEVICE

    if torch_ok:
        import torch
        try:
            xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception as e:
            logger.debug("XPU availability probe failed: %s", e)
            xpu_ok = False
        if xpu_ok:
            DEVICE = DeviceType.XPU
            CHAT_ONLY = False
            device_name = _xpu_device_name_or_placeholder(torch)
            print(f"Hardware detected: XPU — {device_name}")
            return DEVICE

    # Require the full MLX stack to match utils.mlx_repair; partial stacks get self-healed.
    if is_apple_silicon() and _has_usable_mlx_stack():
        DEVICE = DeviceType.MLX
        CHAT_ONLY = False
        # platform.processor() returns "i386" on universal2 / Rosetta builds.
        chip = platform.machine() or "arm64"
        print(f"Hardware detected: MLX — Apple Silicon ({chip})")
        return DEVICE

    DEVICE = DeviceType.CPU
    if (
        is_apple_silicon()
        and _installed_without_torch()
        and (
            _NO_TORCH_SETTLED_EPOCH == current_detection_epoch()
            or not _mlx_distribution_installed()
        )
    ):
        # GGUF-only by request; with mlx on disk the post-warm probe decides instead.
        CHAT_ONLY_REASON = "no_torch"
        _MLX_BLOCKERS_MEASURED = None
        logger.info(
            "Apple Silicon installed --no-torch (GGUF-only); Train/Export are off by "
            "request. Reinstall without --no-torch to enable them."
        )
    elif is_apple_silicon():
        CHAT_ONLY_REASON = "mlx_unavailable"
        CHAT_ONLY_DETAIL = _mlx_stack_detail()
        logger.warning(
            "Apple Silicon detected but the MLX stack is incomplete or too old; "
            "Train/Export disabled (chat-only)%s Run `unsloth studio update` to "
            "restore MLX training.",
            f" ({CHAT_ONLY_DETAIL})." if CHAT_ONLY_DETAIL else ".",
        )
    elif TORCH_IMPORT_ERROR is not None:
        # torch installed but broken, so this host was never measured.
        CHAT_ONLY_REASON = "detection_failed"
        # Classify from the wheel on disk since the import failed, applying the same suppressions.
        _disk_reason = None
        if not (
            _masks_hide_every_accelerator(block_inventory = True) or _expected_cpu_flavor_was_chosen()
        ):
            _disk_reason = _classification_from_disk_label()
        _seed_torch_build_snapshot(_disk_reason)
        _build_reason, _build_detail = _mismatch_verdict_for_this_host(_disk_reason)
        if _build_reason is not None:
            CHAT_ONLY_REASON, CHAT_ONLY_DETAIL = _build_reason, _build_detail
    elif platform.system() == "Darwin":
        CHAT_ONLY_REASON = "intel_mac"
    else:
        # No accelerator from torch is not "no GPU": the wheel may be CPU-only. Ask the OS.
        CHAT_ONLY_REASON = "no_gpu"
        if torch_ok:
            _build_reason, _build_detail = _mismatch_verdict_for_this_host()
            if _build_reason is not None:
                CHAT_ONLY_REASON, CHAT_ONLY_DETAIL = _build_reason, _build_detail
            elif _emptied_cuda_mask_hides_amd_on_a_mixed_host():
                logger.warning(
                    "CUDA_VISIBLE_DEVICES=%r hides the AMD GPU from ROCm torch as well as the "
                    "NVIDIA one: HIP reads it when HIP_VISIBLE_DEVICES is unset. Unset "
                    "CUDA_VISIBLE_DEVICES (or set HIP_VISIBLE_DEVICES=0) before launching. "
                    "To get ROCm torch on an NVIDIA + AMD host without the mask, install "
                    "with UNSLOTH_FORCE_ROCM_TORCH=1.",
                    os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
    print("Hardware detected: CPU training backend (no PyTorch/MLX GPU backend available)")
    return DEVICE


def get_device() -> DeviceType:
    """Return the detected device, auto-detecting if detect_hardware() has not run. Prefer calling detect_hardware() explicitly at startup."""
    return ensure_hardware_detected()


def _torch_reports_a_usable_accelerator() -> bool:
    """Whether torch can open a GPU right now. Never raises."""
    try:
        import torch
        for probe in (
            getattr(getattr(torch, "cuda", None), "is_available", None),
            getattr(getattr(torch, "xpu", None), "is_available", None),
        ):
            try:
                if callable(probe) and probe():
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _request_hardware_redetection() -> None:
    """Ask for a fresh detection pass, at most one per recovery. Never raises. Retiring the epoch is what the rest of this module already uses to mean "the published verdict is stale"; recomputing DEVICE here instead would publish from a request thread and race the detection lock."""
    global _REDETECTION_REQUESTED
    if _REDETECTION_REQUESTED:
        return
    try:
        _REDETECTION_REQUESTED = True
        # invalidate_detection alone leaves DEVICE and the event set; reset both under the lock.
        invalidate_detection()
        invalidate_torch_build_snapshot()
        with _DETECT_LOCK:
            _discard_detection_locked()
        start_background_detection()
        logger.info(
            "An accelerator became usable after startup; discarded the cached hardware "
            "verdict and started a fresh detection pass."
        )
    except Exception as e:
        _REDETECTION_REQUESTED = False
        logger.debug("Could not request hardware re-detection: %s", e)


def _remember_the_vendors_behind_the_mismatch(devices: list[Dict[str, Any]]) -> None:
    """Record which vendors' cards establish the mismatch being reported right now."""
    global CHAT_ONLY_MISMATCH_VENDORS
    CHAT_ONLY_MISMATCH_VENDORS = frozenset(
        device.get("vendor") for device in devices if device.get("vendor")
    )


def _uncertainty_could_hide_the_frozen_mismatch(inventory: Dict[str, Any]) -> bool:
    """Whether an inventory that could not answer might still hold the mismatched card. Only for the vendors the mismatch came from: _carry_unanswered_vendors_forward has already re-added what an unanswered vendor last reported, so nothing left here means every vendor that named a card answered "none" this pass, and holding the mismatch on an unrelated vendor's broken probe would assert a GPU with nothing to point at (a detached AMD eGPU staying "unusable" while nvidia-smi is broken on a host that never had an NVIDIA card). True when no vendor was recorded (every reason but the two mismatches, so those are unchanged), and true for an unknown that names nobody, which is how a cold cache reads."""
    if not CHAT_ONLY_MISMATCH_VENDORS:
        return True
    unanswered = set(inventory.get("unanswered") or ())
    if not unanswered:
        return True
    return bool(unanswered & CHAT_ONLY_MISMATCH_VENDORS)


def current_chat_only_verdict() -> tuple[Optional[str], Optional[str]]:
    """``(reason, detail)``, re-derived when the physical inventory can still change it.

    detect_hardware() runs once at startup, but the inventory it consulted refreshes on a 60 second TTL. An eGPU attached after launch, or a driver that finished restarting after the first probe, flips the answer while the frozen verdict keeps saying ``no_gpu``: /api/system would list the card and publish a mismatch while the sidebar and the Export and Video pages went on insisting no accelerator exists. The reverse is the same bug, a card that goes away leaving a mismatch nobody can act on.

    Only the three inventory-sensitive verdicts are re-derived, plus the one detection_failed that is not really unmeasured: a torch that will not import was classified from its wheel on disk at startup, so the inventory is the only thing still missing. mlx_unavailable, intel_mac and a detection_failed with no importable torch and no readable wheel describe things a 60 second probe cannot change, and re-deriving those would fight detect_hardware() rather than follow it.

    Never raises: a probe that cannot answer keeps the frozen verdict, as does an inventory whose uncertainty is about the vendor the frozen mismatch came from.
    """
    reason, detail = CHAT_ONLY_REASON, CHAT_ONLY_DETAIL
    frozen_but_measurable = reason == "detection_failed" and TORCH_IMPORT_ERROR is not None
    if reason not in ("no_gpu", "torch_cpu_build", "torch_cuda_unavailable"):
        if not frozen_but_measurable:
            return reason, detail
    try:
        snapshot = torch_build_snapshot(block = False)
        if snapshot["unknown"]:
            return reason, detail
        build_reason = snapshot["reason"]
        if build_reason is None and snapshot["usable"]:
            # The accelerator came back: re-detect so DEVICE and CHAT_ONLY are not stale.
            _request_hardware_redetection()
            return reason, detail
        # block=False: health routes reach here and nvidia-smi can hang for 10 s.
        inventory = get_physical_gpu_inventory(block = False)
        establishing = (
            _devices_that_can_establish_a_mismatch(inventory.get("devices") or [])
            if build_reason is not None
            else []
        )
        if establishing:
            # Re-record: the mismatch can move between vendors within one process.
            _remember_the_vendors_behind_the_mismatch(establishing)
            return build_reason, _reported_torch_label(detail)
        if inventory.get("unknown") and _uncertainty_could_hide_the_frozen_mismatch(inventory):
            return reason, detail
    except Exception as e:
        logger.debug("chat-only verdict refresh failed: %s", e)
        return reason, detail
    # A host whose torch will not import never measured "no GPU".
    return (reason, detail) if frozen_but_measurable else ("no_gpu", None)


# Matched on the local part so a plain version can never look like a label.
_WHEEL_LABEL_OTHER_VENDOR_RE = re.compile(r"\+[a-z]*(?:cu\d|xpu)")


def _intel_xpu_pin_hint(vendors: "set[str]") -> str:
    """Linux installers install XPU only when asked, so a plain re-run reinstalls CPU; Windows autodetects Arc."""
    if vendors != {"intel"} or platform.system() != "Linux":
        return ""
    return (
        "On Linux the Unsloth installer installs the Intel XPU build only when asked: re-run "
        "it with UNSLOTH_TORCH_INDEX_FAMILY=xpu set."
    )


def _gpu_present_but_unusable_message(
    feature: str, verdict: Optional[tuple[Optional[str], Optional[str]]] = None
) -> Optional[str]:
    """The capability message for a host whose GPUs are real but unreachable by torch, or ``None`` when this host is not in that state. detect_hardware() records ``torch_cpu_build`` / ``torch_cuda_unavailable`` only after the OS inventory has actually found a card, so reaching this point means the "no supported accelerator was found" wording below would contradict the System tab and send the user after hardware they already own. Both reasons are surfaced verbatim by the Export and Video pages and by rejected export API calls."""
    # Read once: two reads across a TTL boundary can describe different hosts.
    reason, detail = verdict if verdict is not None else current_chat_only_verdict()
    if reason not in ("torch_cpu_build", "torch_cuda_unavailable"):
        return None
    installed = f" (installed {detail})" if detail else ""
    # No reinstall changes group membership, so a closed node is not a mismatch.
    vendors = {str(vendor).lower() for vendor in CHAT_ONLY_MISMATCH_VENDORS}
    _label = (detail or "").lower()
    # Intent is the last resort: it outlives the wheel installed over it.
    wheel_targets_amd = (
        "rocm" in _label
        or "hip" in _label
        or _torch_reports_a_hip_runtime()
        or (
            not _WHEEL_LABEL_OTHER_VENDOR_RE.search(_label)
            # An untagged CUDA build names no vendor, so stale intent must not speak for it.
            and not _torch_reports_another_vendors_runtime()
            and _expected_rocm_flavor_was_chosen()
        )
    )
    amd_is_the_target = vendors == {"amd"} or wheel_targets_amd
    node_hint = None
    if "amd" in vendors and amd_is_the_target:
        try:
            from utils.hardware.amd import (
                amd_closed_nodes_block_the_runtime,
                amd_node_permission_hint,
            )

            # An open sibling render node means ROCm had a path and failed anyway.
            if amd_closed_nodes_block_the_runtime():
                node_hint = amd_node_permission_hint()
        except Exception:
            node_hint = None
    # Replace the reinstall advice only for a ROCm wheel; others need both repairs.
    if node_hint and reason == "torch_cuda_unavailable" and wheel_targets_amd:
        return f"This host has a GPU, but {feature} cannot use it. {node_hint}"
    # Both routes: the repair row exists only in the desktop app for a managed backend.
    if reason == "torch_cpu_build":
        repair = _intel_xpu_pin_hint(vendors) or (
            "Reinstall the GPU build: use Repair installation in Settings in the desktop app, "
            "or re-run the Unsloth installer."
        )
        return (
            f"This host has a GPU, but the installed PyTorch is a CPU-only build{installed}, "
            f"so {feature} cannot use it. {repair}" + (f" {node_hint}" if node_hint else "")
        )
    return (
        f"This host has a GPU, but the installed PyTorch{installed} cannot initialise it, so "
        f"{feature} cannot use it. This is usually a driver or runtime mismatch; reinstalling "
        f"a matching PyTorch build fixes it. Use Repair installation in Settings in the "
        f"desktop app, or re-run the Unsloth installer." + (f" {node_hint}" if node_hint else "")
    )


def export_capability() -> dict:
    """Whether model export can run here, with a torch-aware reason when it cannot. Export runs through Unsloth, which hard-requires an accelerator (it calls ``torch.cuda`` at import and has no CPU path), so it is supported iff ``get_device() in {CUDA, XPU, MLX}``. The reason distinguishes a --no-torch install from a bare-CPU host. Safe to call without torch. Returns {export_supported, export_unsupported_reason, export_unsupported_message, torchao_export_supported}; the last is False only on Windows ROCm without a loadable torchao, where the portable FP8/INT8 formats are hidden."""
    device = get_device()
    torchao_export_supported = True
    if sys.platform == "win32" and IS_ROCM:
        try:
            from core._torchao_stub import torchao_export_loadable
            torchao_export_supported = torchao_export_loadable()
        except Exception:
            torchao_export_supported = False
    if device in (DeviceType.CUDA, DeviceType.XPU, DeviceType.MLX):
        return {
            "export_supported": True,
            "export_unsupported_reason": None,
            "export_unsupported_message": None,
            "torchao_export_supported": torchao_export_supported,
        }
    verdict = current_chat_only_verdict()
    # Detection failure first: later branches assume a measured host.
    if verdict[0] == "detection_failed":
        reason = "detection_failed"
        message = (
            "Hardware detection failed on this host, so export is disabled. The server log records "
            "the underlying error; restart Unsloth Studio to retry detection."
        )
    elif verdict[0] == "no_torch":
        reason = "no_torch"
        message = (
            "This install was set up without the training stack (--no-torch), so export is "
            "disabled. Reinstall Unsloth Studio without --no-torch to enable export."
        )
    elif is_apple_silicon():
        reason = "mlx_unavailable"
        message = (
            "Export on Apple Silicon requires the MLX stack, which is unavailable or too old. Run "
            "`unsloth studio update` to restore MLX and enable export."
        )
    elif _gpu_present_but_unusable_message("export", verdict) is not None:
        # Before _has_torch(), which re-runs the slow failing import.
        reason = verdict[0]
        message = _gpu_present_but_unusable_message("export", verdict)
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
        "torchao_export_supported": torchao_export_supported,
    }


def video_capability() -> dict:
    """Whether video generation can run here, with a torch-aware reason when it cannot. Supported on CUDA and XPU, and on Apple Silicon with a usable MPS device: the pipelines in core/inference/video.py are device-neutral, resolving the device through the shared diffusion device target whose capability flags already decline the CUDA-only options, so Metal needs no branches of its own. Safe to call without torch. Returns {video_supported, video_unsupported_reason, video_unsupported_message}."""
    if get_device() in (DeviceType.CUDA, DeviceType.XPU):
        return {
            "video_supported": True,
            "video_unsupported_reason": None,
            "video_unsupported_message": None,
        }
    # Detection failure first: later branches assume a measured host.
    verdict = current_chat_only_verdict()
    if verdict[0] == "detection_failed":
        reason = "detection_failed"
        message = (
            "Hardware detection failed on this host, so video generation is disabled. The server "
            "log records the underlying error; restart Unsloth Studio to retry detection."
        )
    elif is_apple_silicon() or get_device() == DeviceType.MLX:
        if _torch_mps_available():
            return {
                "video_supported": True,
                "video_unsupported_reason": None,
                "video_unsupported_message": None,
            }
        if TORCH_IMPORT_ERROR is not None:
            # A broken torch reads as absent below, so report detection_failed here.
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
        # Neither torch nor a GPU enables video on an Intel Mac.
        reason = "macos_unsupported"
        message = (
            "Video generation requires Apple Silicon. This Intel Mac has no Metal (MPS) device "
            "for the video pipelines to run on."
        )
    elif _gpu_present_but_unusable_message("video generation", verdict) is not None:
        reason = verdict[0]
        message = _gpu_present_but_unusable_message("video generation", verdict)
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
    """Clear GPU memory cache for the current device. Safe on any platform: no-ops gracefully."""
    gc.collect()

    device = get_device()

    if device == DeviceType.CUDA:
        import torch

        # Skip synchronize when nothing is reserved: it creates a ~612 MiB context.
        # Not wrapped: unload paths need a sticky CUDA fault to propagate.
        if torch.cuda.is_available() and any(
            torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())
        ):
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == DeviceType.XPU:
        # Older torch-xpu builds may lack these; torch.xpu has no ipc_collect().
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
        # Diffusion still runs on Metal when MLX is unavailable, so clear MPS too.
        _clear_mps_cache()


def _clear_mps_cache() -> None:
    """Return torch's MPS reservations to the shared pool: Apple Silicon also runs torch MPS, whose caching allocator keeps freed buffers reserved, and those bytes read as used system memory, so the next load budgets against a pool that looks smaller than it is."""
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
    """Whether GPU_DEVICE_ORDINAL renumbers HIP devices. ROCclr-layer, so it applies on Windows too, and no visibility spec here reads it: a torch ordinal cannot be paired with a physical id while it is set."""
    return bool(os.environ.get("GPU_DEVICE_ORDINAL", "").strip())


def _cuda_order_matches_smi() -> bool:
    """Whether torch ordinals and nvidia-smi rows share one index space. CUDA enumerates FASTEST_FIRST by default and nvidia-smi reports PCI order, and PCI_BUS_ID is only a setdefault here, so an explicit override survives. Equal-sized cards defeat the total-scope check, so nothing else catches it. One GPU is exempt: every ordering is the identity there."""
    if IS_ROCM or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
        return True
    # Use the SMI count: torch counts only visible devices. Cached against transient timeouts.
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
    """Driver used-memory turned into free bytes, in torch's allocatable scope. Subtract from torch's total, never the driver's: NVIDIA's total also spans a reserved framebuffer torch can never hand out (726 MiB on a B200), so driver_total - used reports a full card as free. Clamped both ends, since the parsers accept signed values and a negative used would advertise more free than the card has."""
    return min(total_bytes, max(0, total_bytes - round(used_gb * (1024**3))))


def _mlx_device_info(mx: Any) -> Dict[str, Any]:
    """Metal device properties across the MLX rename: mx.device_info() is current, mlx below 0.30 has only mx.metal.device_info(), and the stack gate accepts mlx >= 0.22.0, so reading just the new name leaves the working set cap silently unapplied."""
    for probe in (
        getattr(mx, "device_info", None),
        getattr(getattr(mx, "metal", None), "device_info", None),
    ):
        if callable(probe):
            try:
                return probe() or {}
            except Exception:
                continue
    return {}


def _apple_unified_free_bytes(available_bytes: int, device_info: Any) -> int:
    """What is available right now, bounded by the Metal working set. Total RAM minus GPU-only usage counts host RAM as free, which the training-method policy reads as room it does not have. The cap is not reduced by the AGX "In use system memory" counter, which is whole-device and only the active subset, while the working set is the per-process budget torch.mps applies. A floor rather than a capacity: macOS reclaims compressed and file-backed pages, and reading low costs a QLoRA suggestion where reading high costs an OOM mid-run."""
    free = max(0, int(available_bytes or 0))
    try:
        recommended = int(device_info.get("max_recommended_working_set_size") or 0)
    except Exception:
        recommended = 0
    return min(free, recommended) if recommended > 0 else free


def _context_free_cuda_memory_info(
    idx: int,
    total_bytes: int,
    unified: bool = False,
) -> Optional[int]:
    """System-wide free bytes without attaching a CUDA/HIP primary context. ``unified`` marks a ROCm APU, whose *total* must still come from HIP but whose *used* is available here (see _rocm_windows_unified_used_bytes)."""
    parent_visible_spec = _get_parent_visible_gpu_spec()

    # Unified parts use WDDM counters: Windows hipMemGetInfo is not system-wide.
    if unified:
        if platform.system() != "Windows" or _rocm_device_ordinal_active():
            return None
        used_bytes = _rocm_windows_unified_used_bytes()
        if used_bytes is None:
            return None
        return _free_in_torch_scope(total_bytes, used_bytes / (1024**3))

    # Prefer the out-of-process vendor CLI so no context stays resident here.
    visible_ids = parent_visible_spec["numeric_ids"]
    if IS_ROCM:
        visible_ids = _amd_smi_ids_for_hip_ids(visible_ids)
    # UUID masks name devices absolutely, so the order gate does not apply.
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

    # DRM sysfs is context-free; the resolver needs the complete physical set.
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

    # GPU_DEVICE_ORDINAL renumbers torch ordinals but not the visible spec.
    if platform.system() == "Windows" and not _rocm_device_ordinal_active():
        numeric_ids = parent_visible_spec.get("numeric_ids")
        device_ids = (
            numeric_ids if numeric_ids else list(range(_torch_get_physical_gpu_count() or 0))
        )
        # Only when counter instances are exactly the visible set, or free_gb can be overstated.
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
    """GPU memory info for CUDA (NVIDIA), MLX (Apple Silicon) and CPU-only."""
    device = get_device()

    if device == DeviceType.CUDA:
        try:
            import torch

            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)

            total = props.total_memory
            allocated = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx)

            # Prefer context-free telemetry: mem_get_info pins a context; WDDM HIP is process-local.
            driver_total_needed = _rocm_props_total_is_carve_out(props)
            free = None
            if driver_total_needed:
                try:
                    free, driver_total = trusted_mem_get_info(idx)
                    # utilization_pct divides by the total, so a zero is not usable.
                    if driver_total:
                        total = driver_total
                except Exception as e:
                    logger.debug("mem_get_info probe failed; free VRAM from reserved: %s", e)
                    free = max(0, total - reserved)
                # Sum Shared Usage only for positively identified UMA, or a discrete GPU understates free.
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

    if device == DeviceType.XPU:
        try:
            import torch

            idx = torch.xpu.current_device()
            props = torch.xpu.get_device_properties(idx)

            total = props.total_memory
            allocated = torch.xpu.memory_allocated(idx)
            reserved = torch.xpu.memory_reserved(idx)

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

    if device == DeviceType.MLX:
        try:
            import mlx.core as mx
            import psutil

            memory = psutil.virtual_memory()
            total = memory.total
            agx = _read_apple_gpu_stats()
            allocated = agx.get("vram_used_bytes", 0) if agx else 0

            info = _mlx_device_info(mx)
            # processor() can return "i386" on native arm64.
            gpu_name = info.get("device_name") or platform.machine() or "arm64"
            free = _apple_unified_free_bytes(getattr(memory, "available", 0), info)

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": 0,
                "device_name": f"Apple Silicon ({gpu_name})",
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": allocated / (1024**3),
                "free_gb": free / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total else 0,
            }
        except Exception as e:
            logger.error(f"Error getting MLX GPU info: {e}")
            return {
                "available": False,
                "backend": _backend_label(device),
                "error": str(e),
            }

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


def get_gpu_summary() -> Dict[str, Any]:
    """Compact summary of the primary GPU: gpu_name (e.g. "NVIDIA L4") and vram_total_gb, either of which may be None."""
    mem = get_gpu_memory_info()
    if mem.get("available"):
        return {
            "gpu_name": mem.get("device_name"),
            "vram_total_gb": round(mem.get("total_gb", 0), 2),
            "vram_free_gb": round(mem.get("free_gb", 0), 2),
        }
    return {"gpu_name": None, "vram_total_gb": None, "vram_free_gb": None}


def get_package_versions() -> Dict[str, Optional[str]]:
    """Installed versions of unsloth / torch / transformers / cuda via importlib.metadata, no subprocess; missing packages yield None."""
    packages = ("unsloth", "torch", "transformers")
    versions: Dict[str, Optional[str]] = {}

    for name in packages:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = None

    try:
        import torch

        versions["cuda"] = getattr(torch.version, "cuda", None)
        versions["rocm"] = getattr(torch.version, "hip", None)
        # Isolated: a broken Intel runtime must not blank the cuda/rocm versions.
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # torch.version.xpu may be None on modern builds.
                xpu_ver = getattr(torch.version, "xpu", None)
                versions["xpu"] = xpu_ver if xpu_ver is not None else "available"
        except Exception:
            versions["xpu"] = None
    except Exception:
        versions["cuda"] = None
        versions["rocm"] = None
        versions["xpu"] = None

    return versions


def _torch_get_device_module():
    """Return the appropriate torch device module (cuda or xpu) and its name.

    No torch at all answers ``(None, None)`` like an unsupported device: raising took the
    exception out through /api/system on a host whose vendor CLI HAD found GPUs.
    """
    device = get_device()
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - no torch is an answer, not a fault
        logger.debug("torch is not importable: %s", e)
        return None, None

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
    """Whether ``mem_get_info``'s FREE half must be treated as an over-report. AMD documents it: on Windows the free memory only accounts for this process's allocations, because WDDM virtualises video memory, so a fresh process sees free at or near total whatever else is resident (ROCm/librocdxg#57 measured 24410 of 24560 MiB free on a filled card; ROCm/TheRock#3724 is the same symptom). Near ``total``, not equal, so callers cap rather than test a sentinel. The TOTAL half is fine, and WSL is deliberately left alone: sys.platform is "linux" there and free tracks physical residency."""
    return sys.platform == "win32" and IS_ROCM


def trusted_mem_get_info(device: Any = None, *, module: Any = None) -> tuple[int, int]:
    """``mem_get_info`` with the Windows ROCm free over-report capped (#8403). Guards that budget against free VRAM cannot be handed a figure saying the whole card is free while a model is resident: on WDDM the overflow does not raise, the driver satisfies it from host RAM, and the process grows past the card. Free is capped at what this process's torch allocator has NOT reserved, a true upper bound but still blind to other processes, so a ceiling rather than a measurement. Off Windows ROCm the driver's own numbers pass through untouched. ``module`` defaults to torch.cuda; exceptions propagate."""
    import torch

    mod = module if module is not None else torch.cuda
    free_bytes, total_bytes = mod.mem_get_info() if device is None else mod.mem_get_info(device)
    free_bytes, total_bytes = int(free_bytes), int(total_bytes)
    if not rocm_windows_free_is_untrusted():
        return free_bytes, total_bytes
    try:
        reserved = int(mod.memory_reserved() if device is None else mod.memory_reserved(device))
    except Exception as e:
        logger.debug("memory_reserved probe failed while capping free VRAM: %s", e)
        return free_bytes, total_bytes
    return min(free_bytes, max(0, total_bytes - reserved)), total_bytes


# clr sets integrated from the APU flag only from 6.1.2, which is indistinguishable from 6.1.0.
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


# Unified-memory APUs not flagged integrated by older clr; gfx1103 never shipped discrete.
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
    """Whether this part is KNOWN to be unified memory, not merely unclassified. _rocm_props_total_is_carve_out folds "uncertain" in with "unified" because a too-small total hides models, but anything ADDING host-shared memory to a used figure needs the stricter question, since shared bytes are not part of a discrete card's props.total_memory. A part the classifier can NAME as an APU is not uncertain: on Windows paldevice.cpp adds the WDDM shared heap for any Pal::GpuType::Integrated part regardless of the props flag, so a gfx1103 Phoenix on a pre-6.2 runtime already carries a pool-scoped total, and answering False pairs that pool with Dedicated Usage alone, which plateaus at the BIOS carve-out (measured: 17.0 GB total, 8 GiB resident, published as 1.90 used)."""
    if not IS_ROCM:
        return False
    return _rocm_props_unified_status(props) is True


def _rocm_props_total_is_carve_out(props: Any) -> bool:
    """True when ``props.total_memory`` MAY understate what torch can use, because on some stacks it is the dedicated carve-out while hipMemGetInfo's total spans the GTT pool. May, not does: on both APUs tested the two agree, so callers must compare and adopt only a LARGER driver total, never adopt it outright. There is no context-free source for the GTT total, so only APUs pay mem_get_info. "Not unified" is not "discrete": the classifier knows an APU by the driver's integrated flag or a hardcoded arch set, so a gfx1103 Phoenix on an older runtime reads as discrete and would publish its carve-out as the whole device -- hence the driver total is worth a context whenever the flag is unfilled or the classifier fails."""
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


def _cuda_props_are_integrated(props: Any, backend: Optional[str] = "cuda") -> bool:
    """Whether ``props`` describes an integrated CUDA part whose VRAM is system RAM.

    Jetson and DGX Spark class parts set ``cudaDeviceProp::integrated`` (torch's
    ``is_integrated``, ``integrated`` on older wheels).

    ROCm and XPU are excluded BY NAME, not by trusting the field to be absent: HIP reuses
    this namespace and left that field unassigned before 6.2, so reading it would call a
    discrete card integrated on exactly the runtimes ``_HIP_INTEGRATED_FLAG_MIN``
    distrusts (``_rocm_props_unified_status`` is the classifier for that hardware), and a
    same-named field on a future Intel wheel must not start rewriting an iGPU's capacity.
    ``torch.version.hip`` as well as ``IS_ROCM`` because that global is published by
    detection, and this inventory is reachable from inside detection.
    """
    if IS_ROCM or backend != "cuda":
        return False
    try:
        import torch
        if getattr(torch.version, "hip", None):
            return False
    except Exception as e:  # noqa: BLE001 - no torch to ask: IS_ROCM alone then
        logger.debug("HIP probe failed while classifying an integrated device: %s", e)
    return bool(getattr(props, "is_integrated", False) or getattr(props, "integrated", False))


def _torch_get_device_inventory(device_indices: list[int]) -> list[Dict[str, Any]]:
    """Per-GPU name and total VRAM only, without creating a driver context: get_device_properties is answered from the driver's device list, while mem_get_info attaches a primary context worth ~612 MiB the process never gives back, and a telemetry poll must not be what pins that, so callers wanting live occupancy go to _torch_get_per_device_info instead. props.total_memory equals mem_get_info's total everywhere except a ROCm APU (see _rocm_props_total_is_carve_out), and used_gb is always None ("telemetry unavailable")."""
    mod, backend = _torch_get_device_module()
    if mod is None:
        return []

    devices = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            props = mod.get_device_properties(ordinal)
            props_total_bytes = int(props.total_memory)
            total_bytes = props_total_bytes
            known_unified = _rocm_props_are_positively_unified(props)
            shared_memory = known_unified and platform.system() == "Windows"
            shared_memory_host_backed_bytes = None
            try:
                if _rocm_props_total_is_carve_out(props) and hasattr(mod, "mem_get_info"):
                    # Only adopt a WIDER total: the classifier fails open, so discrete cards reach here too.
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
                    "_cuda_integrated": _cuda_props_are_integrated(props, backend),
                }
            )
        except Exception as e:
            logger.debug("torch inventory probe failed for ordinal %d: %s", ordinal, e)
    return devices


def _torch_get_per_device_info(device_indices: list[int]) -> list[Dict[str, Any]]:
    """Per-GPU name, total VRAM and used VRAM from torch. Creates a driver context on CUDA/HIP, so call it only when live occupancy is consumed. ``used_gb`` is None on Windows ROCm when the driver reports free == total: that 0 means unknown, not empty, and this is the DISPLAY path, where unknown is right and a pessimistic ceiling is not."""
    mod, _ = _torch_get_device_module()
    if mod is None:
        return []

    device = get_device()
    _win_rocm = rocm_windows_free_is_untrusted()
    devices = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
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
                    # Arc B580 and Lunar Lake can reject free-memory queries; keep the device and total.
                    logger.debug(
                        "XPU free-memory query failed for ordinal %d: %s",
                        ordinal,
                        e,
                    )
                    used_bytes = None
                else:
                    # free==total is the Windows ROCm broken-API sentinel, not an idle GPU.
                    if _win_rocm and free_bytes == total_bytes:
                        used_bytes = None
            elif device == DeviceType.XPU:
                # memory_allocated() is process-local and misleading, so report unknown.
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


def _xpu_hierarchy_is_composite() -> bool:
    """True iff Level Zero runs in COMPOSITE device hierarchy, where numeric ZE_AFFINITY_MASK entries address root GPU IDs (tiles use N.M). Under FLAT, the oneAPI default and the assumption when ZE_FLAT_DEVICE_HIERARCHY is unset, entries address tile handles and mapping them back to root IDs is unsafe."""
    hierarchy = (os.environ.get("ZE_FLAT_DEVICE_HIERARCHY") or "FLAT").strip().upper()
    return hierarchy == "COMPOSITE"


def _parse_ze_mask_roots(mask: str) -> list[int]:
    """Parse ZE_AFFINITY_MASK into ordered root device IDs, one per token, preserving order and duplicates so logical ordinals map 1-to-1 onto physical root IDs ("2.0,0.1,0.2" -> [2, 0, 0]). Only meaningful in COMPOSITE hierarchy, which callers must gate on."""
    roots: list[int] = []
    if not mask:
        return roots
    for token in mask.split(","):
        token = token.strip()
        if not token:
            continue
        root = token.split(".", 1)[0]
        # isdecimal(): isdigit() accepts Unicode superscripts that crash int().
        if root.isdecimal():
            roots.append(int(root))
    return roots


def _smi_query(func_name: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
    """Query the appropriate SMI backend (amd-smi or nvidia-smi); None when unavailable."""
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
    """macOS IORegistry AGX live stats (utilization_pct, and vram_used_bytes when reported; system-wide), or {} on failure. No sudo needed."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator"],
            capture_output = True,
            timeout = 2,
        )
        text = result.stdout.decode("utf-8", errors = "replace")
    except Exception:
        return {}

    m = re.search(r'"PerformanceStatistics" = \{([^}]+)\}', text)
    if not m:
        return {}
    stats_str = m.group(1)
    pairs = re.findall(r'"([^"]+)"=(\d+)', stats_str)
    stats = {k: int(v) for k, v in pairs}

    out = {"utilization_pct": stats.get("Device Utilization %", 0)}
    if "In use system memory" in stats:
        out["vram_used_bytes"] = stats["In use system memory"]
    return out


# psutil <= 7.2.2 misreads M4+ kHz tables as Hz; read via ioreg until psutil#2824 ships.

# Apple clocks are 0.6-4.6 GHz, so a raw Hz entry sits above 1e8 and kHz below.
_CPU_FREQ_UNIT_THRESHOLD = 100_000_000
_MIN_PLAUSIBLE_CPU_MHZ = 500
_MAX_PLAUSIBLE_CPU_MHZ = 20000
# Below this a table is a GPU/NPU rail; the slowest CPU cluster peaks at 2064 MHz.
_CPU_CLUSTER_MIN_PEAK_MHZ = 2000
_VOLTAGE_STATES_KEY = re.compile(r"^voltage-states\d+-sram$")

# Probed once; the sentinel separates "not probed" from "unavailable".
_apple_cpu_peak_mhz: Any = "unprobed"
_apple_cpu_peak_lock = threading.Lock()


def _voltage_state_freqs_mhz(blob: bytes) -> list:
    """Plausible MHz from a voltage-statesN-sram blob: each entry is 8 bytes, little-endian uint32 frequency then uint32 voltage."""
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

    # Locked so concurrent first requests do not each spawn ioreg or poison the cache.
    with _apple_cpu_peak_lock:
        if _apple_cpu_peak_mhz != "unprobed":
            return _apple_cpu_peak_mhz

        peak = None
        try:
            import plistlib

            result = subprocess.run(
                ["ioreg", "-a", "-r", "-c", "AppleARMIODevice", "-d", "1"],
                capture_output = True,
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
    """Current CPU clock in MHz, corrected for the psutil Apple Silicon unit bug. None when no frequency is available (psutil reports nothing inside many containers and VMs)."""
    freq = None
    try:
        import psutil
        freq = psutil.cpu_freq()
    except Exception as e:
        # Not fatal: psutil raises on M5, and the IORegistry read below stands in.
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
    # Recover the magnitude from psutil's kHz-as-Hz value; lands on the GHz step.
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
    """The LUID ``_parse_adapter_luid`` reads, behind a ``pid_<pid>_`` prefix."""
    head = instance_name.lower().find("luid_0x")
    if head < 0:
        return None
    return _parse_adapter_luid(instance_name[head:])


def _rocm_windows_perf_counter_gpu_util_pct(luid: Optional[int] = None) -> Optional[float]:
    """AMD GPU compute utilization via Windows Performance Counters (3D engine nodes). ``luid`` narrows the sum to one adapter's engines, matched here rather than in the counter path so tests can reach it."""
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
    """System-wide AMD GPU VRAM from /sys/class/drm/card*/device/mem_info_vram_*, which the kernel updates across all processes. Returns (used_gb, total_gb), or (None, None) on failure."""
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


# NVIDIA's open module also registers KFD nodes; only AMD nodes take an ordinal.
_AMD_PCI_VENDOR_ID = 4098
_INTEL_PCI_VENDOR_ID = 0x8086


def _rocm_kfd_gpu_pci_ids() -> list[str]:
    """PCI addresses of the GPUs ROCm enumerates, in HIP device order, from the KFD topology ROCm itself enumerates from: AMD GPU nodes (simd_count > 0, vendor_id == AMD) in node-id order, so position N is ROCm physical device N. Unlike DRM sysfs, an amdgpu adapter HIP cannot enumerate has no node here. Returns [] when KFD is absent, and FAILS CLOSED the same way on an unreadable node or an AMD node with no location_id, since dropping one would shift every later ordinal. location_id is the kernel's (bus << 8) | devfn."""
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
            return []  # unreadable node could be a GPU: fail closed
        if props.get("simd_count", 0) <= 0:
            continue
        if props.get("vendor_id") != _AMD_PCI_VENDOR_ID:
            continue
        location_id = props.get("location_id")
        if location_id is None:
            return []
        domain = props.get("domain", 0)
        bus = (location_id >> 8) & 0xFF
        devfn = location_id & 0xFF
        bdf = f"{domain:04x}:{bus:02x}:{(devfn >> 3) & 0x1F:02x}.{devfn & 0x7}"
        nodes.append((int(m.group(1)), bdf))
    nodes.sort(key = lambda n: n[0])
    return [bdf for _node_id, bdf in nodes]


def _rocm_linux_amdgpu_cards() -> list[tuple[str, int, str]]:
    """The amdgpu-bound DRM cards in PCI order: (pci_bdf, card_no, device_dir). Membership is by the BOUND DRIVER, not the VRAM sysfs files, since an AMD device with incomplete sysfs still consumes a ROCm ordinal and dropping it would shift every later card. PCI order is HIP's default enumeration order, so list position is the ROCm ordinal. NOTE this is a superset of the ROCm-visible set, so callers must check the counts agree before assuming a 1:1 mapping."""
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
                continue
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
    """System-wide AMD VRAM via Linux DRM sysfs, keyed by the card's PCI address so every GPU gets its own figure. Keyed by PCI address, not an ordinal, so the caller can join it to _rocm_kfd_gpu_pci_ids() by identity: DRM card numbers include foreign adapters and cards HIP does not enumerate, so an ordinal from this list alone can be shifted relative to ROCm's."""
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


# Windows ROCm: hipMemGetInfo reports free==total, so read used from per-LUID counters.
_ROCM_WIN_ADAPTER_MIN_BYTES = 64 * 1024 * 1024


def _rocm_windows_perf_counter_vram_by_adapter(
    counter: str = "Dedicated Usage",
) -> Optional[list[tuple[str, float]]]:
    """Per-adapter VRAM usage on Windows via Performance Counters. ``counter`` selects the GPU Adapter Memory field; Dedicated Usage is the default and the only one safe to select adapters on (see _rocm_windows_unified_used_bytes for why Shared Usage is read separately). Returns [(instance_name, used_bytes)] per LUID-named adapter, or None when the counter is unavailable so callers fall back."""
    if platform.system() != "Windows":
        return None
    try:
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


# DirectX adapter records join counter LUIDs to torch names and gfx targets.
_WINDOWS_DIRECTX_KEY = r"SOFTWARE\Microsoft\DirectX"


def _parse_adapter_luid(instance_name: str) -> Optional[int]:
    """The 64-bit LUID in a ``GPU Adapter Memory`` instance name, or None. Instances are named luid_0x<high>_0x<low>_phys_<n>, while DirectX stores one 64-bit AdapterLuid, so recombine the halves."""
    m = re.match(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)", instance_name.strip(), re.IGNORECASE)
    if m is None:
        return None
    try:
        return (int(m.group(1), 16) << 32) | int(m.group(2), 16)
    except ValueError:
        return None


# hipDeviceProp_tR0600 prefix layout; the name read back verifies the layout.
_HIP_PROPS_NAME = slice(0, 256)
_HIP_PROPS_LUID = slice(272, 280)
_HIP_PROPS_NODE_MASK = slice(280, 284)
# Oversized on purpose: HIP writes the whole ~2 KiB struct.
_HIP_PROPS_BUFFER_BYTES = 64 * 1024


def _rocm_windows_hip_adapter_ids(
    ordinals: list[int], names: list[str]
) -> Optional[list[tuple[int, int]]]:
    """The (luid, node_mask) HIP itself reports for each visible ordinal, so the counters and torch join on the adapter itself. Windows reassigns LUIDs across a reboot or driver restart and all three sources move together, so the value is only ever compared within one poll. node_mask is luidDeviceNodeMask: which nodes of a linked adapter this ordinal owns, 0 on an ordinary adapter. None when the runtime cannot be asked at all, so the caller falls back to the DirectX join -- all or nothing, since a partially resolved set would let one card's counter answer for another. Route from @pablo86gr in #8793."""
    if platform.system() != "Windows" or not ordinals:
        return None
    try:
        import ctypes

        torch = sys.modules.get("torch")
        major = str(getattr(getattr(torch, "version", None), "hip", "")).split(".", 1)[0]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        # Only an already-loaded module: LoadLibrary could pull in a different runtime.
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
                logger.debug("HIP properties prefix did not read back ordinal %d", ordinal)
                return None
            luid_bytes = blob[_HIP_PROPS_LUID]
            if not any(luid_bytes):
                return None
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
    """Attribute per-adapter used bytes on the LUID HIP reports for each ordinal: the exact key, so unlike the DirectX join this separates two cards of one model. Ordinals come from ``visible_ordinal``, not this list's positions, which compact when a device fails to probe. A linked-node adapter puts several ordinals behind one LUID and the counters index its nodes as phys_N without saying which ordinal owns which, so the node mask can tell "these counters are exactly these ordinals' nodes" from "one belongs to a node HIP is not showing us", but cannot pair them: those ordinals report unknown and feed only the aggregate. The third return is that LUID, only where the device is the whole adapter."""
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
        nodes_by_luid[luid] = nodes_by_luid.get(luid, 0) + (bin(node_mask).count("1") or 1)

    assigned: list[Optional[float]] = [None] * len(dev_meta)
    whole_adapter: list[Optional[int]] = [None] * len(dev_meta)
    total_used = 0.0
    for luid, positions in positions_by_luid.items():
        useds = useds_by_luid.get(luid, [])
        # Counter count must equal node count, or the readings are not exactly this device's.
        if len(useds) != nodes_by_luid[luid]:
            return None
        used = float(sum(useds))
        # The carve-out, not the displayed total, which on an APU is the whole pool.
        capacity = sum(_adapter_counter_capacity(dev_meta[position]) for position in positions)
        if used > capacity:
            return None
        total_used += used
        if len(positions) == 1:
            assigned[positions[0]] = used
            # The node mask names which nodes an ordinal owns; observed phys_N must match it.
            named = {i for i in range(32) if identities[positions[0]][1] >> i & 1}
            if not named or named == physes_by_luid.get(luid, set()):
                whole_adapter[positions[0]] = luid
    return assigned, total_used, whole_adapter


_ADAPTER_NAME_NOISE = re.compile(r"\((?:tm|r)\)|[™®]", re.IGNORECASE)


def _normalize_adapter_name(name: str) -> str:
    """A GPU name in the one spelling both sides of the join can agree on: DirectX takes its Description from the driver INF and HIP fills props.name from the ASIC record, so trademark marks and spacing differ ("AMD Radeon(TM) 780M Graphics" against "AMD Radeon 780M Graphics"). Nothing here merges two different models, and a collision that did normalize alike is caught by the count check in _attribute_adapter_useds_by_key."""
    return " ".join(_ADAPTER_NAME_NOISE.sub(" ", name).split()).casefold()


def _parse_adapter_family_gfx(family: str) -> str:
    """The gfx target in a DirectX ``AdapterFamily``, or "". The AMD driver writes AMD_NAVI44:gfx1200, while torch reports the same target as props.gcnArchName, which on Linux carries feature suffixes the comparison must drop."""
    for token in str(family).split(":"):
        token = token.strip().lower()
        if re.fullmatch(r"gfx[0-9a-f]+", token):
            return token
    return ""


def _windows_amd_adapter_records_by_luid(
    vendor_id_filter: int = _AMD_PCI_VENDOR_ID, *, distinguish_failure: bool = False
) -> "dict[int, Dict[str, Any]] | None":
    """DirectX registry metadata for one vendor's adapters, keyed by LUID; ``vendor_id_filter`` defaults to AMD, and the physical inventory also passes Intel's id so an Arc host whose XPU wheel was replaced is reported. ``gfx`` and ``dedicated_memory_bytes`` are absent when the driver wrote none. All or nothing: an incomplete map is indistinguishable from a complete one at the join and would pair a visible card with a hidden same-named card's counter, so any failure gives up on the whole map and drops the caller back to capacity ranking. Ranking callers get {}; ``distinguish_failure`` returns None instead, because the inventory has to tell a vendor with no adapters from a vendor it could not ask."""
    records = _windows_amd_adapter_records_or_none(vendor_id_filter)
    if records is None and not distinguish_failure:
        return {}
    return records


def _windows_amd_adapter_records_or_none(
    vendor_id_filter: int = _AMD_PCI_VENDOR_ID,
) -> "dict[int, Dict[str, Any]] | None":
    """The read itself. ``None`` whenever the registry could not answer."""
    if platform.system() != "Windows":
        return None
    try:
        import winreg
    except ImportError:
        return None
    by_luid: dict[int, Dict[str, Any]] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_DIRECTX_KEY) as dx_key:
            for index in range(winreg.QueryInfoKey(dx_key)[0]):
                subkey = winreg.EnumKey(dx_key, index)
                # Only GUID-named subkeys are adapter records.
                if not (subkey.startswith("{") and subkey.endswith("}")):
                    continue
                with winreg.OpenKey(dx_key, subkey) as adapter_key:
                    vendor_id, _ = winreg.QueryValueEx(adapter_key, "VendorId")
                    if int(vendor_id) != vendor_id_filter:
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
                    return None
                record = {"name": name}
                gfx = _parse_adapter_family_gfx(str(family))
                if gfx:
                    record["gfx"] = gfx
                if has_dedicated_memory:
                    record["dedicated_memory_bytes"] = dedicated_memory_bytes
                by_luid[int(luid)] = record
    except Exception as e:
        logger.debug("DirectX adapter registry read declined: %s", e)
        return None
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
    """The capacity a ``Dedicated Usage`` counter for this device can actually fill: that counter measures the dedicated segment, so the carve-out is its ceiling, while ``total_bytes`` is what the user is SHOWN and on a unified APU may be the whole driver pool. Rank or bound a counter with this, never total_bytes."""
    return float(meta.get("dedicated_bytes", meta["total_bytes"]))


def _attribute_adapter_useds_by_key(
    useds_by_key: dict[str, list[float]],
    positions_by_key: dict[str, list[int]],
    dev_meta: list[Dict[str, Any]],
) -> Optional[tuple[list[Optional[float]], float]]:
    """Pair each key's counters with the devices carrying that key, or decline: returns (per_device_used_bytes, aggregate_used_bytes), or None when any key's counters are not exactly its devices'. Several devices under one key leave per-device unknown, since nothing says which counter is which ordinal, while still contributing to the aggregate."""
    assigned: list[Optional[float]] = [None] * len(dev_meta)
    total_used = 0.0
    for key, positions in positions_by_key.items():
        useds = useds_by_key.get(key, [])
        # Counters must match cards one-to-one under this key, or they are not its devices'.
        if len(useds) != len(positions):
            return None
        by_capacity = sorted(positions, key = lambda p: -_adapter_counter_capacity(dev_meta[p]))
        for used, position in zip(sorted(useds, reverse = True), by_capacity):
            # Usage above its card's capacity means a stale record, so the key is unreliable.
            if used > _adapter_counter_capacity(dev_meta[position]):
                return None
            total_used += used
        if len(positions) == 1:
            assigned[positions[0]] = useds[0]
    return assigned, total_used


def _match_adapter_used_by_luid(
    adapters: list[tuple[str, float]], dev_meta: list[Dict[str, Any]]
) -> Optional[tuple[list[Optional[float]], float]]:
    """Attribute per-adapter used bytes to torch devices on the adapter LUID: join each counter to a DirectX record by the LUID in its instance name, then to a torch device by what the record and device agree on. Identity, not capacity, so it resolves the single-GPU case ranking never can and stays right when a busy foreign adapter outweighs an idle card. The model name is tried first, since it separates two cards of one arch, with the gfx target as fallback -- and that pass runs only when every AMD record has one, so a driver too old to write AdapterFamily cannot make a hidden card's counter look like the visible card's. Measured on a Windows gfx1151 the name pass carried it and that driver wrote no AdapterFamily at all, so the two keys are not belt and braces: the name pass is load bearing in practice. None when neither key establishes the join, so the caller falls back to capacity ranking."""
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
        if "" in positions_by_key:
            continue
        # Unplaceable AMD usage means the key misidentifies cards; decline rather than mispair.
        if set(useds_by_key) - set(positions_by_key):
            continue
        matched = _attribute_adapter_useds_by_key(useds_by_key, positions_by_key, dev_meta)
        if matched is not None:
            # Rank passes by devices placed, keep the first on a tie.
            resolved = sum(used is not None for used in matched[0])
            if resolved == len(dev_meta):
                return matched
            if resolved > best_resolved:
                best, best_resolved = matched, resolved
    return best


def _match_adapter_used_to_devices(
    adapter_useds: list[float], device_totals: list[float]
) -> list[Optional[float]]:
    """Attribute per-adapter used bytes to torch devices by capacity ranking, since Windows shares no key between LUID counters and torch ordinals: each usage is trusted only when capacity FORCES it (it exceeds every smaller device), and an ambiguous ranking reports None rather than fabricate a per-index free. Extra counters mean a hidden/display adapter and the noise filter may have dropped a real reading, so values are emitted only when the supra-threshold counters number EXACTLY the visible devices. ``device_totals`` must be the DEDICATED capacity each counter can fill, never a unified total: a widened total reranks its device, raises the threshold that forces a pairing, and admits counters belonging to no visible card."""
    n = len(device_totals)
    if n == 0:
        return []
    useds = sorted(adapter_useds, reverse = True)
    ranked_positions = sorted(range(n), key = lambda i: -device_totals[i])
    ranked_totals = [device_totals[pos] for pos in ranked_positions]
    assigned: list[Optional[float]]
    # More counters than devices means a hidden adapter (check before the noise filter).
    if len(useds) > n:
        non_trivial = [u for u in useds if u >= _ROCM_WIN_ADAPTER_MIN_BYTES]
        if len(non_trivial) != n:
            # Not a clean bijection, so no counter maps to a specific card.
            return [None] * n
        dropped = [u for u in useds if u < _ROCM_WIN_ADAPTER_MIN_BYTES]
        useds = non_trivial
        ranked_useds = [useds[rank] for rank in range(n)]
        # Usage above its ranked capacity is a hidden larger GPU; do not clamp onto this one.
        for rank in range(n):
            if ranked_useds[rank] > ranked_totals[rank]:
                return [None] * n
        # Attribute only when every other counter is exactly zero; errs toward understating free.
        if n == 1 and all(u == 0 for u in dropped):
            return [min(ranked_useds[0], device_totals[0])]
        # Capacity forces the mapping only when usage exceeds the next-smaller capacity.
        assigned = [None] * n
        for rank, pos in enumerate(ranked_positions):
            if rank + 1 < n and ranked_useds[rank] > ranked_totals[rank + 1]:
                assigned[pos] = min(ranked_useds[rank], device_totals[pos])
        return assigned
    ranked_useds = [useds[rank] if rank < len(useds) else 0.0 for rank in range(n)]
    # Ambiguous if a larger usage also fits the next smaller card.
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
    """Total VRAM used across the visible devices, when the counters cover them 1:1. Per-device attribution needs capacity to FORCE a pairing, which an asymmetric pair defeats (#7452), but the SUM does not need the pairing, so the System tab keeps a real figure where per-device cannot. Emitted only when the counter list IS the visible set, established by cardinality alone, which rests on an unconfirmed ASSUMPTION that Get-Counter emits exactly one instance per WDDM adapter; it fails closed if that is wrong. Deliberately NOT the noise filter _match_adapter_used_to_devices uses: dropping sub-threshold counters is safe for a capacity-FORCED value but not for a sum, since a retained counter cannot be told from a foreign adapter and the sum would silently gain bytes on no visible card. A host total that is confidently wrong is worse than Unknown. ``device_totals`` must be dedicated capacity, never a unified total."""
    n = len(device_totals)
    if n == 0 or not adapter_useds:
        return None
    # Counter count must equal n, or the sum is not the visible set's.
    if len(adapter_useds) != n:
        return None
    useds = sorted(adapter_useds, reverse = True)
    ranked_totals = sorted(device_totals, reverse = True)
    for rank in range(n):
        if useds[rank] > ranked_totals[rank]:
            return None
    return float(sum(useds))


def _rocm_windows_unified_used_bytes(
    dedicated: Optional[list[tuple[str, float]]] = None,
) -> Optional[float]:
    """Used VRAM for a unified-memory ROCm APU on Windows, from the WDDM counters. Dedicated Usage alone saturates at the carve-out: measured on a gfx1151 Strix Halo host, dedicated plateaus around 30.5 GiB while the overflow lands in Shared (holding 48 GiB: +29.19 dedicated, +19.02 shared), so only the sum tracks the allocation. Adapter SELECTION still keys off Dedicated Usage alone, deliberately: display and placeholder adapters report 0 dedicated while carrying gigabytes of shared, so filtering on the sum would add a foreign adapter's bytes. None unless exactly one adapter clears the noise floor. ``dedicated`` lets a caller hand over a snapshot it already holds, since each query is an out-of-process PowerShell call answering from a different instant."""
    if dedicated is None:
        dedicated = _rocm_windows_perf_counter_vram_by_adapter()
    if not dedicated:
        return None
    candidates = [
        (instance, used) for instance, used in dedicated if used >= _ROCM_WIN_ADAPTER_MIN_BYTES
    ]
    # Need exactly one compute adapter to attribute a figure.
    if len(candidates) != 1:
        return None
    instance, dedicated_used = candidates[0]
    shared = _rocm_windows_perf_counter_vram_by_adapter("Shared Usage")
    # A failed Shared query is not zero: overflow lives in Shared, and zero overstates free.
    if shared is None:
        return None
    shared_used = next((used for name, used in shared if name == instance), 0.0)
    # A negative counter is broken and would publish free above total.
    if dedicated_used < 0.0 or shared_used < 0.0:
        return None
    return dedicated_used + shared_used


def _rocm_windows_unified_used_bytes_for_luid(
    luid: int, dedicated: list[tuple[str, float]], total_bytes: float
) -> Optional[float]:
    """Dedicated + Shared for the adapter with this LUID, clamped to ``total_bytes``; None if the Shared query fails (see _rocm_windows_unified_used_bytes)."""
    shared = _rocm_windows_perf_counter_vram_by_adapter("Shared Usage")
    if shared is None:
        return None
    used = 0.0
    for instance, value in (*dedicated, *shared):
        if _parse_adapter_luid(instance) == luid:
            if value < 0.0:
                return None
            used += value
    return max(0.0, min(used, total_bytes))


def _rocm_windows_per_device_vram(
    device_indices: list[int], adapters: Optional[list[tuple[str, float]]] = None
) -> tuple[list[Dict[str, Any]], Optional[float]]:
    """Per-GPU VRAM on Windows AMD/ROCm: total from torch properties, widened to the driver pool on a unified APU, used from the per-adapter Dedicated Usage counter. Returns ([{index, visible_ordinal, name, used_gb, total_gb}], aggregate_gb), or ([], None) when torch cannot enumerate devices. ``aggregate_gb`` is the visible set's total used VRAM, which survives a pairing no single device can claim (#7452). ``used_gb`` is None when the counter is unavailable, the pairing is not capacity-forced, or this device's total was widened to the unified pool -- that counter measures the dedicated segment, so it is no numerator for a pool-spanning total, which is also why the two totals are kept apart internally."""
    if platform.system() != "Windows":
        return [], None
    mod, _ = _torch_get_device_module()
    if mod is None:
        return [], None
    # Totals and names from torch properties; mem_get_info's free==total zeroes used.
    dev_meta: list[Dict[str, Any]] = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            props = mod.get_device_properties(ordinal)
            total_bytes = int(props.total_memory)
            # Counters are ranked against the carve-out, not the widened total shown to the user.
            dedicated_bytes = total_bytes
            # props.total_memory may be the carve-out on some APU stacks, hence a comparison.
            total_is_pool = False
            # Whether the driver confirmed the total spans the pool, not just failed to widen.
            pool_confirmed = False
            # Gate the probe on positive UMA: mem_get_info costs a ~612 MiB context forever.
            unified = False
            try:
                # Inside the try: a throwing probe must not drop the device from the list.
                unified = _rocm_props_are_positively_unified(props)
                if unified and hasattr(mod, "mem_get_info"):
                    pool_bytes = int(mod.mem_get_info(ordinal)[1])
                    # Equal is the expected case: PAL adds the WDDM shared heap to globalMemSize_ on APUs.
                    pool_confirmed = pool_bytes >= total_bytes
                    # Shrink guard: a shrunk total would report past 100% utilization.
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
                    # Second join key for _match_adapter_used_by_luid.
                    "gfx": str(getattr(props, "gcnArchName", "") or ""),
                    "total_is_pool": total_is_pool,
                    # Whether Shared Usage belongs in the numerator is a property of the part.
                    "positively_unified": unified,
                    "pool_confirmed": pool_confirmed,
                }
            )
        except Exception as e:
            logger.debug("torch property probe failed for ordinal %d: %s", ordinal, e)
    if not dev_meta:
        return [], None

    # Use the validated snapshot passed in; resampling costs ~1.3 s and may differ.
    if adapters is None:
        adapters = _rocm_windows_perf_counter_vram_by_adapter()
    aggregate_gb: Optional[float] = None
    whole_adapter: list[Optional[int]] = [None] * len(dev_meta)
    if adapters:
        # Identity keys first: HIP's LUID, then the DirectX record by name or arch.
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
            totals = [_adapter_counter_capacity(d) for d in dev_meta]
            assigned = _match_adapter_used_to_devices(adapter_useds, totals)
            aggregate_bytes = _rocm_windows_aggregate_used_bytes(adapter_useds, totals)
        if aggregate_bytes is not None:
            aggregate_gb = round(aggregate_bytes / (1024**3), 2)
    else:
        assigned = [None] * len(dev_meta)

    # A pool-scoped total needs a Dedicated+Shared numerator; Dedicated alone saturates.
    pool_scoped = [
        m["total_is_pool"] or (m["positively_unified"] and m["pool_confirmed"]) for m in dev_meta
    ]
    if any(pool_scoped):
        only = dev_meta[0] if len(dev_meta) == 1 else None
        # Skip the sum if the Dedicated query failed or a dropped counter is non-zero.
        wants_sum = only is not None and pool_scoped[0] and only["positively_unified"]
        placeholders_only = bool(adapters) and all(
            used == 0 for _, used in adapters if used < _ROCM_WIN_ADAPTER_MIN_BYTES
        )
        unified_used = (
            _rocm_windows_unified_used_bytes(adapters) if wants_sum and placeholders_only else None
        )
        if unified_used is not None:
            # This reading bypasses the matcher's clamp, so clamp it here.
            unified_used = max(0.0, min(unified_used, float(only["total_bytes"])))
        assigned = [
            (unified_used if scoped else used) for scoped, used in zip(pool_scoped, assigned)
        ]
        if only is None:
            # Beside another GPU only HIP's LUID says which counters are the iGPU's.
            for position, luid in enumerate(whole_adapter):
                meta = dev_meta[position]
                if pool_scoped[position] and meta["positively_unified"] and luid is not None:
                    assigned[position] = _rocm_windows_unified_used_bytes_for_luid(
                        luid, adapters, float(meta["total_bytes"])
                    )
        # The aggregate survives only when every pool-scoped member got a figure.
        if only is not None:
            aggregate_gb = round(unified_used / (1024**3), 2) if unified_used is not None else None
        else:
            aggregate_gb = (
                round(sum(assigned) / (1024**3), 2)
                if all(used is not None for used in assigned)
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
            elif not IS_ROCM:
                # numeric_ids is None under a UUID/MIG mask, which nvidia.py resolves itself.
                _reconcile_cuda_integrated_memory(result, numeric_ids)

            return _gpu_utilization_payload(
                device,
                devices,
                backend_cuda_visible_devices = result.get("backend_cuda_visible_devices"),
                parent_visible_gpu_ids = result.get("parent_visible_gpu_ids", []),
                index_kind = result.get("index_kind"),
            )

        if IS_ROCM and platform.system() == "Windows":
            _win_ids = _get_parent_visible_gpu_spec().get("numeric_ids")
            if not _win_ids:
                _win_ids = list(range(_torch_get_physical_gpu_count() or 0))
            _win_devices, _win_aggregate = _rocm_windows_per_device_vram(_win_ids)
            if _win_devices:
                # Across several GPUs the engine sum is not per-device.
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
    """Per-device reconciliation: when torch reports a larger memory total than amd-smi, overwrite the smi VRAM fields in place. Used by both the multi-device and primary-device reconcilers, so the two endpoints stay in sync on AMD iGPUs with unified memory."""
    torch_total_gb = torch_info["total_gb"]
    torch_used_gb = torch_info.get("used_gb")
    smi_total_gb = device_metrics.get("vram_total_gb") or 0.0
    # torch sees the unified GTT pool, amd-smi only the carve-out; adopt the larger total.
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
    """Fix amd-smi VRAM for ROCm unified-memory GPUs (e.g. Strix Halo): amd-smi reports only the dedicated slice while torch sees the full GTT pool, so where torch's total is larger, overwrite the per-device VRAM fields."""
    torch_devices = _torch_get_per_device_info(device_indices)
    if not torch_devices:
        return
    torch_by_index = {td["index"]: td for td in torch_devices}
    for dev in utilization.get("devices", []):
        td = torch_by_index.get(dev.get("index"))
        if td is None:
            continue
        _apply_unified_memory_correction(dev, td)


def _cuda_join_is_unsafe(device_indices: Optional[list[int]]) -> bool:
    """Whether a torch row cannot be attached to an nvidia-smi row on this host.

    CUDA enumerates FASTEST_FIRST by default while nvidia-smi reports PCI order, and
    equal-sized cards defeat every other check, so with a NUMERIC mask the join is only
    safe behind ``_cuda_order_matches_smi``. A UUID or MIG mask has no physical ids to
    disagree about: nvidia.py resolves the mask itself and torch enumerates that same
    mask in the same order, so ``visible_ordinal`` joins them whatever the order says.
    """
    if not device_indices:
        return False
    return not _cuda_order_matches_smi()


def _integrated_cuda_inventory(
    device_indices: Optional[list[int]],
) -> tuple[Dict[Any, Dict[str, Any]], str]:
    """torch's context-free inventory, keyed the way the SMI rows are indexed.

    Returns ``({key: row}, key_field)``, empty when the two sources cannot be joined.

    Two index spaces, and picking the wrong one attaches another card's capacity to a
    row. With a NUMERIC mask the SMI rows carry physical ids, so the join is on
    ``index`` -- but only behind ``_cuda_order_matches_smi``, because CUDA enumerates
    FASTEST_FIRST by default while nvidia-smi reports PCI order, and equal-sized cards
    defeat every other check. Same gate the SMI VRAM query already applies. With a UUID
    or MIG mask there are no physical ids: nvidia.py resolves the mask itself and
    returns rows ordered by it, and torch enumerates that same mask in that same order,
    so ``visible_ordinal`` joins them whatever CUDA_DEVICE_ORDER says.
    """
    if device_indices is None:
        ordinals = list(range(_torch_get_physical_gpu_count() or 0))
        if not ordinals:
            return {}, "visible_ordinal"
        return {
            td["visible_ordinal"]: td for td in _torch_get_device_inventory(ordinals)
        }, "visible_ordinal"
    if _cuda_join_is_unsafe(device_indices):
        return {}, "index"
    inventory = _torch_get_device_inventory(
        device_indices if device_indices else list(range(_torch_get_physical_gpu_count() or 0))
    )
    return {td["index"]: td for td in inventory}, "index"


# Exact bytes vs MiB rounding; 1% (min 64 MiB) separates that from a real carve-out.
_INTEGRATED_TOTAL_ADOPT_FRACTION = 0.01
_INTEGRATED_TOTAL_ADOPT_FLOOR_GB = 0.0625


def _integrated_total_is_understated(
    cli_total_gb: Optional[float], torch_total_gb: Optional[float]
) -> bool:
    """Whether an integrated part's CLI total is smaller than the pool torch can reach.

    ``None`` from the CLI is the DGX Spark shape: nvidia-smi answers ``[N/A]`` for
    memory.total, which NVIDIA documents, and anything is wider than nothing. A NUMBER
    from the CLI is the Windows RTX Spark N1X shape, where the figure is real, readable
    and scoped to the dedicated carve-out rather than to the CUDA budget.

    Only ever True for a LARGER torch total, which is what makes every caller below a
    widening and never a shrink.
    """
    if torch_total_gb is None or torch_total_gb <= 0:
        return False
    if cli_total_gb is None:
        return True
    return torch_total_gb - cli_total_gb > max(
        cli_total_gb * _INTEGRATED_TOTAL_ADOPT_FRACTION, _INTEGRATED_TOTAL_ADOPT_FLOOR_GB
    )


def _integrated_cuda_rows(
    device_indices: Optional[list[int]],
) -> tuple[Dict[Any, Dict[str, Any]], str]:
    """``_integrated_cuda_inventory`` reduced to the rows torch calls integrated.

    Deliberately NOT cached. The old check was "is a total missing", which a discrete
    host answers no to without touching torch; widening a total nvidia-smi DID answer
    cannot be decided without asking torch, and this runs on the 3-5 s /api/system poll,
    so the cost was measured rather than assumed: 7 microseconds, because
    get_device_properties is answered from the driver's device list and creates no
    primary context (0 MiB on an RTX Spark N1X, against 116 MiB for mem_get_info). A
    memo would buy nothing at that price and would have to be invalidated correctly.

    Empty on a discrete host, and on any host the two sources cannot be joined, so every
    caller keeps whatever the CLI reported.
    """
    inventory, key_field = _integrated_cuda_inventory(device_indices)
    return {k: td for k, td in inventory.items() if td.get("_cuda_integrated")}, key_field


def _cgroup_available_memory_gb() -> Optional[float]:
    """What this process can still charge to an enforcing cgroup, or None.

    Reuses the llama.cpp reader rather than a second copy: it walks the process's
    cgroup AND its ancestors, since an ancestor slice can be the binding limit and
    carries sibling usage a leaf never sees, and it handles v2 and legacy v1.
    Imported lazily and only from the widening branch, so a discrete host, which
    returns before ever reaching here, pays nothing for it.
    """
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        mib = LlamaCppBackend._cgroup_available_memory_mib()
        return None if mib is None else mib / 1024.0
    except Exception as e:  # noqa: BLE001 - no readable limit means keep the host reading
        logger.debug("cgroup budget probe failed while sizing an integrated GPU: %s", e)
        return None


def _host_memory_used_gb() -> Optional[float]:
    """Host memory in use, or None. On one shared pool this is not an approximation of
    the GPU's used half, it is the same measurement."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round((int(vm.total) - int(vm.available)) / (1024**3), 2)
    except Exception as e:  # noqa: BLE001 - a total alone still beats Unknown / 0.00
        logger.debug("host memory probe failed while sizing an integrated GPU: %s", e)
        return None


def _reconcile_cuda_integrated_memory(
    utilization: Dict[str, Any], device_indices: Optional[list[int]]
) -> None:
    """Publish the pool an integrated CUDA SoC can reach, not its dedicated carve-out.

    Two shapes of one fault. On a DGX Spark nvidia-smi answers ``[N/A]`` for
    memory.total and the monitor printed "Unknown / 0.00 GiB" beside a 121 GiB part
    (#10691). On a Windows RTX Spark N1X it answers a number, 8128 MiB, which is the
    carve-out and not the 46477 MiB budget the same device reports through
    ``props.total_memory``: an under-report of about 5.7x, which judged a 270M model
    not to fit. Filling only the blanks repaired the first and left the second standing,
    because a wrong number is not a missing one.

    NOT _torch_get_per_device_info, which the ROCm twin above can afford and this
    cannot: this is the /api/system poll, and mem_get_info pins a primary context for
    the life of the process (test_system_poll_no_cuda_context.py). Both figures here
    are context-free.

    Widens only, in both directions it could be read: a total the CLI reported LARGER
    than torch's is left alone, and the free bytes published here are floored at the
    free bytes the row already promised, so no device loses capacity it was trusted
    with before this ran.
    """
    devices = utilization.get("devices", [])
    if not devices:
        return
    try:
        integrated, key_field = _integrated_cuda_rows(device_indices)
    except Exception as e:  # noqa: BLE001 - reached on a host that HAS nvidia-smi
        logger.debug("torch inventory unavailable while sizing an integrated GPU: %s", e)
        return
    if not integrated:
        return

    host_used_gb = _host_memory_used_gb()

    for dev in devices:
        td = integrated.get(dev.get(key_field))
        if td is None:
            # A row torch does not enumerate, e.g. an NVIDIA NPU with no memory telemetry.
            continue
        total_gb = td["total_gb"]
        cli_total_gb = dev.get("vram_total_gb")
        cli_used_gb = dev.get("vram_used_gb")
        if not _integrated_total_is_understated(cli_total_gb, total_gb):
            continue
        # The row's prior free figure, so widening cannot cost free bytes.
        cli_free_gb = (
            max(cli_total_gb - cli_used_gb, 0.0)
            if cli_total_gb is not None and cli_used_gb is not None
            else None
        )
        dev["vram_total_gb"] = total_gb

        # Both numerators are lower bounds for a pool total, so take the larger.
        numerators = [n for n in (host_used_gb, cli_used_gb) if n is not None]
        if not numerators:
            continue
        pool_used_gb = min(max(numerators), total_gb)
        if host_used_gb is None and cli_free_gb is not None:
            # Without pool-scoped usage, keep the CLI's free budget while widening the total.
            pool_used_gb = max(pool_used_gb, total_gb - cli_free_gb)
        if cli_free_gb is not None:
            pool_used_gb = min(pool_used_gb, max(total_gb - cli_free_gb, 0.0))
        # Last, so it beats the floor: containers charge allocations to memory.max.
        cgroup_free_gb = _cgroup_available_memory_gb()
        if cgroup_free_gb is not None:
            pool_used_gb = min(max(pool_used_gb, total_gb - cgroup_free_gb), total_gb)
        dev["vram_used_gb"] = round(pool_used_gb, 2)
        dev["vram_utilization_pct"] = (
            round((pool_used_gb / total_gb) * 100, 1) if total_gb > 0 else None
        )


def _reconcile_primary_rocm_unified_memory(
    utilization: Dict[str, Any], parent_visible_spec: Dict[str, Any]
) -> None:
    """Same fix as _reconcile_rocm_unified_memory for the flat primary-GPU dict."""
    numeric_ids = parent_visible_spec.get("numeric_ids")
    if numeric_ids is None:
        primary_idx = [0]
    elif len(numeric_ids) == 0:
        # Empty mask: no GPU visible, so do not query torch device 0.
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
    """Decide the system-wide overlay without applying it: {physical index: (used_gb, total_gb)} for every device sysfs can speak for, omitting the rest. Split out from _overlay_system_wide_vram so a caller can ask whether sysfs covers the whole visible set BEFORE paying torch for occupancy it would discard. Never mutates ``devices``."""
    raw = _rocm_linux_sysfs_vram_by_index(devices)
    resolved: Dict[int, tuple[float, float]] = {}
    for dev in devices:
        index = dev.get("index")
        entry = raw.get(index)
        if entry is None:
            continue
        used, total = entry
        dev_total = dev.get("vram_total_gb") or 0.0
        # Overlay only when torch and sysfs totals agree within 10%; else memory scopes differ.
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
        if sysfs_total <= 0:
            # sysfs unreadable (e.g. WSL): the split is unknown.
            continue
        excess = torch_total - sysfs_total
        # Torch not exceeding sysfs means host-backed is a measured zero; omitting reads as all-host.
        # sysfs far above torch is a partition scope change, so leave it unknown.
        if -excess > 0.1 * torch_total:
            continue
        shared[index] = round(excess, 2) if excess > 0.1 * torch_total else 0.0
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
    """Replace process-local torch VRAM with system-wide Linux ROCm figures: the torch fallback is process-local, so a model served by the separate llama-server process reads as ~0 used with the GPU full (#7072), while DRM sysfs figures are kernel-updated across processes. Matched by PHYSICAL index, never list position, and only with NO visibility mask active and the device count equal to the host GPU count, since under a mask the index is not a verifiable host ordinal. Best-effort and in place: a device with no matching card, or an APU whose sysfs total is below torch's GTT-backed total, keeps torch's. Windows is intentionally not overlaid: its per-adapter counters cannot be mapped to ROCm ordinals and miss WDDM shared memory."""
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
                _reconcile_rocm_unified_memory(result, numeric_ids)
            elif not IS_ROCM:
                # /api/system reads this path, so the monitor's figures are repaired here.
                _reconcile_cuda_integrated_memory(result, numeric_ids)
            return result

        # Windows ROCm: torch reports used==0, so read per-adapter Dedicated Usage.
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
                    # Host total, known even when no single device's usage is attributable.
                    "vram_used_gb_aggregate": win_aggregate,
                }

    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_ids = get_parent_visible_gpu_ids()
        if parent_ids:
            torch_indices = parent_ids
            index_kind = "physical"
        else:
            visible_count = _torch_get_physical_gpu_count() or 0
            torch_indices = list(range(visible_count))
            index_kind = "relative"

        # Linux ROCm: sysfs first, avoiding a permanent torch context, only if it covers every device.
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
                # used=None means telemetry unavailable; propagate None.
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
                # System-wide sysfs shows memory held by llama-server; physical indices only.
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
                    # Unified memory: free is not total - used, so publish it.
                    "vram_free_gb": round(mem.get("free_gb", 0), 2),
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


_physical_gpu_count: Optional[int] = None
# Only an SMI count can answer "is this host single-GPU".
_physical_gpu_count_from_smi: bool = False
_visible_gpu_count: Optional[int] = None


def _get_parent_visible_gpu_spec() -> Dict[str, Any]:
    # Intel XPU visibility is ZE_AFFINITY_MASK, not CUDA_VISIBLE_DEVICES.
    if get_device() == DeviceType.XPU:
        xpu_mask_raw = os.environ.get("ZE_AFFINITY_MASK")
        composite = _xpu_hierarchy_is_composite()

        if xpu_mask_raw is None:
            if composite:
                return {
                    "raw": None,
                    "numeric_ids": list(range(get_physical_gpu_count())),
                    "supports_explicit_gpu_ids": True,
                }
            # FLAT ordinals are tile handles, not physical IDs; pinning needs COMPOSITE.
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

        # Subdevice syntax ("N.M") cannot be addressed by root-ID selection.
        has_subdevice = any("." in token.strip() for token in xpu_mask.split(",") if token.strip())
        if has_subdevice:
            return {
                "raw": xpu_mask,
                "numeric_ids": None,
                "supports_explicit_gpu_ids": False,
            }

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

        roots_with_dupes = _parse_ze_mask_roots(xpu_mask)
        if not roots_with_dupes:
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

    # ROCm masks layer on top of CUDA_VISIBLE_DEVICES; "" means no visible GPUs.
    cuda_visible = None
    # A stale HIP_VISIBLE_DEVICES on NVIDIA must not override CUDA_VISIBLE_DEVICES.
    _is_rocm_spec = IS_ROCM or (
        "CUDA_VISIBLE_DEVICES" not in os.environ
        and ("HIP_VISIBLE_DEVICES" in os.environ or "ROCR_VISIBLE_DEVICES" in os.environ)
    )
    if _is_rocm_spec:
        hip_vis = os.environ.get("HIP_VISIBLE_DEVICES")
        # ROCR_VISIBLE_DEVICES is Linux-only: Windows HIP has no ROCr layer.
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
        # nvidia-smi indices are PCI order, so they only name the same cards a numeric mask written back to a child would under PCI_BUS_ID (#8873).
        if not _is_rocm_spec and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID":
            from . import nvidia
            resolved_ids = nvidia.resolve_uuid_mask(cuda_visible)
            if resolved_ids is not None:
                return {
                    "raw": cuda_visible,
                    "numeric_ids": resolved_ids,
                    "supports_explicit_gpu_ids": True,
                }
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
        # Vulkan ordinals are a separate index space; only reject malformed ones.
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

    negative_ids = [gpu_id for gpu_id in requested_ids if gpu_id < 0]
    if negative_ids:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: GPU IDs must be non-negative. "
            f"Rejected IDs: {negative_ids}. Parent-visible GPUs: {parent_visible_ids}"
        )

    # Enforce the upper bound only for an SMI count; torch counts only visible devices.
    if physical_gpu_count > 0 and parent_visible_ids:
        max_parent_id = max(parent_visible_ids)
        if physical_gpu_count > max_parent_id:
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


_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")
_WEIGHT_COUNTER = re.compile(r"(?:-\d+-of-\d+|\.\d+)$")
_TRAINER_BOOKKEEPING = re.compile(
    r"^(?:optimizer|scheduler|scaler|rng_state|training_args|trainer_state)"
    r"(?:[-_]\d+(?:-of-\d+)?)?$"
)
# A variant is these weights again and is never opened by a no-variant load.
_WEIGHT_VARIANT = re.compile(r"\.([A-Za-z][\w-]*)$")
# The order from_pretrained tries, direct file ahead of the index per spelling.
_TRANSFORMERS_ARCHIVES = (
    ("model", ".safetensors"),
    ("pytorch_model", ".bin"),
    ("consolidated", ".safetensors"),
    ("consolidated", ".pth"),
)
# The folder's declared class decides which table it loads by.
_DIFFUSERS_ARCHIVES = (
    ("diffusion_pytorch_model", ".safetensors"),
    ("diffusion_pytorch_model", ".bin"),
)
_MODEL_ARCHIVES = _TRANSFORMERS_ARCHIVES + _DIFFUSERS_ARCHIVES
# An adapter loads on top of the base model, so it never stands in for one.
_ADAPTER_ARCHIVES = (
    ("adapter_model", ".safetensors"),
    ("adapter_model", ".bin"),
)
_WEIGHT_ARCHIVES = _MODEL_ARCHIVES + _ADAPTER_ARCHIVES
# The only indexes a load resolves; any other *.index.json is never opened.
_WEIGHT_INDEX_NAMES = frozenset(f"{base}{ext}.index.json" for base, ext in _WEIGHT_ARCHIVES)


def _archive_stem(stem: str) -> tuple:
    """``(base, variant)`` of a weight stem, the shard counter and the precision variant
    stripped in either order: model-00001-of-00002.fp16 and model.fp16-00001-of-00002 are
    both ``("model", "fp16")``; consolidated.00 is ``("consolidated", None)``."""
    stem = _WEIGHT_COUNTER.sub("", stem)
    variant = _WEIGHT_VARIANT.search(stem)
    if variant is None:
        return stem, None
    return _WEIGHT_COUNTER.sub("", stem[: variant.start()]), variant.group(1)


def _declared_library(directories: list) -> Optional[str]:
    """Which loader a folder's own config.json says opens it: a diffusers component carries
    ``_class_name``/``_diffusers_version``, a transformers model ``architectures``/``model_type``.
    ``None`` when the folder declares nothing, and its spellings are then two payloads."""
    for directory in directories:
        try:
            config = json.loads((directory / "config.json").read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        if "_class_name" in config or "_diffusers_version" in config:
            return "diffusers"
        if "architectures" in config or "model_type" in config:
            return "transformers"
    return None


def _index_targets(index: Path, directory: Path) -> set:
    """What an index names, joined onto its folder the way from_pretrained joins it."""
    try:
        weight_map = json.loads(index.read_text(encoding = "utf-8")).get("weight_map") or {}
        return {Path(os.path.normpath(directory / name)) for name in weight_map.values()}
    except (OSError, ValueError, AttributeError, TypeError):
        return set()


def _indexed_archive(directories: list, base: str, ext: str, tree: dict) -> tuple:
    """The shards from_pretrained opens here, and every shard any of these indexes names."""
    files, settled, read = tree["files"], tree["settled"], tree["read"]
    chosen: dict = {}
    every: dict = {}
    for directory in directories:
        index = directory / f"{base}{ext}.index.json"
        if index not in read:
            read[index] = _index_targets(index, directory)
        shards = {
            path: files[path]
            for path in read[index]
            if path in files and (path.parent, path.stem) not in settled
        }
        every.update(shards)
        if shards and not chosen:
            chosen = shards
    return chosen, every


def _archive_candidates(directories: list, pool: dict, tree: dict, table: tuple) -> tuple:
    """Every spelling of the weights present here, in the order from_pretrained tries them,
    and every file those spellings account for, a variant-only spelling included."""
    candidates = []
    held: dict = {}
    for base, ext in table:
        direct = {path: size for path, size in pool.items() if path.name == f"{base}{ext}"}
        indexed, all_indexed = _indexed_archive(directories, base, ext, tree)
        if base == "diffusion_pytorch_model" and ext == ".bin":
            # A default diffusers load never reaches a .bin.index.json.
            indexed = {}
        counted: dict = {}
        variants: dict = {}
        for path, size in pool.items():
            if path.suffix != ext:
                continue
            stem, variant = _archive_stem(path.stem)
            if stem == base:
                (variants if variant else counted)[path] = size
        # An index naming the direct file is not stale, so it decides.
        names_the_direct_file = bool(direct) and set(direct) <= set(indexed)
        # diffusers: its index settles a component whenever one exists.
        index_first = names_the_direct_file or (base, ext) == (
            "diffusion_pytorch_model",
            ".safetensors",
        )
        opens = indexed if (indexed and index_first) else (direct or indexed)
        held.update({**direct, **all_indexed, **counted, **variants})
        if opens or counted:
            candidates.append((opens or counted, bool(opens)))
    return candidates, held


def _selected_archive(homes: list, sizes: dict, tree: dict, vendor: set, table: tuple) -> tuple:
    """The one archive from ``table`` these folders open, and every spelling of it."""
    directories = [folder for folder, _ in homes]
    candidates, held = _archive_candidates(directories, sizes, tree, table)
    # A vendor copy never outranks native weights, and drops out as a whole archive.
    native_pool = {path: size for path, size in sizes.items() if path not in vendor}
    native, _ = _archive_candidates(
        [f for f, is_vendor in homes if not is_vendor], native_pool, tree, table
    )

    archive: dict = {}
    for choices in (native, candidates):
        if choices:
            opens = [entry for entry in choices if entry[1]]
            archive = (opens or choices)[0][0]
            break

    return archive, (set(held) if archive else set())


def _directory_weight_bytes(homes: list, sizes: dict, tree: dict, vendor: set) -> tuple:
    """What one directory costs, and every file its spellings account for.

    ``homes`` are the ``(folder, is_vendor)`` pairs answering to it, decided together because
    splitting them lets a single archive lose in halves. ``tree`` carries every file, since an
    index may name a shard below itself; the second return is what it accounted for.
    """
    # Resolved apart so an adapter never stands in for its base model.
    transformers_model, transformers_held = _selected_archive(
        homes, sizes, tree, vendor, _TRANSFORMERS_ARCHIVES
    )
    diffusers_model, diffusers_held = _selected_archive(
        homes, sizes, tree, vendor, _DIFFUSERS_ARCHIVES
    )
    library = _declared_library([folder for folder, _ in homes])
    if library == "diffusers" and diffusers_model:
        model = diffusers_model
    elif library == "transformers" and transformers_model:
        model = transformers_model
    else:
        model = {**transformers_model, **diffusers_model}
    model_held = transformers_held | diffusers_held
    adapter, adapter_held = _selected_archive(homes, sizes, tree, vendor, _ADAPTER_ARCHIVES)
    archive = {**model, **adapter}

    # model.pt beside model.safetensors is the same archive, not a second payload.
    alternatives = model_held | adapter_held
    archive_stems = {(path.parent, path.stem) for path in archive}
    alternatives |= {path for path in sizes if (path.parent, path.stem) in archive_stems}
    rest = {path: size for path, size in sizes.items() if path not in alternatives}
    here = {folder for folder, _ in homes}
    above_an_archive = any(
        folder != other and other.is_relative_to(folder)
        for folder in here
        for other in tree.get("archive_dirs", ())
    )
    if archive or above_an_archive:
        rest = {p: s for p, s in rest.items() if not _TRAINER_BOOKKEEPING.match(p.stem)}
    components: dict = {}
    ordered = sorted(rest.items(), key = lambda i: (i[0].suffix != ".safetensors", i[0].name))
    for path, size in ordered:
        components.setdefault(path.stem, size)
    accounted = {path for path in alternatives if path.parent in here} | set(archive)
    # Shards an index no longer names are obsolete, not components.
    for shard in [path for path in archive if path.parent not in here]:
        base, _ = _archive_stem(shard.stem)
        accounted |= {
            path
            for path in tree["files"]
            if path.parent == shard.parent
            and path.suffix == shard.suffix
            and _archive_stem(path.stem)[0] == base
        }
    return sum(archive.values()) + sum(components.values()), accounted


def _get_local_weight_size_bytes(model_name: str) -> Optional[int]:
    # Lexical normpath so `..` cannot make a named shard look outside the model.
    model_path = Path(os.path.normpath(model_name))
    if not model_path.exists():
        return None

    # Skip intermediate checkpoints: export loads only the root model.
    skip_prefixes = ("checkpoint-", "global_step")
    found = []
    indexed_dirs = []
    index_files = []
    weight_sizes: dict = {}
    homes_by_directory: dict = {}
    vendor: set = set()
    placed: dict = {}
    for file in model_path.rglob("*"):
        if not file.is_file():
            continue
        parent = file.parent
        if parent not in placed:
            rel_parent = parent.relative_to(model_path)
            # A top-level original/ answers to the directory above it.
            is_vendor = rel_parent.parts[:1] == ("original",)
            placed[parent] = (
                any(part.startswith(skip_prefixes) for part in rel_parent.parts),
                is_vendor,
                Path(*rel_parent.parts[1:]) if is_vendor else rel_parent,
                rel_parent,
            )
        skipped, is_vendor, home, rel_parent = placed[parent]
        if skipped or file.name.startswith(skip_prefixes):
            continue
        if is_vendor:
            vendor.add(file)
        homes_by_directory.setdefault(home, {})[parent] = is_vendor
        if file.suffix in _WEIGHT_EXTS:
            try:
                weight_sizes[file] = file.stat().st_size
            except OSError:
                continue
            found.append(rel_parent / file.name)
        elif file.name in _WEIGHT_INDEX_NAMES:
            index_files.append(file)
            indexed_dirs.append(home)

    sizes_by_directory: dict = {}
    names_by_directory: dict = {}
    for rel in sorted(found, key = lambda r: r.parts[:1] == ("original",)):
        directory = Path(*rel.parent.parts[1:]) if rel.parts[:1] == ("original",) else rel.parent
        names = names_by_directory.setdefault(directory, set())
        if rel.name in names:
            continue
        names.add(rel.name)
        sizes_by_directory.setdefault(directory, {})[model_path / rel] = weight_sizes[
            model_path / rel
        ]

    # An index may name shards without a recognised suffix.
    for directory in indexed_dirs:
        sizes_by_directory.setdefault(directory, {})

    # A shallower index can name deeper shards, so it decides first, by stem.
    files = dict(weight_sizes)
    read: dict = {}
    for index in index_files:
        read[index] = _index_targets(index, index.parent)
        for target in read[index]:
            if target in files:
                continue
            try:
                relative = target.relative_to(model_path)
                if not target.is_file():
                    continue
                files[target] = target.stat().st_size
            except (OSError, ValueError):
                continue
            if any(part.startswith(skip_prefixes) for part in relative.parts):
                del files[target]
            elif relative.parts[:1] == ("original",):
                vendor.add(target)

    settled: set = set()
    # Trainer state above archive folders is bookkeeping, not weights.
    archive_bases = {base for base, _ in _WEIGHT_ARCHIVES}
    archive_dirs = {index.parent for index in index_files} | {
        path.parent for path in weight_sizes if _archive_stem(path.stem)[0] in archive_bases
    }
    tree = {"files": files, "settled": settled, "read": read, "archive_dirs": archive_dirs}
    total = 0
    for directory in sorted(sizes_by_directory, key = lambda d: (len(d.parts), d.as_posix())):
        unclaimed = {
            path: size
            for path, size in sizes_by_directory[directory].items()
            if (path.parent, path.stem) not in settled
        }
        charged, accounted = _directory_weight_bytes(
            sorted(homes_by_directory.get(directory, {model_path / directory: False}).items()),
            unclaimed,
            tree,
            vendor,
        )
        settled |= {(path.parent, path.stem) for path in accounted}
        total += charged
    return total if total > 0 else None


def _get_hf_safetensors_total_params(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[int]:
    try:
        from utils.utils import hf_env_offline

        if hf_env_offline():
            return None

        from huggingface_hub import model_info as hf_model_info
        from hub.utils.hf_tokens import call_hub_with_anonymous_retry

        # A refused token must not drop this to the text-tower-only estimate.
        info = call_hub_with_anonymous_retry(hf_model_info, hf_token, model_name)
        safetensors = getattr(info, "safetensors", None)
        if isinstance(safetensors, dict):
            total = safetensors.get("total")
            if total:
                return int(total)
    except Exception as e:
        logger.warning("Could not get safetensors metadata for '%s': %s", model_name, e)
    return None


def _load_config_for_gpu_estimate(model_name: str, hf_token: Optional[str] = None):
    # Read raw config.json (never run auto_map code): only declarative fields are needed.
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
        # A 5.x-only config fails on the default transformers as expected; warn only there.
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
    # torch.distributed is incomplete on Windows ROCm, so stub it before importing.
    if sys.platform == "win32" and IS_ROCM:

        class _Dummy:
            pass

        for _c10d_name in (
            "torch._C._distributed_c10d",
            "torch._C._distributed_autograd",
            "torch._C._distributed_rpc",
        ):
            if _c10d_name not in sys.modules:
                _stub = types.ModuleType(_c10d_name)
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
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    # Deep copy: attention resolution mutates nested sub-configs.
    config_copy = copy.deepcopy(config)
    model_type = getattr(config_copy, "model_type", None)
    config_class = (
        CONFIG_MAPPING[model_type] if model_type in CONFIG_MAPPING else config_copy.__class__
    )

    model_class = None
    for auto_model in (AutoModelForCausalLM, AutoModel):
        mapping = getattr(auto_model, "_model_mapping", None)
        if mapping is None:
            continue
        try:
            if config_class in mapping:
                model_class = mapping[config_class]
                break
        except Exception:
            continue

    impl = resolve_attention_implementation(model_class, config_copy)
    # Inlined because callers stub the unsloth import.
    if isinstance(impl, dict):
        named = [value for key, value in impl.items() if key != "" and value is not None]
        impl = named[0] if named else impl.get("", "eager")
    return impl


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

    # Config bytes cover only the text tower, so take the larger of the two.
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
            # Debug level: expected on every Windows ROCm estimate.
            logger.debug(
                "Could not resolve attention implementation for '%s': %s",
                estimate_model,
                e,
            )
            # Charge the quadratic eager path while flash attention is unproven.
            vram_config.attention_implementation = "eager"
    arch = extract_arch_config(config) if config is not None else None

    if arch is not None:
        breakdown = estimate_training_vram(arch, vram_config)
        # extract_arch_config only sees text_config; add the vision/audio tower bytes.
        arch_fp16_bytes = compute_total_params(arch) * 2
        extra_bytes = max(0, int(model_size_bytes) - arch_fp16_bytes)
        if extra_bytes > 0:
            breakdown.model_weights += extra_bytes
            if training_method == "full":
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


# Excludes generic code objects and family labels, which no device reports.
_CONCRETE_GFX_ARCH = re.compile(r"^gfx[0-9][0-9a-f]{2,4}$")


def _props_gfx_arch(props) -> str:
    # gcnArchName alone leaves the map empty on AMD SDK / Radeon wheels.
    for attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        arch = (getattr(props, attr, "") or "").split(":")[0].strip().lower()
        if arch:
            return arch
    return ""


def _torch_ordinal_physical_ids(device_count: int) -> Optional[list[int]]:
    """Map torch ordinals to physical GPU IDs, or return None if uncertain."""
    visible_spec = _get_parent_visible_gpu_spec()
    physical_ids = visible_spec["numeric_ids"]
    if physical_ids is None:
        return None
    if device_count > len(physical_ids):
        # Without a mask, torch ordinals include GPUs AMD SMI may miss.
        if visible_spec["raw"] is None:
            return list(range(device_count))
        logger.debug(
            "Skipping torch arch gate: %s torch devices but mask %r names %s ids",
            device_count,
            visible_spec["raw"],
            len(physical_ids),
        )
        return None
    return list(physical_ids)


def rocm_gpu_ids_without_torch_kernels() -> set[int]:
    """PHYSICAL ids of visible ROCm GPUs the installed torch wheel has no kernels for. Compares what the device PRESENTS, not its silicon, so HSA_OVERRIDE_GFX_VERSION keeps working (#7624). Every uncertainty fails OPEN, the opposite of the bf16 gate: one unreadable device is skipped rather than voiding the probe, which would restore the known-uncovered card and re-break #8792."""
    try:
        import torch

        if not (
            getattr(torch.version, "hip", None) is not None
            or "rocm" in getattr(torch, "__version__", "").lower()
        ):
            return set()
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return set()

        tokens = [
            token
            for token in (
                str(arch).split(":")[0].strip().lower()
                for arch in (torch.cuda.get_arch_list() or ())
            )
            if token
        ]
        # A generic code object covers devices no exact token names.
        unknown = sorted(t for t in tokens if not _CONCRETE_GFX_ARCH.match(t))
        if unknown:
            logger.debug("torch arch list carries non-concrete tokens %s; not gating", unknown)
            return set()
        if not tokens:
            return set()
        supported = set(tokens)

        # Under ROCR="1,0" + CUDA="1" the spec reports [1,0] but ordinal 0 is physical 0.
        if _rocm_device_ordinal_active() or _rocm_visibility_masks_are_stacked():
            logger.debug("Skipping torch arch gate: torch ordinals are renumbered by a mask")
            return set()

        device_count = torch.cuda.device_count()
        physical_ids = _torch_ordinal_physical_ids(device_count)
        if physical_ids is None:
            return set()

        unsupported: set[int] = set()
        unsupported_ordinals = 0
        readable = 0
        for ordinal in range(device_count):
            try:
                props = torch.cuda.get_device_properties(ordinal)
            except Exception:
                continue
            arch = _props_gfx_arch(props)
            if not arch:
                logger.debug("Torch arch gate: device %s reports no arch; not gating it", ordinal)
                continue
            readable += 1
            if arch not in supported:
                unsupported_ordinals += 1
                unsupported.add(physical_ids[ordinal])

        # Never drop every device (silently forces CPU); count ordinals, as a mask may repeat an id.
        if readable and readable == device_count and unsupported_ordinals >= readable:
            logger.warning(
                "The installed PyTorch build has no kernels for any GPU on this host "
                "(built for %s); leaving device selection alone.",
                sorted(supported),
            )
            return set()
        return unsupported
    except Exception as e:
        logger.debug("torch arch coverage probe failed: %s", e)
        return set()


def _torch_kernel_arch_tokens() -> list[str]:
    try:
        import torch
        return sorted(
            {
                str(arch).split(":")[0].strip().lower()
                for arch in (torch.cuda.get_arch_list() or ())
                if str(arch).strip()
            }
        )
    except Exception:
        return []


def _describe_rocm_gpus(gpu_ids) -> list[str]:
    """Best-effort labels keyed by PHYSICAL id, for an error message only; never a gate."""
    wanted = {int(gpu_id) for gpu_id in gpu_ids}
    labels: Dict[int, str] = {}
    try:
        import torch

        count = torch.cuda.device_count()
        physical_ids = _get_parent_visible_gpu_spec()["numeric_ids"]
        if physical_ids is None or count > len(physical_ids):
            physical_ids = list(range(count))
        for ordinal, physical in enumerate(physical_ids[:count]):
            if physical not in wanted:
                continue
            props = torch.cuda.get_device_properties(ordinal)
            arch = _props_gfx_arch(props)
            detail = ", ".join(
                part for part in (str(getattr(props, "name", "") or ""), arch) if part
            )
            labels[physical] = f"GPU {physical} ({detail})" if detail else f"GPU {physical}"
    except Exception as e:
        logger.debug("Could not describe GPUs %s: %s", sorted(wanted), e)
    return [labels.get(gpu_id, f"GPU {gpu_id}") for gpu_id in sorted(wanted)]


def reject_gpu_ids_without_torch_kernels(gpu_ids) -> None:
    """Explicit picks bypass the #8792 auto-select skip; without this the worker dies with hipErrorInvalidImage."""
    uncovered = sorted(
        set(int(gpu_id) for gpu_id in gpu_ids) & rocm_gpu_ids_without_torch_kernels()
    )
    if not uncovered:
        return
    built_for = ", ".join(_torch_kernel_arch_tokens()) or "other GPU architectures"
    raise ValueError(
        f"{', '.join(_describe_rocm_gpus(uncovered))} cannot run the PyTorch build this Unsloth Studio installed, "
        f"which has kernels for {built_for} only. Pick another GPU, or reinstall Unsloth "
        f"Studio for that card."
    )


def gpu_ids_with_torch_kernels() -> Optional[list[int]]:
    """Exclude GPUs with known missing torch kernels; None preserves visibility."""
    uncovered = rocm_gpu_ids_without_torch_kernels()
    if not uncovered:
        return None
    try:
        import torch
        visible = _torch_ordinal_physical_ids(torch.cuda.device_count())
    except Exception as e:
        logger.debug("Could not map torch devices to physical ids: %s", e)
        return None
    covered = [gpu_id for gpu_id in visible or () if gpu_id not in uncovered]
    if not covered:
        return None
    logger.warning(
        "Hiding GPU(s) %s from this worker: the installed PyTorch build has no kernels "
        "for their architecture.",
        sorted(uncovered),
    )
    return covered


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
    required_override_gb: Optional[float] = None,
) -> tuple[Optional[list[int]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {"selection_mode": "auto"}

    # Auto-selection needs per-device free-VRAM telemetry (CUDA, XPU only).
    if get_device() not in (DeviceType.CUDA, DeviceType.XPU):
        metadata["selection_mode"] = "non_accelerator"
        return None, metadata

    if required_override_gb is None:
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
    else:
        required_gb = float(required_override_gb)
        estimate_metadata = {
            "mode": "inference" if training_type is None else "training",
            "model_size_source": "override",
            "required_gb": round(required_gb, 3),
        }
    metadata.update(estimate_metadata)
    parent_visible_spec = _get_parent_visible_gpu_spec()
    metadata["parent_cuda_visible_devices"] = parent_visible_spec["raw"]

    if not parent_visible_spec["supports_explicit_gpu_ids"]:
        metadata["selection_mode"] = "inherit_parent_visible"
        metadata["selected_gpu_ids"] = None
        return None, metadata

    # Before the free-memory rank and every fallback, which return the whole visible set.
    _uncovered = rocm_gpu_ids_without_torch_kernels()
    if _uncovered:
        logger.warning(
            "Excluding GPU(s) %s from automatic selection: the installed PyTorch "
            "build has no kernels for their architecture.",
            sorted(_uncovered),
        )

    def _covered(ids):
        return [gpu_id for gpu_id in ids if gpu_id not in _uncovered]

    if required_gb is None:
        parent_ids = _covered(get_parent_visible_gpu_ids())
        metadata["selection_mode"] = "fallback_all"
        metadata["selected_gpu_ids"] = parent_ids
        return parent_ids, metadata

    utilization = get_visible_gpu_utilization()
    devices = [d for d in utilization.get("devices", []) if d.get("index") not in _uncovered]
    parent_ids = _covered(get_parent_visible_gpu_ids())

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
    # Empirical sharding overhead on 2-8 GPUs (NCCL buffers, bubbles, fragmentation).
    multi_gpu_overhead = 0.85

    # Activations do not shard, so each GPU needs its shard plus full activations.
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
    required_override_gb: Optional[float] = None,
) -> tuple[Optional[list[int]], Dict[str, Any]]:
    """Resolve which physical GPUs to use for a model load. Explicit (gpu_ids=[5, 6, 7]) uses exactly those, sharded via device_map="balanced" even if fewer would do, validated against the parent-visible set; auto (None or []) lets auto_select_gpu_ids estimate VRAM and pick the minimum, preferring the most free memory. The result is passed to get_device_map() and to apply_gpu_ids() in the worker, which narrows CUDA_VISIBLE_DEVICES before torch init."""
    if gpu_ids and get_device() not in (DeviceType.CUDA, DeviceType.XPU):
        raise ValueError(
            f"gpu_ids {list(gpu_ids)} is only supported on CUDA and Intel XPU "
            f"devices, but the current backend is '{get_device().value}'."
        )

    if gpu_ids:
        resolved = resolve_requested_gpu_ids(gpu_ids)
        reject_gpu_ids_without_torch_kernels(resolved)
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
        required_override_gb = required_override_gb,
    )
    return selected_gpu_ids, metadata


def get_physical_gpu_count() -> int:
    """Number of physical GPUs on the machine, from `nvidia-smi -L` (unaffected by CUDA_VISIBLE_DEVICES) with a torch fallback for AMD ROCm and Intel XPU. Cached after the first call."""
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
    """The raw visibility env string that applies to this backend: ZE_AFFINITY_MASK on XPU, and HIP_/ROCR_VISIBLE_DEVICES ahead of CUDA_VISIBLE_DEVICES on ROCm. Mirrors _get_parent_visible_gpu_spec, so backend_cuda_visible_devices reports the value actually narrowing the visible set."""
    if get_device() == DeviceType.XPU:
        return os.environ.get("ZE_AFFINITY_MASK")
    if IS_ROCM:
        return _get_parent_visible_gpu_spec().get("raw")
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def get_vulkan_inference_gpu_info() -> Optional[Dict[str, Any]]:
    """llama.cpp Vulkan devices, or None when Vulkan is not installed."""
    # Vulkan is a llama.cpp inference backend, not a training device.
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
                    # ggml Vulkan ordinals are what `--device Vulkan<i>` pins, so they are selectable.
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


def _installed_llama_backend() -> Optional[str]:
    """The backend the installed llama.cpp prebuilt records, as Settings shows it."""
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.llama_cpp_freshness import read_install_marker
    from utils.prebuilt.llama_backend import marker_backend

    return marker_backend(read_install_marker(LlamaCppBackend._find_llama_server_binary()))


def _smi_inference_device(
    index: int,
    ordinal: int,
    name: Optional[str],
    total_gb: Optional[float],
    used_gb: Optional[float],
) -> Dict[str, Any]:
    known = total_gb is not None and used_gb is not None
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": ordinal,
        "name": name,
        "memory_total_gb": total_gb,
        "vram_used_gb": used_gb,
        "vram_free_gb": round(max(0.0, total_gb - used_gb), 2) if known else None,
        "vram_utilization_pct": round((used_gb / total_gb) * 100, 1)
        if known and total_gb > 0
        else None,
        "shared_memory": False,
    }


def _nvidia_inference_devices() -> list[Dict[str, Any]]:
    from core.inference.llama_cpp import LlamaCppBackend

    from . import nvidia

    # Same mask llama.cpp's nvidia-smi probe applies.
    allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
    if allowed is None and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return []  # UUID / MIG masks cannot be matched to nvidia-smi rows
    physical = [
        row
        for row in (nvidia.get_physical_gpu_inventory().get("devices") or [])
        if isinstance(row.get("index"), int)
    ]
    # nvidia-smi rows are PCI order; ordinals only match them under PCI_BUS_ID.
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" and (
        allowed is not None or len(physical) > 1
    ):
        return []
    rows = [row for row in physical if allowed is None or row["index"] in allowed]
    if not rows:
        return []
    if allowed is not None:
        # visible_ordinal is the child's numbering, which follows the mask's order.
        try:
            order = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
        except ValueError:
            order = nvidia.resolve_uuid_mask(os.environ["CUDA_VISIBLE_DEVICES"].strip()) or []
        rows.sort(
            key = lambda row: order.index(row["index"]) if row["index"] in order else len(order)
        )
    usage = nvidia.get_visible_gpu_utilization([row["index"] for row in rows])
    usage_by_index = {d.get("index"): d for d in usage.get("devices") or []}
    devices = []
    for ordinal, row in enumerate(rows):
        util = usage_by_index.get(row["index"], {})
        devices.append(
            _smi_inference_device(
                row["index"],
                ordinal,
                row.get("name"),
                row.get("memory_total_gb") or util.get("vram_total_gb"),
                util.get("vram_used_gb"),
            )
        )
    return devices


def get_cross_vendor_inference_gpu_info() -> Optional[Dict[str, Any]]:
    """The NVIDIA cards a CUDA llama.cpp runs on when torch is another backend, else None.

    nvidia-smi only: a CUDA context here would pin VRAM. No ROCm counterpart: amd-smi cannot
    prove the memory scope HIP sees (an APU reports only its carve-out).
    """
    try:
        llama_backend = _installed_llama_backend()
    except Exception as e:
        logger.debug("Could not read the installed llama.cpp backend: %s", e)
        return None
    if llama_backend != "cuda" or _backend_label(get_device()) == "cuda":
        return None
    try:
        devices = _nvidia_inference_devices()
    except Exception as e:
        logger.debug("CUDA inference GPU query failed: %s", e)
        return None
    # Not []: the load estimate reads that as "no GPU".
    if not devices or not all((d["memory_total_gb"] or 0) > 0 for d in devices):
        return None
    return {
        "available": True,
        "backend": llama_backend,
        "backend_cuda_visible_devices": None,
        "parent_visible_gpu_ids": [],
        "devices": devices,
        "index_kind": "physical",
    }


def _repair_smi_visible_devices(
    devices: list[Dict[str, Any]], parent_visible_ids: Optional[list[int]]
) -> bool:
    """Fill in what nvidia-smi could not answer for, from torch's context-free inventory.

    Returns whether every device now carries a capacity.

    nvidia-smi answers ``[N/A]`` for memory.total on a DGX Spark, which NVIDIA documents.
    Keeping that row is right, but a missing total is indistinguishable from no GPU
    downstream: the frontend maps it to zero, the fit classifier returns ``ram``, and the
    picker warns "No GPU detected" on a 121 GiB Blackwell (#10691). ``props.total_memory``
    answers on the same host for no driver context.

    The integrated flag rides along because the reason the capacity is unreadable is that
    this part has no memory of its own, and a total published without it is counted twice.

    A readable total is not a right one. nvidia-smi answers 8128 MiB on a Windows RTX
    Spark N1X, the dedicated carve-out of a part whose CUDA budget is 46477 MiB, so the
    old shortcut here -- return early whenever every row carried a number -- published a
    45 GiB device as a 7.94 GiB one on the System tab while the About tab, which reads
    torch, showed 45.39 GiB for the same machine. So a BLANK total is still filled on any
    part, as it always was, and a READABLE one is additionally widened on a confirmed
    integrated part, and on nothing else.
    """
    if not devices:
        return False
    try:
        inventory, key_field = _integrated_cuda_inventory(parent_visible_ids)
    except Exception as e:  # noqa: BLE001 - the caller keeps the rows nvidia-smi found
        logger.debug("torch inventory unavailable while repairing a GPU capacity: %s", e)
        return all(dev.get("memory_total_gb") is not None for dev in devices)
    for dev in devices:
        td = inventory.get(dev.get(key_field))
        if td is None:
            continue
        # A blank total is always filled, or MIG and vGPU rows hit the torch fallback.
        if dev.get("memory_total_gb") is None:
            dev["memory_total_gb"] = td["total_gb"]
        if not td.get("_cuda_integrated"):
            continue
        # A readable total is only widened on a confirmed integrated part.
        if _integrated_total_is_understated(dev.get("memory_total_gb"), td["total_gb"]):
            dev["memory_total_gb"] = td["total_gb"]
        dev["unified_memory"] = True
        # Set shared_memory too: gpu-vram.ts splits pools on that flag alone.
        dev["shared_memory"] = True
        dev["shared_memory_host_backed_gb"] = dev["memory_total_gb"]
    return all(dev.get("memory_total_gb") is not None for dev in devices)


# (probe start, inventory or None once confirmed empty) per (device, mask); newer wins.
_last_good_visible_info: Dict[tuple, tuple] = {}
_last_good_visible_lock = threading.Lock()


def get_backend_visible_gpu_info() -> Dict[str, Any]:
    """Backend-visible GPU inventory; an unproven empty probe returns the last one, marked ``stale``."""
    device = get_device()
    key = (
        str(device),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("HIP_VISIBLE_DEVICES"),
        os.environ.get("ROCR_VISIBLE_DEVICES"),
        os.environ.get("ZE_AFFINITY_MASK"),
    )
    started = time.monotonic()
    info = _probe_backend_visible_gpu_info(device)
    confirmed_empty = info.pop("_confirmed_empty", False)
    info.pop("probe_failed", None)
    info.pop("smi_absent", None)
    found = bool(info.get("available") and info.get("devices"))
    if found or confirmed_empty:
        with _last_good_visible_lock:
            prior = _last_good_visible_info.get(key)
            if prior is None or prior[0] <= started:
                _last_good_visible_info[key] = (started, copy.deepcopy(info) if found else None)
        return info
    # NVIDIA only: elsewhere no probe can prove a device went away.
    if device != DeviceType.CUDA or IS_ROCM:
        return info
    with _last_good_visible_lock:
        last = (_last_good_visible_info.get(key) or (0.0, None))[1]
    if last is None:
        return info
    logger.warning(
        "GPU inventory probe came back empty after an earlier read found %d device(s); "
        "keeping that inventory (marked stale) instead of reporting no GPU.",
        len(last.get("devices") or []),
    )
    stale = copy.deepcopy(last)
    stale["stale"] = True
    for dev in stale.get("devices") or []:
        for k in ("vram_used_gb", "vram_free_gb", "vram_utilization_pct"):
            if k in dev:
                dev[k] = None
    return stale


def _probe_backend_visible_gpu_info(device) -> Dict[str, Any]:
    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_visible_ids = get_parent_visible_gpu_ids()
        # Kept in case torch cannot size them either: losing the card is worse.
        unrepaired_smi_result: Optional[Dict[str, Any]] = None
        # The only proof of "no GPU" under this mask.
        smi_answered_empty = False
        if device == DeviceType.CUDA and not IS_ROCM:
            try:
                from . import nvidia

                parent_visible_spec = _get_parent_visible_gpu_spec()
                result = nvidia.get_backend_visible_gpu_info(
                    parent_visible_spec["numeric_ids"],
                    parent_visible_spec["raw"],
                )
                smi_answered_empty = (
                    not result.get("available")
                    and not result.get("probe_failed")
                    and not result.get("smi_absent")
                    and result.get("index_kind") != "unresolved"
                )
                if result.get("available"):
                    if _repair_smi_visible_devices(
                        result.get("devices") or [], parent_visible_spec["numeric_ids"]
                    ):
                        result["backend"] = _backend_label(device)
                        return result
                    # Fall through to torch unless the join is unsafe: under FASTEST_FIRST torch rows
                    # would mislabel cards.
                    if _cuda_join_is_unsafe(parent_visible_spec["numeric_ids"]):
                        logger.debug(
                            "Keeping the nvidia-smi rows: CUDA_DEVICE_ORDER does not "
                            "match nvidia-smi, so no torch row can be trusted here."
                        )
                        result["backend"] = _backend_label(device)
                        return result
                    unrepaired_smi_result = result
            except Exception as e:
                logger.warning("Backend GPU visibility query failed: %s", e)

        # Empty parent_visible_ids (UUID/MIG mask): enumerate by torch ordinal.
        if parent_visible_ids:
            torch_indices = parent_visible_ids
            index_kind = "physical"
        else:
            visible_count = _torch_get_physical_gpu_count() or 0
            torch_indices = list(range(visible_count))
            index_kind = "relative"
        # Inventory only, so avoid a permanent driver context.
        torch_devices = _torch_get_device_inventory(torch_indices)
        if torch_devices:
            if IS_ROCM and platform.system() == "Linux":
                shared_host_gb = _rocm_linux_shared_pool_host_gb_by_index(torch_devices)
                for td in torch_devices:
                    if td["index"] in shared_host_gb:
                        host_gb = shared_host_gb[td["index"]]
                        # A measured zero is published: an absent figure renders as all host memory.
                        td["shared_memory_host_backed_gb"] = host_gb
                        # shared_memory also collapses rows into one pool; set it only when host_gb > 0.
                        if host_gb > 0:
                            td["shared_memory"] = True
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
                    "shared_memory": bool(td.get("shared_memory") or td.get("_cuda_integrated")),
                    # An integrated part's total is host memory; publishing stops double counting.
                    "shared_memory_host_backed_gb": (
                        td["total_gb"]
                        if td.get("_cuda_integrated")
                        else td.get("shared_memory_host_backed_gb")
                    ),
                    # A ROCm APU total is the GTT pool, not a VRAM ceiling; distinct from shared_memory.
                    "unified_memory": bool(
                        td.get("_rocm_known_unified") or td.get("_cuda_integrated")
                    ),
                }
                for td in torch_devices
            ]
            # Never report fewer devices than nvidia-smi saw.
            if unrepaired_smi_result is not None and len(devices) < len(
                unrepaired_smi_result.get("devices") or []
            ):
                unrepaired_smi_result["backend"] = _backend_label(device)
                return unrepaired_smi_result

            return {
                "available": True,
                "backend": _backend_label(device),
                "backend_cuda_visible_devices": _backend_visible_devices_env(),
                "parent_visible_gpu_ids": parent_visible_ids,
                "devices": devices,
                "index_kind": index_kind,
            }

        if unrepaired_smi_result is not None:
            # Unknown capacity beats reporting no GPU.
            unrepaired_smi_result["backend"] = _backend_label(device)
            return unrepaired_smi_result

        return {
            "available": False,
            "backend": _backend_label(device),
            "backend_cuda_visible_devices": _backend_visible_devices_env(),
            "parent_visible_gpu_ids": parent_visible_ids,
            "devices": [],
            "index_kind": "physical",
            "_confirmed_empty": smi_answered_empty,
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
        # Cards this PyTorch cannot open, reported alongside, never merged into, devices.
        **_torch_gpu_mismatch_report(),
    }


def get_visible_gpu_count() -> int:
    """Number of GPUs visible to this process, respecting CUDA_VISIBLE_DEVICES and falling back to the physical count when it is unset or torch is unavailable. Cached after the first call."""
    global _visible_gpu_count
    if _visible_gpu_count is not None:
        return _visible_gpu_count

    # Level Zero interprets ZE_AFFINITY_MASK correctly (subdevices collapse to roots).
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
                # Count unique roots: "0.0,0.1" is 1 root.
                if xpu_visible == "*":
                    _visible_gpu_count = get_physical_gpu_count()
                else:
                    roots = _parse_ze_mask_roots(xpu_visible)
                    # Unparseable masks count as 0 visible, not all.
                    _visible_gpu_count = len(set(roots))
            else:
                _visible_gpu_count = get_physical_gpu_count()
        return _visible_gpu_count

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

    try:
        import torch
        _visible_gpu_count = torch.cuda.device_count()
    except Exception:
        _visible_gpu_count = get_physical_gpu_count()

    return _visible_gpu_count


def _rocr_relative_visibility(value: str) -> Optional[str]:
    """``value`` (physical ids) as ordinals into the ROCr-filtered agent list, else None. HIP indexes what ROCr left: under ROCR_VISIBLE_DEVICES=1,0 a HIP mask of "1" is physical GPU 0. Clearing the ROCr mask instead would re-expose the agents it hides, which HSA can crash on merely enumerating."""
    if sys.platform == "win32" or "HIP_VISIBLE_DEVICES" in os.environ:
        return None
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
    if rocr is None:
        return None
    # rocclr reads CUDA_VISIBLE_DEVICES as the HIP mask when HIP's is empty; do not translate twice.
    if _rocm_visibility_masks_are_stacked():
        return None
    try:
        agents = [int(token.strip()) for token in rocr.split(",") if token.strip()]
        wanted = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError:
        return None
    if not agents or not wanted or any(gpu_id not in agents for gpu_id in wanted):
        return None
    return ",".join(str(agents.index(gpu_id)) for gpu_id in wanted)


def apply_gpu_ids(gpu_ids, backend: Optional[str] = None) -> None:
    if gpu_ids is None:
        return

    # CUDA_VISIBLE_DEVICES="" disables CUDA entirely, so treat [] as inherit.
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) == 0:
        return

    global _visible_gpu_count

    if isinstance(gpu_ids, (list, tuple)):
        value = ",".join(str(g) for g in gpu_ids)
    else:
        value = str(gpu_ids)

    # Do not call get_device(): a lazy detect would latch enumeration before the mask is written.
    _is_xpu = DEVICE == DeviceType.XPU
    if backend is not None:
        # The parent's detected backend: exact and probe-free.
        _is_xpu = backend == DeviceType.XPU.value
    elif DEVICE is None:
        # version.xpu can be None on working builds; _is_compiled() needs no runtime init.
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
                # Mirror detect_hardware so hidden CUDA is not re-exposed.
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
        # Leave CUDA_VISIBLE_DEVICES alone so hybrid hosts do not flip back to CUDA.
        _visible_gpu_count = None
        logger.info("Applied gpu_ids: ZE_AFFINITY_MASK='%s'", value)
        return

    # From the build: a stale ROCR var on NVIDIA would move a CUDA pin.
    _rocm_build = IS_ROCM
    if not _rocm_build:
        try:
            import torch as _torch
            _rocm_build = (
                getattr(_torch.version, "hip", None) is not None
                or "rocm" in getattr(_torch, "__version__", "").lower()
            )
        except Exception as e:
            logger.debug(
                "apply_gpu_ids: torch ROCm probe skipped (%s: %s)",
                type(e).__name__,
                e,
            )

    if _rocm_build:
        _relative = _rocr_relative_visibility(value)
        if _relative is not None:
            value = _relative

    os.environ["CUDA_VISIBLE_DEVICES"] = value
    # Writing HIP on NVIDIA is inert; a missed mirror is not.
    _inherits_rocm_visibility = (
        "HIP_VISIBLE_DEVICES" in os.environ or "ROCR_VISIBLE_DEVICES" in os.environ
    )
    _is_rocm = _rocm_build or _inherits_rocm_visibility
    if _is_rocm:
        os.environ["HIP_VISIBLE_DEVICES"] = value
        # ROCR stays inherited: narrowing it made torch.cuda.is_available() False.
    _visible_gpu_count = None
    if _is_rocm:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s' (rocm)", value)
    else:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s'", value)


def get_device_map(gpu_ids: Optional[list[int]] = None) -> str:
    """The Hugging Face ``device_map`` string for model loading: "unsloth_balanced" on CUDA or "balanced" on XPU when gpu_ids lists >1 GPU, or when the visibility mask uses UUID/MIG/wildcard identifiers and >1 GPU is visible; "sequential" otherwise, including CPU/MLX.

    CUDA asks for unsloth's head-aware planner rather than accelerate's "balanced", which caps every device but the last at about model_size / n_devices: the unquantized lm_head then does not fit on cuda:0, every module lands on the last card, and prepare_model refuses a 4bit model wholly on a non-zero card. "unsloth_balanced" rather than "unsloth" because the planner declines several shapes and the plain name falls back to "sequential", which fills cuda:0 to its whole free budget. XPU keeps plain "balanced": the planner has no non-CUDA memory budgets. Use prepare_gpu_selection() upstream to determine gpu_ids.
    """
    device = get_device()
    if device in (DeviceType.CUDA, DeviceType.XPU):
        multi_gpu = gpu_ids is not None and len(gpu_ids) > 1

        if not multi_gpu:
            parent_visible_spec = _get_parent_visible_gpu_spec()
            if device == DeviceType.CUDA:
                # UUID/MIG masks cannot be split into numeric IDs.
                if parent_visible_spec["numeric_ids"] is None and get_visible_gpu_count() > 1:
                    multi_gpu = True
            elif device == DeviceType.XPU and gpu_ids is None:
                # An explicit gpu_ids=[0] must stay sequential.
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
            return "unsloth_balanced" if device == DeviceType.CUDA else "balanced"

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
    """The torch device string for the detected hardware, e.g. "cuda", "xpu" or "cpu"."""
    device = get_device()
    if device == DeviceType.CUDA:
        return "cuda"
    elif device == DeviceType.XPU:
        return "xpu"
    return "cpu"


# Mirrors AUTO_NUM_PROC_CAP in unsloth_zoo.dataset_num_proc; a test canary checks drift.
_STUDIO_NUM_PROC_CAP = 8


def safe_num_proc(desired: Optional[int] = None) -> int:
    """A safe ``num_proc`` for ``dataset.map()``, auto-computed from os.cpu_count() when ``desired`` is None, always >= 1. Always 1 on Windows, which spawns rather than forks, so re-importing torch/transformers/unsloth per worker beats single-process only for huge datasets. Capped to 4 when several GPUs are VISIBLE to this process, since the NVIDIA driver's background threads make os.fork() deadlock-prone; the cap does not apply when CUDA_VISIBLE_DEVICES restricts to one GPU."""
    # spawn platforms: re-importing per worker is usually slower than single-process.
    if sys.platform in ("win32", "darwin"):
        return 1

    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)

    # Downstream treats this as user intent and only clamps by free memory, so cap it here.
    if desired > _STUDIO_NUM_PROC_CAP:
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
    """A safe worker count for ThreadPoolExecutor, auto-computed from os.cpu_count() when ``desired`` is None, always >= 1. Unlike safe_num_proc() it does NOT cap to 1 on macOS/Windows: threads share the parent address space, unaffected by spawn vs fork."""
    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)

    return max(1, desired)


def dataset_map_num_proc(
    desired: Optional[int] = None, *, serial_as_none: bool = True
) -> Optional[int]:
    """A safe ``num_proc`` for Dataset.map()/filter(). None on spawn platforms (Windows, macOS) -- None, not 1, is the disable sentinel, since datasets >= 4.1 takes the pool branch for any num_proc >= 1. Also None on XPU once its runtime is initialized here, because os.fork() corrupts the Level-Zero context; pre-init XPU hosts can still parallelize CPU-side preprocessing. There is deliberately no CUDA equivalent: the child only runs the tokenizer, 300 forced-fork map() runs on an initialized CUDA context produced no failures, and detect_hardware() always initializes CUDA, so such a guard would serialize every CUDA run for nothing.

    ``serial_as_none`` says how to spell "run in-process" for the layer reading the value back. Leave it True at a map() call site, where None is the only value that builds no pool. Pass False when the result is written into a config (SFTConfig.dataset_num_proc): a config None means "auto-size me" to every downstream reader, so only 1 survives that round trip.
    """
    if sys.platform in ("win32", "darwin"):
        # UNSLOTH_DATASET_NUM_PROC is an unvetoed escape hatch; honour it.
        if _num_proc_override_is_set():
            return _bounded_by_the_shared_policy(desired, serial_as_none)
        # None vetoes at every layer; 1 would make other trainers spawn a Pool(1).
        return None

    if get_device() == DeviceType.XPU:
        try:
            import torch
        except Exception:
            # Still bounded, so torch-less containers respect the memory ceiling and hatch.
            return _bounded_by_the_shared_policy(desired, serial_as_none)

        xpu = getattr(torch, "xpu", None)
        is_initialized = getattr(xpu, "is_initialized", None)
        if callable(is_initialized):
            try:
                if is_initialized():
                    if _num_proc_override_is_set():
                        return _bounded_by_the_shared_policy(desired, serial_as_none)
                    # Fork is available here, so None would be auto-sized up; encode serial.
                    return None if serial_as_none else 1
            except Exception as e:
                logger.debug("torch.xpu.is_initialized() probe failed: %s", e)

    return _bounded_by_the_shared_policy(desired, serial_as_none)


# Not "unsloth.dataset_num_proc": claiming the package name would shadow it.
_LOCAL_POLICY_MODULE = "unsloth_studio_local_dataset_num_proc"


def _shared_policy():
    """The shared num_proc policy module, or None on an installation without it. The Zoo owns it, and unsloth.dataset_num_proc is a byte-identical fallback for a Zoo that predates the module. `import unsloth.dataset_num_proc` would run the package __init__, which patches torch and loads the model stack, so that form is used only when the package is already imported; otherwise the file is loaded straight off disk, which is safe because the module is stdlib-only by design."""
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
    # Memoised so the once-per-process warning and cgroup read are not repeated.
    cached = sys.modules.get(_LOCAL_POLICY_MODULE)
    if cached is not None:
        return cached
    try:
        import importlib.util

        # find_spec locates unsloth/ without importing it.
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
    """Whether the escape hatch decided the count, not merely whether it is set: the policy ignores an unparseable or negative value with a warning, so reading the variable directly would let UNSLOTH_DATASET_NUM_PROC=-1 skip the multi-GPU cap while contributing nothing."""
    policy = _shared_policy()
    if policy is None:
        return False
    parsed = getattr(policy, "environment_override", None)
    if parsed is None:
        # Older policy without the public reader: presence errs toward honouring the hatch.
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
    """Apply the training-side num_proc policy to an Unsloth request. format_conversion.py and chat_templates.py hand this straight to Dataset.map, so without it a container with 2GB and eight cores still got eight tokenizer workers. ``desired`` is passed through as written: materializing an auto request with safe_num_proc first would hide it from the policy, whose auto path reads this process's CPU affinity and cgroup quota while safe_num_proc reads the host's os.cpu_count(). Unsloth's own caps are then applied to whatever the policy chose, except over the escape hatch, which is uncapped by contract."""
    policy = _shared_policy()
    if policy is None:
        return safe_num_proc(desired)

    try:
        bounded = policy.get_dataset_num_proc(desired, serial_as_none = serial_as_none)
    except Exception as e:
        logger.debug("dataset_num_proc policy unavailable: %s", e)
        return safe_num_proc(desired)

    if isinstance(bounded, int) and bounded > 1 and not _num_proc_override_is_set():
        bounded = safe_num_proc(bounded)
    return bounded
