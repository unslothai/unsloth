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
import types
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
IS_ROCM: bool = False  # True when running on AMD ROCm (HIP) -- routes GPU monitoring to amd.py


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


def _has_torch() -> bool:
    """True if PyTorch is importable."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_mlx() -> bool:
    """True if MLX is importable."""
    try:
        import mlx.core
        return True
    except ImportError:
        return False


def _has_usable_mlx_stack() -> bool:
    """True only when the FULL Unsloth MLX training/export stack is usable
    (mlx + mlx-lm + mlx-vlm at the minimum versions unsloth-zoo requires), not
    just a bare ``import mlx.core``. A backtracked/old mlx-vlm still imports but
    breaks VLM Train/Export, so the training gate must match the self-heal's own
    criterion (utils.mlx_repair.mlx_stack_available) -- otherwise detect_hardware
    would enable Train/Export on exactly the inadequate stack the MLX self-heal
    is trying to repair, leaving the user with greyed-in-but-broken buttons."""
    try:
        from utils.mlx_repair import mlx_stack_available
        return mlx_stack_available()
    except Exception as exc:
        # mlx_repair should always import; if it somehow cannot, fall back to the
        # bare import check rather than forcing a working host into chat-only.
        logger.debug("MLX stack availability check failed, using bare import: %s", exc)
        return _has_mlx()


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
      1. CUDA  (NVIDIA GPU, requires torch)
      2. MLX   (Apple Silicon via MLX framework)
      3. CPU   (fallback)
    """
    global DEVICE, CHAT_ONLY, CHAT_ONLY_REASON, IS_ROCM
    CHAT_ONLY = True  # reset -- only CUDA/ROCm/XPU/MLX sets it to False
    CHAT_ONLY_REASON = None
    IS_ROCM = False

    # --- CUDA / ROCm: try PyTorch ---
    if _has_torch():
        import torch
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
    if _has_torch():
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
        logger.warning(
            "Apple Silicon detected but the MLX stack is incomplete or too old; "
            "Train/Export disabled (chat-only). Run `unsloth studio update` to "
            "restore MLX training."
        )
    elif platform.system() == "Darwin":
        CHAT_ONLY_REASON = "intel_mac"  # Intel Mac: no PyTorch/MLX -> GGUF-only by design.
    else:
        CHAT_ONLY_REASON = "no_gpu"
    print("Hardware detected: CPU (no GPU backend available)")
    return DEVICE


# ========== Convenience helpers ==========


def get_device() -> DeviceType:
    """
    Return the detected device, auto-detecting if detect_hardware() hasn't run.
    Prefer calling detect_hardware() explicitly at startup.
    """
    global DEVICE
    if DEVICE is None:
        detect_hardware()
    return DEVICE


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
    # No accelerator: name the blocker. Apple Silicon first -- its path is MLX, so "install PyTorch"
    # would be wrong advice on a Mac even when torch is also absent.
    if is_apple_silicon():
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


def clear_gpu_cache():
    """
    Clear GPU memory cache for the current device.
    Safe on any platform — no-ops gracefully.
    """
    gc.collect()

    device = get_device()

    if device == DeviceType.CUDA:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == DeviceType.XPU:
        import torch
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    elif device == DeviceType.MLX:
        # MLX manages memory automatically; gc.collect() above is enough.
        pass


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

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": idx,
                "device_name": props.name,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
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

            return {
                "available": True,
                "backend": _backend_label(device),
                "device": idx,
                "device_name": props.name,
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
                "free_gb": (total - allocated) / (1024**3),
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

    # GPU runtime version bundled with torch
    try:
        import torch
        versions["cuda"] = getattr(torch.version, "cuda", None)
        versions["rocm"] = getattr(torch.version, "hip", None)
    except Exception:
        versions["cuda"] = None
        versions["rocm"] = None

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


def _torch_get_per_device_info(device_indices: list[int]) -> list[Dict[str, Any]]:
    """Query torch for per-GPU name, total VRAM, and used VRAM."""
    mod, _ = _torch_get_device_module()
    if mod is None:
        return []

    devices = []
    for ordinal, phys_idx in enumerate(device_indices):
        try:
            # torch ordinals are 0-based relative to CUDA_VISIBLE_DEVICES.
            props = mod.get_device_properties(ordinal)
            total_bytes = props.total_memory
            # Prefer mem_get_info (system-wide) so auto-select sees other consumers.
            if hasattr(mod, "mem_get_info"):
                free_bytes, total_bytes = mod.mem_get_info(ordinal)
                used_bytes = total_bytes - free_bytes
            else:
                used_bytes = mod.memory_allocated(ordinal)
            devices.append(
                {
                    "index": phys_idx,
                    "visible_ordinal": ordinal,
                    "name": props.name,
                    "total_gb": round(total_bytes / (1024**3), 2),
                    "used_gb": round(used_bytes / (1024**3), 2),
                }
            )
        except Exception as e:
            logger.debug("torch device query failed for ordinal %d: %s", ordinal, e)
    return devices


# ========== Live GPU Utilization ==========


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


def _rocm_linux_sysfs_gpu_busy_pct() -> Optional[float]:
    """Query AMD GPU compute utilization via Linux DRM sysfs gpu_busy_percent."""
    if platform.system() != "Linux":
        return None
    try:
        files = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        if not files:
            return None
        values = [int(open(f).read().strip()) for f in files]
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
        temps = [int(open(f).read().strip()) / 1000.0 for f in files]
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
                watts = sum(int(open(f).read().strip()) / 1_000_000.0 for f in files)
                return round(watts, 1)
        return None
    except Exception:
        return None


def _rocm_windows_perf_counter_gpu_util_pct() -> Optional[float]:
    """Query AMD GPU compute utilization via Windows Performance Counters (3D engine nodes)."""
    if platform.system() != "Windows":
        return None
    try:
        ps = (
            "$s=(Get-Counter '\\GPU Engine(*engtype_3D*)\\Utilization Percentage'"
            " -ErrorAction SilentlyContinue).CounterSamples;"
            "if($s){[math]::Min(($s|Measure-Object CookedValue -Sum).Sum,100)}else{-1}"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            timeout = 5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        val = float(r.stdout.strip())
        return round(val, 1) if val >= 0 else None
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
        used_bytes = sum(int(open(f).read().strip()) for f in used_files)
        total_bytes = sum(int(open(f).read().strip()) for f in total_files)
        if total_bytes == 0:
            return None, None
        return round(used_bytes / (1024**3), 2), round(total_bytes / (1024**3), 2)
    except Exception:
        return None, None


# 0x1002. The NVIDIA open kernel module (560+) also registers KFD topology nodes,
# with vendor_id 4318 (0x10DE) -- install.sh filters on this same id for exactly
# that reason. A non-AMD node is not a HIP device and must never take an ordinal.
_AMD_PCI_VENDOR_ID = 4098


def _rocm_kfd_gpu_pci_ids() -> list[str]:
    """PCI addresses of the GPUs ROCm enumerates, in HIP device order.

    Reads /sys/class/kfd/kfd/topology/nodes/<N>/properties -- the topology the
    ROCm runtime itself enumerates from. AMD GPU nodes (``simd_count`` > 0, which
    excludes CPU nodes, and ``vendor_id`` == AMD, which excludes NVIDIA nodes)
    taken in node-id order are HIP's device order, and each carries its PCI
    location, so position N here IS ROCm physical device N.

    This is the authoritative device->identity link that DRM sysfs alone cannot
    provide: an amdgpu-bound adapter HIP cannot enumerate (an unsupported older
    AMD GPU) has no GPU node here, so it never consumes an ordinal.

    Returns [] -- disabling the overlay rather than guessing -- when KFD is
    unavailable, and FAILS CLOSED the same way if any node cannot be read or an
    AMD GPU node has no ``location_id``: dropping one would silently shift every
    later ordinal, and a similar-capacity GPU would then pass the total-size
    guard while showing another card's usage.

    ``location_id`` is the kernel's ``(bus << 8) | devfn``; ``domain`` is separate.
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
            with open(os.path.join(node_dir, "properties")) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            props[parts[0]] = int(parts[1])
                        except ValueError:
                            continue
        except OSError:
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
    """The amdgpu-bound DRM cards, in PCI order: ``(pci_bdf, card_no, device_dir)``.

    Membership is decided by the BOUND DRIVER (``device/driver`` resolves to
    amdgpu), not by the presence of the VRAM sysfs files: an AMD device with
    incomplete sysfs support (some APUs expose no mem_info_vram_* at all) is still
    a device that consumes a ROCm ordinal, and dropping it would shift every later
    card down one. PCI order is ROCm/HIP's default enumeration order, so the
    position in this list is the ROCm ordinal; the DRM card number is a stable
    tiebreak (the kernel assigns it in PCI-probe order too) for when the device
    symlink cannot be resolved to a BDF.

    NOTE this is a superset of the ROCm-visible set: an amdgpu-bound but
    HIP-unsupported adapter appears here too. Callers must therefore not assume a
    1:1 mapping onto torch devices without checking the counts agree.
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

    Reads /sys/class/drm/card<N>/device/mem_info_vram_{used,total} (amdgpu-only
    files, kernel-updated across all processes) so each GPU gets its own figure,
    unlike _rocm_linux_sysfs_vram_gb which sums the host. Returns
    ``{pci_bdf: (used_gb, total_gb)}``; empty on failure or off Linux.

    Keyed by PCI address rather than any ordinal, so the caller can join it to the
    ROCm device order from _rocm_kfd_gpu_pci_ids() by identity. Position-based
    keying cannot express this correctly: DRM card numbers include foreign
    adapters, and the amdgpu set includes cards HIP does not enumerate, so any
    ordinal derived from this list alone can be shifted relative to ROCm's.
    A card whose figures are missing, unreadable or zero-total simply has no entry.
    """
    if platform.system() != "Linux":
        return {}

    try:
        by_pci: dict[str, tuple[float, float]] = {}
        for bdf, _card_no, dev_dir in _rocm_linux_amdgpu_cards():
            if not bdf:
                continue
            try:
                with open(os.path.join(dev_dir, "mem_info_vram_used")) as f:
                    used_bytes = int(f.read().strip())
                with open(os.path.join(dev_dir, "mem_info_vram_total")) as f:
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


def _rocm_windows_perf_counter_vram_gb() -> tuple[Optional[float], Optional[float]]:
    """Query system-wide dedicated GPU VRAM via Windows Performance Counters.

    Same data source as Task Manager, so cross-process usage is accurate.
    Works for any GPU vendor without amd-smi or nvidia-smi.
    Returns (used_gb, total_gb) or (None, None) on failure.
    """
    if platform.system() != "Windows":
        return None, None
    try:
        ps = (
            "$s=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage'"
            " -ErrorAction SilentlyContinue).CounterSamples;"
            "if($s){($s|Measure-Object CookedValue -Sum).Sum}else{-1}"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True,
            text = True,
            timeout = 5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None, None
        used_bytes = float(r.stdout.strip())
        if used_bytes < 0:
            return None, None
        import torch as _torch

        total_bytes = _torch.cuda.get_device_properties(0).total_memory
        return round(used_bytes / (1024**3), 2), round(total_bytes / (1024**3), 2)
    except Exception:
        return None, None


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

        # Fallback Windows ROCm
        if IS_ROCM and platform.system() == "Windows":
            _win_used, _win_total = _rocm_windows_perf_counter_vram_gb()
            if _win_used is not None and _win_total is not None:
                _win_util = _rocm_windows_perf_counter_gpu_util_pct()
                return _gpu_utilization_payload(
                    device,
                    [
                        {
                            "available": True,
                            "backend": _backend_label(device),
                            "index": 0,
                            "visible_ordinal": 0,
                            "gpu_utilization_pct": _win_util,
                            "temperature_c": None,
                            "vram_used_gb": _win_used,
                            "vram_total_gb": _win_total,
                            "vram_utilization_pct": round((_win_used / _win_total) * 100, 1)
                            if _win_total > 0
                            else None,
                            "power_draw_w": None,
                            "power_limit_w": None,
                            "power_utilization_pct": None,
                        }
                    ],
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
                        if _total > 0
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
    smi_total_gb = device_metrics.get("vram_total_gb") or 0.0
    if torch_total_gb > smi_total_gb:
        torch_used_gb = torch_info["used_gb"]
        device_metrics["vram_total_gb"] = torch_total_gb
        device_metrics["vram_used_gb"] = torch_used_gb
        device_metrics["vram_utilization_pct"] = (
            round((torch_used_gb / torch_total_gb) * 100, 1) if torch_total_gb > 0 else None
        )
        logger.debug(
            "ROCm unified memory: replaced amd-smi VRAM (%.2f GB) with "
            "torch mem_get_info total (%.2f GB) for device %s",
            smi_total_gb,
            torch_total_gb,
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


def _overlay_system_wide_vram(devices: list[Dict[str, Any]]) -> None:
    """Replace process-local torch VRAM figures with system-wide ones on Linux
    ROCm.

    The torch fallback is process-local, so a model served by the separate
    llama-server process reads as ~0 used even with the GPU full (#7072). On
    Linux, DRM sysfs (``mem_info_vram_{used,total}``) gives per-card system-wide
    figures the kernel updates across all processes. Sources are matched by the
    device's PHYSICAL index (never by list position), so a reordering visibility
    mask (HIP_VISIBLE_DEVICES=1,0) keeps each card's figures on the right GPU.
    Best-effort, in place; a device with no matching card keeps its torch
    figures, and a unified-memory APU whose sysfs total is below torch's
    GTT-backed total keeps torch's (mirrors _apply_unified_memory_correction).

    Windows is intentionally not overlaid: the per-adapter Performance Counters
    cannot be reliably mapped to ROCm device ordinals (the wildcard query also
    returns non-ROCm/iGPU adapters, LUID order is not device order) and read
    only Dedicated Usage, missing WDDM shared memory on unified-memory GPUs -- so
    the multi-GPU view keeps the torch fallback there rather than risk
    misattributing another adapter's usage."""
    if not devices or platform.system() != "Linux":
        return
    # Match devices to cards by PCI IDENTITY, never by position. KFD topology is
    # what the ROCm runtime enumerates from, so index N there is ROCm physical
    # device N and carries that GPU's PCI address; DRM sysfs supplies the
    # system-wide figures under the same address.
    #
    # That join is only meaningful while the reported ``index`` really is a
    # HOST-physical ordinal, and torch cannot tell us its device's PCI id to check
    # directly. So overlay only when host visibility is positively verified:
    #
    #   * No visibility mask. Any of them makes ``index`` something other than a
    #     host-physical ordinal -- a layered HIP-over-ROCR value is ROCR-relative,
    #     GPU_DEVICE_ORDINAL is not consulted when the index is assigned at all,
    #     and under device-cgroup filtering a mask indexes the container's set.
    #   * The device count matches the host's GPU count. A container that exposes
    #     only some render devices through device cgroups sets no env var, yet
    #     torch compacts what it can see to ordinals from zero while the
    #     host-mounted KFD/DRM trees still list every GPU. Equal counts is what
    #     rules that out and pins the indices to host-physical 0..n-1.
    #
    # Anything else keeps torch's process-local figures: less informative, but
    # never another card's usage attributed to this GPU.
    pci_by_ordinal = _rocm_kfd_gpu_pci_ids()
    if not pci_by_ordinal:
        return
    if _rocm_visibility_mask_active() or len(devices) != len(pci_by_ordinal):
        return
    vram_by_pci = _rocm_linux_sysfs_vram_by_pci_gb()
    for dev in devices:
        index = dev.get("index")
        if not isinstance(index, int) or not (0 <= index < len(pci_by_ordinal)):
            continue
        entry = vram_by_pci.get(pci_by_ordinal[index].lower())
        if entry is None:
            continue
        used, total = entry
        dev_total = dev.get("vram_total_gb") or 0.0
        # Overlay only a device that maps 1:1 to this whole physical card: its
        # torch total must match the card's sysfs total (within ~10%). A mismatch
        # in EITHER direction means sysfs describes a different memory scope than
        # the torch device and must not overwrite it:
        #   * unified-memory APU (Strix Halo): sysfs reports only the small
        #     dedicated VRAM slice while torch sees the larger GTT-backed pool;
        #   * partitioned ROCm device (MI300 CPX mode): HIP exposes several
        #     logical devices per card while sysfs reports the whole card's
        #     aggregate, so the card total dwarfs a partition's torch total.
        # Overlaying either would misstate the device's capacity and free VRAM
        # (a partition would look like it has the whole card free).
        if dev_total <= 0 or abs(total - dev_total) > 0.1 * dev_total:
            continue
        dev["vram_used_gb"] = used
        dev["vram_total_gb"] = total
        dev["vram_utilization_pct"] = round((used / total) * 100, 1) if total > 0 else None


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
        torch_devices = _torch_get_per_device_info(torch_indices)
        if torch_devices:
            devices = []
            for td in torch_devices:
                total = td["total_gb"]
                used = td["used_gb"]
                devices.append(
                    {
                        "index": td["index"],
                        "index_kind": index_kind,
                        "visible_ordinal": td["visible_ordinal"],
                        "gpu_utilization_pct": None,
                        "temperature_c": None,
                        "vram_used_gb": used,
                        "vram_total_gb": total,
                        "vram_utilization_pct": round((used / total) * 100, 1)
                        if total > 0
                        else None,
                        "power_draw_w": None,
                        "power_limit_w": None,
                        "power_utilization_pct": None,
                    }
                )
            if IS_ROCM and index_kind == "physical":
                # Torch VRAM here is process-local; swap in system-wide readings
                # (Linux DRM sysfs) so a model held by the separate llama-server
                # process shows up (#7072). Mirrors the fallbacks
                # get_gpu_utilization already applies to the primary GPU.
                # Physical-index only: under a UUID/MIG mask ``index`` is a visible
                # ordinal (index_kind == "relative"), so it is not a host GPU id.
                # The overlay verifies the rest of that claim itself before
                # touching anything.
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
        "index_kind": "relative",
    }


# ========== Multi-GPU Detection & Safe num_proc ==========

_physical_gpu_count: Optional[int] = None
_visible_gpu_count: Optional[int] = None


def _get_parent_visible_gpu_spec() -> Dict[str, Any]:
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
        rocr_vis = os.environ.get("ROCR_VISIBLE_DEVICES")
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


def resolve_requested_gpu_ids(gpu_ids: Optional[list[int]]) -> list[int]:
    parent_visible_spec = _get_parent_visible_gpu_spec()
    parent_visible_ids = get_parent_visible_gpu_ids()
    physical_gpu_count = get_physical_gpu_count()

    if gpu_ids is None:
        return parent_visible_ids

    requested_ids = list(gpu_ids)
    if len(requested_ids) == 0:
        return parent_visible_ids

    if not parent_visible_spec["supports_explicit_gpu_ids"]:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: explicit physical GPU IDs are "
            f"unsupported when CUDA_VISIBLE_DEVICES uses UUID/MIG entries "
            f"({parent_visible_spec['raw']!r}). Omit gpu_ids to use the "
            "parent-visible devices."
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

    if get_device() != DeviceType.CUDA:
        metadata["selection_mode"] = "non_cuda"
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
                "Selected GPUs automatically",
                model_name = model_name,
                selected_gpu_ids = selected,
                usable_gb = metadata["usable_gb"],
                required_gb = metadata.get("required_gb"),
                multi_gpu_overhead = multi_gpu_overhead,
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
        "Falling back to all visible GPUs -- model may not fit",
        model_name = model_name,
        selected_gpu_ids = fallback_all,
        usable_gb = metadata["usable_gb"],
        required_gb = metadata.get("required_gb"),
        multi_gpu_overhead = multi_gpu_overhead,
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
    if gpu_ids and get_device() != DeviceType.CUDA:
        raise ValueError(
            f"gpu_ids {list(gpu_ids)} is only supported on CUDA devices, "
            f"but the current backend is '{get_device().value}'."
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
    global _physical_gpu_count
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

    On ROCm, HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES take precedence over
    CUDA_VISIBLE_DEVICES; this mirrors ``_get_parent_visible_gpu_spec`` so
    ``backend_cuda_visible_devices`` reports the value actually narrowing the
    visible device set.
    """
    if IS_ROCM:
        return _get_parent_visible_gpu_spec().get("raw")
    return os.environ.get("CUDA_VISIBLE_DEVICES")


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
        torch_devices = _torch_get_per_device_info(torch_indices)
        if torch_devices:
            devices = [
                {
                    "index": td["index"],
                    "index_kind": index_kind,
                    "visible_ordinal": td["visible_ordinal"],
                    "name": td["name"],
                    "memory_total_gb": td["total_gb"],
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
                    "memory_total_gb": round(mem.get("total_gb", 0), 2),
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
        "index_kind": "relative",
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

    # No visibility env var set -- try torch, else physical count
    try:
        import torch
        if get_device() == DeviceType.XPU and hasattr(torch, "xpu"):
            _visible_gpu_count = torch.xpu.device_count()
        else:
            _visible_gpu_count = torch.cuda.device_count()
    except Exception:
        _visible_gpu_count = get_physical_gpu_count()

    return _visible_gpu_count


def apply_gpu_ids(gpu_ids) -> None:
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
      - ``CUDA_VISIBLE_DEVICES`` uses UUID/MIG identifiers (non-numeric) and
        >1 GPU is visible (fallback: numeric IDs unresolvable, so assume
        multi-GPU is intended).

    Returns ``"sequential"`` (single device) otherwise, including non-CUDA
    backends (CPU, MLX).

    Use ``prepare_gpu_selection()`` upstream to determine ``gpu_ids`` -- it
    handles auto-selecting the minimum GPUs needed for a model.
    """
    device = get_device()
    if device == DeviceType.CUDA:
        multi_gpu = gpu_ids is not None and len(gpu_ids) > 1

        if not multi_gpu:
            # UUID/MIG masks can't be split into numeric IDs; >1 visible GPU
            # means multi-GPU sharding is intended.
            parent_visible_spec = _get_parent_visible_gpu_spec()
            if parent_visible_spec["numeric_ids"] is None and get_visible_gpu_count() > 1:
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


def dataset_map_num_proc(desired: Optional[int] = None) -> Optional[int]:
    """
    Return a safe ``num_proc`` for ``Dataset.map()`` and ``Dataset.filter()``.

    Returns ``None`` on spawn platforms (Windows, macOS) because ``datasets``
    treats ``num_proc=1`` as multiprocessing (creates ``Pool(1)``); only
    ``num_proc=None`` guarantees in-process execution.
    """
    if sys.platform in ("win32", "darwin"):
        return None
    return safe_num_proc(desired)
