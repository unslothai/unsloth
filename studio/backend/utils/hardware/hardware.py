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

import platform
import structlog
from loggers import get_logger
from enum import Enum
from typing import Optional, Dict, Any

logger = get_logger(__name__)


# ========== Device Enum ==========


class DeviceType(str, Enum):
    """Supported compute backends. Inherits from str so it serializes cleanly in JSON."""

    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"


# ========== Global State (set once by detect_hardware) ==========

DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True  # No CUDA GPU → GGUF chat only (Mac, CPU-only, etc.)


# ========== Detection ==========


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon hardware (pure platform check, no ML imports)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _has_torch() -> bool:
    """Check if PyTorch is importable."""
    try:
        import torch

        return True
    except ImportError:
        return False


def _has_mlx() -> bool:
    """Check if MLX is importable."""
    try:
        import mlx.core

        return True
    except ImportError:
        return False


def detect_hardware() -> DeviceType:
    """
    Detect the best available compute device and set the module-level DEVICE global.

    Should be called exactly once during FastAPI lifespan startup.
    Safe to call multiple times (idempotent).

    Detection order:
      1. CUDA  (NVIDIA GPU, requires torch)
      2. MLX   (Apple Silicon via MLX framework)
      3. CPU   (fallback)
    """
    global DEVICE, CHAT_ONLY
    CHAT_ONLY = True  # reset — only CUDA sets it to False

    # --- CUDA: try PyTorch ---
    if _has_torch():
        import torch

        if torch.cuda.is_available():
            DEVICE = DeviceType.CUDA
            CHAT_ONLY = False
            device_name = torch.cuda.get_device_properties(0).name
            print(f"Hardware detected: CUDA — {device_name}")
            return DEVICE

    # --- MLX: Apple Silicon ---
    if is_apple_silicon() and _has_mlx():
        DEVICE = DeviceType.MLX
        chip = platform.processor() or platform.machine()
        print(f"Hardware detected: MLX — Apple Silicon ({chip})")
        return DEVICE

    # --- Fallback ---
    DEVICE = DeviceType.CPU
    print("Hardware detected: CPU (no GPU backend available)")
    return DEVICE


# ========== Convenience helpers ==========


def get_device() -> DeviceType:
    """
    Return the detected device. Auto-detects if detect_hardware() hasn't been called yet.
    Prefer calling detect_hardware() explicitly at startup instead.
    """
    global DEVICE
    if DEVICE is None:
        detect_hardware()
    return DEVICE


def clear_gpu_cache():
    """
    Clear GPU memory cache for the current device.
    Safe to call on any platform — no-ops gracefully.
    """
    import gc

    gc.collect()

    device = get_device()

    if device == DeviceType.CUDA:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == DeviceType.MLX:
        # MLX manages memory automatically; no explicit cache clear needed.
        # mlx.core has no empty_cache equivalent — gc.collect() above is enough.
        pass


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    Supports CUDA (NVIDIA), MLX (Apple Silicon), and CPU-only environments.
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
                "backend": device.value,
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
            return {"available": False, "backend": device.value, "error": str(e)}

    # ---- MLX path (Apple Silicon) ----
    if device == DeviceType.MLX:
        try:
            import mlx.core as mx
            import psutil

            # MLX uses unified memory — report system memory as the pool
            total = psutil.virtual_memory().total
            # MLX doesn't expose per-process GPU allocation; report 0 as allocated
            allocated = 0

            return {
                "available": True,
                "backend": device.value,
                "device": 0,
                "device_name": f"Apple Silicon ({platform.processor() or platform.machine()})",
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": 0,
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total else 0,
            }
        except Exception as e:
            logger.error(f"Error getting MLX GPU info: {e}")
            return {"available": False, "backend": device.value, "error": str(e)}

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
        }
    return {"gpu_name": None, "vram_total_gb": None}


def get_package_versions() -> Dict[str, Optional[str]]:
    """
    Return the installed versions of key ML packages.

    Uses importlib.metadata (stdlib) so no subprocess is needed.
    CUDA version comes from torch.version.cuda.

    Returns dict with keys: unsloth, torch, transformers, cuda.
    Missing packages yield None.
    """
    from importlib.metadata import version as pkg_version, PackageNotFoundError

    packages = ("unsloth", "torch", "transformers")
    versions: Dict[str, Optional[str]] = {}

    for name in packages:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = None

    # CUDA toolkit version bundled with torch
    try:
        import torch

        versions["cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        versions["cuda"] = None

    return versions


# ========== Live GPU Utilization (nvidia-smi) ==========


def get_gpu_utilization() -> Dict[str, Any]:
    """
    Return a live snapshot of GPU utilization via ``nvidia-smi``.

    Designed to be polled by the frontend during training (not streaming).
    Uses ``nvidia-smi --query-gpu`` which is the most accurate source for
    utilization %, temperature, and power draw – stats that PyTorch does
    not expose.

    Returns dict with keys:
        available          – bool, whether stats could be retrieved
        gpu_utilization_pct – GPU core utilization %
        temperature_c      – GPU temperature in °C
        vram_used_gb       – VRAM currently used (GiB)
        vram_total_gb      – VRAM total (GiB)
        vram_utilization_pct – VRAM used / total * 100
        power_draw_w       – current power draw (W)
        power_limit_w      – power limit (W)
        power_utilization_pct – power draw / limit * 100
    """
    device = get_device()

    if device != DeviceType.CUDA:
        return {"available": False, "backend": device.value}

    def _parse_smi_value(raw: str):
        """Parse a single nvidia-smi CSV value. Returns float or None for [N/A]."""
        raw = raw.strip()
        if not raw or raw == "[N/A]":
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    # ── nvidia-smi (most complete source) ───────────────────────
    smi_data = {}
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            timeout = 5,
        )

        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi outputs one line per GPU; take GPU 0
            first_line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 6:
                smi_data = {
                    "gpu_util": _parse_smi_value(parts[0]),
                    "temp": _parse_smi_value(parts[1]),
                    "vram_used_mb": _parse_smi_value(parts[2]),
                    "vram_total_mb": _parse_smi_value(parts[3]),
                    "power_draw": _parse_smi_value(parts[4]),
                    "power_limit": _parse_smi_value(parts[5]),
                }

    except FileNotFoundError:
        logger.debug("nvidia-smi not found, falling back to torch.cuda")
    except Exception as e:
        logger.warning(f"nvidia-smi query failed: {e}")

    # ── Backfill VRAM from torch.cuda if nvidia-smi returned [N/A] ──
    vram_used_mb = smi_data.get("vram_used_mb")
    vram_total_mb = smi_data.get("vram_total_mb")

    if vram_used_mb is None or vram_total_mb is None:
        try:
            import torch

            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            if vram_total_mb is None:
                vram_total_mb = props.total_memory / (1024**2)  # bytes → MiB
            if vram_used_mb is None:
                vram_used_mb = torch.cuda.memory_allocated(idx) / (1024**2)
        except Exception as e:
            logger.debug(f"torch.cuda VRAM backfill failed: {e}")

    # ── Build response ──────────────────────────────────────────
    gpu_util = smi_data.get("gpu_util")
    temp = smi_data.get("temp")
    power_draw = smi_data.get("power_draw")
    power_limit = smi_data.get("power_limit")

    vram_used_gb = round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None
    vram_total_gb = (
        round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None
    )
    vram_pct = (
        round((vram_used_mb / vram_total_mb) * 100, 1)
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
        else None
    )
    power_pct = (
        round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit and power_limit > 0
        else None
    )

    # If we got at least something useful, report available
    has_any = any(v is not None for v in [gpu_util, temp, vram_used_gb, power_draw])
    if not has_any:
        return {"available": False, "backend": device.value}

    return {
        "available": True,
        "backend": device.value,
        "gpu_utilization_pct": gpu_util,
        "temperature_c": temp,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_utilization_pct": vram_pct,
        "power_draw_w": power_draw,
        "power_limit_w": power_limit,
        "power_utilization_pct": power_pct,
    }


# ========== Multi-GPU Detection & Safe num_proc ==========

_physical_gpu_count: Optional[int] = None


def get_physical_gpu_count() -> int:
    """
    Return the number of physical NVIDIA GPUs on the machine.

    Uses ``nvidia-smi -L`` which is NOT affected by CUDA_VISIBLE_DEVICES,
    so it always reflects the true hardware count.
    Result is cached after the first call.
    """
    global _physical_gpu_count
    if _physical_gpu_count is not None:
        return _physical_gpu_count

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output = True,
            text = True,
            timeout = 5,
        )
        if result.returncode == 0 and result.stdout.strip():
            _physical_gpu_count = len(result.stdout.strip().splitlines())
        else:
            _physical_gpu_count = 1
    except Exception:
        _physical_gpu_count = 1

    return _physical_gpu_count


def safe_num_proc(desired: Optional[int] = None) -> int:
    """
    Return a safe ``num_proc`` for ``dataset.map()`` calls.

    On Windows, always returns 1 because Python uses ``spawn`` instead of
    ``fork`` for multiprocessing — the overhead of re-importing torch,
    transformers, unsloth etc. per worker is typically slower than
    single-process for normal dataset sizes.

    On multi-GPU machines the NVIDIA driver spawns extra background threads,
    making ``os.fork()`` prone to deadlocks when many workers are created.
    This helper caps ``num_proc`` to 4 on such machines.

    On single-GPU (or CPU-only) machines the original value is returned
    unchanged.

    Args:
        desired: The num_proc you *want*. If None, auto-computes from
                 ``os.cpu_count()``.

    Returns:
        A safe integer ≥ 1.
    """
    import os
    import sys

    # Windows uses 'spawn' for multiprocessing — the overhead of re-importing
    # torch/transformers/unsloth per worker is typically slower than single-process.
    if sys.platform == "win32":
        return 1

    if desired is None or not isinstance(desired, int):
        desired = max(1, os.cpu_count() // 3)

    if get_physical_gpu_count() > 1:
        capped = min(4, desired)
        logger.info(
            f"⚙️ Multi-GPU detected ({get_physical_gpu_count()} GPUs) — "
            f"capping num_proc {desired} → {capped} to avoid fork deadlocks"
        )
        return capped

    return desired
