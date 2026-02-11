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
import logging
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ========== Device Enum ==========

class DeviceType(str, Enum):
    """Supported compute backends. Inherits from str so it serializes cleanly in JSON."""
    CUDA = "cuda"
    MPS  = "mps"
    CPU  = "cpu"


# ========== Global State (set once by detect_hardware) ==========

DEVICE: Optional[DeviceType] = None


# ========== Detection ==========

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon hardware (pure platform check, no ML imports)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
pass


def detect_hardware() -> DeviceType:
    """
    Detect the best available compute device and set the module-level DEVICE global.

    Should be called exactly once during FastAPI lifespan startup.
    Safe to call multiple times (idempotent).

    Detection order:
      1. CUDA  (NVIDIA GPU, requires torch)
      2. MPS   (Apple Silicon via PyTorch MPS backend)
      3. CPU   (fallback)
    """
    global DEVICE

    # --- Try PyTorch first (covers CUDA and MPS) ---
    try:
        import torch

        if torch.cuda.is_available():
            DEVICE = DeviceType.CUDA
            device_name = torch.cuda.get_device_properties(0).name
            logger.info(f"Hardware detected: CUDA — {device_name}")
            return DEVICE

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = DeviceType.MPS
            chip = platform.processor() or platform.machine()
            logger.info(f"Hardware detected: MPS — Apple Silicon ({chip})")
            return DEVICE

    except ImportError:
        logger.warning("PyTorch not installed — falling back to CPU")

    DEVICE = DeviceType.CPU
    logger.info("Hardware detected: CPU (no GPU backend available)")
    return DEVICE
pass


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
pass


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
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == DeviceType.MPS:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
pass


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get GPU memory information.
    Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU-only environments.
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

    # ---- MPS path (Apple Silicon) ----
    if device == DeviceType.MPS:
        try:
            import torch
            import psutil
            allocated = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
            total = psutil.virtual_memory().total

            return {
                "available": True,
                "backend": device.value,
                "device": 0,
                "device_name": f"Apple Silicon ({platform.processor() or platform.machine()})",
                "total_gb": total / (1024**3),
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": 0,  # MPS doesn't have a separate reserved pool
                "free_gb": (total - allocated) / (1024**3),
                "utilization_pct": (allocated / total) * 100 if total else 0,
            }
        except Exception as e:
            logger.error(f"Error getting MPS GPU info: {e}")
            return {"available": False, "backend": device.value, "error": str(e)}

    # ---- CPU-only ----
    return {"available": False, "backend": "cpu"}
pass


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
pass
