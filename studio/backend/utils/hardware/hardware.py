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

import os
import platform
import structlog
from loggers import get_logger
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

logger = get_logger(__name__)


# ========== Device Enum ==========


class DeviceType(str, Enum):
    """Supported compute backends. Inherits from str so it serializes cleanly in JSON."""

    CUDA = "cuda"
    XPU = "xpu"
    MLX = "mlx"
    CPU = "cpu"


# ========== Global State (set once by detect_hardware) ==========

DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True  # No CUDA GPU -> GGUF chat only (Mac, CPU-only, etc.)
IS_ROCM: bool = (
    False  # True when running on AMD ROCm (HIP) -- routes GPU monitoring to amd.py
)


def _backend_label(device: DeviceType) -> str:
    """Return the user-facing backend name for API responses.

    Internally we still represent ROCm hosts as ``DeviceType.CUDA`` because
    ROCm torch sets ``torch.cuda.is_available() = True`` and reuses the whole
    ``torch.cuda.*`` API surface, so branching on ``DeviceType`` stays
    consistent with the rest of the codebase. For the JSON responses served
    to the Studio frontend and other clients, however, "cuda" is misleading
    on an AMD machine. This helper swaps the label to ``"rocm"`` when the
    module-level ``IS_ROCM`` flag is set so the UI can render the correct
    backend name without every caller having to duplicate the check.
    """
    if IS_ROCM and device == DeviceType.CUDA:
        return "rocm"
    return device.value


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
    global DEVICE, CHAT_ONLY, IS_ROCM
    CHAT_ONLY = True  # reset -- only CUDA/ROCm sets it to False
    IS_ROCM = False

    # --- CUDA / ROCm: try PyTorch ---
    if _has_torch():
        import torch

        if torch.cuda.is_available():
            DEVICE = DeviceType.CUDA
            CHAT_ONLY = False
            device_name = torch.cuda.get_device_properties(0).name

            # Distinguish AMD ROCm (HIP) from NVIDIA CUDA for display purposes.
            # DeviceType stays CUDA since torch.cuda.* works on ROCm via HIP.
            if getattr(torch.version, "hip", None) is not None:
                IS_ROCM = True
                print(
                    f"Hardware detected: ROCm (HIP {torch.version.hip}) -- {device_name}"
                )
            else:
                print(f"Hardware detected: CUDA -- {device_name}")
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
    elif device == DeviceType.XPU:
        import torch

        torch.xpu.synchronize()
        torch.xpu.empty_cache()
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

            # MLX uses unified memory — report system memory as the pool
            total = psutil.virtual_memory().total
            # MLX doesn't expose per-process GPU allocation; report 0 as allocated
            allocated = 0

            return {
                "available": True,
                "backend": _backend_label(device),
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
            # torch uses 0-based ordinals relative to CUDA_VISIBLE_DEVICES
            props = mod.get_device_properties(ordinal)
            total_bytes = props.total_memory
            # Prefer mem_get_info (reports system-wide usage, not just this
            # process) so auto-selection accounts for other GPU consumers.
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
    """Run a query against the appropriate SMI backend (amd-smi or nvidia-smi).

    Returns the result dict if available, or None on failure/unavailability.
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
        if result.get("available"):
            return result
    except Exception as e:
        logger.warning("%s %s query failed: %s", backend_name, func_name, e)
    return None


def get_gpu_utilization() -> Dict[str, Any]:
    """Return a live snapshot of device utilization information."""
    device = get_device()

    if device == DeviceType.CUDA:
        result = _smi_query("get_primary_gpu_utilization")
        if result is not None:
            result["backend"] = _backend_label(device)
            return result

    mem = get_gpu_memory_info()
    if device != DeviceType.CPU and mem.get("available"):
        return {
            "available": True,
            "backend": _backend_label(device),
            "gpu_utilization_pct": None,
            "temperature_c": None,
            "vram_used_gb": round(mem.get("allocated_gb", 0), 2),
            "vram_total_gb": round(mem.get("total_gb", 0), 2),
            "vram_utilization_pct": round(mem.get("utilization_pct", 0), 1),
            "power_draw_w": None,
            "power_limit_w": None,
            "power_utilization_pct": None,
        }

    return {"available": False, "backend": _backend_label(device)}


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
            return result

    # Torch-based fallback for CUDA (nvidia-smi unavailable, AMD ROCm) and XPU (Intel)
    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_ids = get_parent_visible_gpu_ids()
        # When parent_visible_ids is empty (UUID/MIG mask or no CVD set),
        # enumerate torch-visible ordinals so the UI still shows devices.
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
    # ROCm uses HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES in addition to
    # CUDA_VISIBLE_DEVICES (which HIP also respects).  Check ROCm-specific
    # env vars first so multi-GPU AMD setups are handled correctly.
    # Use explicit None checks (not `or`) so empty string "" is honoured
    # as "no visible GPUs" rather than falling through to CUDA_VISIBLE_DEVICES.
    cuda_visible = None
    if IS_ROCM:
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

    # Reject negative IDs unconditionally.
    negative_ids = [gpu_id for gpu_id in requested_ids if gpu_id < 0]
    if negative_ids:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: GPU IDs must be non-negative. "
            f"Rejected IDs: {negative_ids}. Parent-visible GPUs: {parent_visible_ids}"
        )

    # Only enforce the physical upper bound when we have a reliable count
    # from nvidia-smi. When the count comes from torch, it reflects visible
    # devices (filtered by CUDA_VISIBLE_DEVICES), not the physical total,
    # so high physical indices like 3 would be falsely rejected on a
    # CUDA_VISIBLE_DEVICES="2,3" machine that reports device_count()=2.
    # The parent-visible check below is authoritative in all cases.
    if physical_gpu_count > 0 and parent_visible_ids:
        max_parent_id = max(parent_visible_ids)
        if physical_gpu_count > max_parent_id:
            # Count is plausibly physical (not just visible), so enforce it
            out_of_range = [
                gpu_id for gpu_id in requested_ids if gpu_id >= physical_gpu_count
            ]
            if out_of_range:
                raise ValueError(
                    f"Invalid gpu_ids {requested_ids}: IDs must be physical GPU IDs "
                    f"between 0 and {physical_gpu_count - 1}. "
                    f"Rejected IDs: {out_of_range}. Parent-visible GPUs: {parent_visible_ids}"
                )

    disallowed_ids = [
        gpu_id for gpu_id in requested_ids if gpu_id not in parent_visible_ids
    ]
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
        logger.debug(
            "Could not resolve base model for GPU estimate '%s': %s", model_name, e
        )
        return model_name


def _get_local_weight_size_bytes(model_name: str) -> Optional[int]:
    model_path = Path(model_name)
    if not model_path.exists():
        return None

    weight_exts = (".safetensors", ".bin", ".pt", ".pth")
    total = 0
    for file in model_path.rglob("*"):
        if file.is_file() and file.suffix in weight_exts:
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
    try:
        from transformers import AutoConfig

        trust_remote_code = model_name.lower().startswith("unsloth/")
        return AutoConfig.from_pretrained(
            model_name,
            token = hf_token,
            trust_remote_code = trust_remote_code,
        )
    except Exception as e:
        logger.warning("Could not load config for '%s': %s", model_name, e)
        return None


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
            _, _, _, memory_left_for_kv_cache_gb = (
                _vllm_utils.approximate_vllm_memory_usage(
                    config,
                    load_in_4bit = False,
                    load_in_8bit = False,
                    max_seq_length = 1,
                    gpu_memory_utilization = 1.0,
                    enable_lora = False,
                    account_for_gradients = False,
                    cuda_graph_overhead = False,
                )
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
    estimate_model = _resolve_model_identifier_for_gpu_estimate(
        model_name, hf_token = hf_token
    )

    total_params = None
    if "/" in estimate_model and not Path(estimate_model).exists():
        total_params = _get_hf_safetensors_total_params(
            estimate_model, hf_token = hf_token
        )
    if total_params:
        return int(total_params * 2), "safetensors"

    config = _load_config_for_gpu_estimate(estimate_model, hf_token = hf_token)
    if config is not None:
        config_bytes = _estimate_fp16_model_size_bytes_from_config(config)
        if config_bytes is not None:
            return config_bytes, "config"

    local_bytes = _get_local_weight_size_bytes(estimate_model)
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
        CUDA_OVERHEAD_BYTES,
        QUANT_4BIT_FACTOR,
        DEFAULT_TARGET_MODULES,
    )

    model_size_bytes, source = estimate_fp16_model_size_bytes(
        model_name, hf_token = hf_token
    )
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
        "full"
        if training_type == "Full Finetuning"
        else ("qlora" if load_in_4bit else "lora")
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

    estimate_model = _resolve_model_identifier_for_gpu_estimate(
        model_name, hf_token = hf_token
    )
    config = _load_config_for_gpu_estimate(estimate_model, hf_token = hf_token)
    arch = extract_arch_config(config) if config is not None else None

    if arch is not None:
        breakdown = estimate_training_vram(arch, vram_config)
        required_gb = breakdown.total / (1024**3)
        metadata["required_gb"] = round(required_gb, 3)
        metadata["estimation_mode"] = "detailed"
        metadata["vram_breakdown"] = breakdown.to_gb_dict()
        max_gpus = max(1, get_visible_gpu_count())
        for n_gpus in range(1, max_gpus + 1):
            metadata["vram_breakdown"][f"min_per_gpu_{n_gpus}"] = round(
                breakdown.min_gpu_vram(n_gpus) / (1024**3), 3
            )
        return required_gb, metadata

    # Fallback when model config is unavailable
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
        # Cannot estimate model size -- fall back to all visible GPUs
        # rather than risk loading on a single GPU that may not have
        # enough memory.
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
    # Multi-GPU sharding has overhead from inter-GPU communication (NCCL
    # all-reduce, PCIe/NVLink transfers, synchronization barriers), so each
    # additional GPU contributes less than its raw free memory. The first GPU
    # keeps its full capacity (no cross-device overhead). 0.85 was calibrated
    # empirically on 2-8 GPU setups with NVLink and PCIe topologies -- the
    # 15% discount accounts for NCCL buffers (~2-5% of VRAM), pipeline bubble
    # overhead, and memory fragmentation from non-uniform shard sizes.
    multi_gpu_overhead = 0.85

    # Per-GPU check: activations don't shard, so each GPU needs its weight
    # shard + full activation cost. Use precomputed min_per_gpu_N values.
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

    # Use only GPUs with verified VRAM data (from gpu_candidates, not raw devices)
    fallback_all = (
        [c["index"] for c in gpu_candidates] if gpu_candidates else parent_ids
    )
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
      - **Explicit** (``gpu_ids=[5, 6, 7]``): the caller chooses exact GPUs.
        All listed GPUs are used and the model is sharded across them via
        ``device_map="balanced"``, regardless of whether the model would fit
        on fewer GPUs.  IDs are validated against the parent-visible set.
      - **Auto** (``gpu_ids=None`` or ``[]``): ``auto_select_gpu_ids`` estimates
        VRAM requirements and picks the *minimum* number of GPUs needed,
        preferring GPUs with the most free memory.

    The returned ``gpu_ids`` list is later passed to ``get_device_map()`` which
    maps it to a Hugging Face ``device_map`` string, and to ``apply_gpu_ids()``
    in the worker subprocess which narrows ``CUDA_VISIBLE_DEVICES`` before any
    torch/CUDA initialisation.
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
    with a torch-based fallback for AMD ROCm and Intel XPU.
    Result is cached after the first call.
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
        # SMI tool unavailable or failed -- fall back to torch
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

    On ROCm, HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES take precedence
    over CUDA_VISIBLE_DEVICES; the helper mirrors the resolution logic in
    ``_get_parent_visible_gpu_spec`` so ``backend_cuda_visible_devices``
    reports the value that is actually narrowing the visible device set.
    """
    if IS_ROCM:
        return _get_parent_visible_gpu_spec().get("raw")
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def get_backend_visible_gpu_info() -> Dict[str, Any]:
    device = get_device()
    if device in (DeviceType.CUDA, DeviceType.XPU):
        parent_visible_ids = get_parent_visible_gpu_ids()
        # Try native SMI tool first (nvidia-smi for NVIDIA, skipped for ROCm)
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

        # Torch fallback (AMD ROCm, Intel XPU, nvidia-smi missing/failed)
        # When parent_visible_ids is empty (UUID/MIG mask), enumerate by
        # torch ordinal so the UI still shows devices.
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
    Falls back to physical count if the env var is unset or torch is
    unavailable.  Result is cached after the first call.
    """
    global _visible_gpu_count
    if _visible_gpu_count is not None:
        return _visible_gpu_count

    # Use _get_parent_visible_gpu_spec() which already handles
    # HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES on ROCm.
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

    # No visibility env var set -- try torch, fall back to physical count
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

    # Empty list means "no GPUs visible" -- treat the same as None
    # (inherit parent) to avoid setting CUDA_VISIBLE_DEVICES="" which
    # disables CUDA entirely and crashes downstream torch calls.
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) == 0:
        return

    global _visible_gpu_count

    if isinstance(gpu_ids, (list, tuple)):
        value = ",".join(str(g) for g in gpu_ids)
    else:
        value = str(gpu_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = value
    # Keep ROCm visibility env vars in sync so _get_parent_visible_gpu_spec()
    # picks up the narrowed set on AMD systems. Workers can call
    # apply_gpu_ids() before detect_hardware() runs (so IS_ROCM is still
    # its default False), so also mirror the selection whenever the
    # parent process already set a ROCm visibility variable -- that
    # way a downstream ROCm process inherits the narrowed mask even
    # before Studio's hardware detection has classified the host.
    _inherits_rocm_visibility = (
        "HIP_VISIBLE_DEVICES" in os.environ or "ROCR_VISIBLE_DEVICES" in os.environ
    )
    if IS_ROCM or _inherits_rocm_visibility:
        os.environ["HIP_VISIBLE_DEVICES"] = value
        os.environ["ROCR_VISIBLE_DEVICES"] = value
    _visible_gpu_count = None
    if IS_ROCM or _inherits_rocm_visibility:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s' (rocm)", value)
    else:
        logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s'", value)


def get_device_map(
    gpu_ids: Optional[list[int]] = None,
) -> str:
    """Return the Hugging Face ``device_map`` string for model loading.

    Returns ``"balanced"`` (shard evenly across GPUs) when:
      - ``gpu_ids`` explicitly lists >1 GPU, **or**
      - ``CUDA_VISIBLE_DEVICES`` uses UUID/MIG identifiers (non-numeric) and
        more than one GPU is visible (fallback: we cannot resolve numeric IDs,
        so we assume the caller intends multi-GPU).

    Returns ``"sequential"`` (single device) in all other cases, including
    non-CUDA backends (CPU, MLX).

    Callers should use ``prepare_gpu_selection()`` upstream to determine the
    ``gpu_ids`` list -- that function handles the smart auto-selection of the
    minimum number of GPUs needed for a given model.
    """
    device = get_device()
    if device == DeviceType.CUDA:
        multi_gpu = gpu_ids is not None and len(gpu_ids) > 1

        if not multi_gpu:
            # UUID/MIG masks cannot be split into numeric IDs, so if multiple
            # GPUs are visible we assume multi-GPU sharding is intended.
            parent_visible_spec = _get_parent_visible_gpu_spec()
            if (
                parent_visible_spec["numeric_ids"] is None
                and get_visible_gpu_count() > 1
            ):
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


def raise_if_offloaded(model, device_map: str, context: str = "Loading") -> None:
    """Raise ``ValueError`` if *model* has modules offloaded to CPU or disk."""
    offloaded = get_offloaded_device_map_entries(model)
    if not offloaded:
        return
    example = ", ".join(
        f"{name}={placement}" for name, placement in list(offloaded.items())[:5]
    )
    raise ValueError(
        f"{context} does not support models loaded with CPU or disk offload. "
        f"device_map='{device_map}' produced offloaded modules: {example}"
    )


def safe_num_proc(desired: Optional[int] = None) -> int:
    """
    Return a safe ``num_proc`` for ``dataset.map()`` calls.

    On Windows, always returns 1 because Python uses ``spawn`` instead of
    ``fork`` for multiprocessing -- the overhead of re-importing torch,
    transformers, unsloth etc. per worker is typically slower than
    single-process for normal dataset sizes.

    On multi-GPU machines (where multiple GPUs are *visible* to this
    process) the NVIDIA driver spawns extra background threads, making
    ``os.fork()`` prone to deadlocks when many workers are created.
    This helper caps ``num_proc`` to 4 on such machines.

    When ``CUDA_VISIBLE_DEVICES`` restricts to a single GPU, the cap
    does not apply.

    Args:
        desired: The num_proc you *want*. If None, auto-computes from
                 ``os.cpu_count()``.

    Returns:
        A safe integer ≥ 1.
    """
    import sys

    # Windows and macOS use 'spawn' for multiprocessing -- the overhead of
    # re-importing torch/transformers/unsloth per worker is typically slower
    # than single-process.
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

    Unlike ``safe_num_proc()``, this does NOT cap to 1 on macOS/Windows.
    Threads share the parent process address space and are unaffected by
    the ``spawn`` vs ``fork`` distinction.

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

    Returns ``None`` on spawn-based platforms (Windows, macOS) because
    ``datasets`` treats ``num_proc=1`` as multiprocessing (creates ``Pool(1)``).
    Only ``num_proc=None`` guarantees in-process execution.
    """
    import sys

    if sys.platform in ("win32", "darwin"):
        return None
    return safe_num_proc(desired)
