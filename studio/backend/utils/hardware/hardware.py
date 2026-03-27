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


def _parse_smi_value(raw: str):
    raw = raw.strip()
    if not raw or raw == "[N/A]":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ========== Device Enum ==========


class DeviceType(str, Enum):
    """Supported compute backends. Inherits from str so it serializes cleanly in JSON."""

    CUDA = "cuda"
    MLX = "mlx"
    CPU = "cpu"


# ========== Global State (set once by detect_hardware) ==========

DEVICE: Optional[DeviceType] = None
CHAT_ONLY: bool = True  # No CUDA GPU -> GGUF chat only (Mac, CPU-only, etc.)


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
    CHAT_ONLY = True  # reset -- only CUDA sets it to False

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


def get_visible_gpu_utilization() -> Dict[str, Any]:
    device = get_device()

    if device != DeviceType.CUDA:
        return {
            "available": False,
            "backend": device.value,
            "parent_visible_gpu_ids": [],
            "devices": [],
        }

    parent_visible_ids = get_parent_visible_gpu_ids()
    allowed_indices = set(parent_visible_ids)
    devices = []

    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,temperature.gpu,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            timeout = 5,
        )

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue

                idx = int(parts[0])
                if idx not in allowed_indices:
                    continue

                vram_used_mb = _parse_smi_value(parts[3])
                vram_total_mb = _parse_smi_value(parts[4])
                power_draw = _parse_smi_value(parts[5])
                power_limit = _parse_smi_value(parts[6])

                devices.append(
                    {
                        "index": idx,
                        "gpu_utilization_pct": _parse_smi_value(parts[1]),
                        "temperature_c": _parse_smi_value(parts[2]),
                        "vram_used_gb": round(vram_used_mb / 1024, 2)
                        if vram_used_mb is not None
                        else None,
                        "vram_total_gb": round(vram_total_mb / 1024, 2)
                        if vram_total_mb is not None
                        else None,
                        "vram_utilization_pct": round(
                            (vram_used_mb / vram_total_mb) * 100, 1
                        )
                        if vram_used_mb is not None
                        and vram_total_mb
                        and vram_total_mb > 0
                        else None,
                        "power_draw_w": power_draw,
                        "power_limit_w": power_limit,
                        "power_utilization_pct": round(
                            (power_draw / power_limit) * 100, 1
                        )
                        if power_draw is not None and power_limit and power_limit > 0
                        else None,
                    }
                )

    except FileNotFoundError:
        logger.debug("nvidia-smi not found for multi-GPU utilization query")
    except Exception as e:
        logger.warning(f"nvidia-smi multi-GPU query failed: {e}")

    return {
        "available": len(devices) > 0,
        "backend": device.value,
        "parent_visible_gpu_ids": parent_visible_ids,
        "devices": devices,
    }


# ========== Multi-GPU Detection & Safe num_proc ==========

_physical_gpu_count: Optional[int] = None
_visible_gpu_count: Optional[int] = None


def get_parent_visible_gpu_ids() -> list[int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        cuda_visible = cuda_visible.strip()
        if cuda_visible == "" or cuda_visible == "-1":
            return []
        try:
            return [
                int(value.strip()) for value in cuda_visible.split(",") if value.strip()
            ]
        except ValueError:
            pass

    return list(range(get_physical_gpu_count()))


def resolve_requested_gpu_ids(gpu_ids: Optional[list[int]]) -> list[int]:
    parent_visible_ids = get_parent_visible_gpu_ids()
    physical_gpu_count = get_physical_gpu_count()

    if gpu_ids is None:
        return parent_visible_ids

    requested_ids = list(gpu_ids)
    if len(requested_ids) == 0:
        raise ValueError(
            "Invalid gpu_ids []: gpu_ids must be a non-empty list. "
            f"Parent-visible GPUs: {parent_visible_ids}"
        )

    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: duplicate GPU IDs are not allowed. "
            f"Parent-visible GPUs: {parent_visible_ids}"
        )

    invalid_ids = [
        gpu_id for gpu_id in requested_ids if gpu_id < 0 or gpu_id >= physical_gpu_count
    ]
    if invalid_ids:
        raise ValueError(
            f"Invalid gpu_ids {requested_ids}: IDs must be unique physical GPU IDs "
            f"between 0 and {max(physical_gpu_count - 1, 0)}. "
            f"Rejected IDs: {invalid_ids}. Parent-visible GPUs: {parent_visible_ids}"
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
        logger.debug("Could not get safetensors metadata for '%s': %s", model_name, e)
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
        logger.debug("Could not load config for '%s': %s", model_name, e)
        return None


def _estimate_fp16_model_size_bytes_from_config(config) -> Optional[int]:
    text_config = getattr(config, "text_config", None) or config

    vocab_size = getattr(text_config, "vocab_size", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    intermediate_size = getattr(text_config, "intermediate_size", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    num_heads = getattr(text_config, "num_attention_heads", None)

    if isinstance(intermediate_size, (list, tuple)):
        intermediate_size = intermediate_size[0] if intermediate_size else None
    if intermediate_size is None and hidden_size is not None:
        intermediate_size = hidden_size * 4

    if not all(
        value is not None
        for value in (vocab_size, hidden_size, intermediate_size, num_layers, num_heads)
    ):
        return None
    if num_heads <= 0:
        return None

    num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
    kv_size = (hidden_size // num_heads) * num_kv_heads

    qkvo = (hidden_size + kv_size + kv_size + hidden_size) * hidden_size
    mlp = (hidden_size * intermediate_size) * 3
    layernorms = 2 * hidden_size
    embed_tokens = vocab_size * hidden_size
    lm_head = 0 if getattr(text_config, "tie_word_embeddings", True) else vocab_size * hidden_size

    total_elements = (qkvo + mlp + layernorms) * num_layers + embed_tokens + lm_head
    return int(total_elements * 2)


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

    return None, "unavailable"


def estimate_required_model_memory_gb(
    model_name: str,
    *,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
) -> tuple[Optional[float], Dict[str, Any]]:
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
    min_qlora_total_gb = 3.0

    if training_type is None:
        required_gb = model_size_gb + max(model_size_gb * 0.3, min_buffer_gb)
    elif training_type == "Full Finetuning":
        required_gb = model_size_gb * 6.0
    elif load_in_4bit:
        base_4bit_gb = model_size_gb / 4.0
        required_gb = max(
            base_4bit_gb + max(base_4bit_gb * 0.5, min_buffer_gb),
            min_qlora_total_gb,
        )
    else:
        required_gb = model_size_gb + max(model_size_gb * 0.3, min_buffer_gb)

    metadata["required_gb"] = round(required_gb, 3)
    metadata["min_buffer_gb"] = min_buffer_gb
    metadata["min_qlora_total_gb"] = min_qlora_total_gb
    return required_gb, metadata


def auto_select_gpu_ids(
    model_name: str,
    *,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
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
    )
    metadata.update(estimate_metadata)
    if required_gb is None:
        metadata["selection_mode"] = "unavailable"
        return None, metadata

    utilization = get_visible_gpu_utilization()
    devices = utilization.get("devices", [])
    if not devices:
        metadata["selection_mode"] = "fallback_all"
        return None, metadata

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
        return None, metadata

    ranked = sorted(gpu_candidates, key = lambda item: (-item["free_gb"], item["index"]))
    free_by_index = {item["index"]: item["free_gb"] for item in ranked}
    selected: list[int] = []
    usable_gb = 0.0

    for candidate in ranked:
        selected.append(candidate["index"])
        if len(selected) == 1:
            usable_gb = candidate["free_gb"]
        else:
            usable_gb = sum(free_by_index[gpu_id] * 0.8 for gpu_id in selected)

        if usable_gb >= required_gb:
            metadata["usable_gb"] = round(usable_gb, 3)
            metadata["selection_mode"] = "auto"
            metadata["selected_gpu_ids"] = selected
            return selected, metadata

    fallback_all = [device["index"] for device in devices]
    metadata["selection_mode"] = "fallback_all"
    metadata["usable_gb"] = round(
        sum(candidate["free_gb"] * 0.8 for candidate in ranked),
        3,
    )
    metadata["selected_gpu_ids"] = fallback_all
    return fallback_all, metadata


def prepare_gpu_selection(
    gpu_ids: Optional[list[int]],
    *,
    model_name: str,
    hf_token: Optional[str] = None,
    training_type: Optional[str] = None,
    load_in_4bit: bool = True,
) -> tuple[Optional[list[int]], Dict[str, Any]]:
    if gpu_ids is not None:
        resolved = resolve_requested_gpu_ids(gpu_ids)
        return resolved, {
            "selection_mode": "explicit",
            "selected_gpu_ids": resolved,
        }

    return auto_select_gpu_ids(
        model_name,
        hf_token = hf_token,
        training_type = training_type,
        load_in_4bit = load_in_4bit,
    )


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

    import os

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        # "" means zero GPUs, "0" means 1, "0,1,2" means 3
        cuda_visible = cuda_visible.strip()
        if cuda_visible == "" or cuda_visible == "-1":
            _visible_gpu_count = 0
        else:
            _visible_gpu_count = len([x for x in cuda_visible.split(",") if x.strip()])
        return _visible_gpu_count

    # CUDA_VISIBLE_DEVICES not set -- try torch, fall back to physical count
    try:
        import torch

        _visible_gpu_count = torch.cuda.device_count()
    except Exception:
        _visible_gpu_count = get_physical_gpu_count()

    return _visible_gpu_count


def apply_gpu_ids(gpu_ids) -> None:
    if gpu_ids is None:
        return

    global _visible_gpu_count

    if isinstance(gpu_ids, (list, tuple)):
        value = ",".join(str(g) for g in gpu_ids)
    else:
        value = str(gpu_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = value
    _visible_gpu_count = None
    logger.info("Applied gpu_ids: CUDA_VISIBLE_DEVICES='%s'", value)


def get_device_map(gpu_ids: Optional[list[int]] = None) -> str:
    device = get_device()
    if device == DeviceType.CUDA and gpu_ids is not None and len(gpu_ids) > 1:
        return "balanced_low_0"
    return "sequential"


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
    import os
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
    import os

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
