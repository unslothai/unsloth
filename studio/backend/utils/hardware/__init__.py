# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hardware detection and GPU utilities."""

from typing import Optional

from . import hardware as _hardware
from .hardware import (
    DeviceType,
    DEVICE,
    CHAT_ONLY,
    detect_hardware,
    get_device,
    is_apple_silicon,
    clear_gpu_cache,
    get_gpu_memory_info,
    log_gpu_memory,
    get_gpu_summary,
    get_package_versions,
    get_gpu_utilization,
    get_visible_gpu_utilization,
    get_backend_visible_gpu_info,
    get_vulkan_inference_gpu_info,
    get_physical_gpu_count,
    get_visible_gpu_count,
    get_parent_visible_gpu_ids,
    resolve_requested_gpu_ids,
    estimate_fp16_model_size_bytes,
    estimate_required_model_memory_gb,
    auto_select_gpu_ids,
    prepare_gpu_selection,
    safe_num_proc,
    safe_thread_num_proc,
    dataset_map_num_proc,
    get_device_map,
    get_offloaded_device_map_entries,
    raise_if_offloaded,
    apply_gpu_ids,
)

from .vram_estimation import (
    ModelArchConfig,
    TrainingVramConfig,
    VramBreakdown,
    extract_arch_config,
    estimate_training_vram,
)


def ensure_hardware_detected(epoch: Optional[int] = None) -> DeviceType:
    """Detect once, from any thread; delegate so the live function always runs.

    Wrapper rather than re-export, like export_capability() below: a re-export is an
    unused module-level import, which scripts/verify_import_hoist.py flags. It has to
    carry the epoch: the warm calls this name, and a wrapper that dropped the argument
    would raise into _run_stage, which logs and moves on, leaving detection undone.
    """
    return _hardware.ensure_hardware_detected(epoch)


def start_background_detection() -> None:
    """Put detection on a daemon thread if nothing is running it yet."""
    _hardware.start_background_detection()


def export_capability() -> dict:
    """Return live export capability from the hardware module."""
    return _hardware.export_capability()


def get_torch_device_str() -> str:
    """Return the torch device string ("cuda", "xpu", "cpu") for the detected hardware."""
    return _hardware.get_torch_device_str()


__all__ = [
    "DeviceType",
    "DEVICE",
    "CHAT_ONLY",
    "IS_ROCM",
    "detect_hardware",
    "ensure_hardware_detected",
    "start_background_detection",
    "get_device",
    "export_capability",
    "is_apple_silicon",
    "clear_gpu_cache",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_gpu_summary",
    "get_package_versions",
    "get_gpu_utilization",
    "get_visible_gpu_utilization",
    "get_backend_visible_gpu_info",
    "get_vulkan_inference_gpu_info",
    "get_physical_gpu_count",
    "get_visible_gpu_count",
    "get_parent_visible_gpu_ids",
    "resolve_requested_gpu_ids",
    "estimate_fp16_model_size_bytes",
    "estimate_required_model_memory_gb",
    "auto_select_gpu_ids",
    "prepare_gpu_selection",
    "get_torch_device_str",
    "safe_num_proc",
    "safe_thread_num_proc",
    "dataset_map_num_proc",
    "get_device_map",
    "get_offloaded_device_map_entries",
    "raise_if_offloaded",
    "apply_gpu_ids",
    "ModelArchConfig",
    "TrainingVramConfig",
    "VramBreakdown",
    "extract_arch_config",
    "estimate_training_vram",
]


def __getattr__(name: str):
    """Resolve IS_ROCM lazily so callers see the live value detect_hardware()
    sets in hardware.py."""
    if name == "IS_ROCM":
        return getattr(_hardware, "IS_ROCM")
    raise AttributeError(name)
