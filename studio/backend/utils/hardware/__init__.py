# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Hardware detection and GPU utilities
"""

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

__all__ = [
    "DeviceType",
    "DEVICE",
    "CHAT_ONLY",
    "IS_ROCM",
    "detect_hardware",
    "get_device",
    "is_apple_silicon",
    "clear_gpu_cache",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_gpu_summary",
    "get_package_versions",
    "get_gpu_utilization",
    "get_visible_gpu_utilization",
    "get_backend_visible_gpu_info",
    "get_physical_gpu_count",
    "get_visible_gpu_count",
    "get_parent_visible_gpu_ids",
    "resolve_requested_gpu_ids",
    "estimate_fp16_model_size_bytes",
    "estimate_required_model_memory_gb",
    "auto_select_gpu_ids",
    "prepare_gpu_selection",
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
    """Resolve IS_ROCM at access time so callers always see the live value
    after detect_hardware() runs (it flips the flag in hardware.py)."""
    if name == "IS_ROCM":
        return getattr(_hardware, "IS_ROCM")
    raise AttributeError(name)
