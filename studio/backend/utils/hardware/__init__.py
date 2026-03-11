# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Hardware detection and GPU utilities
"""

from .hardware import (
    DeviceType,
    DEVICE,
    detect_hardware,
    get_device,
    is_apple_silicon,
    clear_gpu_cache,
    get_gpu_memory_info,
    log_gpu_memory,
    get_gpu_summary,
    get_package_versions,
    get_gpu_utilization,
    get_physical_gpu_count,
    safe_num_proc,
)

__all__ = [
    "DeviceType",
    "DEVICE",
    "detect_hardware",
    "get_device",
    "is_apple_silicon",
    "clear_gpu_cache",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_gpu_summary",
    "get_package_versions",
    "get_gpu_utilization",
    "get_physical_gpu_count",
    "safe_num_proc",
]
