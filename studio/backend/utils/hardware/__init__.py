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
)

__all__ = [
    'DeviceType',
    'DEVICE',
    'detect_hardware',
    'get_device',
    'is_apple_silicon',
    'clear_gpu_cache',
    'get_gpu_memory_info',
    'log_gpu_memory',
]
