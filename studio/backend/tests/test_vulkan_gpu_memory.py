# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for Vulkan GPU memory detection fallback.

Covers _get_gpu_free_memory (orchestrator) and _get_gpu_free_memory_vulkan
(vulkaninfo parser): single/multi GPU, multi-heap, nvidia-smi priority,
CUDA_VISIBLE_DEVICES filtering, and graceful failure paths.

Requires no GPU, network, or external libraries beyond pytest.
Cross-platform: Linux, macOS, Windows, WSL.
"""

import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test.  Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend

# ---------------------------------------------------------------------------
# Vulkaninfo output fixtures
# ---------------------------------------------------------------------------

VULKANINFO_SINGLE_GPU_SINGLE_HEAP = """\
GPU0:
\tdeviceName        = AMD Radeon RX 5700 XT
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920 (0x7fbd40000) (31.93 GiB)
\t\tbudget = 33484447744 (0x7cbd42000) (31.18 GiB)
\t\tusage  = 200704 (0x00031000) (196.00 KiB)
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 8573157376 (0x1ff000000) (7.98 GiB)
\t\tbudget = 6441484288 (0x17ff14000) (6.00 GiB)
\t\tusage  = 0 (0x00000000) (0.00 B)
\t\tflags: count = 2
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\t\t\tMEMORY_HEAP_MULTI_INSTANCE_BIT
"""

VULKANINFO_SINGLE_GPU_MULTI_HEAP = """\
GPU0:
\tdeviceName        = AMD Radeon RX 7900 XTX
memoryHeaps: count = 3
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 8573157376
\t\tbudget = 6441484288
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
\tmemoryHeaps[2]:
\t\tsize   = 268435456
\t\tbudget = 268435456
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
"""

VULKANINFO_MULTI_GPU = """\
GPU0:
\tdeviceName        = AMD Radeon RX 5700 XT
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 8573157376
\t\tbudget = 6441484288
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
GPU1:
\tdeviceName        = AMD Radeon RX 6800
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 17179869184
\t\tbudget = 16777216000
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
"""

VULKANINFO_MULTI_GPU_INLINE_NAME = """\
GPU0: AMD Radeon RX 5700 XT
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 8573157376
\t\tbudget = 6441484288
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
GPU1: NVIDIA GeForce RTX 3090
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 25769803776
\t\tbudget = 25200000000
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
"""

VULKANINFO_NO_DEVICE_LOCAL = """\
GPU0:
\tdeviceName        = llvmpipe (LLVM 15.0.0, 256 bits)
memoryHeaps: count = 1
\tmemoryHeaps[0]:
\t\tsize   = 2147483648
\t\tbudget = 2147483648
\t\tflags:
\t\t\tNone
"""

VULKANINFO_NO_GPU_HEADERS = """\
memoryHeaps: count = 2
\tmemoryHeaps[0]:
\t\tsize   = 34289745920
\t\tbudget = 33484447744
\t\tflags:
\t\t\tNone
\tmemoryHeaps[1]:
\t\tsize   = 8573157376
\t\tbudget = 6441484288
\t\tflags: count = 1
\t\t\tMEMORY_HEAP_DEVICE_LOCAL_BIT
"""


# ---------------------------------------------------------------------------
# Tests: _get_gpu_free_memory_vulkan
# ---------------------------------------------------------------------------


class TestVulkanParser:
    """Tests for _get_gpu_free_memory_vulkan (vulkaninfo parser)."""

    def test_single_gpu_single_heap(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_SINGLE_GPU_SINGLE_HEAP
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert result == [(0, 6441484288 // (1024 * 1024))]

    def test_single_gpu_multi_heap_takes_max(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_SINGLE_GPU_MULTI_HEAP
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert len(result) == 1
        assert result[0][0] == 0
        # Should take the larger heap (6441484288), not the smaller (268435456)
        assert result[0][1] == 6441484288 // (1024 * 1024)

    def test_multi_gpu(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_MULTI_GPU
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert len(result) == 2
        assert result[0] == (0, 6441484288 // (1024 * 1024))
        assert result[1] == (1, 16777216000 // (1024 * 1024))

    def test_multi_gpu_inline_device_name(self):
        """GPU headers like 'GPU0: AMD Radeon...' (not just 'GPU0:')."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_MULTI_GPU_INLINE_NAME
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert len(result) == 2
        assert result[0] == (0, 6441484288 // (1024 * 1024))
        assert result[1] == (1, 25200000000 // (1024 * 1024))

    def test_no_device_local_heaps(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_NO_DEVICE_LOCAL
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert result == []

    def test_vulkaninfo_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode = 1, stdout = "")
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert result == []

    def test_no_gpu_headers_fallback(self):
        """When vulkaninfo has no GPU0:/GPU1: headers, treat as single device."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_NO_GPU_HEADERS
            )
            result = LlamaCppBackend._get_gpu_free_memory_vulkan()
        assert len(result) == 1
        assert result[0] == (0, 6441484288 // (1024 * 1024))


# ---------------------------------------------------------------------------
# Tests: _get_gpu_free_memory (orchestrator)
# ---------------------------------------------------------------------------

NVIDIA_SMI_OUTPUT = "0, 8000\n1, 4000\n"


class TestGpuMemoryOrchestrator:
    """Tests for _get_gpu_free_memory (nvidia-smi + vulkan fallback)."""

    def test_nvidia_smi_success(self):
        """nvidia-smi works -> return its results, don't call vulkan."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = NVIDIA_SMI_OUTPUT
            )
            result = LlamaCppBackend._get_gpu_free_memory()
        assert result == [(0, 8000), (1, 4000)]
        # Only one call (nvidia-smi), no vulkaninfo
        assert mock_run.call_count == 1

    def test_nvidia_smi_success_empty_no_fallback(self):
        """nvidia-smi succeeds but returns empty -> return [], no vulkan fallback."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode = 0, stdout = "")
            result = LlamaCppBackend._get_gpu_free_memory()
        assert result == []
        # Only nvidia-smi called, NOT vulkaninfo
        assert mock_run.call_count == 1

    def test_nvidia_smi_fail_vulkan_fallback(self):
        """nvidia-smi not found -> fall back to vulkaninfo."""

        def side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError("nvidia-smi not found")
            return SimpleNamespace(
                returncode = 0, stdout = VULKANINFO_SINGLE_GPU_SINGLE_HEAP
            )

        with patch("subprocess.run", side_effect = side_effect):
            result = LlamaCppBackend._get_gpu_free_memory()
        assert len(result) == 1
        assert result[0][0] == 0

    def test_both_fail(self):
        """Both nvidia-smi and vulkaninfo fail -> return []."""

        def side_effect(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError("nvidia-smi not found")
            return SimpleNamespace(returncode = 1, stdout = "")

        with patch("subprocess.run", side_effect = side_effect):
            result = LlamaCppBackend._get_gpu_free_memory()
        assert result == []

    def test_cuda_visible_devices_filtering(self):
        """CUDA_VISIBLE_DEVICES filters nvidia-smi results."""
        with (
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}),
        ):
            mock_run.return_value = SimpleNamespace(
                returncode = 0, stdout = NVIDIA_SMI_OUTPUT
            )
            result = LlamaCppBackend._get_gpu_free_memory()
        assert result == [(1, 4000)]
