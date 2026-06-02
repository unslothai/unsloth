# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for utils/hardware and utils/utils — device detection, GPU memory, error formatting.

These tests are designed to pass on ANY platform:
  • NVIDIA GPU  (CUDA backend, requires torch)
  • Apple Silicon (MLX backend, requires mlx)
  • CPU-only     (no GPU at all)

No ML framework is imported at the top level.
Tests that need torch/mlx internals for mocking are skipped when unavailable.

Run with:
    cd studio/backend
    python -m pytest tests/test_utils.py -v
"""

import platform
from unittest.mock import patch, MagicMock

import pytest

# --- Conditional framework imports ---
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason = "PyTorch not installed")
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason = "MLX not installed")

from utils.hardware import (
    get_device,
    detect_hardware,
    is_apple_silicon,
    clear_gpu_cache,
    get_gpu_memory_info,
    log_gpu_memory,
    DeviceType,
)
import utils.hardware.hardware as _hw_module
from utils.utils import format_error_message


# ========== Helpers ==========


def _actual_device() -> str:
    """Return the real device string for the current machine."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    if is_apple_silicon() and HAS_MLX:
        return "mlx"
    return "cpu"


def _reset_and_detect():
    """Reset the cached DEVICE global and re-run detection."""
    _hw_module.DEVICE = None
    return detect_hardware()


# ========== get_device() ==========


class TestGetDevice:
    """Tests for get_device() — should agree with the real hardware."""

    def setup_method(self):
        self._saved_device = _hw_module.DEVICE

    def teardown_method(self):
        _hw_module.DEVICE = self._saved_device

    def test_returns_valid_device_type(self):
        result = get_device()
        assert result in (DeviceType.CUDA, DeviceType.MLX, DeviceType.CPU)

    def test_matches_actual_hardware(self):
        assert get_device().value == _actual_device()

    # --- Mocked paths ---

    @needs_torch
    def test_returns_cuda_when_cuda_available(self):
        with (
            patch("utils.hardware.hardware._has_torch", return_value = True),
            patch("torch.cuda.is_available", return_value = True),
        ):
            assert _reset_and_detect() == DeviceType.CUDA

    @needs_mlx
    def test_returns_mlx_when_on_apple_silicon_with_mlx(self):
        with (
            patch("utils.hardware.hardware._has_torch", return_value = False),
            patch("utils.hardware.hardware.is_apple_silicon", return_value = True),
            patch("utils.hardware.hardware._has_mlx", return_value = True),
        ):
            assert _reset_and_detect() == DeviceType.MLX

    def test_returns_cpu_when_nothing_available(self):
        with (
            patch("utils.hardware.hardware._has_torch", return_value = False),
            patch("utils.hardware.hardware.is_apple_silicon", return_value = False),
            patch("utils.hardware.hardware._has_mlx", return_value = False),
        ):
            assert _reset_and_detect() == DeviceType.CPU


# ========== is_apple_silicon() ==========


class TestIsAppleSilicon:
    def test_returns_bool(self):
        assert isinstance(is_apple_silicon(), bool)

    def test_true_on_darwin_arm64(self):
        with patch("utils.hardware.hardware.platform") as mock_plat:
            mock_plat.system.return_value = "Darwin"
            mock_plat.machine.return_value = "arm64"
            assert is_apple_silicon() is True

    def test_false_on_linux_x86(self):
        with patch("utils.hardware.hardware.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            mock_plat.machine.return_value = "x86_64"
            assert is_apple_silicon() is False

    def test_false_on_darwin_x86(self):
        """Intel Mac should return False."""
        with patch("utils.hardware.hardware.platform") as mock_plat:
            mock_plat.system.return_value = "Darwin"
            mock_plat.machine.return_value = "x86_64"
            assert is_apple_silicon() is False


# ========== clear_gpu_cache() ==========


class TestClearGpuCache:
    """clear_gpu_cache() must never raise, regardless of platform."""

    def test_does_not_raise(self):
        clear_gpu_cache()

    @needs_torch
    def test_calls_cuda_cache_when_cuda(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("torch.cuda.empty_cache") as mock_empty,
            patch("torch.cuda.ipc_collect") as mock_ipc,
        ):
            clear_gpu_cache()
            mock_empty.assert_called_once()
            mock_ipc.assert_called_once()

    @needs_mlx
    def test_mlx_does_not_raise(self):
        """MLX cache clear is a no-op — should just succeed."""
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.MLX):
            clear_gpu_cache()

    def test_noop_on_cpu(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.CPU):
            clear_gpu_cache()


# ========== get_gpu_memory_info() ==========


class TestGetGpuMemoryInfo:
    def test_returns_dict(self):
        result = get_gpu_memory_info()
        assert isinstance(result, dict)

    def test_has_available_key(self):
        assert "available" in get_gpu_memory_info()

    def test_has_backend_key(self):
        assert "backend" in get_gpu_memory_info()

    def test_backend_matches_device(self):
        # The backend field uses _backend_label, which swaps "cuda" for
        # "rocm" when running on an AMD host (IS_ROCM=True) so the UI
        # can render the correct label. On CUDA / XPU / MLX / CPU hosts
        # it is equivalent to `get_device().value`.
        from utils.hardware.hardware import _backend_label

        result = get_gpu_memory_info()
        assert result["backend"] == _backend_label(get_device())

    # --- When a GPU IS available ---

    @pytest.mark.skipif(
        _actual_device() == "cpu", reason = "No GPU available on this machine"
    )
    def test_gpu_available_fields(self):
        result = get_gpu_memory_info()
        assert result["available"] is True
        assert result["total_gb"] > 0
        assert result["allocated_gb"] >= 0
        assert result["free_gb"] >= 0
        assert 0 <= result["utilization_pct"] <= 100
        assert "device_name" in result

    # --- CUDA-specific mocked test ---

    @needs_torch
    def test_cuda_path_returns_correct_fields(self):
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)
        mock_props.name = "NVIDIA Test GPU"

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("torch.cuda.current_device", return_value = 0),
            patch("torch.cuda.get_device_properties", return_value = mock_props),
            patch("torch.cuda.memory_allocated", return_value = 4 * (1024**3)),
            patch("torch.cuda.memory_reserved", return_value = 6 * (1024**3)),
        ):
            result = get_gpu_memory_info()

        assert result["available"] is True
        assert result["backend"] == "cuda"
        assert result["device_name"] == "NVIDIA Test GPU"
        assert abs(result["total_gb"] - 16.0) < 0.01
        assert abs(result["allocated_gb"] - 4.0) < 0.01
        assert abs(result["free_gb"] - 12.0) < 0.01
        assert abs(result["utilization_pct"] - 25.0) < 0.1

    # --- MLX-specific mocked test ---

    @needs_mlx
    def test_mlx_path_returns_correct_fields(self):
        mock_psutil_mem = MagicMock()
        mock_psutil_mem.total = 32 * (1024**3)  # 32 GB unified

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_psutil_mem

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.MLX),
            patch.dict("sys.modules", {"psutil": mock_psutil}),
        ):
            result = get_gpu_memory_info()

        assert result["available"] is True
        assert result["backend"] == "mlx"
        assert "Apple Silicon" in result["device_name"]
        assert abs(result["total_gb"] - 32.0) < 0.01

    # --- CPU-only path ---

    def test_cpu_path_returns_unavailable(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.CPU):
            result = get_gpu_memory_info()
        assert result["available"] is False
        assert result["backend"] == "cpu"

    # --- Error resilience ---

    @needs_torch
    def test_cuda_error_returns_unavailable(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "torch.cuda.current_device",
                side_effect = RuntimeError("CUDA init failed"),
            ),
        ):
            result = get_gpu_memory_info()
        assert result["available"] is False
        assert "error" in result


# ========== log_gpu_memory() ==========


class TestLogGpuMemory:
    def test_does_not_raise(self):
        log_gpu_memory("test")

    def test_logs_gpu_info_when_available(self, capfd):
        fake_info = {
            "available": True,
            "backend": "cuda",
            "device_name": "FakeGPU",
            "allocated_gb": 2.0,
            "total_gb": 16.0,
            "utilization_pct": 12.5,
            "free_gb": 14.0,
        }

        with patch(
            "utils.hardware.hardware.get_gpu_memory_info", return_value = fake_info
        ):
            log_gpu_memory("unit-test")

        captured = capfd.readouterr()
        assert "unit-test" in captured.out
        assert "CUDA" in captured.out
        assert "FakeGPU" in captured.out

    def test_logs_cpu_fallback_when_no_gpu(self, capfd):
        fake_info = {"available": False, "backend": "cpu"}

        with patch(
            "utils.hardware.hardware.get_gpu_memory_info", return_value = fake_info
        ):
            log_gpu_memory("cpu-test")

        captured = capfd.readouterr()
        assert "No GPU available" in captured.out


# ========== format_error_message() ==========


class TestFormatErrorMessage:
    def test_not_found(self):
        err = Exception("Repository not found for unsloth/test")
        msg = format_error_message(err, "unsloth/test")
        assert "not found" in msg.lower()
        assert "test" in msg

    def test_unauthorized(self):
        err = Exception("401 Unauthorized")
        msg = format_error_message(err, "some/model")
        assert "authentication" in msg.lower() or "unauthorized" in msg.lower()

    def test_gated_model(self):
        err = Exception("Access to model requires authentication")
        msg = format_error_message(err, "meta/llama")
        assert "authentication" in msg.lower()

    def test_invalid_token(self):
        err = Exception("Invalid user token")
        msg = format_error_message(err, "any/model")
        assert "invalid" in msg.lower()

    # --- OOM on CUDA ---

    @needs_torch
    def test_cuda_oom(self):
        err = Exception("CUDA out of memory")
        with patch("utils.hardware.get_device", return_value = DeviceType.CUDA):
            msg = format_error_message(err, "big/model")
        assert "GPU" in msg
        assert "big/model" not in msg
        assert "model" in msg

    # --- OOM on MLX ---

    @needs_mlx
    def test_mlx_oom(self):
        err = Exception("MLX backend out of memory")
        with patch("utils.hardware.get_device", return_value = DeviceType.MLX):
            msg = format_error_message(err, "unsloth/huge-model")
        assert "Apple Silicon" in msg

    # --- OOM on CPU ---

    def test_cpu_oom(self):
        err = Exception("not enough memory to allocate")
        with patch("utils.hardware.get_device", return_value = DeviceType.CPU):
            msg = format_error_message(err, "any/model")
        assert "system" in msg.lower()

    # --- Generic fallback ---

    def test_generic_error(self):
        err = Exception("Something completely unexpected")
        msg = format_error_message(err, "any/model")
        assert msg == "Something completely unexpected"
