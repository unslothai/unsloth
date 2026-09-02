# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for utils/hardware and utils/utils: device detection, GPU memory, error formatting.

Passes on any platform (NVIDIA/CUDA, Apple Silicon/MLX, CPU-only). No ML framework
is imported at top level; tests needing torch/mlx internals skip when unavailable.
"""

import platform
import sys
import types
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
from utils.utils import format_error_message, is_hf_authentication_error


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

    @needs_torch
    def test_detect_survives_device0_probe_failure(self, capsys):
        # is_available() True but the device-0 name probe raises: startup must
        # still resolve CUDA rather than crash.
        with (
            patch("utils.hardware.hardware._has_torch", return_value = True),
            patch("torch.cuda.is_available", return_value = True),
            patch("torch.cuda.device_count", return_value = 1),
            patch("torch.cuda.get_device_properties", side_effect = RuntimeError("probe")),
        ):
            assert _reset_and_detect() == DeviceType.CUDA
        assert "<unavailable>" in capsys.readouterr().out

    @needs_mlx
    def test_returns_mlx_when_on_apple_silicon_with_mlx(self):
        with (
            patch("utils.hardware.hardware._has_torch", return_value = False),
            patch("utils.hardware.hardware.is_apple_silicon", return_value = True),
            patch("utils.hardware.hardware._has_mlx", return_value = True),
            patch("utils.hardware.hardware._has_usable_mlx_stack", return_value = True),
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

    @needs_torch
    def test_clears_mps_on_apple_silicon_without_mlx(self):
        """An Apple Silicon host with a broken MLX stack reports CPU, but diffusion and video
        still run on Metal, so the MPS allocator has to be released on that path too."""
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CPU),
            patch("utils.hardware.hardware.is_apple_silicon", return_value = True),
            patch("torch.mps.empty_cache") as mock_empty,
        ):
            clear_gpu_cache()
            mock_empty.assert_called_once()

    @needs_torch
    def test_does_not_clear_mps_on_a_non_apple_cpu_host(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CPU),
            patch("utils.hardware.hardware.is_apple_silicon", return_value = False),
            patch("torch.mps.empty_cache") as mock_empty,
        ):
            clear_gpu_cache()
            mock_empty.assert_not_called()


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
        # _backend_label swaps "cuda" for "rocm" on AMD hosts; elsewhere it
        # equals get_device().value.
        from utils.hardware.hardware import _backend_label
        result = get_gpu_memory_info()
        assert result["backend"] == _backend_label(get_device())

    # --- When a GPU IS available ---

    @pytest.mark.skipif(_actual_device() == "cpu", reason = "No GPU available on this machine")
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
            # Driver truth from a context-free SMI/sysfs probe: another process
            # and torch's cache leave only 9 of 16 GiB free.
            patch(
                "utils.hardware.hardware._context_free_cuda_memory_info",
                return_value = 9 * (1024**3),
            ),
            patch(
                "utils.hardware.hardware.trusted_mem_get_info",
                side_effect = AssertionError("native telemetry must avoid mem_get_info"),
            ),
        ):
            result = get_gpu_memory_info()

        assert result["available"] is True
        assert result["backend"] == "cuda"
        assert result["device_name"] == "NVIDIA Test GPU"
        assert abs(result["total_gb"] - 16.0) < 0.01
        assert abs(result["allocated_gb"] - 4.0) < 0.01
        assert abs(result["free_gb"] - 9.0) < 0.01
        assert abs(result["utilization_pct"] - 25.0) < 0.1

    @needs_torch
    def test_cuda_free_falls_back_to_reserved_when_probe_fails(self):
        mock_props = MagicMock()
        mock_props.total_memory = 16 * (1024**3)
        mock_props.name = "NVIDIA Test GPU"

        def _boom():
            raise RuntimeError("driver unavailable")

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("torch.cuda.current_device", return_value = 0),
            patch("torch.cuda.get_device_properties", return_value = mock_props),
            patch("torch.cuda.memory_allocated", return_value = 4 * (1024**3)),
            patch("torch.cuda.memory_reserved", return_value = 6 * (1024**3)),
            patch("utils.hardware.hardware._context_free_cuda_memory_info", return_value = None),
            patch("utils.hardware.hardware.trusted_mem_get_info", side_effect = _boom),
        ):
            result = get_gpu_memory_info()

        # Reserved includes allocated, so the fallback bound is 16 - 6, not
        # the old allocated-only 12.
        assert abs(result["free_gb"] - 10.0) < 0.01

    @needs_torch
    def test_rocm_apu_free_uses_the_matching_driver_total(self):
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024**3)
        mock_props.name = "AMD Radeon 8060S Graphics"

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.hardware.IS_ROCM", True),
            patch("torch.cuda.current_device", return_value = 0),
            patch("torch.cuda.get_device_properties", return_value = mock_props),
            patch("torch.cuda.memory_allocated", return_value = 1 * (1024**3)),
            patch("torch.cuda.memory_reserved", return_value = 2 * (1024**3)),
            patch("utils.hardware.hardware._rocm_props_total_is_carve_out", return_value = True),
            patch(
                "utils.hardware.hardware._context_free_cuda_memory_info",
                side_effect = AssertionError("an APU needs hipMemGetInfo's GTT total"),
            ),
            patch(
                "utils.hardware.hardware.trusted_mem_get_info",
                return_value = (98 * (1024**3), 100 * (1024**3)),
            ),
        ):
            result = get_gpu_memory_info()

        assert abs(result["total_gb"] - 100.0) < 0.01
        assert abs(result["free_gb"] - 98.0) < 0.01

    # --- XPU (Intel GPU) ---

    def _xpu_torch(self, mem_get_info):
        """A torch stub exposing only what the XPU branch touches."""
        props = types.SimpleNamespace(total_memory = 16 * (1024**3), name = "Intel Arc A770")
        xpu = types.SimpleNamespace(
            current_device = lambda: 0,
            get_device_properties = lambda _o: props,
            memory_allocated = lambda _o: 2 * (1024**3),
            memory_reserved = lambda _o: 3 * (1024**3),
        )
        if mem_get_info is not None:
            xpu.mem_get_info = mem_get_info
        return types.SimpleNamespace(xpu = xpu)

    def _xpu_result(self, monkeypatch, mem_get_info):
        monkeypatch.setitem(sys.modules, "torch", self._xpu_torch(mem_get_info))
        monkeypatch.setattr(_hw_module, "get_device", lambda: DeviceType.XPU)
        monkeypatch.setattr(_hw_module, "rocm_windows_free_is_untrusted", lambda: False)
        return get_gpu_memory_info()

    def test_xpu_free_comes_from_the_driver(self, monkeypatch):
        # 12 of 16 GiB free system-wide, against 2 GiB allocated by this process:
        # the old total - allocated would have claimed 14.
        result = self._xpu_result(monkeypatch, lambda _o: (12 * (1024**3), 16 * (1024**3)))
        assert abs(result["free_gb"] - 12.0) < 0.01
        assert abs(result["total_gb"] - 16.0) < 0.01

    def test_xpu_falls_back_to_reserved_when_the_probe_fails(self, monkeypatch):
        def _boom(_o):
            raise RuntimeError("level zero unavailable")

        result = self._xpu_result(monkeypatch, _boom)
        assert abs(result["free_gb"] - 13.0) < 0.01

    def test_xpu_falls_back_on_a_torch_without_mem_get_info(self, monkeypatch):
        # torch.xpu.mem_get_info is newer than the floor this backend supports,
        # so its absence must degrade, not raise.
        result = self._xpu_result(monkeypatch, None)
        assert abs(result["free_gb"] - 13.0) < 0.01

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

        with patch("utils.hardware.hardware.get_gpu_memory_info", return_value = fake_info):
            log_gpu_memory("unit-test")

        captured = capfd.readouterr()
        assert "unit-test" in captured.out
        assert "CUDA" in captured.out
        assert "FakeGPU" in captured.out

    def test_logs_cpu_fallback_when_no_gpu(self, capfd):
        fake_info = {"available": False, "backend": "cpu"}

        with patch("utils.hardware.hardware.get_gpu_memory_info", return_value = fake_info):
            log_gpu_memory("cpu-test")

        captured = capfd.readouterr()
        assert "No GPU available" in captured.out


# ========== CUDA_DEVICE_ORDER pinning ==========


class TestCudaDeviceOrder:
    """Importing the hardware module pins CUDA_DEVICE_ORDER=PCI_BUS_ID when unset,
    but setdefault keeps an explicit user override, so nvidia-smi indices, torch
    ordinals, and CUDA_VISIBLE_DEVICES agree on a mixed-GPU host."""

    @staticmethod
    def _order_after_fresh_import(preset):
        # Fresh interpreter so the module-level setdefault runs against a clean env.
        import os, subprocess, sys
        from pathlib import Path

        env = os.environ.copy()
        backend = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH", "")
        # Avoid a trailing os.pathsep (empty entry -> cwd on sys.path) when unset.
        env["PYTHONPATH"] = (backend + os.pathsep + existing) if existing else backend
        if preset is None:
            env.pop("CUDA_DEVICE_ORDER", None)
        else:
            env["CUDA_DEVICE_ORDER"] = preset
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, utils.hardware.hardware; print(os.environ.get('CUDA_DEVICE_ORDER'))",
            ],
            env = env,
            capture_output = True,
            text = True,
            check = True,
        )
        return out.stdout.strip().splitlines()[-1]

    def test_import_pins_pci_bus_id_when_unset(self):
        assert self._order_after_fresh_import(None) == "PCI_BUS_ID"

    def test_import_respects_explicit_user_override(self):
        assert self._order_after_fresh_import("FASTEST_FIRST") == "FASTEST_FIRST"


# ========== _print_cuda_device_list() ==========


class TestPrintCudaDeviceList:
    """The startup console lists every CUDA GPU with its index, not just
    device 0, so a multi-GPU host shows the full available set."""

    @needs_torch
    def test_lists_all_devices_when_multi_gpu(self, capsys):
        props = [
            MagicMock(name = "p0"),
            MagicMock(name = "p1"),
        ]
        props[0].name = "NVIDIA GeForce RTX 5090"
        props[1].name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        with (
            patch("torch.cuda.device_count", return_value = 2),
            patch("torch.cuda.get_device_properties", side_effect = lambda i: props[i]),
        ):
            _hw_module._print_cuda_device_list(is_rocm = False)
        out = capsys.readouterr().out
        assert "[0] NVIDIA GeForce RTX 5090" in out
        assert "[1] NVIDIA RTX PRO 6000 Blackwell Workstation Edition" in out
        assert "CUDA_DEVICE_ORDER=" in out

    @needs_torch
    def test_silent_on_single_gpu(self, capsys):
        with patch("torch.cuda.device_count", return_value = 1):
            _hw_module._print_cuda_device_list(is_rocm = False)
        assert capsys.readouterr().out == ""

    @needs_torch
    def test_never_raises_on_probe_failure(self, capsys):
        with patch("torch.cuda.device_count", side_effect = RuntimeError("no cuda")):
            _hw_module._print_cuda_device_list(is_rocm = False)
        assert capsys.readouterr().out == ""

    @needs_torch
    def test_rocm_label_omits_cuda_device_order(self, capsys):
        # CUDA_DEVICE_ORDER governs CUDA only, so the ROCm listing must not claim it.
        props = [MagicMock(), MagicMock()]
        props[0].name = "AMD Instinct MI300X"
        props[1].name = "AMD Instinct MI300X"
        with (
            patch("torch.cuda.device_count", return_value = 2),
            patch("torch.cuda.get_device_properties", side_effect = lambda i: props[i]),
        ):
            _hw_module._print_cuda_device_list(is_rocm = True)
        out = capsys.readouterr().out
        assert "ROCm devices (2):" in out
        assert "CUDA_DEVICE_ORDER" not in out
        assert "[0] AMD Instinct MI300X" in out


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

    def test_hf_authentication_error_follows_wrapped_401(self):
        response = type("Response", (), {"status_code": 401})()
        auth_error = Exception("request failed")
        auth_error.response = response
        wrapper = RuntimeError("model validation failed")
        wrapper.__cause__ = auth_error
        assert is_hf_authentication_error(wrapper) is True

    def test_hf_authentication_error_does_not_treat_429_as_invalid(self):
        response = type("Response", (), {"status_code": 429})()
        rate_error = Exception("too many requests")
        rate_error.response = response
        assert is_hf_authentication_error(rate_error) is False

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


class TestIsTlsVerificationError:
    def _make(self):
        from utils.utils import is_tls_verification_error
        return is_tls_verification_error

    def test_direct_ssl_error(self):
        import ssl
        assert self._make()(ssl.SSLCertVerificationError(1, "certificate verify failed")) is True

    def test_wrapped_two_cause_hops(self):
        import ssl

        inner = ssl.SSLCertVerificationError(1, "certificate verify failed")
        mid = ConnectionError("transport")
        mid.__cause__ = inner
        outer = RuntimeError("Connection failed")
        outer.__cause__ = mid
        assert self._make()(outer) is True

    def test_context_chain(self):
        import ssl

        inner = ssl.SSLCertVerificationError(1, "certificate verify failed")
        outer = RuntimeError("Connection failed")
        outer.__context__ = inner
        assert self._make()(outer) is True

    def test_cycle_is_safe(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert self._make()(a) is False

    def test_string_fallback(self):
        err = RuntimeError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        assert self._make()(err) is True

    def test_non_tls_error(self):
        assert self._make()(RuntimeError("connection refused")) is False
