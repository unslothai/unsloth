"""Tests for the optional FlashAttention installer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips
from backend.utils import wheel_utils


def _smi_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, "")


class TestHasBlackwellGpu:
    def setup_method(self):
        wheel_utils.has_blackwell_gpu.cache_clear()

    def teardown_method(self):
        wheel_utils.has_blackwell_gpu.cache_clear()

    def test_returns_false_when_nvidia_smi_missing(self):
        with mock.patch.object(wheel_utils.shutil, "which", return_value = None):
            assert wheel_utils.has_blackwell_gpu() is False

    def test_returns_true_for_sm_100(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("10.0\n")
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is True

    def test_returns_true_for_sm_120(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("12.0\n")
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is True

    def test_returns_true_for_sm_121(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("12.1\n")
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is True

    def test_returns_false_for_sm_90(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("9.0\n")
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is False

    def test_returns_false_for_sm_89(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("8.9\n")
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is False

    def test_mixed_gpus_with_one_blackwell_returns_true(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                return_value = _smi_result("8.0\n10.0\n"),
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is True

    def test_returns_false_when_nvidia_smi_fails(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                return_value = _smi_result("", returncode = 1),
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is False

    def test_returns_false_on_subprocess_timeout(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                side_effect = subprocess.TimeoutExpired(cmd = "nvidia-smi", timeout = 10),
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is False

    def test_returns_false_on_malformed_output(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                return_value = _smi_result("not-a-number\n\n"),
            ),
        ):
            assert wheel_utils.has_blackwell_gpu() is False


class TestHasNvidiaGpu:
    def setup_method(self):
        wheel_utils.has_nvidia_gpu.cache_clear()

    def teardown_method(self):
        wheel_utils.has_nvidia_gpu.cache_clear()

    def test_returns_false_when_nvidia_smi_missing_and_no_torch_cuda(self):
        # nvidia-smi absent + torch fallback False -> False.
        with (
            mock.patch.object(wheel_utils.shutil, "which", return_value = None),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = False
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is False

    def test_returns_true_when_nvidia_smi_reports_gpu(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                return_value = _smi_result("NVIDIA H100 80GB HBM3\n"),
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is True

    def test_returns_false_when_nvidia_smi_returns_no_gpus_and_no_torch_cuda(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("")
            ),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = False
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is False

    def test_returns_false_when_nvidia_smi_fails_and_no_torch_cuda(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                return_value = _smi_result("", returncode = 1),
            ),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = False
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is False

    def test_returns_false_on_subprocess_timeout_and_no_torch_cuda(self):
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess,
                "run",
                side_effect = subprocess.TimeoutExpired(cmd = "nvidia-smi", timeout = 10),
            ),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = False
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is False

    def test_falls_back_to_torch_when_nvidia_smi_missing(self):
        # Containerised CUDA host: no nvidia-smi but torch.cuda is_available.
        with (
            mock.patch.object(wheel_utils.shutil, "which", return_value = None),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = True
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is True

    def test_falls_back_to_torch_when_nvidia_smi_returns_empty(self):
        # nvidia-smi present but returns no GPUs (driver glitch): torch rescues.
        with (
            mock.patch.object(
                wheel_utils.shutil, "which", return_value = "/usr/bin/nvidia-smi"
            ),
            mock.patch.object(
                wheel_utils.subprocess, "run", return_value = _smi_result("")
            ),
            mock.patch.object(
                wheel_utils, "_torch_nvidia_cuda_available", return_value = True
            ),
        ):
            assert wheel_utils.has_nvidia_gpu() is True


class TestTorchNvidiaCudaAvailable:
    def test_returns_false_when_torch_missing(self):
        # Simulate setup-time call before torch is installed.
        import sys as _sys
        saved = _sys.modules.pop("torch", None)
        _sys.modules["torch"] = None  # forces ImportError on `import torch`
        try:
            assert wheel_utils._torch_nvidia_cuda_available() is False
        finally:
            if saved is not None:
                _sys.modules["torch"] = saved
            else:
                _sys.modules.pop("torch", None)

    def test_returns_false_on_rocm_torch(self):
        fake_torch = mock.MagicMock()
        fake_torch.version.hip = "6.2"
        fake_torch.cuda.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            assert wheel_utils._torch_nvidia_cuda_available() is False

    def test_returns_true_on_nvidia_cuda_torch(self):
        fake_torch = mock.MagicMock()
        fake_torch.version.hip = None
        fake_torch.cuda.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            assert wheel_utils._torch_nvidia_cuda_available() is True

    def test_returns_false_when_cuda_unavailable(self):
        fake_torch = mock.MagicMock()
        fake_torch.version.hip = None
        fake_torch.cuda.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            assert wheel_utils._torch_nvidia_cuda_available() is False


class TestFlashAttnWheelSelection:
    def test_torch_210_maps_to_v281(self):
        assert ips._select_flash_attn_version("2.10") == "2.8.1"

    def test_torch_29_maps_to_v283(self):
        assert ips._select_flash_attn_version("2.9") == "2.8.3"

    def test_unsupported_torch_has_no_wheel_mapping(self):
        assert ips._select_flash_attn_version("2.11") is None

    def test_exact_wheel_url_uses_full_env_tuple(self):
        url = ips._build_flash_attn_wheel_url(
            {
                "python_tag": "cp313",
                "torch_mm": "2.10",
                "cuda_major": "12",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            }
        )
        assert url is not None
        assert "v2.8.1" in url
        assert (
            "flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
            in url
        )

    def test_missing_cuda_major_disables_wheel_lookup(self):
        assert (
            ips._build_flash_attn_wheel_url(
                {
                    "python_tag": "cp313",
                    "torch_mm": "2.10",
                    "cuda_major": "",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                }
            )
            is None
        )


class TestEnsureFlashAttn:
    def test_setup_calls_generic_flash_attn_installer_without_pypi_fallback(self):
        install_mock = mock.Mock(return_value = False)
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", True),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = False),
            mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = True),
            mock.patch.object(ips, "install_optional_kernel", install_mock),
        ):
            ips._ensure_flash_attn()

        install_mock.assert_called_once()
        args, kwargs = install_mock.call_args
        assert args == (ips.FLASH_ATTN_SPEC,)
        assert kwargs["python_executable"] == sys.executable
        assert kwargs["use_uv"] is True
        assert kwargs["uv_needs_system"] is True
        assert kwargs["allow_pypi_fallback"] is False
        assert callable(kwargs["status"])

    def test_setup_skips_generic_install_when_no_torch(self):
        with (
            mock.patch.object(ips, "NO_TORCH", True),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()

    def test_setup_skips_generic_install_on_macos(self):
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", True),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = False),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()

    def test_setup_skips_generic_install_on_windows(self):
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", True),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = False),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()

    def test_skip_env_disables_setup_install(self):
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.dict(os.environ, {"UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL": "1"}),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()

    def test_blackwell_gpu_skips_install_with_warning(self):
        step_messages: list[tuple[str, str]] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = True),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
            mock.patch.object(ips, "_step", side_effect = fake_step),
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()
        assert any(
            label == "warning" and "Blackwell" in msg for label, msg in step_messages
        )

    def test_blackwell_gpu_on_windows_emits_blackwell_warning(self):
        step_messages: list[tuple[str, str]] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", True),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = True),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
            mock.patch.object(ips, "_step", side_effect = fake_step),
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()
        assert any(
            label == "warning" and "Blackwell" in msg for label, msg in step_messages
        )

    def test_setup_skips_install_without_nvidia_gpu(self):
        # AMD/Intel/CPU Linux: warn and skip, no install_optional_kernel call.
        step_messages: list[tuple[str, str]] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = False),
            mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = False),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
            mock.patch.object(ips, "_step", side_effect = fake_step),
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()
        assert any(
            label == "warning" and "no NVIDIA GPU" in msg
            for label, msg in step_messages
        )

    def test_non_blackwell_windows_does_not_emit_blackwell_warning(self):
        step_messages: list[tuple[str, str]] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", True),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "has_blackwell_gpu", return_value = False),
            mock.patch.object(ips, "install_optional_kernel") as install_mock,
            mock.patch.object(ips, "_step", side_effect = fake_step),
        ):
            ips._ensure_flash_attn()

        install_mock.assert_not_called()
        assert not any("Blackwell" in msg for _, msg in step_messages)


class TestInstallPythonStackFlashAttnIntegration:
    def _run_install(self, *, no_torch: bool, is_macos: bool, is_windows: bool) -> int:
        flash_attn_calls = 0

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        def count_flash_attn():
            nonlocal flash_attn_calls
            flash_attn_calls += 1

        with (
            mock.patch.object(ips, "NO_TORCH", no_torch),
            mock.patch.object(ips, "IS_MACOS", is_macos),
            mock.patch.object(ips, "IS_WINDOWS", is_windows),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", False),
            mock.patch.object(ips, "VERBOSE", False),
            mock.patch.object(ips, "_bootstrap_uv", return_value = True),
            mock.patch.object(ips, "_ensure_flash_attn", side_effect = count_flash_attn),
            mock.patch("subprocess.run", side_effect = fake_run),
            mock.patch.object(ips, "_has_usable_nvidia_gpu", return_value = False),
            mock.patch.object(ips, "_has_rocm_gpu", return_value = False),
            mock.patch.object(
                ips, "LOCAL_DD_UNSTRUCTURED_PLUGIN", Path("/fake/plugin")
            ),
            mock.patch("pathlib.Path.is_dir", return_value = True),
            mock.patch("pathlib.Path.is_file", return_value = True),
            mock.patch.dict(os.environ, {"SKIP_STUDIO_BASE": "1"}, clear = False),
        ):
            ips.install_python_stack()

        return flash_attn_calls

    def test_linux_torch_install_calls_flash_attn_step(self):
        assert self._run_install(no_torch = False, is_macos = False, is_windows = False) == 1

    def test_no_torch_install_skips_flash_attn_step(self):
        assert self._run_install(no_torch = True, is_macos = False, is_windows = False) == 0

    def test_macos_install_skips_flash_attn_step(self):
        assert self._run_install(no_torch = False, is_macos = True, is_windows = False) == 0

    def test_windows_install_skips_flash_attn_step(self):
        assert self._run_install(no_torch = False, is_macos = False, is_windows = True) == 0
