"""Tests for the optional FlashAttention installer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips


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
    def _import_check(self, code: int = 1):
        return subprocess.CompletedProcess(["python", "-c", "import flash_attn"], code)

    def test_prefers_exact_match_wheel(self):
        install_calls = []

        def fake_install_wheel(*args, **kwargs):
            install_calls.append((args, kwargs))
            return [("uv", subprocess.CompletedProcess(["uv"], 0, ""))]

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", False),
            mock.patch.object(
                ips,
                "probe_torch_wheel_env",
                return_value = {
                    "python_tag": "cp313",
                    "torch_mm": "2.10",
                    "cuda_major": "12",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                },
            ),
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(ips, "install_wheel", side_effect = fake_install_wheel),
            mock.patch("subprocess.run", return_value = self._import_check()),
        ):
            ips._ensure_flash_attn()

        assert len(install_calls) == 1
        args, kwargs = install_calls[0]
        assert args == (
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        )
        assert kwargs["python_executable"] == sys.executable
        assert kwargs["use_uv"] is True
        assert kwargs["uv_needs_system"] is False

    def test_uv_install_respects_system_flag(self):
        install_calls = []

        def fake_install_wheel(*args, **kwargs):
            install_calls.append((args, kwargs))
            return [("uv", subprocess.CompletedProcess(["uv"], 0, ""))]

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", True),
            mock.patch.object(
                ips,
                "probe_torch_wheel_env",
                return_value = {
                    "python_tag": "cp313",
                    "torch_mm": "2.10",
                    "cuda_major": "12",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                },
            ),
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(ips, "install_wheel", side_effect = fake_install_wheel),
            mock.patch("subprocess.run", return_value = self._import_check()),
        ):
            ips._ensure_flash_attn()

        assert len(install_calls) == 1
        _, kwargs = install_calls[0]
        assert kwargs["uv_needs_system"] is True

    def test_wheel_failure_warns_and_continues(self):
        step_messages: list[tuple[str, str]] = []
        printed_failures: list[str] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", False),
            mock.patch.object(
                ips,
                "probe_torch_wheel_env",
                return_value = {
                    "python_tag": "cp313",
                    "torch_mm": "2.10",
                    "cuda_major": "12",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                },
            ),
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(
                ips,
                "install_wheel",
                return_value = [
                    ("uv", subprocess.CompletedProcess(["uv"], 1, "uv wheel failed")),
                    (
                        "pip",
                        subprocess.CompletedProcess(["pip"], 1, "pip wheel failed"),
                    ),
                ],
            ),
            mock.patch.object(
                ips,
                "_print_optional_install_failure",
                side_effect = lambda label, result: printed_failures.append(label),
            ),
            mock.patch.object(ips, "_step", side_effect = fake_step),
            mock.patch("subprocess.run", return_value = self._import_check()),
        ):
            ips._ensure_flash_attn()

        assert printed_failures == [
            "Installing flash-attn prebuilt wheel with uv",
            "Installing flash-attn prebuilt wheel with pip",
        ]
        assert ("warning", "Continuing without flash-attn") in step_messages

    def test_wheel_missing_skips_install_at_setup_time(self):
        step_messages: list[tuple[str, str]] = []

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(
                ips,
                "probe_torch_wheel_env",
                return_value = {
                    "python_tag": "cp313",
                    "torch_mm": "2.10",
                    "cuda_major": "13",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                },
            ),
            mock.patch.object(ips, "url_exists", return_value = False),
            mock.patch.object(ips, "install_wheel") as mock_install_wheel,
            mock.patch.object(ips, "_step", side_effect = fake_step),
            mock.patch("subprocess.run", return_value = self._import_check()),
        ):
            ips._ensure_flash_attn()

        mock_install_wheel.assert_not_called()
        assert (
            "warning",
            "No published flash-attn prebuilt wheel found",
        ) in step_messages

    def test_skip_env_disables_setup_install(self):
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.dict(os.environ, {"UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL": "1"}),
            mock.patch.object(ips, "probe_torch_wheel_env") as mock_probe,
            mock.patch.object(ips, "install_wheel") as mock_install_wheel,
            mock.patch("subprocess.run", return_value = self._import_check()),
        ):
            ips._ensure_flash_attn()

        mock_probe.assert_not_called()
        mock_install_wheel.assert_not_called()


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
