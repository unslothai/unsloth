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

     def test_skip_env_var_disables_install(self):
        with mock.patch.dict(os.environ, {"UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL": "1"}):
            assert ips._flash_attn_install_disabled() is True

    def test_fallback_install_uses_pinned_version_when_mapped(self):
        assert ips._flash_attn_install_spec("2.8.1") == "flash-attn==2.8.1"

    def test_fallback_install_is_unpinned_when_version_unknown(self):
        assert ips._flash_attn_install_spec(None) == "flash-attn"


class TestEnsureFlashAttn:
    def _completed(
        self, cmd: list[str], code: int = 0, stdout: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, code, stdout)

    def _import_check(self, code: int = 1):
        return subprocess.CompletedProcess(["python", "-c", "import flash_attn"], code)

    def test_prefers_exact_match_wheel(self):
        calls: list[tuple[str, str, bool, bool]] = []

        def fake_install_wheel(
            wheel_url: str,
            *,
            python_executable: str,
            use_uv: bool,
            uv_needs_system: bool,
        ):
            calls.append((wheel_url, python_executable, use_uv, uv_needs_system))
            return [("uv", self._completed(["uv"], 0))]

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

        assert len(calls) == 1
        assert calls[0][0].endswith(
            "flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )
        assert calls[0][1] == sys.executable
        assert calls[0][2:] == (True, False)

    def test_uv_install_respects_system_flag(self):
        calls: list[tuple[str, str, bool, bool]] = []

        def fake_install_wheel(
            wheel_url: str,
            *,
            python_executable: str,
            use_uv: bool,
            uv_needs_system: bool,
        ):
            calls.append((wheel_url, python_executable, use_uv, uv_needs_system))
            return [("uv", self._completed(["uv"], 0))]

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

        assert calls[0][1] == sys.executable
        assert calls[0][2:] == (True, True)

    def test_falls_back_to_pinned_source_install_when_wheel_missing(self):
        run_calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if cmd[:3] == [sys.executable, "-c", "import flash_attn"]:
                return self._import_check()
            return self._completed(list(cmd), 0)

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
                    "cuda_major": "13",
                    "cxx11abi": "TRUE",
                    "platform_tag": "linux_x86_64",
                },
            ),
            mock.patch.object(ips, "url_exists", return_value = False),
            mock.patch.object(ips, "install_wheel") as install_wheel,
            mock.patch("shutil.which", return_value = None),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        install_wheel.assert_not_called()
        assert run_calls[1] == [sys.executable, "-m", "pip", "install", "ninja"]
        assert run_calls[2] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-binary=flash-attn",
            "flash-attn==2.8.1",
        ]

    def test_uv_source_failure_falls_back_to_pip_source_install(self):
        run_calls: list[list[str]] = []
        printed_failures: list[str] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if cmd[:3] == [sys.executable, "-c", "import flash_attn"]:
                return self._import_check()
            if cmd[:3] == ["uv", "pip", "install"]:
                return self._completed(list(cmd), 1, "uv source failed")
            return self._completed(list(cmd), 0)

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", True),
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
            mock.patch.object(ips, "install_wheel") as install_wheel,
            mock.patch("shutil.which", return_value = "uv"),
            mock.patch.object(
                ips,
                "_print_optional_install_failure",
                side_effect = lambda label, result: printed_failures.append(label),
            ),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        install_wheel.assert_not_called()
        assert printed_failures == ["Installing flash-attn from source with uv"]
        assert run_calls[1][:6] == [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-build-isolation",
        ]
        assert run_calls[1][-3:] == ["--no-binary", "flash-attn", "flash-attn==2.8.1"]
        assert run_calls[2] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-binary=flash-attn",
            "flash-attn==2.8.1",
        ]

    def test_total_failure_warns_and_continues(self):
        run_calls: list[list[str]] = []
        step_messages: list[tuple[str, str]] = []

        def fake_run(cmd, **kwargs):
            run_calls.append(list(cmd))
            if cmd[:3] == [sys.executable, "-c", "import flash_attn"]:
                return self._import_check()
            return self._completed(list(cmd), 1, "install failed")

        def fake_step(label: str, value: str, color_fn = None):
            step_messages.append((label, value))

        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", False),
            mock.patch.object(ips, "IS_MACOS", False),
            mock.patch.object(ips, "USE_UV", False),
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
            mock.patch.object(ips, "install_wheel") as install_wheel,
            mock.patch("shutil.which", return_value = None),
            mock.patch.object(ips, "_step", side_effect = fake_step),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        install_wheel.assert_not_called()
        assert run_calls[1] == [sys.executable, "-m", "pip", "install", "ninja"]
        assert run_calls[2] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-binary=flash-attn",
            "flash-attn==2.8.1",
        ]
        assert ("warning", "Continuing without flash-attn") in step_messages

    def test_skip_env_var_avoids_install_work(self):
        with (
            mock.patch.dict(
                os.environ,
                {"UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL": "1"},
                clear = False,
            ),
            mock.patch("subprocess.run") as run,
            mock.patch.object(ips, "install_wheel") as install_wheel,
        ):
            ips._ensure_flash_attn()

        run.assert_not_called()
        install_wheel.assert_not_called()


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
            mock.patch.object(ips, "LOCAL_DD_UNSTRUCTURED_PLUGIN", Path("/fake/plugin")),
            mock.patch("pathlib.Path.is_dir", return_value = True),
            mock.patch("pathlib.Path.is_file", return_value = True),
            mock.patch.dict(os.environ, {"SKIP_STUDIO_BASE": "1"}, clear = False),
        ):
            ips.install_python_stack()

        return flash_attn_calls

    def test_linux_torch_install_calls_flash_attn_step(self):
        assert self._run_install(
            no_torch = False, is_macos = False, is_windows = False
        ) == 1

    def test_no_torch_install_skips_flash_attn_step(self):
        assert self._run_install(
            no_torch = True, is_macos = False, is_windows = False
        ) == 0

    def test_macos_install_skips_flash_attn_step(self):
        assert self._run_install(
            no_torch = False, is_macos = True, is_windows = False
        ) == 0

    def test_windows_install_skips_flash_attn_step(self):
        assert self._run_install(
            no_torch = False, is_macos = False, is_windows = True
        ) == 0
