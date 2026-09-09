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
from utils import wheel_utils


class TestPrebuiltWheelTorchMapping:
    def test_torch_211_maps_to_torch210(self):
        assert wheel_utils.prebuilt_wheel_torch_mm("2.11") == "2.10"

    def test_torch_212_maps_to_torch210(self):
        assert wheel_utils.prebuilt_wheel_torch_mm("2.12") == "2.10"

    def test_other_versions_pass_through(self):
        # 2.13 stays unmapped on purpose: a torch minor only joins the reuse table once its wheels have actually been
        # measured.
        for torch_mm in ("2.9", "2.10", "2.13"):
            assert wheel_utils.prebuilt_wheel_torch_mm(torch_mm) == torch_mm

    def test_reuse_never_targets_a_pre_210_wheel(self):
        # torch broke extension ABI between 2.9 and 2.10, so the torch2.9 .so raises "undefined symbol" on 2.10+.
        assert set(wheel_utils._PREBUILT_WHEEL_TORCH_MM.values()) == {"2.10"}

    def test_direct_wheel_url_reuses_torch210_on_211(self):
        # causal-conv1d / mamba go through direct_wheel_url; torch 2.11 reuses the torch2.10 wheel filename just like
        # flash-attn does.
        url = wheel_utils.direct_wheel_url(
            filename_prefix = "causal_conv1d",
            package_version = "1.6.1",
            release_tag = "v1.6.1.post4",
            release_base_url = "https://example.test/download",
            env = {
                "python_tag": "cp313",
                "torch_mm": "2.11",
                "cuda_major": "13",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            },
        )
        assert url is not None
        assert "causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" in url

    def test_direct_wheel_url_reuses_torch210_on_212(self):
        url = wheel_utils.direct_wheel_url(
            filename_prefix = "mamba_ssm",
            package_version = "2.3.1",
            release_tag = "v2.3.1",
            release_base_url = "https://example.test/download",
            env = {
                "python_tag": "cp312",
                "torch_mm": "2.12",
                "cuda_major": "13",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            },
        )
        assert url is not None
        assert "mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" in url


class TestFlashAttnWheelSelection:
    def test_torch_210_maps_to_v281(self):
        # v2.8.1 is the newest release still publishing the full torch2.10 asset matrix (cu12 + cu13, cp312 + cp313,
        # x86_64 + aarch64).
        assert ips._select_flash_attn_version("2.10") == "2.8.1"

    def test_selected_version_is_never_a_post_release(self):
        # The v2.8.3.post1 respin dropped every torch2.10 asset and stops at torch2.9, whose .so will not load on
        # torch 2.10+. A future "just take the newest release" bump must fail here instead of shipping that.
        for torch_mm in ("2.4", "2.7", "2.9", "2.10"):
            version = ips._select_flash_attn_version(torch_mm)
            assert version is not None
            assert ".post" not in version

    def test_torch_29_maps_to_v283(self):
        assert ips._select_flash_attn_version("2.9") == "2.8.3"

    def test_torch_211_has_no_native_version_entry(self):
        # The raw version table has no torch2.11-tagged wheel;
        # the URL builder reuses the torch2.10 wheel instead (see test_torch_211_reuses_torch210_wheel).
        assert ips._select_flash_attn_version("2.11") is None

    def test_torch_211_reuses_torch210_wheel(self):
        url = ips._build_flash_attn_wheel_url(
            {
                "python_tag": "cp313",
                "torch_mm": "2.11",
                "cuda_major": "13",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            }
        )
        assert url is not None
        assert "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" in url

    def test_torch_212_reuses_torch210_wheel(self):
        url = ips._build_flash_attn_wheel_url(
            {
                "python_tag": "cp313",
                "torch_mm": "2.12",
                "cuda_major": "13",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            }
        )
        assert url is not None
        assert "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" in url

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
        assert "flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl" in url

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


class TestFlashAttnImportProbe:
    """The probe is bounded: a native extension can hang in its initialiser, not just fail."""

    def test_clean_exit_is_importable(self):
        with mock.patch(
            "subprocess.run",
            return_value = subprocess.CompletedProcess(["python"], 0),
        ):
            assert ips._flash_attn_importable() is True

    def test_non_zero_exit_is_not_importable(self):
        with mock.patch(
            "subprocess.run",
            return_value = subprocess.CompletedProcess(["python"], 1),
        ):
            assert ips._flash_attn_importable() is False

    def test_a_hung_import_is_not_importable(self):
        with mock.patch(
            "subprocess.run",
            side_effect = subprocess.TimeoutExpired(cmd = "python", timeout = 300),
        ):
            assert ips._flash_attn_importable() is False

    def test_the_probe_is_bounded(self):
        with mock.patch(
            "subprocess.run",
            return_value = subprocess.CompletedProcess(["python"], 0),
        ) as run:
            ips._flash_attn_importable()

        assert run.call_args.kwargs["timeout"] == ips._FLASH_ATTN_IMPORT_PROBE_TIMEOUT


class TestEnsureFlashAttn:
    def _import_check(self, code: int = 1):
        return subprocess.CompletedProcess(["python", "-c", "import flash_attn"], code)

    def _import_fails_removal_works(self, cmd, **_kwargs):
        """flash_attn never imports, but uninstalling it succeeds."""
        if "uninstall" in cmd:
            return subprocess.CompletedProcess(cmd, 0)
        return self._import_check()

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

    def test_wheel_that_does_not_import_is_not_trusted(self):
        """pip exits 0 on a wrong-arch/ABI wheel; the import is what decides.

        The #5420 / #6961 Blackwell shape: setup must report that rather than claim
        flash-attn is ready.
        """
        step_messages: list[tuple[str, str]] = []

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
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(
                ips,
                "install_wheel",
                return_value = [("uv", subprocess.CompletedProcess(["uv"], 0, ""))],
            ),
            mock.patch.object(
                ips,
                "_step",
                side_effect = lambda label, value, color_fn = None: step_messages.append(
                    (label, value)
                ),
            ),
            # Import never succeeds, before or after the install; the removal does.
            mock.patch("subprocess.run", side_effect = self._import_fails_removal_works),
        ):
            ips._ensure_flash_attn()

        assert (
            "warning",
            "flash-attn wheel installed but is not importable on this GPU; removed it",
        ) in step_messages
        assert ("warning", "Continuing without flash-attn") in step_messages

    def test_rejected_wheel_is_uninstalled(self):
        """Leaving it installed is not "continuing without flash-attn".

        unsloth/models/_utils.py finds it by metadata (_package_available) and then imports
        the native module in process, so a wheel that killed the probe kills training too.
        """
        step_messages: list[tuple[str, str]] = []
        removals: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if "uninstall" in cmd:
                removals.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0)
            return self._import_check()

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
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(
                ips,
                "install_wheel",
                return_value = [("uv", subprocess.CompletedProcess(["uv"], 0, ""))],
            ),
            mock.patch.object(
                ips,
                "_step",
                side_effect = lambda label, value, color_fn = None: step_messages.append(
                    (label, value)
                ),
            ),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        assert removals, "the rejected wheel must be uninstalled, not left in site-packages"
        assert any("flash-attn" in cmd for cmd in removals), removals
        assert ("warning", "Continuing without flash-attn") in step_messages

    def test_uninstall_targets_the_interpreter_install_wheel_used(self):
        """install_wheel passes --python as well as --system, and its pip fallback runs
        sys.executable directly, so --system alone would uninstall from the wrong Python."""
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", True),
            mock.patch.object(ips.shutil, "which", return_value = "/usr/bin/uv"),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            assert ips._remove_rejected_flash_attn() is True

        assert commands == [
            ["uv", "pip", "uninstall", "--system", "--python", sys.executable, "flash-attn"]
        ]

    def test_uninstall_targets_the_interpreter_without_system_mode(self):
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        with (
            mock.patch.object(ips, "USE_UV", True),
            mock.patch.object(ips, "UV_NEEDS_SYSTEM", False),
            mock.patch.object(ips.shutil, "which", return_value = "/usr/bin/uv"),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            assert ips._remove_rejected_flash_attn() is True

        assert commands == [["uv", "pip", "uninstall", "--python", sys.executable, "flash-attn"]]

    def test_a_failed_removal_is_not_reported_as_removed(self):
        """Still importable in process, so it must not read like a clean skip."""
        step_messages: list[tuple[str, str]] = []

        def fake_run(cmd, **kwargs):
            if "uninstall" in cmd:
                return subprocess.CompletedProcess(cmd, 1)
            return self._import_check()

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
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(
                ips,
                "install_wheel",
                return_value = [("uv", subprocess.CompletedProcess(["uv"], 0, ""))],
            ),
            mock.patch.object(
                ips,
                "_step",
                side_effect = lambda label, value, color_fn = None: step_messages.append(
                    (label, value)
                ),
            ),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        warnings = [value for _, value in step_messages]
        assert any("could not be removed" in value for value in warnings), warnings
        assert not any("removed it" in value for value in warnings), warnings

    def test_working_wheel_reports_no_warning(self):
        """The happy path stays silent: install exits 0 and the module imports."""
        step_messages: list[tuple[str, str]] = []
        import_calls: list[int] = []

        def fake_run(cmd, **kwargs):
            # First call is the pre-install check (missing), the second verifies the install.
            import_calls.append(1)
            return self._import_check(1 if len(import_calls) == 1 else 0)

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
            mock.patch.object(ips, "url_exists", return_value = True),
            mock.patch.object(
                ips,
                "install_wheel",
                return_value = [("uv", subprocess.CompletedProcess(["uv"], 0, ""))],
            ),
            mock.patch.object(
                ips,
                "_step",
                side_effect = lambda label, value, color_fn = None: step_messages.append(
                    (label, value)
                ),
            ),
            mock.patch("subprocess.run", side_effect = fake_run),
        ):
            ips._ensure_flash_attn()

        assert step_messages == []
        assert len(import_calls) == 2, "expected a verification import after the install"

    def test_wheel_failure_warns_and_continues(self):
        step_messages: list[tuple[str, str]] = []
        printed_failures: list[str] = []

        def fake_step(
            label: str,
            value: str,
            color_fn = None,
        ):
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

        def fake_step(
            label: str,
            value: str,
            color_fn = None,
        ):
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
        assert ("warning", "No published flash-attn prebuilt wheel found") in step_messages

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

    def test_windows_skips_install_without_probing(self):
        # flash-attn is Linux-only: on Windows the installer returns before probing the torch env or resolving a
        # wheel (no Windows wheels are published upstream).
        with (
            mock.patch.object(ips, "NO_TORCH", False),
            mock.patch.object(ips, "IS_WINDOWS", True),
            mock.patch.object(ips, "IS_MACOS", False),
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
            mock.patch.object(ips, "LOCAL_DD_UNSTRUCTURED_PLUGIN", Path("/fake/plugin")),
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
