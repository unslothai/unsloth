# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import subprocess
import sys
from unittest import mock

from utils import wheel_utils
from utils.native_path_leases import LEASE_SECRET_ENV
from utils.wheel_utils import KernelPackageSpec


def _spec() -> KernelPackageSpec:
    return KernelPackageSpec(
        import_name = "definitely_missing_kernel_for_test",
        display_name = "test-kernel",
        pypi_spec = "test-kernel==1.0.0",
        filename_prefix = "test_kernel",
        package_version = "1.0.0",
        release_tag = "v1.0.0",
        release_base_url = "https://example.com/releases/download",
    )


def _env() -> dict[str, str]:
    return {
        "python_tag": "cp313",
        "torch_mm": "2.10",
        "cuda_major": "12",
        "cxx11abi": "TRUE",
        "platform_tag": "linux_x86_64",
    }


def _patch_missing_package(monkeypatch):
    monkeypatch.setattr(wheel_utils, "_package_is_importable", lambda name: False)
    monkeypatch.setattr(wheel_utils, "probe_torch_wheel_env", lambda timeout = 30: _env())


def test_direct_wheel_success_returns_true_and_skips_pypi(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            _spec(),
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )

    assert len(calls) == 1
    assert calls[0][:4] == ["uv", "pip", "install", "--python"]
    assert calls[0][-1].endswith(
        "test_kernel-1.0.0+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )
    assert "test-kernel==1.0.0" not in calls[0]
    assert "--no-deps" not in calls[0]
    assert "--upgrade" not in calls[0]
    assert "--reinstall" not in calls[0]


def test_direct_wheel_lets_installer_resolve_wheel_dependencies(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = False,
            run = fake_run,
        )
        is True
    )

    assert calls[0][:5] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]
    assert "--no-deps" not in calls[0]
    assert "einops" not in calls[0]
    assert calls[0][-1].endswith(".whl")


def test_direct_wheel_pip_fallback_resolves_wheel_dependencies(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "uv":
            return subprocess.CompletedProcess(cmd, 1, "uv failed")
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = False,
            run = fake_run,
        )
        is True
    )

    assert calls[1][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" not in calls[1]
    assert "--upgrade" not in calls[1]
    assert "--force-reinstall" not in calls[1]
    assert "--ignore-installed" not in calls[1]
    assert "einops" not in calls[1]
    assert calls[1][-1].endswith(".whl")


def test_mamba_wheel_uses_only_direct_wheel_requirement(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.MAMBA_SSM_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = False,
            run = fake_run,
        )
        is True
    )

    assert calls[0][:5] == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]
    assert "--no-deps" not in calls[0]
    assert calls[0].count(calls[0][-1]) == 1
    assert calls[0][-1].endswith(".whl")


def test_missing_direct_wheel_without_fallback_returns_false(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: False)
    run_mock = mock.Mock()

    assert (
        wheel_utils.install_optional_kernel(
            _spec(),
            python_executable = sys.executable,
            use_uv = False,
            allow_pypi_fallback = False,
            run = run_mock,
        )
        is False
    )

    run_mock.assert_not_called()


def test_missing_direct_wheel_with_fallback_runs_pypi_install(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: False)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: None)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            _spec(),
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "test-kernel==1.0.0",
        ]
    ]


def test_flash_linear_attention_spec_is_pypi_only():
    assert wheel_utils.FLASH_LINEAR_ATTN_SPEC.build_wheel_url(_env()) is None


def test_flash_linear_attention_fallback_runs_plain_pip_install(monkeypatch):
    _patch_missing_package(monkeypatch)
    url_exists = mock.Mock(side_effect = AssertionError("url_exists called"))
    monkeypatch.setattr(wheel_utils, "url_exists", url_exists)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: None)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_LINEAR_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "flash-linear-attention==0.5.0",
        ]
    ]
    assert "--no-deps" not in calls[0]
    assert "--no-build-isolation" not in calls[0]
    assert "--no-cache-dir" not in calls[0]
    url_exists.assert_not_called()


def test_flash_linear_attention_fallback_runs_plain_uv_install(monkeypatch):
    _patch_missing_package(monkeypatch)
    url_exists = mock.Mock(side_effect = AssertionError("url_exists called"))
    monkeypatch.setattr(wheel_utils, "url_exists", url_exists)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_LINEAR_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )

    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "flash-linear-attention==0.5.0",
        ]
    ]
    assert "--no-deps" not in calls[0]
    assert "--no-build-isolation" not in calls[0]
    url_exists.assert_not_called()


def test_pypi_fallback_tries_pip_after_uv_failure(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: False)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "uv":
            return subprocess.CompletedProcess(cmd, 1, "uv failed")
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_LINEAR_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )

    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "flash-linear-attention==0.5.0",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "flash-linear-attention==0.5.0",
        ],
    ]


def test_already_importable_package_returns_true_without_install(monkeypatch):
    probe_mock = mock.Mock()
    monkeypatch.setattr(wheel_utils, "probe_torch_wheel_env", probe_mock)

    importable_spec = KernelPackageSpec(
        import_name = "sys",
        display_name = "sys",
        pypi_spec = "sys",
    )
    assert (
        wheel_utils.install_optional_kernel(
            importable_spec,
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = True,
            run = mock.Mock(),
        )
        is True
    )
    probe_mock.assert_not_called()


def test_direct_wheel_tries_uv_then_pip(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "uv":
            return subprocess.CompletedProcess(cmd, 1, "uv failed")
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            _spec(),
            python_executable = sys.executable,
            use_uv = True,
            allow_pypi_fallback = False,
            run = fake_run,
        )
        is True
    )

    assert calls[0][0] == "uv"
    assert calls[0][1:3] == ["pip", "install"]
    assert "--no-deps" not in calls[0]
    assert calls[1][:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" not in calls[1]
    assert calls[1][-1] == calls[0][-1]


def _hip_env() -> dict[str, str]:
    # ROCm env: hip_version set, cuda_major empty -> no wheel candidate.
    return {
        "python_tag": "cp313",
        "torch_mm": "2.10",
        "cuda_major": "",
        "cxx11abi": "TRUE",
        "platform_tag": "linux_x86_64",
        "hip_version": "6.2",
    }


def test_rocm_without_hipcc_skips_source_build(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(
        wheel_utils, "probe_torch_wheel_env", lambda timeout = 30: _hip_env()
    )
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: False)
    # No hipcc on PATH.
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: None)

    statuses: list[str] = []
    run_mock = mock.Mock()
    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = False,
            allow_pypi_fallback = True,
            run = run_mock,
            status = statuses.append,
        )
        is False
    )
    run_mock.assert_not_called()
    assert any("hipcc" in s.lower() for s in statuses)


def test_rocm_with_hipcc_injects_gcc_install_dir_into_subprocess_env(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(
        wheel_utils, "probe_torch_wheel_env", lambda timeout = 30: _hip_env()
    )
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: False)
    monkeypatch.setattr(
        wheel_utils.shutil,
        "which",
        lambda name: "/usr/bin/hipcc" if name == "hipcc" else None,
    )
    monkeypatch.setattr(
        wheel_utils,
        "_hipcc_gcc_install_dir",
        lambda: "/usr/lib/gcc/x86_64-linux-gnu/13",
    )
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, "")

    assert (
        wheel_utils.install_optional_kernel(
            wheel_utils.FLASH_ATTN_SPEC,
            python_executable = sys.executable,
            use_uv = False,
            allow_pypi_fallback = True,
            run = fake_run,
        )
        is True
    )
    assert captured["kwargs"]["timeout"] == 1800
    env = captured["kwargs"]["env"]
    assert (
        "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
        in env["HIPCC_COMPILE_FLAGS_APPEND"]
    )


def test_install_subprocess_env_strips_native_path_lease_secret(monkeypatch):
    _patch_missing_package(monkeypatch)
    monkeypatch.setattr(wheel_utils, "probe_torch_wheel_env", lambda timeout = 30: _env())
    monkeypatch.setattr(wheel_utils, "url_exists", lambda url: True)
    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: None)
    monkeypatch.setenv(LEASE_SECRET_ENV, "test-only-secret-value")

    seen_env: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    wheel_utils.install_optional_kernel(
        wheel_utils.FLASH_ATTN_SPEC,
        python_executable = sys.executable,
        use_uv = False,
        allow_pypi_fallback = False,
        run = fake_run,
    )
    assert seen_env, "subprocess.run must receive a non-empty env"
    assert LEASE_SECRET_ENV not in seen_env
