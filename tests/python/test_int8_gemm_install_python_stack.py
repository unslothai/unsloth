# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The optional int8 GEMM wheel step must not import torch where no wheel is published."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))
sys.path.insert(0, str(STUDIO_DIR / "backend"))

import install_python_stack as ips
import pytest

_HOSTS = {
    "linux_x86_64": ("linux", "x86_64"),
    "linux_aarch64": ("linux", "aarch64"),
    "win_amd64": ("win32", "AMD64"),
    "win_arm64": ("win32", "ARM64"),
}


def _run(host, probe_env = None):
    sys_platform, machine = _HOSTS[host]
    probe = mock.Mock(return_value = probe_env)
    install = mock.Mock(return_value = "installed")
    with (
        mock.patch.object(sys, "platform", sys_platform),
        mock.patch("platform.machine", return_value = machine),
        mock.patch.object(ips, "IS_WINDOWS", sys_platform == "win32"),
        mock.patch.object(ips, "IS_MACOS", False),
        mock.patch.object(ips, "NO_TORCH", False),
        mock.patch.object(ips, "_int8_gemm_install_disabled", return_value = False),
        mock.patch.object(ips, "_installed_distribution_version", return_value = None),
        mock.patch.object(ips, "probe_torch_wheel_env", probe),
        mock.patch.object(ips, "url_exists", return_value = True),
        mock.patch.object(ips, "install_prebuilt", install),
        mock.patch.object(ips, "_count_install_action"),
    ):
        ips._ensure_int8_gemm()
    return probe, install


@pytest.mark.parametrize("host", ["linux_aarch64", "win_arm64"])
def test_no_torch_probe_where_no_wheel_is_published(host):
    probe, install = _run(host)
    probe.assert_not_called()
    install.assert_not_called()


def test_windows_probe_is_bounded():
    probe, _ = _run("win_amd64")
    assert probe.call_args.kwargs["timeout"] == ips._FLASH_ATTN_IMPORT_PROBE_TIMEOUT


def test_linux_probe_keeps_no_timeout():
    probe, _ = _run("linux_x86_64")
    assert probe.call_args.kwargs.get("timeout") is None


@pytest.mark.parametrize(
    "host, leaf",
    [("linux_x86_64", "manylinux_2_28_x86_64"), ("win_amd64", "win_amd64")],
)
def test_nvidia_x86_64_still_installs_the_same_wheel(host, leaf):
    env = {"cuda_major": "12", "hip_version": "", "platform_tag": host}
    probe, install = _run(host, env)
    probe.assert_called_once()
    url = install.call_args.args[0]
    assert url.endswith(f"unsloth_studio_i8gemm-{ips.I8GEMM_VERSION}-py3-none-{leaf}.whl")


def test_rocm_torch_on_x86_64_installs_nothing():
    env = {"cuda_major": "", "hip_version": "6.4", "platform_tag": "linux_x86_64"}
    _, install = _run("linux_x86_64", env)
    install.assert_not_called()
