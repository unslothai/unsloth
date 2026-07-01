# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the prebuilt sd-cli asset resolver (``install_sd_cpp_prebuilt``).

Pure: the host -> release-asset matrix is exercised against a fixed asset list
(a real stable-diffusion.cpp release), no network. The installer lives under
``studio/`` (not ``studio/backend``), so the test puts that dir on the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import zipfile  # noqa: E402

import pytest  # noqa: E402

from install_sd_cpp_prebuilt import (  # noqa: E402
    _safe_extractall,
    default_install_dir,
    resolve_release_asset,
)

# A real stable-diffusion.cpp latest-release asset list.
_ASSETS = [
    "cudart-sd-bin-win-cu12-x64.zip",
    "sd-master-8caa3f9-bin-Darwin-macOS-15.7.7-arm64.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.13.0.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.2.1.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64.zip",
    "sd-master-8caa3f9-bin-win-avx-x64.zip",
    "sd-master-8caa3f9-bin-win-avx2-x64.zip",
    "sd-master-8caa3f9-bin-win-avx512-x64.zip",
    "sd-master-8caa3f9-bin-win-cuda12-x64.zip",
    "sd-master-8caa3f9-bin-win-noavx-x64.zip",
    "sd-master-8caa3f9-bin-win-rocm-7.13.0-x64.zip",
    "sd-master-8caa3f9-bin-win-vulkan-x64.zip",
]


def _resolve(
    system,
    machine,
    accelerator = "auto",
):
    return resolve_release_asset(_ASSETS, system = system, machine = machine, accelerator = accelerator)


# ── macOS (the key Apple-Silicon target) ────────────────────────────────────


def test_macos_arm64_picks_darwin_arm64():
    assert _resolve("Darwin", "arm64") == "sd-master-8caa3f9-bin-Darwin-macOS-15.7.7-arm64.zip"
    # aarch64 spelling resolves the same
    assert _resolve("Darwin", "aarch64").startswith("sd-master") and "arm64" in _resolve(
        "Darwin", "aarch64"
    )


def test_macos_intel_has_no_prebuilt():
    # only an arm64 Darwin asset exists -> Intel Macs must build from source
    assert _resolve("Darwin", "x86_64") is None


# ── Linux (CPU is the default tier) ─────────────────────────────────────────


def test_linux_x86_64_auto_picks_plain_cpu_build():
    # the plain x86_64 zip, NOT a rocm/vulkan one
    assert _resolve("Linux", "x86_64") == "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64.zip"


def test_linux_vulkan_and_rocm_select_accelerator_builds():
    assert (
        _resolve("Linux", "x86_64", "vulkan")
        == "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"
    )
    assert "rocm" in _resolve("Linux", "x86_64", "rocm")


def test_linux_arm64_has_no_prebuilt():
    assert _resolve("Linux", "aarch64") is None


# ── Windows ─────────────────────────────────────────────────────────────────


def test_windows_auto_picks_avx2():
    assert _resolve("Windows", "AMD64") == "sd-master-8caa3f9-bin-win-avx2-x64.zip"


def test_windows_cuda_picks_cuda12():
    assert _resolve("Windows", "AMD64", "cuda") == "sd-master-8caa3f9-bin-win-cuda12-x64.zip"


def test_windows_vulkan_picks_vulkan():
    assert _resolve("Windows", "AMD64", "vulkan") == "sd-master-8caa3f9-bin-win-vulkan-x64.zip"


# ── cudart helper archive is never chosen as the engine ─────────────────────


def test_cudart_runtime_archive_never_selected():
    for accel in ("auto", "cuda", "vulkan", "rocm"):
        chosen = _resolve("Windows", "AMD64", accel)
        assert chosen is None or not chosen.startswith("cudart")


# ── install dir ─────────────────────────────────────────────────────────────


def test_default_install_dir_is_sibling_of_llama(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    d = default_install_dir()
    assert d.name == "stable-diffusion.cpp"
    assert d.parent.name == ".unsloth"


# ── safe extraction (Zip-Slip guard) ─────────────────────────────────────────


def test_safe_extractall_rejects_path_traversal(tmp_path):
    target = tmp_path / "install"
    target.mkdir()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sd-cli", b"ok")
        zf.writestr("../escape.txt", b"pwned")  # escapes the install dir
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(RuntimeError, match = "unsafe path"):
            _safe_extractall(zf, target)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extractall_extracts_normal_members(tmp_path):
    target = tmp_path / "install"
    target.mkdir()
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("build/bin/sd-cli", b"ok")
    with zipfile.ZipFile(archive) as zf:
        _safe_extractall(zf, target)
    assert (target / "build" / "bin" / "sd-cli").read_bytes() == b"ok"


def test_find_sd_cpp_binary_honors_studio_home(tmp_path, monkeypatch):
    # A binary installed under a custom Studio root must be discovered without also
    # setting UNSLOTH_SD_CPP_PATH (matches default_install_dir's env handling).
    from core.inference import sd_cpp_engine as eng

    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)
    studio_home = tmp_path / "studio_root"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    binary = tmp_path / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"x")
    assert eng.find_sd_cpp_binary() == str(binary)
