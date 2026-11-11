# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CUDA install with no cudart anywhere gets NVIDIA's runtime wheels; anything else gets none."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "studio"))

import install_llama_prebuilt as ip  # noqa: E402


def _install(tmp_path, asset, kind):
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"asset": asset, "install_kind": kind})
    )
    return tmp_path


@pytest.fixture(autouse = True)
def _no_host_runtime(monkeypatch, tmp_path_factory):
    site = tmp_path_factory.mktemp("site")
    monkeypatch.setattr(ip.sysconfig, "get_paths", lambda: {"purelib": str(site)})
    monkeypatch.delenv("CUDA_PATH", raising = False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path_factory.mktemp("nocuda")))
    monkeypatch.setattr(ip.ctypes.util, "find_library", lambda name: None)
    return site


@pytest.mark.parametrize(
    "asset, kind, wheels",
    [
        ("app-b1-windows-x64-cuda13-newer.zip", "windows-cuda", ("nvidia-cuda-runtime", "nvidia-cublas")),
        ("app-b1-windows-x64-cuda12-newer.zip", "windows-cuda", ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12==12.9.2.10")),
        ("app-b1-linux-x64-cuda12-newer.tar.gz", "linux-cuda", ("nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12")),
        ("app-b1-windows-x64-cpu.zip", "windows-cpu", ()),
        ("app-b1-windows-arm64-cuda13-portable.zip", "windows-arm64-cuda", ()),
    ],
)
def test_wheels_for_a_bare_install(tmp_path, asset, kind, wheels):
    assert ip._missing_cuda_runtime_wheels(_install(tmp_path, asset, kind)) == wheels


def test_bundled_cudart_needs_nothing(tmp_path):
    install = _install(tmp_path, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    (install / "build" / "bin" / "Release").mkdir(parents = True)
    (install / "build" / "bin" / "Release" / "cudart64_13.dll").touch()
    assert ip._missing_cuda_runtime_wheels(install) == ()


def test_torch_runtime_needs_nothing(tmp_path, _no_host_runtime):
    (_no_host_runtime / "torch" / "lib").mkdir(parents = True)
    (_no_host_runtime / "torch" / "lib" / "cudart64_12.dll").touch()
    install = _install(tmp_path, "app-b1-windows-x64-cuda12-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ()


def test_other_major_does_not_count(tmp_path, _no_host_runtime):
    (_no_host_runtime / "torch" / "lib").mkdir(parents = True)
    (_no_host_runtime / "torch" / "lib" / "cudart64_12.dll").touch()
    install = _install(tmp_path, "app-b1-windows-x64-cuda13-newer.zip", "windows-cuda")
    assert ip._missing_cuda_runtime_wheels(install) == ("nvidia-cuda-runtime", "nvidia-cublas")
