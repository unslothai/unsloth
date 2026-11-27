# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CUDA 13.5 driver (R630) on Windows on ARM: torch 2.15.0+cu135 passes the installer's driver
gate, and the llama.cpp arm64 CUDA bundle follows the newest 13.x minor the driver can run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from woa_ps_harness import (  # noqa: E402
    INSTALL_SRC,
    _ps_kv,
    _script,
    functions,
    requires_pwsh,
    slice_between,
    substep_collector,
)

_SPEC = importlib.util.spec_from_file_location(
    "woa_gating_install_llama_prebuilt", PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
)
LLAMA = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LLAMA
_SPEC.loader.exec_module(LLAMA)

R630_BANNER = (
    "| NVIDIA-SMI 630.46    Driver Version: 630.46    CUDA UMD Version: 13.5 |\\n"
    "| GPU  Name   ...                                                     |"
)


def _arm64_host(driver):
    return LLAMA.HostInfo(
        system = "Windows",
        machine = "ARM64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = "C:/Windows/System32/nvidia-smi.exe",
        driver_cuda_version = driver,
        compute_caps = ["121"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )


def _assets(*names):
    return {name: f"https://example.invalid/{name}" for name in names}


@pytest.mark.parametrize(
    "driver, names, expected",
    [
        ((13, 5), ("13.4", "13.5"), "13.5"),
        ((13, 5), ("13.4",), "13.4"),
        # Never above the driver: a 13.6 bundle would need the minor-version path.
        ((13, 5), ("13.4", "13.5", "13.6"), "13.5"),
        ((13, 4), ("13.4", "13.5"), "13.4"),
    ],
)
def test_arm64_bundle_follows_the_driver_minor(driver, names, expected):
    assets = _assets(*(f"llama-b9999-bin-win-cuda-{v}-arm64.zip" for v in names))
    # An x64 bundle of a newer minor must not leak into the arm64 answer.
    assets.update(_assets("llama-b9999-bin-win-cuda-13.9-x64.zip"))
    assert LLAMA._published_windows_cuda_runtime(assets, 13, driver, "arm64") == expected


def test_a_cuda_13_5_driver_selects_the_13_5_arm64_bundle(monkeypatch):
    monkeypatch.setattr(LLAMA, "detected_windows_runtime_lines", lambda: ([], {}))
    host = _arm64_host((13, 5))
    assert LLAMA.compatible_windows_runtime_lines(host) == ["cuda13", "cuda12"]
    assets = _assets(
        "llama-b9999-bin-win-cuda-13.4-arm64.zip",
        "llama-b9999-bin-win-cuda-13.5-arm64.zip",
    )
    attempts = LLAMA.windows_cuda_attempts(host, "b9999", assets, "cuda13", arch = "arm64")
    assert attempts and attempts[0].name == "llama-b9999-bin-win-cuda-13.5-arm64.zip"
    assert attempts[0].install_kind == "windows-arm64-cuda"


def test_cu135_torch_reads_as_cuda_13():
    assert LLAMA.runtime_line_from_cuda_version("13.5") == "cuda13"


@requires_pwsh
@pytest.mark.parametrize(
    "torch_version, driver, passes",
    [
        ("2.15.0+cu135", "@(13, 5)", True),
        ("2.15.0+cu135", "@(13, 4)", True),
        ("2.15.0+cu135", "@(12, 8)", False),
        ("2.14.0+cu134", "@(13, 5)", True),
    ],
)
def test_install_ps1_driver_gate_accepts_cu135_on_a_13_x_driver(torch_version, driver, passes):
    gate = slice_between(
        INSTALL_SRC,
        "$_woaDriver = Get-WoaDriverCudaVersion",
        "# torch alone is not enough",
    )
    script = _script(
        substep_collector(),
        f"function Get-WoaDriverCudaVersion {{ return {driver} }}",
        "function Remove-IndexUrlCredentials { param($u) return $u }",
        "function Probe {",
        "  $torchIndex = 'https://index.invalid'",
        f"  $_woaTorchVersion = '{torch_version}'",
        gate,
        "  return 'passed'",
        "}",
        "Write-Output ('GATE=' + [string](Probe))",
    )
    assert (_ps_kv(script)["GATE"] == "passed") is passes


@requires_pwsh
def test_install_ps1_reads_the_r630_banner_as_cuda_13_5():
    script = _script(
        functions(INSTALL_SRC, "Get-WoaDriverCudaVersion", "Get-WoaDriverCudaLeaf"),
        "function Get-WoaNvidiaSmiPath { return 'nvidia-smi.exe' }",
        f'function Invoke-NvidiaSmiBounded {{ param($exe, $a) return "{R630_BANNER}" }}',
        "$v = Get-WoaDriverCudaVersion",
        "Write-Output ('VERSION=' + $v[0] + '.' + $v[1])",
        "Write-Output ('LEAF=' + (Get-WoaDriverCudaLeaf))",
    )
    out = _ps_kv(script)
    assert out["VERSION"] == "13.5"
    # The download.pytorch.org probe leaf; the win_arm64 torch itself comes from the NVIDIA indexes.
    assert out["LEAF"] == "cu130"
