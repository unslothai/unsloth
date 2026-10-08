# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama.cpp CUDA source build must load on a driver older than CUDA 12.4 (#12842).

ggml passes nvcc -compress-mode=size for toolkit >= 12.8, and nvcc documents that mode as
"not compatible with drivers released before CUDA Toolkit's 12.4 Release": the driver
rejects every kernel with "device kernel image is invalid". Both setup scripts must build
uncompressed kernels for a known driver below 12.4 and keep ggml's default otherwise.
Part one runs the decision helpers sliced out of setup.sh / setup.ps1; part two pins the
wiring into the CUDA cmake arguments.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH_TEXT = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
SETUP_PS1_TEXT = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

BASH = shutil.which("bash")
PWSH = shutil.which("pwsh")

# (driver CUDA version as the scripts parse it, needs uncompressed kernels)
CASES = [
    ("11.8", True),
    ("12.0", True),
    ("12.2", True),
    ("12.3", True),
    ("12.4", False),
    ("12.8", False),
    ("13.0", False),
    ("13.1", False),
    ("", False),
    ("unknown", False),
]


def _sh_function(name):
    start = SETUP_SH_TEXT.index(f"{name}() {{")
    end = SETUP_SH_TEXT.index("\n}\n", start) + len("\n}\n")
    return SETUP_SH_TEXT[start:end]


def _ps1_function(name):
    match = re.search(
        rf"^function {re.escape(name)} \{{.*?^\}}\n", SETUP_PS1_TEXT, flags = re.DOTALL | re.MULTILINE
    )
    assert match is not None, f"setup.ps1 function not found: {name}"
    return match.group(0)


@pytest.mark.skipif(BASH is None, reason = "a working bash is required")
@pytest.mark.parametrize(("driver", "expected"), CASES)
def test_setup_sh_decision(tmp_path, driver, expected):
    script = tmp_path / "fns.sh"
    script.write_text(
        _sh_function("_cuda_version_gt") + _sh_function("_cuda_driver_needs_uncompressed_fatbin"),
        encoding = "utf-8",
    )
    result = subprocess.run(
        [
            BASH,
            "-c",
            'set -u; . "$1"; if _cuda_driver_needs_uncompressed_fatbin "$2"; then echo yes; else echo no; fi',
            "_",
            str(script),
            driver,
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    assert result.stdout.strip() == ("yes" if expected else "no")


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is unavailable")
def test_setup_ps1_decision():
    function = _ps1_function("Test-CudaDriverNeedsUncompressedFatbin")
    calls = "".join(
        f"Write-Output ('{driver}=' + (Test-CudaDriverNeedsUncompressedFatbin -DriverMaxCuda '{driver}'))\n"
        for driver, _ in CASES
    )
    # $null is what the script holds when no driver was detected.
    calls += (
        "Write-Output ('null=' + (Test-CudaDriverNeedsUncompressedFatbin -DriverMaxCuda $null))\n"
    )
    result = run_pwsh(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Set-StrictMode -Version Latest\n" + function + calls,
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    lines = result.stdout.strip().splitlines()
    expected = [f"{driver}={expected}" for driver, expected in CASES] + ["null=False"]
    assert lines == expected


def test_setup_sh_wires_the_flag_into_the_cuda_build():
    on = SETUP_SH_TEXT.index('CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=')
    block = SETUP_SH_TEXT[on : SETUP_SH_TEXT.index("_BUILD_DESC=", on)]
    assert '_cuda_driver_needs_uncompressed_fatbin "$_DRIVER_MAX_CUDA"' in block
    assert "-DGGML_CUDA_COMPRESSION_MODE=none" in block
    # The driver version is read in the same branch, before the build arguments.
    assert SETUP_SH_TEXT.rindex('_DRIVER_MAX_CUDA="$(_cuda_driver_max_version)"', 0, on) > 0


def test_setup_ps1_wires_the_flag_into_the_cuda_build():
    on = SETUP_PS1_TEXT.index("$CmakeArgs += '-DGGML_CUDA=ON'")
    block = SETUP_PS1_TEXT[on : SETUP_PS1_TEXT.index('$CmakeArgs += "-DCMAKE_CUDA_COMPILER=', on)]
    assert "Test-CudaDriverNeedsUncompressedFatbin -DriverMaxCuda $script:DriverMaxCuda" in block
    assert "-DGGML_CUDA_COMPRESSION_MODE=none" in block
    # $DriverMaxCuda is local to Resolve-CudaToolkit; the build reads the published copy.
    # Its body holds column-0 braces, so slice to the publish block that ends it.
    start = SETUP_PS1_TEXT.index("function Resolve-CudaToolkit {")
    resolve = SETUP_PS1_TEXT[
        start : SETUP_PS1_TEXT.index("$script:CudaToolkitReady = $true\n}", start)
    ]
    assert "$script:DriverMaxCuda = $DriverMaxCuda" in resolve
    assert (
        "$script:DriverMaxCuda = $null"
        in SETUP_PS1_TEXT[: SETUP_PS1_TEXT.index("function Resolve-CudaToolkit")]
    )
