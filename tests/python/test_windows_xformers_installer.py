# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 must install the xFormers wheel built for the torch it installed.

Before this, install.ps1 never mentioned xFormers at all: it installed torch from a
cu126 / cu128 / cu130 index and then plain `unsloth`, whose `windows` extra pulled
xFormers from PyPI -- which publishes only the CUDA-12.8 flavour. On a cu130 host that
produced the NVIDIA QA report "xFormers was built for PyTorch 2.10.0+cu128 with CUDA
1208 (you have 2.10.0+cu130)", with memory-efficient attention silently disabled.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _source() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL | re.MULTILINE)
    assert match is not None, f"installer block not found: {pattern}"
    return match.group(0)


def _selector_harness() -> str:
    """`$script:XformersWheelVersions` + the two pure helpers, ready to dot-source."""
    source = _source()
    return "\n".join(
        (
            _extract(r"^    \$script:XformersWheelVersions = @\{.*?^    \}", source),
            _extract(r"^    function ConvertTo-TorchFlavorTag \{.*?^    \}", source),
            _extract(r"^    function Get-XformersWheelVersion \{.*?^    \}", source),
        )
    )


def _run_pwsh(script: str) -> str:
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
    )
    return result.stdout.strip()


# (torch.__version__, expected xFormers version or "" for "no wheel, install nothing").
# The live wheels behind each row were HEAD-verified on download.pytorch.org and their
# xformers/cpp_lib.json read back -- cu130/xformers-0.0.34 reports {"torch":
# "2.10.0+cu130"}, cu128/xformers-0.0.34 reports {"torch": "2.10.0+cu128"}.
SELECTION_CASES = [
    ("2.10.0+cu130", "0.0.34"),
    ("2.10.0+cu128", "0.0.34"),
    ("2.10.0+cu126", "0.0.34"),
    ("2.9.1+cu130", "0.0.33.post2"),
    ("2.9.1+cu128", "0.0.33.post2"),
    ("2.9.0+cu130", "0.0.33.post1"),
    ("2.9.0+cu126", "0.0.33.post1"),
    ("2.8.0+cu129", "0.0.32.post2"),
    ("2.7.1+cu128", "0.0.31.post1"),
    # No cu130 build of xFormers exists for torch 2.8 or earlier, and no cu118 /
    # cu124 win_amd64 build exists at all -- refuse rather than serve a neighbour.
    ("2.8.0+cu130", ""),
    ("2.9.0+cu118", ""),
    ("2.10.0+cu124", ""),
    # torch 2.11 has no xFormers wheel on any index; a 2.10 wheel will not load there.
    ("2.11.0+cu130", ""),
    # Non-CUDA and nightly builds must miss the table outright.
    ("2.10.0+cpu", ""),
    ("2.10.0+rocm6.4", ""),
    ("2.10.0.dev20260101+cu130", ""),
]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(("torch_version", "expected"), SELECTION_CASES)
def test_selector_picks_the_matching_wheel(torch_version: str, expected: str):
    out = _run_pwsh(
        f"{_selector_harness()}\n"
        f"$v = '{torch_version}'\n"
        "$tag = ConvertTo-TorchFlavorTag $v\n"
        "$sel = Get-XformersWheelVersion -TorchVersion $v -CudaTag $tag\n"
        "Write-Output \"[$sel]\"\n"
    )
    assert out == f"[{expected}]", f"{torch_version} selected {out}, expected [{expected}]"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_selector_refuses_a_missing_or_blank_input():
    out = _run_pwsh(
        f"{_selector_harness()}\n"
        "Write-Output \"[$(Get-XformersWheelVersion -TorchVersion '' -CudaTag 'cu130')]\"\n"
        "Write-Output \"[$(Get-XformersWheelVersion -TorchVersion '2.10.0+cu130' -CudaTag '')]\"\n"
    )
    assert out.splitlines() == ["[]", "[]"]


def test_installer_installs_xformers_from_the_torch_index():
    source = _source()
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", source
    )
    assert "--no-deps" in block, (
        "the xFormers wheel declares torch==<exact release>; without --no-deps uv can pull "
        "a PyPI (CUDA 12.8) torch over the CUDA build just installed"
    )
    assert "--reinstall-package xformers" in block, (
        "cu126/cu128/cu130 publish the SAME xformers version string, so a wrong-CUDA wheel "
        "is invisible to a version check and must be force-replaced"
    )
    assert "--default-index $_xfIndexUrl" in block
    assert 'xformers==$_xfVersion' in block
    assert "UNSLOTH_SKIP_XFORMERS" in block


def test_installer_never_installs_an_unpinned_xformers():
    """A bare `xformers` (or a floor) resolves to whatever the index serves newest -- on the
    cu130 index that is 0.0.35, a py39-none wheel with no compiled extension at all."""
    source = _source()
    for match in re.finditer(r'"xformers[^"]*"', source):
        spec = match.group(0)
        assert spec == '"xformers==$_xfVersion"', f"unpinned xFormers spec in install.ps1: {spec}"


def test_xformers_step_runs_after_the_torch_flavor_repair():
    """The repair can reinstall torch from a different index; selecting the wheel before it
    would pin against a torch build that is about to be replaced."""
    source = _source()
    repair = source.index("Enforce the installed torch flavor matches the detected GPU build")
    xformers = source.index("Pin xFormers to the wheel built for the torch")
    overlay = source.index("CI only: overlay a source checkout")
    assert repair < xformers < overlay


def test_xformers_step_is_skipped_for_no_torch_installs():
    block = _extract(
        r"    # ── Pin xFormers to the wheel built for the torch.*?^    \}\n", _source()
    )
    assert "if (-not $SkipTorch" in block
    # cu<digits> only: cpu / rocm / xpu torch has no xFormers wheel on any index.
    assert "'^cu\\d+$'" in block


def test_installed_build_probe_reads_cpp_lib_json():
    """The version string cannot distinguish cu128 from cu130, so the repair check has to read
    the build metadata xFormers itself reports in its error message."""
    block = _extract(r"    function Get-InstalledXformersBuild \{.*?^    \}", _source())
    assert "cpp_lib.json" in block
    assert "find_spec" in block
    # Reading the file rather than importing xformers keeps a mismatched .pyd from writing
    # its own warning into the probe output, and makes 0.0.35 (no cpp_lib.json) read as unbuilt.
    assert "import xformers" not in block
    # Invoke-BoundedPythonProbe interpolates the code into a double-quoted -c argument.
    assert '"' not in block.split("$code = ", 1)[1].split("\n", 1)[0].strip("'")
