# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama.cpp mismatch gate and the llama.cpp selector have to agree, or updates churn.

Before reinstalling, setup.ps1 deletes the existing llama.cpp tree when its recorded
`install_kind` is not in the set the gate considers correct for this host; then
install_llama_prebuilt.py picks a bundle. Nothing makes those two agree, and a disagreement
is permanent: every update deletes the tree and refetches the identical bundle, forever, on
a working host. The only relationship that matters is "the kind the selector installs is one
the gate accepts", so that is asserted directly, over the product of what either side
branches on, rather than the text of either.

Both churning combinations this found are on Windows ARM64 and invisible from an x64 box:

  * NVIDIA. The gate expected windows-cuda; on an ARM64 venv the selector installs
    windows-arm64. That is every Windows ARM64 machine with an NVIDIA GPU.
  * ROCm. No ROCm bundle exists for Windows ARM64 at all, so the selector falls through to
    the ARM64 CPU bundle while the gate still expected windows-rocm or windows-hip.

Offline. direct_upstream_release_plan takes a release dict, and the gate's own block is lifted
out of setup.ps1 and run with only its inputs replaced.
"""

from __future__ import annotations

import itertools
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
SETUP_SRC = SETUP_PS1.read_text(encoding = "utf-8")

sys.path.insert(0, str(REPO_ROOT / "studio"))

import install_llama_prebuilt as ip  # noqa: E402

PWSH = shutil.which("pwsh") or shutil.which("powershell")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "PowerShell is unavailable")

TAG = "b9334"

# Everything upstream publishes for Windows, so no row is starved of a candidate and a
# disagreement is a disagreement rather than a missing asset. Note what is NOT here and
# cannot be: there is no ROCm or CUDA bundle for Windows ARM64 upstream.
ASSET_NAMES = (
    f"llama-{TAG}-bin-win-cpu-arm64.zip",
    f"llama-{TAG}-bin-win-cpu-x64.zip",
    f"llama-{TAG}-bin-win-cuda-12.4-x64.zip",
    f"llama-{TAG}-bin-win-cuda-13.0-x64.zip",
    f"llama-{TAG}-bin-win-vulkan-x64.zip",
    f"llama-{TAG}-bin-win-hip-radeon-x64.zip",
)
RELEASE = {
    "tag_name": TAG,
    "assets": [
        {"name": name, "browser_download_url": f"https://example.invalid/{name}"}
        for name in ASSET_NAMES
    ],
}


def _host(*, arm64: bool, nvidia: bool, rocm: bool) -> ip.HostInfo:
    """The interpreter's own view of the machine, which is what the selector reads: an
    emulated x64 Python on an ARM64 box reports AMD64, and is an x64 host here."""
    return ip.HostInfo(
        system = "Windows",
        machine = "ARM64" if arm64 else "AMD64",
        is_windows = True,
        is_linux = False,
        is_macos = False,
        is_x86_64 = not arm64,
        is_arm64 = arm64,
        nvidia_smi = "nvidia-smi" if nvidia else None,
        driver_cuda_version = (13, 0) if nvidia else None,
        compute_caps = ["12.1"] if nvidia else [],
        visible_cuda_devices = None,
        has_physical_nvidia = nvidia,
        has_usable_nvidia = nvidia,
        has_rocm = rocm,
        rocm_gfx_target = "gfx1201" if rocm else None,
    )


def _gate_block() -> str:
    """setup.ps1's own expected-kinds computation, from the opt-out read to the delete."""
    start = SETUP_SRC.index("$_arm64CudaOptOut = ")
    end = SETUP_SRC.index("if ($existingKind -and", start)
    return SETUP_SRC[start:end]


def _expected_kinds(*, arm64_venv: bool, nvidia: bool, rocm: bool, opt_out: bool) -> list[str]:
    block = _gate_block()
    # The parentheses stay: the source writes `if (Test-WinArm64Venv)`, and replacing the
    # whole parenthesised form would leave `if $true {`, which does not parse.
    block = block.replace("Test-WinArm64Venv", "$true" if arm64_venv else "$false")
    # The evidence index is read through a chain of markers and a venv on disk. Its only use
    # is Test-WoaPersistableIndex, so supply the answer and stub the predicate.
    block = re.sub(
        r"(?s)\$_woaEvidenceIndex = if.*?\n(\s*)\$_nvidiaEvidence",
        "$_woaEvidenceIndex = '{}'\n\\1$_nvidiaEvidence".format(
            "https://pypi.nvidia.com/nvtorch_oot" if nvidia else ""
        ),
        block,
    )
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "function Test-WoaPersistableIndex { param($i) return ($i -like '*nvidia*') }",
            f"$env:UNSLOTH_LLAMA_ARM64_CUDA = '{'0' if opt_out else '1'}'",
            f"$HasNvidiaSmi = ${str(nvidia).lower()}",
            f"$HasROCm = ${str(rocm).lower()}",
            "$script:ROCmGfxArch = " + ("'gfx1201'" if rocm else "$null"),
            block,
            "Write-Output ('<<<' + ($expectedKinds -join ',') + '>>>')",
        ]
    )
    done = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        timeout = 120,
    )
    out = done.stdout.decode("utf-8", "replace")
    assert "<<<" in out, done.stderr.decode("utf-8", "replace")
    return out[out.index("<<<") + 3 : out.rindex(">>>")].split(",")


def _installed_kinds(host: ip.HostInfo) -> list[str]:
    plan = ip.direct_upstream_release_plan(RELEASE, host, "ggml-org/llama.cpp", "latest")
    return [attempt.install_kind for attempt in plan.attempts] if plan else []


# nvidia and rocm together is not a host we ship to: the ROCm arm wins outright on both sides.
COMBINATIONS = [
    (arm64_venv, nvidia, rocm, opt_out)
    for arm64_venv, nvidia, rocm, opt_out in itertools.product([True, False], repeat = 4)
    if not (nvidia and rocm)
]


@requires_pwsh
@pytest.mark.parametrize(
    ("arm64_venv", "nvidia", "rocm", "opt_out"),
    COMBINATIONS,
    ids = [
        "{}-{}{}".format(
            "arm64" if a else "x64",
            "nvidia" if n else ("rocm" if r else "cpu"),
            "-optout" if o else "",
        )
        for a, n, r, o in COMBINATIONS
    ],
)
def test_the_gate_accepts_what_the_selector_installs(arm64_venv, nvidia, rocm, opt_out):
    expected = _expected_kinds(arm64_venv = arm64_venv, nvidia = nvidia, rocm = rocm, opt_out = opt_out)
    installed = _installed_kinds(_host(arm64 = arm64_venv, nvidia = nvidia, rocm = rocm))
    assert installed, "no candidate at all, so this row proves nothing"
    assert installed[0] in expected, (
        f"the gate would delete what the selector just installed, on every update: "
        f"selector picks {installed[0]}, gate accepts {expected}"
    )


@requires_pwsh
def test_no_arm64_row_expects_a_kind_only_published_for_x64():
    """The shape behind both bugs: on an ARM64 venv a gate that expects an x64-only kind and
    nothing else is a delete on every update whatever the selector does. windows-vulkan counts
    as x64-only, since upstream's bundle is vulkan-x64."""
    x64_only = {"windows-cuda", "windows-rocm", "windows-hip", "windows-cpu", "windows-vulkan"}
    for nvidia, rocm, opt_out in itertools.product([True, False], repeat = 3):
        if nvidia and rocm:
            continue
        expected = set(_expected_kinds(arm64_venv = True, nvidia = nvidia, rocm = rocm, opt_out = opt_out))
        assert expected - x64_only, (
            f"nvidia={nvidia} rocm={rocm} optout={opt_out}: the gate expects only x64 kinds "
            f"on an ARM64 venv: {sorted(expected)}"
        )
