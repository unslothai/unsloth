# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh and setup.ps1 must map UNSLOTH_LLAMA_CPP_BACKEND=cpu to
install_llama_prebuilt.py's --cpu-fallback so users can force the CPU-only
prebuilt on GPU hosts (#7213). The match is case-insensitive and
whitespace-trimmed, and an unrecognized value warns instead of silently falling
back. Runs the real block extracted from each script so the tests track the
shipped logic.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
_SETUP_SH = _STUDIO / "setup.sh"
_SETUP_PS1 = _STUDIO / "setup.ps1"
_SKIP_NO_BASH = pytest.mark.skipif(shutil.which("bash") is None, reason = "bash unavailable")
_SKIP_NO_PWSH = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh unavailable")


def _backend_block() -> str:
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(r"_llama_backend=.*?esac", text, re.DOTALL)
    assert m, "UNSLOTH_LLAMA_CPP_BACKEND block not found in setup.sh"
    return m.group(0)


def _run(value: str | None) -> tuple[list[str], str]:
    # Pass the value through env (not the script text) so whitespace survives, and
    # stub the setup.sh logging helpers the unknown-value branch calls.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_LLAMA_CPP_BACKEND"}
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = (
        '_PREBUILT_CMD=()\nC_WARN=""\nstep() { printf "STEP: %s\\n" "$*" >&2; }\n'
        f"{_backend_block()}\n"
        'printf "%s\\n" "${_PREBUILT_CMD[@]}"'
    )
    out = subprocess.run(
        ["bash", "-c", harness], capture_output = True, text = True, env = env, check = True
    )
    return out.stdout.split(), out.stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu", " cpu ", "CPU\t"])
def test_backend_cpu_appends_flag(value):
    args, stderr = _run(value)
    assert "--cpu-fallback" in args
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_backend_auto_no_flag_no_warn(value):
    args, stderr = _run(value)
    assert "--cpu-fallback" not in args
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["vulkan", "gpu", "cuda"])
def test_backend_unknown_warns_and_no_flag(value):
    args, stderr = _run(value)
    assert "--cpu-fallback" not in args
    assert "Ignoring" in stderr


def _ps1_search(pattern: str, flags = 0) -> str:
    m = re.search(pattern, _SETUP_PS1.read_text(encoding = "utf-8"), flags)
    assert m, f"setup.ps1 block not found: {pattern}"
    return m.group(0)


def _run_ps1(value: str | None) -> str:
    # The override is normalized (assign + warn) at the top of the prebuilt block and
    # applied to $prebuiltArgs lower down; compose both real snippets.
    normalize = _ps1_search(
        r'\$llamaBackend = "\$\(\$env:UNSLOTH_LLAMA_CPP_BACKEND\)".*?Write-Host.*?\n\s*\}',
        re.DOTALL,
    )
    apply_flag = _ps1_search(
        r'if \(\$llamaBackend -eq "cpu"\) \{\s*\$prebuiltArgs \+= "--cpu-fallback"\s*\}'
    )
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_LLAMA_CPP_BACKEND"}
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = f'$prebuiltArgs = @()\n{normalize}\n{apply_flag}\n"ARGS:" + ($prebuiltArgs -join ",")'
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", harness],
        capture_output = True,
        text = True,
        env = env,
        check = True,
    )
    return out.stdout


def _run_ps1_expected_kinds(backend: str, has_nvidia: bool) -> str:
    # The mismatch-prune must treat a forced-CPU install as expected on a GPU host, or
    # persisting install_kind deletes a valid windows-cpu bundle every rerun (#7228).
    kinds = _ps1_search(
        r"\$expectedKinds = if .*?else \{ @\(\"windows-cpu\", \"windows-arm64\"\) \}"
    )
    harness = (
        f"$HasNvidiaSmi=${str(has_nvidia).lower()}; $HasROCm=$false; $script:ROCmGfxArch=$null\n"
        f'$llamaBackend="{backend}"\n{kinds}\n"KINDS:" + ($expectedKinds -join ",")'
    )
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", harness], capture_output = True, text = True, check = True
    )
    return out.stdout


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu", " cpu ", "CPU\t"])
def test_ps1_backend_cpu_appends_flag(value):
    out = _run_ps1(value)
    assert "--cpu-fallback" in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_ps1_backend_auto_no_flag_no_warn(value):
    out = _run_ps1(value)
    assert "--cpu-fallback" not in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["vulkan", "gpu", "cuda"])
def test_ps1_backend_unknown_warns_and_no_flag(value):
    out = _run_ps1(value)
    assert "--cpu-fallback" not in out
    assert "Ignoring" in out


@_SKIP_NO_PWSH
def test_ps1_cpu_override_keeps_cpu_expected_on_gpu_host():
    # Forced CPU on an NVIDIA host: windows-cpu must stay expected so the mismatch
    # prune does not delete a deliberate CPU install (#7228).
    assert "windows-cpu" in _run_ps1_expected_kinds("cpu", has_nvidia = True)


@_SKIP_NO_PWSH
def test_ps1_auto_prunes_cpu_on_nvidia_host():
    # Without the override an NVIDIA host still expects only windows-cuda.
    out = _run_ps1_expected_kinds("auto", has_nvidia = True)
    assert "windows-cuda" in out
    assert "windows-cpu" not in out
