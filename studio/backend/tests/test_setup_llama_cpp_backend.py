# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh and setup.ps1 must map UNSLOTH_LLAMA_CPP_BACKEND=cpu to
install_llama_prebuilt.py's --force-cpu so users can force the CPU-only prebuilt
on GPU hosts (#7213). The match is case-insensitive and whitespace-trimmed, an
unrecognized value warns instead of silently falling back, and macOS warns (no
CPU-only bundle). Runs the real block extracted from each script so the tests
track the shipped logic.
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
_SKIP_NO_BASH = pytest.mark.skipif(
    shutil.which("bash") is None, reason = "bash unavailable"
)
_SKIP_NO_PWSH = pytest.mark.skipif(
    shutil.which("pwsh") is None, reason = "pwsh unavailable"
)


def _backend_block() -> str:
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(r"_llama_backend=.*?esac", text, re.DOTALL)
    assert m, "UNSLOTH_LLAMA_CPP_BACKEND block not found in setup.sh"
    return m.group(0)


def _run(value: str | None, system: str = "Linux") -> tuple[list[str], str]:
    # Pass the value through env (not the script text) so whitespace survives, and
    # stub the setup.sh logging helpers the unknown-value branch calls. system sets
    # _HOST_SYSTEM so the macOS (Darwin) no-op branch can be exercised.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_LLAMA_CPP_BACKEND"}
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = (
        f'_PREBUILT_CMD=()\nC_WARN=""\n_HOST_SYSTEM="{system}"\n'
        'step() { printf "STEP: %s\\n" "$*" >&2; }\n'
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
    # A deliberate CPU choice persists, so it uses --force-cpu (not the transient
    # --cpu-fallback the arm64 GPU-build recovery uses).
    args, stderr = _run(value)
    assert "--force-cpu" in args
    assert "--cpu-fallback" not in args
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["cpu", "CPU", " cpu "])
def test_backend_cpu_macos_warns_no_flag(value):
    # macOS has no CPU-only bundle (the universal build already runs on CPU), so the
    # override warns instead of writing a misleading forced-CPU marker.
    args, stderr = _run(value, system = "Darwin")
    assert "--force-cpu" not in args
    assert "--cpu-fallback" not in args
    assert "macOS" in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_backend_auto_no_flag_no_warn(value):
    args, stderr = _run(value)
    assert "--force-cpu" not in args
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["vulkan", "gpu", "cuda"])
def test_backend_unknown_warns_and_no_flag(value):
    args, stderr = _run(value)
    assert "--force-cpu" not in args
    assert "Ignoring" in stderr


@_SKIP_NO_BASH
def test_arm64_recovery_uses_transient_cpu_fallback():
    # The arm64 Linux GPU-build recovery must stay transient (--cpu-fallback), never
    # the persisted --force-cpu, so a later update can still heal to a GPU bundle (#6097).
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(r"_ARM64_CPU_CMD=\((.*?)\)", text, re.DOTALL)
    assert m, "arm64 CPU recovery command not found in setup.sh"
    block = m.group(1)
    assert "--cpu-fallback" in block
    assert "--force-cpu" not in block


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
        r'if \(\$llamaBackend -eq "cpu"\) \{\s*\$prebuiltArgs \+= "--force-cpu"\s*\}'
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


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu", " cpu ", "CPU\t"])
def test_ps1_backend_cpu_appends_flag(value):
    out = _run_ps1(value)
    assert "--force-cpu" in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_ps1_backend_auto_no_flag_no_warn(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["vulkan", "gpu", "cuda"])
def test_ps1_backend_unknown_warns_and_no_flag(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "Ignoring" in out
