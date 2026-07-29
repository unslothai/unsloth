# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend selector coverage for setup.sh and setup.ps1.

cpu maps to install_llama_prebuilt.py's persisted --force-cpu option. vulkan is
accepted and passed through in the environment for the installer to consume.
The match is case-insensitive and whitespace-trimmed, unknown values warn, and
macOS warns for the CPU-only choice.
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


def _run(value: str | None, system: str = "Linux") -> tuple[list[str], str]:
    # Pass the value through env (not the script text) so whitespace survives, and
    # stub the setup.sh logging helpers the unknown-value branch calls. system sets
    # _HOST_SYSTEM so the macOS (Darwin) no-op branch can be exercised.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = (
        f'set -u\n_PREBUILT_CMD=()\nC_WARN=""\nC_OK=""\n_HOST_SYSTEM="{system}"\n'
        '_source_backend_choice="$(printf \'%s\' "${UNSLOTH_LLAMA_CPP_BACKEND:-auto}" '
        "| awk '{$1=$1; print tolower($0)}')\"\n"
        '_source_legacy_force_vulkan="$(printf \'%s\' "${UNSLOTH_FORCE_VULKAN:-}" '
        "| awk '{$1=$1; print tolower($0)}')\"\n"
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
@pytest.mark.parametrize("value", ["vulkan", "VULKAN", " vulkan "])
def test_backend_vulkan_is_accepted(value):
    args, stderr = _run(value)
    assert "--force-cpu" not in args
    assert args[-2:] == ["--llama-backend", "vulkan"]
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["hip", "HIP", "rocm", " ROCM "])
def test_backend_hip_opt_out_is_accepted(value):
    args, stderr = _run(value)
    assert "--force-cpu" not in args
    assert "--llama-backend" not in args
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["gpu", "cuda"])
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


def test_explicit_vulkan_prebuilt_failure_does_not_change_backend():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    failure = sh.index('step "llama.cpp" "prebuilt install failed"')
    source_build = sh.index("_NEED_LLAMA_SOURCE_BUILD=true", failure)
    guard = sh.index('if [ "$_explicit_vulkan_backend" = true ]', failure)
    assert guard < source_build
    guarded = sh[guard:source_build]
    assert "will not substitute a ROCm or CPU source build" in guarded
    assert "exit 1" in guarded

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    failure = ps1.index('step "llama.cpp" "prebuilt install failed"')
    source_build = ps1.index("$NeedLlamaSourceBuild = $true", failure)
    guard = ps1.index("if ($explicitVulkanBackend)", failure)
    assert guard < source_build
    guarded = ps1[guard:source_build]
    assert "will not substitute a CUDA, ROCm, or CPU source build" in guarded
    assert "exit 1" in guarded


def test_explicit_vulkan_source_build_fails_closed():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    local_branch = sh.index('if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then')
    prebuilt_branch = sh.index('elif [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then', local_branch)
    guarded = sh[local_branch:prebuilt_branch]
    assert (
        'elif [ "$_explicit_vulkan_source_build" = true ] && '
        '[ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then'
    ) in guarded
    assert "Vulkan source builds are not supported" in guarded
    assert "exit 1" in guarded

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    local_branch = ps1.index("if ($LocalLlamaCppLinked) {")
    prebuilt_branch = ps1.index(
        '} elseif ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {', local_branch
    )
    guarded = ps1[local_branch:prebuilt_branch]
    assert "} elseif ($explicitVulkanSourceBuild -and $NeedLlamaSourceBuild) {" in guarded
    assert "Vulkan source builds are not supported" in guarded
    assert "exit 1" in guarded


def test_force_compile_sets_need_source_build_before_vulkan_guard():
    # UNSLOTH_LLAMA_FORCE_COMPILE=1 combined with an explicit Vulkan backend must
    # hit the "Vulkan source builds are not supported" rejection, not silently
    # fall through to a CUDA/ROCm/CPU source build. That only holds if
    # _NEED_LLAMA_SOURCE_BUILD/$NeedLlamaSourceBuild is already true by the time
    # the explicit-Vulkan elif guard runs, i.e. set earlier in the script.
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    force_compile_set = sh.index(
        'if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then\n    _NEED_LLAMA_SOURCE_BUILD=true'
    )
    vulkan_guard = sh.index('elif [ "$_explicit_vulkan_source_build" = true ] && ')
    assert force_compile_set < vulkan_guard

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    force_compile_set = ps1.index(
        'if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {\n    $NeedLlamaSourceBuild = $true'
    )
    vulkan_guard = ps1.index("} elseif ($explicitVulkanSourceBuild -and $NeedLlamaSourceBuild) {")
    assert force_compile_set < vulkan_guard


def test_legacy_force_vulkan_gets_the_same_strict_fallback():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    assert "_legacy_force_vulkan=" in sh
    assert "1|true|yes|on) _explicit_vulkan_backend=true" in sh

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    assert "$legacyForceVulkan = $sourceLegacyForceVulkan" in ps1
    assert '$legacyForceVulkan -in @("1", "true", "yes", "on")' in ps1


def _source_backend_choice_block() -> str:
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(
        r'_source_backend_choice="\$\(printf.*?\n.*?_explicit_vulkan_source_build=false\n'
        r".*?\nfi\n",
        text,
        re.DOTALL,
    )
    assert m, "_source_backend_choice block not found in setup.sh"
    return m.group(0)


@_SKIP_NO_BASH
@pytest.mark.parametrize(
    "backend, force_vulkan, expected_backend, expected_explicit",
    [
        (None, None, "auto", "false"),
        ("vulkan", None, "vulkan", "true"),
        ("auto", "on", "auto", "true"),
        ("banana", "1", "banana", "true"),
        ("cpu", "1", "cpu", "false"),
        ("hip", "1", "hip", "false"),
        ("rocm", "1", "rocm", "false"),
    ],
)
def test_llama_backend_source_choice_in_setup_sh(
    backend, force_vulkan, expected_backend, expected_explicit
):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
    if force_vulkan is not None:
        env["UNSLOTH_FORCE_VULKAN"] = force_vulkan
    harness = (
        "set -u\n_HOST_SYSTEM=Linux\n"
        f"{_source_backend_choice_block()}\n"
        'printf "%s:%s" "$_source_backend_choice" "$_explicit_vulkan_source_build"'
    )
    out = subprocess.run(
        ["bash", "-c", harness], capture_output = True, text = True, env = env, check = True
    )
    assert out.stdout == f"{expected_backend}:{expected_explicit}"


def _ps1_search(pattern: str, flags = 0) -> str:
    m = re.search(pattern, _SETUP_PS1.read_text(encoding = "utf-8"), flags)
    assert m, f"setup.ps1 block not found: {pattern}"
    return m.group(0)


@_SKIP_NO_PWSH
@pytest.mark.parametrize(
    "backend, force_vulkan, expected_explicit",
    [
        (None, None, "False"),
        ("vulkan", None, "True"),
        ("auto", "on", "True"),
        ("cpu", "1", "False"),
        ("hip", "1", "False"),
        ("rocm", "1", "False"),
    ],
)
def test_llama_backend_source_choice_in_setup_ps1(backend, force_vulkan, expected_explicit):
    normalize = _ps1_search(
        r'\$sourceLlamaBackend = "\$\(\$env:UNSLOTH_LLAMA_CPP_BACKEND\)".*?'
        r"\$explicitVulkanSourceBuild = \(.*?\n\)\n",
        re.DOTALL,
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    if backend is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = backend
    if force_vulkan is not None:
        env["UNSLOTH_FORCE_VULKAN"] = force_vulkan
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            # Brace the name: PowerShell reads "$var:" as a scope qualifier and
            # fails to parse, so the probe never ran on a host that has pwsh.
            f'{normalize}\n"RESULT:${{sourceLlamaBackend}}:$explicitVulkanSourceBuild"',
        ],
        capture_output = True,
        text = True,
        env = env,
        check = True,
    )
    expected_backend = (backend or "").strip().lower()
    assert out.stdout.strip() == f"RESULT:{expected_backend}:{expected_explicit}"


def _run_ps1(value: str | None) -> str:
    # One snippet, not two: NORMALIZE already spans the whole if/elseif chain up to
    # and including the "Ignoring UNSLOTH_LLAMA_CPP_BACKEND" warn, so the --force-cpu
    # append is inside it. Composing the apply snippet on top of it made the harness
    # emit the flag twice while setup.ps1 appends it once.
    normalize = _ps1_search(
        r"\$llamaBackend = \$sourceLlamaBackend.*?Ignoring UNSLOTH_LLAMA_CPP_BACKEND.*?\n\s*\}",
        re.DOTALL,
    )
    assert normalize.count('$prebuiltArgs += "--force-cpu"') == 1, normalize
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = (
        "$prebuiltArgs = @()\n"
        '$sourceLlamaBackend = "$($env:UNSLOTH_LLAMA_CPP_BACKEND)".Trim().ToLowerInvariant()\n'
        '$sourceLegacyForceVulkan = "$($env:UNSLOTH_FORCE_VULKAN)".Trim().ToLowerInvariant()\n'
        f'{normalize}\n"ARGS:" + ($prebuiltArgs -join ",")'
    )
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
    # The whole argv, not a substring: an `in` check passed while the harness was
    # emitting --force-cpu twice. setup.ps1 appends it exactly once, and nothing
    # else, for a cpu override.
    assert out.strip() == "ARGS:--force-cpu"


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_ps1_backend_auto_no_flag_no_warn(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["vulkan", "VULKAN", " vulkan "])
def test_ps1_backend_vulkan_is_accepted(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "--llama-backend,vulkan" in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["hip", "HIP", "rocm", " ROCM "])
def test_ps1_backend_hip_opt_out_is_accepted(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "--llama-backend" not in out
    assert "Ignoring" not in out


def test_ps1_forced_vulkan_fails_closed_on_windows_arm64():
    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    arm64_guard = ps1.index("elseif ($windowsArm64)")
    vulkan_flag = ps1.index(
        '$prebuiltArgs += @("--llama-backend", "vulkan")',
        arm64_guard,
    )
    assert arm64_guard < vulkan_flag
    assert "no Windows ARM64 Vulkan bundle is published" in ps1
    assert "Unset UNSLOTH_FORCE_VULKAN" in ps1


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["gpu", "cuda"])
def test_ps1_backend_unknown_warns_and_no_flag(value):
    out = _run_ps1(value)
    assert "--force-cpu" not in out
    assert "Ignoring" in out
