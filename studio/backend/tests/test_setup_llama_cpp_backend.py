# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend selector coverage for setup.sh and setup.ps1.

The scripts do not act on UNSLOTH_LLAMA_CPP_BACKEND: install_llama_prebuilt.py
reads it directly, and it is also the only side that can see a choice recorded in
the install marker. What is left here is reporting -- the match is
case-insensitive and whitespace-trimmed, unknown values warn, and macOS says so
for the two choices its universal Metal build cannot honour.
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
    m = re.search(r"# Reporting only:.*?esac", text, re.DOTALL)
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
        '_source_backend_choice="$(printf \'%s\' "${UNSLOTH_LLAMA_CPP_BACKEND:-}" '
        "| awk '{$1=$1; print tolower($0)}')\"\n"
        '_source_legacy_force_vulkan="$(printf \'%s\' "${UNSLOTH_FORCE_VULKAN:-}" '
        "| awk '{$1=$1; print tolower($0)}')\"\n"
        '_explicit_llama_source_backend=""\n'
        'step() { printf "STEP: %s\\n" "$*" >&2; }\n'
        f"{_backend_block()}\n"
        'printf "%s\\n" "${_PREBUILT_CMD[@]}"'
    )
    out = subprocess.run(
        ["bash", "-c", harness], capture_output = True, text = True, env = env, check = True
    )
    return out.stdout.split(), out.stderr


def test_backend_block_forwards_nothing_to_the_installer():
    # One owner for the selection. install_llama_prebuilt.py reads the variable
    # itself and is the only side that can also read the marker's recorded choice,
    # so a second copy assembled here could only ever contradict it.
    assert "_PREBUILT_CMD" not in _backend_block()


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu", " cpu ", "CPU\t"])
def test_backend_cpu_is_accepted(value):
    args, stderr = _run(value)
    assert args == []
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["cpu", "CPU", " cpu "])
def test_backend_cpu_macos_warns(value):
    # macOS has no CPU-only bundle: the universal build already runs on CPU.
    args, stderr = _run(value, system = "Darwin")
    assert args == []
    assert "macOS" in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  "])
def test_backend_auto_is_silent(value):
    args, stderr = _run(value)
    assert args == []
    assert stderr == ""


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["vulkan", "VULKAN", " vulkan "])
def test_backend_vulkan_is_reported(value):
    args, stderr = _run(value)
    assert args == []
    assert "Vulkan selected for GGUF inference" in stderr
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["vulkan", "VULKAN"])
def test_backend_vulkan_macos_warns(value):
    args, stderr = _run(value, system = "Darwin")
    assert args == []
    assert "Metal" in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["hip", "HIP", "rocm", " ROCM ", "cuda", " CUDA "])
def test_backend_gpu_opt_out_is_accepted(value):
    args, stderr = _run(value)
    assert args == []
    assert "Ignoring" not in stderr


@_SKIP_NO_BASH
@pytest.mark.parametrize("value", ["gpu", "sycl"])
def test_backend_unknown_warns(value):
    args, stderr = _run(value)
    assert args == []
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


def test_ordinary_prebuilt_failure_does_not_re_derive_the_backend():
    # Exit 2 is reached only when no concrete backend was in play: a request the
    # installer could not honour -- named in the environment or recorded in the
    # marker the scripts cannot read -- arrives as exit 5 and fails closed there.
    # Re-deriving it here from the environment alone would miss the marker case
    # and disagree with the installer on the rest.
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    failure = sh.index('step "llama.cpp" "prebuilt install failed"')
    branch = sh[failure : sh.index("_NEED_LLAMA_SOURCE_BUILD=true", failure)]
    assert "_explicit_llama" not in branch

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    failure = ps1.index('step "llama.cpp" "prebuilt install failed"')
    branch = ps1[failure : ps1.index("$NeedLlamaSourceBuild = $true", failure)]
    assert "explicitLlama" not in branch


def test_unavailable_named_backend_never_falls_through_to_source():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    unavailable = sh.index('elif [ "$_PREBUILT_STATUS" -eq 5 ]; then')
    ordinary_fallback = sh.index('elif [ "$_PREBUILT_STATUS" -eq 2 ]; then', unavailable)
    guarded = sh[unavailable:ordinary_fallback]
    assert "will not substitute a different source backend" in guarded
    assert "_NEED_LLAMA_SOURCE_BUILD=true" not in guarded
    assert "setup_fail 1" in guarded

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    unavailable = ps1.index("} elseif ($prebuiltExit -eq 5) {")
    ordinary_fallback = ps1.index("} elseif ($prebuiltExit -eq 2) {", unavailable)
    guarded = ps1[unavailable:ordinary_fallback]
    assert "will not substitute a different source backend" in guarded
    assert "$NeedLlamaSourceBuild = $true" not in guarded
    assert "Exit-SetupFailure" in guarded


def test_explicit_backend_source_build_fails_closed():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    local_branch = sh.index('if [ "$_LOCAL_LLAMA_CPP_LINKED" = true ]; then')
    prebuilt_branch = sh.index('elif [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then', local_branch)
    guarded = sh[local_branch:prebuilt_branch]
    assert (
        'elif [ -n "$_explicit_llama_source_backend" ] && '
        '[ "$_NEED_LLAMA_SOURCE_BUILD" = true ]; then'
    ) in guarded
    assert "Explicit backend selection requires a matching prebuilt bundle" in guarded
    assert "setup_fail 1" in guarded

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    local_branch = ps1.index("if ($LocalLlamaCppLinked) {")
    prebuilt_branch = ps1.index(
        '} elseif ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {', local_branch
    )
    guarded = ps1[local_branch:prebuilt_branch]
    assert "} elseif ($explicitLlamaSourceBackend -and $NeedLlamaSourceBuild) {" in guarded
    assert "Explicit backend selection requires a matching prebuilt bundle" in guarded
    assert "Exit-SetupFailure" in guarded


def test_force_compile_sets_need_source_build_before_backend_guard():
    # A forced source build combined with any concrete backend must reach the
    # explicit-backend rejection. The source-build state must therefore be set
    # before the guard runs.
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    force_compile_set = sh.index(
        'if [ "$_LLAMA_FORCE_COMPILE" = "1" ]; then\n    _NEED_LLAMA_SOURCE_BUILD=true'
    )
    backend_guard = sh.index('elif [ -n "$_explicit_llama_source_backend" ] && ')
    assert force_compile_set < backend_guard

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    force_compile_set = ps1.index(
        'if ($env:UNSLOTH_LLAMA_FORCE_COMPILE -eq "1") {\n    $NeedLlamaSourceBuild = $true'
    )
    backend_guard = ps1.index("} elseif ($explicitLlamaSourceBackend -and $NeedLlamaSourceBuild) {")
    assert force_compile_set < backend_guard


def test_legacy_force_vulkan_gets_the_same_strict_fallback():
    sh = _SETUP_SH.read_text(encoding = "utf-8")
    assert '1|true|yes|on) _explicit_llama_source_backend="vulkan"' in sh

    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    assert '$sourceLegacyForceVulkan -in @("1", "true", "yes", "on")' in ps1
    assert '$explicitLlamaSourceBackend = "vulkan"' in ps1


def _source_backend_choice_block() -> str:
    text = _SETUP_SH.read_text(encoding = "utf-8")
    m = re.search(
        r'_source_backend_choice="\$\(printf.*?\n.*?_explicit_llama_source_backend=""\n'
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
        (None, None, "", ""),
        (None, "on", "", "vulkan"),
        ("vulkan", None, "vulkan", "vulkan"),
        ("auto", "on", "auto", ""),
        ("banana", "1", "banana", "vulkan"),
        ("cpu", "1", "cpu", "cpu"),
        ("cuda", "1", "cuda", "cuda"),
        ("hip", "1", "hip", "rocm"),
        ("rocm", "1", "rocm", "rocm"),
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
        'printf "%s:%s" "$_source_backend_choice" "$_explicit_llama_source_backend"'
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
        (None, None, ""),
        (None, "on", "vulkan"),
        ("vulkan", None, "vulkan"),
        ("auto", "on", ""),
        ("cpu", "1", "cpu"),
        ("cuda", "1", "cuda"),
        ("hip", "1", "rocm"),
        ("rocm", "1", "rocm"),
    ],
)
def test_llama_backend_source_choice_in_setup_ps1(backend, force_vulkan, expected_explicit):
    normalize = _ps1_search(
        r'\$sourceLlamaBackend = "\$\(\$env:UNSLOTH_LLAMA_CPP_BACKEND\)".*?'
        r"\$explicitLlamaSourceBackend = \$null.*?\n\}",
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
            f'{normalize}\n"RESULT:${{sourceLlamaBackend}}:$explicitLlamaSourceBackend"',
        ],
        capture_output = True,
        text = True,
        env = env,
        check = True,
    )
    expected_backend = (backend or "").strip().lower()
    assert out.stdout.strip() == f"RESULT:{expected_backend}:{expected_explicit}"


def _run_ps1(value: str | None) -> str:
    normalize = _ps1_search(
        r"\$llamaBackend = \$sourceLlamaBackend.*?Ignoring UNSLOTH_LLAMA_CPP_BACKEND.*?\n\s*\}",
        re.DOTALL,
    )
    # The mirror of test_backend_block_forwards_nothing_to_the_installer.
    assert "$prebuiltArgs" not in normalize, normalize
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN")
    }
    if value is not None:
        env["UNSLOTH_LLAMA_CPP_BACKEND"] = value
    harness = (
        # The spliced snippet warns through setup.ps1's output sink, which the real
        # script defines above every call site. Stub it to Write-Host so the warning
        # lands on stdout, where the assertions below read it: without this the call
        # errors out and the harness returns a bare "ARGS:".
        "function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) "
        "Write-Host $Message }\n"
        "$prebuiltArgs = @()\n"
        '$sourceLlamaBackend = "$($env:UNSLOTH_LLAMA_CPP_BACKEND)".Trim().ToLowerInvariant()\n'
        '$sourceLegacyForceVulkan = "$($env:UNSLOTH_FORCE_VULKAN)".Trim().ToLowerInvariant()\n'
        "$explicitLlamaSourceBackend = $null\n"
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
@pytest.mark.parametrize(
    "value", ["cpu", "CPU", "Cpu", " cpu ", "CPU\t", None, "", "auto", "AUTO", "  "]
)
def test_ps1_backend_forwards_no_arguments(value):
    out = _run_ps1(value)
    assert out.strip() == "ARGS:"


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["vulkan", "VULKAN", " vulkan "])
def test_ps1_backend_vulkan_is_reported(value):
    out = _run_ps1(value)
    assert "Vulkan selected for GGUF inference" in out
    assert "Ignoring" not in out


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["cuda", " CUDA ", "hip", "HIP", "rocm", " ROCM "])
def test_ps1_backend_gpu_opt_out_is_accepted(value):
    out = _run_ps1(value)
    assert out.strip() == "ARGS:"


def test_ps1_forced_vulkan_fails_closed_on_windows_arm64():
    ps1 = _SETUP_PS1.read_text(encoding = "utf-8")
    branch = ps1.index(
        'if ($llamaBackend -eq "vulkan" -or $explicitLlamaSourceBackend -eq "vulkan")'
    )
    guarded = ps1[branch : ps1.index("Vulkan selected for GGUF inference", branch)]
    assert "elseif ($windowsArm64)" in guarded
    assert "no Windows ARM64 Vulkan bundle is published" in guarded
    assert "UNSLOTH_FORCE_VULKAN" in guarded


@_SKIP_NO_PWSH
@pytest.mark.parametrize("value", ["gpu", "sycl"])
def test_ps1_backend_unknown_warns(value):
    out = _run_ps1(value)
    assert "Ignoring" in out
