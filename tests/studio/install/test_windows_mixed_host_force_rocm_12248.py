# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows parity for mixed NVIDIA + AMD hosts (#12248): install.ps1 / setup.ps1 read
UNSLOTH_FORCE_ROCM_TORCH and treat a CUDA_VISIBLE_DEVICES that hides every NVIDIA device the
way install.sh does, and install.ps1 prints install.sh's mixed-host notice otherwise."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from woa_ps_harness import (  # noqa: E402
    INSTALL_SRC,
    SETUP_SRC,
    _function_source,
    _ps_copies,
    _ps_kv,
    _script,
    functions,
    requires_pwsh,
    slice_between,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_spec = importlib.util.spec_from_file_location("studio_stack_mixed_host_12248", _STACK_PY)
stack_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stack_mod
_spec.loader.exec_module(stack_mod)

_MASK_ENV = (
    "UNSLOTH_FORCE_ROCM_TORCH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "SKIP_STUDIO_BASE",
)

INSTALL_HELPERS = functions(
    INSTALL_SRC,
    "Test-CudaMaskHidesNvidia",
    "Test-RocmTorchExplicitlyRequested",
    "Get-MixedHostTorchRoute",
    "Get-MixedHostNoticeArch",
    "Write-CudaMaskHidesAmdWarning",
    "Write-MixedHostCudaNotice",
)
SETUP_HELPERS = functions(
    SETUP_SRC,
    "Test-CudaMaskHidesNvidia",
    "Test-RocmTorchExplicitlyRequested",
    "Write-CudaMaskHidesAmdWarning",
)
# The wheel tables Get-MixedHostNoticeArch reads, sliced from the live script.
INSTALL_WHEEL_MAPS = slice_between(
    INSTALL_SRC, "    $archFamilyMap = @{", "    $MultiArchIndexBase"
)

COLLECT = _script(
    "$script:Said = @()",
    "function Write-StudioLine { param($m, $ForegroundColor) $script:Said += [string]$m }",
    "function substep { param($m, $c) $script:Said += [string]$m }",
)

INSTALL_DECISION = slice_between(
    INSTALL_SRC, '    # "rocm": from here on this is an AMD host', "    # ── Intel GPU detection"
)
INSTALL_NOTICE = slice_between(
    INSTALL_SRC,
    "    $TorchIndexUrl = Get-TorchIndexUrl\n",
    "    # Must run AFTER Get-TorchIndexUrl",
)
SETUP_DECISION = slice_between(
    SETUP_SRC,
    "if ($script:ProbeAmdBesideNvidia) {\n    if ($AmdHasGpuWheels)",
    "# Mirrors install.ps1's Intel scan",
)


def _env(**values) -> str:
    lines = [f"Remove-Item Env:{n} -ErrorAction SilentlyContinue" for n in _MASK_ENV]
    lines += [f"$env:{k} = '{v}'" for k, v in values.items()]
    return _script(*lines)


def _install_decision(
    arch,
    wheels,
    hip_arches = (),
    **env,
):
    """The install.ps1 decision on an NVIDIA host whose AMD probe found `arch` (hipinfo listed `hip_arches`)."""
    listed = ", ".join(repr(a) for a in hip_arches)
    return _ps_kv(
        _script(
            COLLECT,
            INSTALL_HELPERS,
            INSTALL_WHEEL_MAPS,
            f"$_hipAllArches = @({listed})",
            _env(**env),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            "$HasNvidiaSmi = $true; $ProbeAmdBesideNvidia = $true",
            f"$ROCmGfxArch = {'$null' if arch is None else repr(arch)}",
            f"$AmdHasGpuWheels = ${str(wheels).lower()}",
            "$HasROCm = $true; $HipSdkInstalled = $true; $ROCmGpuLabel = 'x'; $ROCmGpuName = 'x'",
            "$ROCmUnsupportedGfxArch = $null",
            INSTALL_DECISION,
            'Write-Output "ROUTE=$MixedHostRoute"',
            'Write-Output "OVER=$RocmTorchOverNvidia"',
            'Write-Output "NVIDIA=$HasNvidiaSmi"',
            'Write-Output "ARCH=$ROCmGfxArch"',
            'Write-Output "HASROCM=$HasROCm"',
            'Write-Output "NOTICE=$MixedHostAmdArch"',
            "Write-Output (\"SAID=\" + ($script:Said -join ' | '))",
        )
    )


@requires_pwsh
class TestInstallPs1Decision:
    def test_force_on_a_mixed_host_picks_rocm(self):
        kv = _install_decision("gfx1201", True, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert kv["ROUTE"] == "rocm" and kv["OVER"] == "True"
        assert kv["NVIDIA"] == "False" and kv["ARCH"] == "gfx1201"
        assert "selecting ROCm PyTorch over CUDA" in kv["SAID"]

    @pytest.mark.parametrize("value", ["true", " Yes ", "ON"])
    def test_force_takes_the_install_sh_truthy_values(self, value):
        assert _install_decision("gfx1201", True, UNSLOTH_FORCE_ROCM_TORCH = value)["ROUTE"] == "rocm"

    def test_without_force_cuda_stays_and_the_amd_probe_is_dropped(self):
        kv = _install_decision("gfx1201", True)
        assert kv["ROUTE"] == "cuda-notice" and kv["OVER"] == "False"
        assert kv["NVIDIA"] == "True" and kv["ARCH"] == "" and kv["HASROCM"] == "False"
        assert kv["NOTICE"] == "gfx1201"

    @pytest.mark.parametrize("arch", ["gfx1036", "gfx1103", "gfx1151", "gfx1150"])
    def test_an_integrated_gpu_gets_no_notice(self, arch):
        kv = _install_decision(arch, True)
        assert kv["ROUTE"] == "cuda" and kv["NOTICE"] == ""

    def test_an_igpu_listed_first_does_not_hide_the_discrete_card(self):
        kv = _install_decision("gfx1036", True, hip_arches = ("gfx1036", "gfx1201"))
        assert kv["ROUTE"] == "cuda-notice" and kv["NOTICE"] == "gfx1201"

    def test_a_card_with_no_wheels_is_not_named(self):
        kv = _install_decision("gfx1036", True, hip_arches = ("gfx1036", "gfx803"))
        assert kv["ROUTE"] == "cuda" and kv["NOTICE"] == ""

    def test_force_without_a_wheel_route_keeps_cuda(self):
        kv = _install_decision("gfx803", False, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert kv["ROUTE"] == "cuda-forced" and kv["NVIDIA"] == "True" and kv["ARCH"] == ""

    @pytest.mark.parametrize("mask", ["", "-1", " -1 "])
    def test_a_cuda_mask_hiding_nvidia_picks_rocm_and_warns(self, mask):
        kv = _install_decision("gfx1201", True, CUDA_VISIBLE_DEVICES = mask)
        assert kv["ROUTE"] == "rocm" and kv["NVIDIA"] == "False"
        assert "ROCm reads CUDA_VISIBLE_DEVICES too" in kv["SAID"]

    def test_force_with_a_hiding_mask_also_warns(self):
        kv = _install_decision(
            "gfx1201", True, UNSLOTH_FORCE_ROCM_TORCH = "1", CUDA_VISIBLE_DEVICES = "-1"
        )
        assert "selecting ROCm PyTorch over CUDA" in kv["SAID"]
        assert "ROCm reads CUDA_VISIBLE_DEVICES too" in kv["SAID"]

    def test_force_alone_does_not_warn_about_a_mask(self):
        kv = _install_decision("gfx1201", True, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert "ROCm reads" not in kv["SAID"]

    def test_the_mask_warning_is_skipped_when_hip_chooses_the_card(self):
        kv = _install_decision("gfx1201", True, CUDA_VISIBLE_DEVICES = "-1", HIP_VISIBLE_DEVICES = "0")
        assert kv["ROUTE"] == "rocm" and "ROCm reads" not in kv["SAID"]

    def test_a_mask_hiding_nvidia_with_no_amd_route_keeps_cuda(self):
        # install.sh would fall to CPU here; Windows keeps the NVIDIA host's CUDA install.
        kv = _install_decision(None, False, CUDA_VISIBLE_DEVICES = "-1")
        assert kv["ROUTE"] == "cuda" and kv["NVIDIA"] == "True"

    def test_a_cuda_mask_naming_a_device_changes_nothing(self):
        assert (
            _install_decision("gfx1201", True, CUDA_VISIBLE_DEVICES = "0")["ROUTE"] == "cuda-notice"
        )


def _install_notice(route, arch, index, **env):
    return _ps_kv(
        _script(
            COLLECT,
            INSTALL_HELPERS,
            _function_source(INSTALL_SRC, "Get-TorchIndexLeafName"),
            _env(**env),
            f"function Get-TorchIndexUrl {{ return '{index}' }}",
            f"$MixedHostRoute = '{route}'",
            f"$MixedHostAmdArch = {'$null' if arch is None else repr(arch)}",
            INSTALL_NOTICE,
            "Write-Output (\"SAID=\" + ($script:Said -join ' | '))",
        )
    )["SAID"]


@requires_pwsh
class TestInstallPs1Notice:
    def test_mixed_host_on_cuda_prints_the_install_sh_notice(self):
        said = _install_notice("cuda-notice", "gfx1201", "https://download.pytorch.org/whl/cu130")
        assert (
            "An AMD GPU (gfx1201) was also found. CUDA PyTorch was installed for the NVIDIA GPU,"
            in said
        )
        assert '$env:UNSLOTH_FORCE_ROCM_TORCH = "1"' in said
        assert "Settings > System > GGUF inference engine > Compute backend." in said
        assert "HIP_VISIBLE_DEVICES" not in said

    def test_a_hip_mask_adds_the_flag_is_still_needed_line(self):
        said = _install_notice(
            "cuda-notice", "gfx1201", "https://x/whl/cu128", HIP_VISIBLE_DEVICES = "0"
        )
        assert "UNSLOTH_FORCE_ROCM_TORCH=1 is needed as well" in said

    def test_a_cpu_index_on_a_pre_r450_driver_stays_quiet(self):
        assert _install_notice("cuda-notice", "gfx1201", "https://x/whl/cpu") == ""

    def test_nothing_for_an_nvidia_only_host(self):
        assert _install_notice("cuda", None, "https://x/whl/cu130") == ""

    def test_force_without_a_route_warns_and_keeps_cuda(self):
        said = _install_notice("cuda-forced", None, "https://x/whl/cu130")
        assert "no ROCm wheel index could be selected" in said and "An AMD GPU" not in said


@requires_pwsh
def test_get_torch_index_url_hands_a_forced_host_to_the_amd_route():
    kv = _ps_kv(
        _script(
            _env(),
            "Remove-Item Env:UNSLOTH_TORCH_INDEX_URL, Env:UNSLOTH_TORCH_INDEX_FAMILY, Env:UNSLOTH_PYTORCH_MIRROR -ErrorAction SilentlyContinue",
            "function Invoke-NvidiaSmiBounded { throw 'nvidia-smi must not be asked' }",
            _function_source(INSTALL_SRC, "Get-TorchIndexUrl"),
            "$NvidiaSmiExe = 'C:\\nvidia-smi.exe'; $script:WoaNativeCudaTorch = $false",
            "$RocmTorchOverNvidia = $true",
            'Write-Output "URL=$(Get-TorchIndexUrl)"',
        )
    )
    assert kv["URL"] == "https://download.pytorch.org/whl/cpu"


def _setup_decision(arch, wheels, **env):
    return _ps_kv(
        _script(
            COLLECT,
            SETUP_HELPERS,
            _env(**env),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            "$HasNvidiaSmi = $true; $script:ProbeAmdBesideNvidia = $true",
            f"$script:ROCmGfxArch = {'$null' if arch is None else repr(arch)}",
            f"$AmdHasGpuWheels = ${str(wheels).lower()}",
            "$HasROCm = $true; $HipSdkInstalled = $true; $ROCmGpuLabel = 'x'; $ROCmGpuName = 'x'",
            SETUP_DECISION,
            'Write-Output "NVIDIA=$HasNvidiaSmi"',
            'Write-Output "ARCH=$script:ROCmGfxArch"',
            'Write-Output "HASROCM=$HasROCm"',
            "Write-Output (\"SAID=\" + ($script:Said -join ' | '))",
        )
    )


@requires_pwsh
class TestSetupPs1Decision:
    def test_force_keeps_rocm_instead_of_restoring_cuda(self):
        kv = _setup_decision("gfx1201", True, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert kv["NVIDIA"] == "False" and kv["ARCH"] == "gfx1201"
        assert "UNSLOTH_FORCE_ROCM_TORCH is set" in kv["SAID"]

    def test_mask_alone_also_keeps_rocm_and_warns_on_update(self):
        kv = _setup_decision("gfx1100", True, CUDA_VISIBLE_DEVICES = "-1")
        assert kv["NVIDIA"] == "False" and "CUDA_VISIBLE_DEVICES hides" in kv["SAID"]
        assert "ROCm reads CUDA_VISIBLE_DEVICES too" in kv["SAID"]

    def test_force_with_a_hiding_mask_still_warns_on_update(self):
        kv = _setup_decision(
            "gfx1100", True, UNSLOTH_FORCE_ROCM_TORCH = "1", CUDA_VISIBLE_DEVICES = "-1"
        )
        assert "ROCm reads CUDA_VISIBLE_DEVICES too" in kv["SAID"]

    def test_force_alone_does_not_warn_about_a_mask(self):
        kv = _setup_decision("gfx1100", True, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert "ROCm reads" not in kv["SAID"]

    def test_no_second_mask_warning_inside_install_ps1(self):
        kv = _setup_decision("gfx1100", True, CUDA_VISIBLE_DEVICES = "-1", SKIP_STUDIO_BASE = "1")
        assert "ROCm reads" not in kv["SAID"]

    def test_no_mask_warning_when_hip_chooses_the_card(self):
        kv = _setup_decision("gfx1100", True, CUDA_VISIBLE_DEVICES = "-1", HIP_VISIBLE_DEVICES = "0")
        assert "ROCm reads" not in kv["SAID"]

    def test_force_without_a_wheel_route_resets_the_amd_probe(self):
        kv = _setup_decision("gfx803", False, UNSLOTH_FORCE_ROCM_TORCH = "1")
        assert kv["NVIDIA"] == "True" and kv["ARCH"] == "" and kv["HASROCM"] == "False"
        assert "keeping CUDA" in kv["SAID"]


@requires_pwsh
@pytest.mark.parametrize(
    "src, gate",
    [
        (INSTALL_SRC, "$ProbeAmdBesideNvidia = "),
        (SETUP_SRC, "$script:ProbeAmdBesideNvidia = "),
    ],
    ids = ["install.ps1", "setup.ps1"],
)
@pytest.mark.parametrize(
    "env, nvidia, expected",
    [
        ({"UNSLOTH_FORCE_ROCM_TORCH": "1"}, True, "True"),
        ({"UNSLOTH_FORCE_ROCM_TORCH": "1"}, False, "False"),
        (
            {"UNSLOTH_FORCE_ROCM_TORCH": "1", "UNSLOTH_TORCH_INDEX_URL": "https://x/whl/cu130"},
            True,
            "False",
        ),
        ({"CUDA_VISIBLE_DEVICES": "-1"}, True, "True"),
        ({}, True, None),
    ],
)
def test_the_amd_probe_runs_beside_nvidia_only_when_it_can_matter(src, gate, env, nvidia, expected):
    if expected is None:
        # install.ps1 also probes without the flag, for the notice; setup.ps1 (an update) does not.
        expected = "True" if src is INSTALL_SRC else "False"
    stmt = slice_between(src, gate, "\nif (" if src is SETUP_SRC else "\n    if (")
    kv = _ps_kv(
        _script(
            SETUP_HELPERS,
            functions(SETUP_SRC, "Trim-IndexPathSlashes", "Get-PinnedTorchIndexUrl"),
            _env(),
            "Remove-Item Env:UNSLOTH_TORCH_INDEX_URL, Env:UNSLOTH_TORCH_INDEX_FAMILY -ErrorAction SilentlyContinue",
            *[f"$env:{k} = '{v}'" for k, v in env.items()],
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            f"$HasNvidiaSmi = ${str(nvidia).lower()}; $SkipTorch = $false",
            stmt,
            "Write-Output \"PROBE=$(if ($script:ProbeAmdBesideNvidia -or $ProbeAmdBesideNvidia) { 'True' } else { 'False' })\"",
        )
    )
    assert kv["PROBE"] == expected


@pytest.mark.parametrize(
    "name",
    [
        "Test-CudaMaskHidesNvidia",
        "Test-RocmTorchExplicitlyRequested",
        "Write-CudaMaskHidesAmdWarning",
    ],
)
def test_install_and_setup_keep_one_copy_of_each_helper(name):
    install, setup = _ps_copies(name)
    assert install == setup


@requires_pwsh
@pytest.mark.parametrize(
    "value", ["1", "true", " TRUE ", "yes", "on", "0", "false", "", "2", "enabled"]
)
def test_the_flag_parses_like_install_python_stack(value):
    kv = _ps_kv(
        _script(
            SETUP_HELPERS,
            _env(UNSLOTH_FORCE_ROCM_TORCH = value),
            'Write-Output "FORCE=$(Test-RocmTorchExplicitlyRequested)"',
        )
    )
    with patch.dict(os.environ, {"UNSLOTH_FORCE_ROCM_TORCH": value}):
        assert kv["FORCE"] == str(stack_mod._rocm_torch_explicitly_requested())


class TestStackWindowsForce:
    """install_python_stack.py's Windows ROCm repair used to return on any usable NVIDIA GPU, since
    install.ps1 / setup.ps1 would have put CUDA back. Now they honour the flag, so the stack does too."""

    def _run(self, force: bool):
        detect = MagicMock(return_value = None)
        env = {"UNSLOTH_FORCE_ROCM_TORCH": "1"} if force else {}
        with patch.dict(os.environ, env, clear = False):
            if not force:
                os.environ.pop("UNSLOTH_FORCE_ROCM_TORCH", None)
            os.environ.pop("UNSLOTH_ROCM_TORCH_INSTALLED", None)
            with (
                patch.object(stack_mod, "IS_WINDOWS", True),
                patch.object(stack_mod, "IS_MACOS", False),
                patch.object(stack_mod, "_TORCH_BACKEND", ""),
                patch.object(stack_mod, "_explicit_rocm_torch_index_url", return_value = None),
                patch.object(
                    stack_mod, "_explicit_unknown_family_torch_index_url", return_value = None
                ),
                patch.object(stack_mod, "_explicit_torch_index_is_unusable", return_value = False),
                patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True),
                patch.object(stack_mod, "_detect_windows_gfx_arch", detect),
            ):
                stack_mod._ensure_rocm_torch()
        return detect

    def test_forced_mixed_host_looks_for_the_amd_card(self):
        assert self._run(force = True).called

    def test_unforced_mixed_host_keeps_cuda_untouched(self):
        assert not self._run(force = False).called


HIPINFO_CALL = slice_between(
    INSTALL_SRC,
    "                # Beside NVIDIA a device-naming CUDA mask",
    "                if ($hipOut -match",
)


@requires_pwsh
@pytest.mark.parametrize(
    "beside, env, seen",
    [
        (True, {"CUDA_VISIBLE_DEVICES": "0"}, "<unset>"),
        (False, {"CUDA_VISIBLE_DEVICES": "0"}, "0"),
        (True, {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"}, "0"),
        (True, {"CUDA_VISIBLE_DEVICES": "-1"}, "-1"),
    ],
    ids = ["nvidia-mask-beside-nvidia", "amd-only-host", "hip-mask-wins", "hiding-mask"],
)
def test_hipinfo_beside_nvidia_ignores_a_cuda_mask_naming_nvidia_cards(beside, env, seen):
    kv = _ps_kv(
        _script(
            SETUP_HELPERS,
            _env(**env),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            f"$ProbeAmdBesideNvidia = ${str(beside).lower()}",
            "function Invoke-FakeHipinfo { if ($null -eq $env:CUDA_VISIBLE_DEVICES) { '<unset>' } else { $env:CUDA_VISIBLE_DEVICES } }",
            "$hipinfoExe = [pscustomobject]@{ Source = 'Invoke-FakeHipinfo' }",
            HIPINFO_CALL,
            'Write-Output "SEEN=$($hipOut.Trim())"',
            'Write-Output "AFTER=$env:CUDA_VISIBLE_DEVICES"',
        )
    )
    assert kv["SEEN"] == seen
    assert kv["AFTER"] == env["CUDA_VISIBLE_DEVICES"]
