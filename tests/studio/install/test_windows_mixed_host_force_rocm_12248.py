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
    "Resolve-MixedHostShadowingArch",
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

    def test_force_installs_for_the_discrete_card_behind_a_shadowing_igpu(self):
        kv = _install_decision(
            "gfx1036", True, hip_arches = ("gfx1036", "gfx1201"), UNSLOTH_FORCE_ROCM_TORCH = "1"
        )
        assert kv["ROUTE"] == "rocm" and kv["ARCH"] == "gfx1201"

    def test_force_reaches_a_discrete_card_behind_an_igpu_with_no_wheels(self):
        kv = _install_decision(
            "gfx90c", False, hip_arches = ("gfx90c", "gfx1100"), UNSLOTH_FORCE_ROCM_TORCH = "1"
        )
        assert kv["ROUTE"] == "rocm" and kv["ARCH"] == "gfx1100"

    @pytest.mark.parametrize("mask", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
    def test_a_mask_keeps_the_igpu_pick_as_setup_ps1_does(self, mask):
        kv = _install_decision(
            "gfx1036",
            True,
            hip_arches = ("gfx1036", "gfx1201"),
            UNSLOTH_FORCE_ROCM_TORCH = "1",
            **{mask: "0"},
        )
        assert kv["ARCH"] == "gfx1036"

    def test_strix_is_not_a_shadowing_igpu(self):
        kv = _install_decision(
            "gfx1151", True, hip_arches = ("gfx1151", "gfx1201"), UNSLOTH_FORCE_ROCM_TORCH = "1"
        )
        assert kv["ARCH"] == "gfx1151"

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
    stmt = slice_between(
        src, gate, "\nif (" if src is SETUP_SRC else "\n    # Beside NVIDIA with neither"
    )
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
    "                # Notice-only probe: a device-naming CUDA mask",
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
        (True, {"CUDA_VISIBLE_DEVICES": "1", "UNSLOTH_FORCE_ROCM_TORCH": "1"}, "1"),
    ],
    ids = [
        "nvidia-mask-beside-nvidia",
        "amd-only-host",
        "hip-mask-wins",
        "hiding-mask",
        "forced-keeps-mask",
    ],
)
def test_hipinfo_beside_nvidia_ignores_a_cuda_mask_naming_nvidia_cards(beside, env, seen):
    kv = _ps_kv(
        _script(
            SETUP_HELPERS,
            _env(**env),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            f"$ProbeAmdBesideNvidia = ${str(beside).lower()}",
            slice_between(INSTALL_SRC, "    $script:StudioMuteLines = $ProbeAmdBesideNvidia", "\n"),
            "function Invoke-FakeHipinfo { if ($null -eq $env:CUDA_VISIBLE_DEVICES) { '<unset>' } else { $env:CUDA_VISIBLE_DEVICES } }",
            "$hipinfoExe = [pscustomobject]@{ Source = 'Invoke-FakeHipinfo' }",
            HIPINFO_CALL,
            'Write-Output "SEEN=$($hipOut.Trim())"',
            'Write-Output "AFTER=$env:CUDA_VISIBLE_DEVICES"',
        )
    )
    assert kv["SEEN"] == seen
    assert kv["AFTER"] == env["CUDA_VISIBLE_DEVICES"]


def test_install_ps1_shadowing_list_matches_setup_ps1():
    import re

    setup = slice_between(SETUP_SRC, "$script:ShadowingIntegratedGfx = @(", ")")
    install = slice_between(
        _function_source(INSTALL_SRC, "Resolve-MixedHostShadowingArch"), "$shadowing = @(", ")"
    )
    assert re.findall(r"gfx\w+", install) == re.findall(r"gfx\w+", setup)


PROBE_MUTE = slice_between(INSTALL_SRC, "    $script:StudioMuteLines = $ProbeAmdBesideNvidia", "\n")


@requires_pwsh
@pytest.mark.parametrize(
    "beside, env, heard",
    [
        (True, {}, False),
        (True, {"UNSLOTH_FORCE_ROCM_TORCH": "1"}, True),
        (True, {"CUDA_VISIBLE_DEVICES": "-1"}, True),
        (False, {}, True),
    ],
    ids = ["default-mixed-host", "forced", "hiding-mask", "amd-only-host"],
)
def test_the_amd_probe_is_silent_only_where_cuda_always_wins(beside, env, heard):
    kv = _ps_kv(
        _script(
            functions(INSTALL_SRC, "Write-StudioLine"),
            INSTALL_HELPERS,
            _env(**env),
            "$script:StudioStdoutRedirected = $true",
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            f"$ProbeAmdBesideNvidia = ${str(beside).lower()}",
            PROBE_MUTE,
            "Write-StudioLine 'PROBE=said'",
            "$script:StudioMuteLines = $false",
            "Write-StudioLine 'AFTER=said'",
        )
    )
    assert ("PROBE" in kv) == heard
    assert kv["AFTER"] == "said"


class TestStackWindowsForcedFlavor:
    """A forced ROCm swap must survive the rest of a direct run: the flavor invariant and setup.ps1's
    device-pack fast path read the flag like _ensure_rocm_torch, not the NVIDIA probe alone."""

    def _patches(
        self,
        gfx = "gfx1201",
        nvidia = True,
        **over,
    ):
        return tuple(patch.object(stack_mod, k, return_value = v) for k, v in over.items()) + (
            patch.object(stack_mod, "IS_WINDOWS", True),
            patch.object(stack_mod, "IS_MACOS", False),
            patch.object(stack_mod, "NO_TORCH", False),
            patch.object(stack_mod, "_TORCH_BACKEND", ""),
            patch.object(stack_mod, "_RECORDED_TORCH_TAG", "cu128"),
            patch.object(stack_mod, "_handover_torch_flavor_tag", return_value = ""),
            patch.object(stack_mod, "_explicit_torch_index_family", return_value = None),
            patch.object(stack_mod, "_explicit_unknown_family_torch_index_url", return_value = None),
            patch.object(stack_mod, "_explicit_torch_index_is_unusable", return_value = False),
            patch.object(stack_mod, "_explicit_rocm_torch_index_url", return_value = None),
            patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = nvidia),
            patch.object(stack_mod, "_detect_windows_gfx_arch", return_value = gfx),
        )

    def _run(self, fn, force, **kw):
        from contextlib import ExitStack
        env = {"UNSLOTH_FORCE_ROCM_TORCH": "1"} if force else {}
        with patch.dict(os.environ, env, clear = False), ExitStack() as stack:
            if not force:
                os.environ.pop("UNSLOTH_FORCE_ROCM_TORCH", None)
            seen = set()
            for p in self._patches(**kw):
                if p.attribute not in seen:
                    seen.add(p.attribute)
                    stack.enter_context(p)
            return fn()

    @pytest.mark.parametrize(
        "force, kw, expected",
        [
            (True, {}, "rocm"),
            (False, {}, "cu128"),
            (True, {"gfx": None}, "cu128"),
            (True, {"gfx": "gfx803"}, "cu128"),
        ],
        ids = ["forced", "unforced", "no-amd-card", "no-wheel-route"],
    )
    def test_the_flavor_invariant_keeps_a_forced_swap(self, force, kw, expected):
        assert self._run(stack_mod._expected_torch_flavor_tag, force, **kw) == expected

    def test_a_forced_rocm_torch_is_not_reinstalled_as_cuda(self):
        pip = MagicMock()
        with (
            patch.object(
                stack_mod,
                "_probe_torch_runtime",
                return_value = (True, True, "2.11.0+rocm7.14.0", "7.14", None),
            ),
            patch.object(stack_mod, "pip_install", pip),
            patch.object(stack_mod, "_is_win_arm64_interpreter", return_value = False),
        ):
            assert self._run(stack_mod._ensure_expected_torch_flavor, True) is True
        assert not pip.called

    @pytest.mark.parametrize("force, expected", [(True, True), (False, False)])
    def test_the_device_pack_fast_path_reads_the_flag(self, force, expected):
        with (
            patch.object(stack_mod, "_windows_routes_multiarch", return_value = True),
            patch.object(stack_mod, "_multiarch_device_pack_installed", return_value = False),
            patch("importlib.metadata.version", return_value = "2.11.0+rocm7.14.0"),
        ):
            assert (
                self._run(stack_mod._windows_rocm_device_packs_need_dependency_pass, force)
                is expected
            )

    def test_an_opaque_pin_keeps_its_own_expectation(self):
        pin = {"_explicit_unknown_family_torch_index_url": "https://m/simple"}
        assert self._run(stack_mod._expected_torch_flavor_tag, True, **pin) == ""

    def test_a_cuda_handover_from_setup_ps1_wins_over_the_python_probe(self):
        cuda = {"_handover_torch_flavor_tag": "cu128"}
        assert not self._run(stack_mod._windows_forced_rocm_is_viable, True, **cuda)
        install = MagicMock()
        with patch.object(stack_mod, "pip_install", install):
            self._run(stack_mod._ensure_rocm_torch, True, **cuda)
        assert not install.called


AMD_SMI_GATE = slice_between(INSTALL_SRC, "        $amdSmiAllowed = ", "\n")


@requires_pwsh
@pytest.mark.parametrize("muted, expected", [(True, "False"), (False, "True")])
def test_the_notice_only_probe_never_runs_amd_smi(muted, expected):
    kv = _ps_kv(
        _script(
            "Remove-Item Env:UNSLOTH_ENABLE_AMD_SMI -ErrorAction SilentlyContinue",
            "$amdSmiOptOut = $false; $HipSdkInstalled = $true",
            f"$script:StudioMuteLines = ${str(muted).lower()}",
            AMD_SMI_GATE,
            'Write-Output "ALLOWED=$amdSmiAllowed"',
        )
    )
    assert kv["ALLOWED"] == expected


NAME_ARCH_TABLE = slice_between(
    INSTALL_SRC,
    "                $nameArchTable = @(",
    "                foreach ($row in $nameArchTable)",
)


@requires_pwsh
@pytest.mark.parametrize(
    "names, env, arch, notice",
    [
        (
            ["AMD Radeon 780M Graphics", "AMD Radeon RX 9070 XT"],
            {"UNSLOTH_FORCE_ROCM_TORCH": "1"},
            "gfx1201",
            "",
        ),
        (["AMD Radeon 780M Graphics", "AMD Radeon RX 9070 XT"], {}, "", "gfx1201"),
        (
            ["AMD Radeon 780M Graphics", "AMD Radeon RX 9070 XT"],
            {"UNSLOTH_FORCE_ROCM_TORCH": "1", "UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
            "gfx1103",
            "",
        ),
        (["AMD Radeon 780M Graphics"], {"UNSLOTH_FORCE_ROCM_TORCH": "1"}, "gfx1103", ""),
    ],
    ids = ["forced-picks-discrete", "notice-names-discrete", "declared-arch-wins", "igpu-only"],
)
def test_a_wmi_only_mixed_host_weighs_every_amd_adapter(names, env, arch, notice):
    listed = ", ".join(repr(n) for n in names)
    kv = _ps_kv(
        _script(
            COLLECT,
            INSTALL_HELPERS,
            INSTALL_WHEEL_MAPS,
            NAME_ARCH_TABLE,
            "$_hipAllArches = @()",
            f"$wmiAmdNames = @({listed})",
            _env(**env),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            "$HasNvidiaSmi = $true; $ProbeAmdBesideNvidia = $true",
            "$ROCmGfxArch = 'gfx1103'",
            "$AmdHasGpuWheels = $true",
            "$HasROCm = $false; $HipSdkInstalled = $false; $ROCmGpuLabel = 'x'; $ROCmGpuName = 'x'",
            "$ROCmUnsupportedGfxArch = $null",
            INSTALL_DECISION,
            'Write-Output "ARCH=$ROCmGfxArch"',
            'Write-Output "NOTICE=$MixedHostAmdArch"',
        )
    )
    assert kv["ARCH"] == arch and kv["NOTICE"] == notice


@requires_pwsh
def test_a_forced_mixed_host_weighs_the_amd_smi_arches():
    kv = _ps_kv(
        _script(
            COLLECT,
            INSTALL_HELPERS,
            INSTALL_WHEEL_MAPS,
            "$_hipAllArches = @(); $_asicGfxTokens = @(); $_smiGfxTokens = @('gfx90c', 'gfx1100')",
            _env(UNSLOTH_FORCE_ROCM_TORCH = "1"),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            "$HasNvidiaSmi = $true; $ProbeAmdBesideNvidia = $true",
            "$ROCmGfxArch = 'gfx90c'; $AmdHasGpuWheels = $false",
            "$HasROCm = $true; $HipSdkInstalled = $true; $ROCmGpuLabel = 'x'; $ROCmGpuName = 'x'",
            "$ROCmUnsupportedGfxArch = $null",
            INSTALL_DECISION,
            'Write-Output "ROUTE=$MixedHostRoute"',
            'Write-Output "ARCH=$ROCmGfxArch"',
        )
    )
    assert kv["ROUTE"] == "rocm" and kv["ARCH"] == "gfx1100"


SMI_VIS_IDX = slice_between(INSTALL_SRC, "                        $_smiVisIdx = ", "\n")


@requires_pwsh
@pytest.mark.parametrize(
    "beside, env, idx",
    [
        (True, {"CUDA_VISIBLE_DEVICES": "1"}, "1"),
        (False, {"CUDA_VISIBLE_DEVICES": "1"}, "0"),
        (True, {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"}, "0"),
    ],
    ids = ["forced-route-reads-cuda-mask", "amd-only-host-unchanged", "hip-mask-wins"],
)
def test_the_amd_smi_pick_applies_a_cuda_mask_beside_nvidia(beside, env, idx):
    kv = _ps_kv(
        _script(
            _env(**env),
            f"$ProbeAmdBesideNvidia = ${str(beside).lower()}",
            SMI_VIS_IDX,
            'Write-Output "IDX=$_smiVisIdx"',
        )
    )
    assert kv["IDX"] == idx


@requires_pwsh
@pytest.mark.parametrize("src", [INSTALL_SRC, SETUP_SRC], ids = ["install.ps1", "setup.ps1"])
def test_the_logger_runs_under_strict_mode(src):
    init = "$script:StudioMuteLines = $false"
    assert init in src.split("function Write-StudioLine")[0]
    kv = _ps_kv(
        _script(
            "Set-StrictMode -Version Latest",
            "$script:StudioStdoutRedirected = $true",
            init,
            _function_source(src, "Write-StudioLine"),
            "Write-StudioLine 'OK=yes'",
        )
    )
    assert kv["OK"] == "yes"


AMD_WMI_SCAN = slice_between(
    INSTALL_SRC, "                $amdWmiErr = $null\n", "                $healthyAdapters = "
)


@requires_pwsh
@pytest.mark.parametrize(
    "beside, source", [(True, "bounded"), (False, "raw")], ids = ["beside-nvidia", "amd-only-host"]
)
def test_the_amd_wmi_scan_beside_nvidia_is_bounded(beside, source):
    kv = _ps_kv(
        _script(
            "function Invoke-BoundedVideoControllerScan { [pscustomobject]@{ Ok = $true; Adapters = @([pscustomobject]@{ Name = 'AMD Radeon RX 9070 XT (bounded)' }) } }",
            "function Get-CimInstance { [pscustomobject]@{ Name = 'AMD Radeon RX 9070 XT (raw)' } }",
            f"$ProbeAmdBesideNvidia = ${str(beside).lower()}",
            AMD_WMI_SCAN,
            'Write-Output "NAME=$($amdAdapters[0].Name)"',
            'Write-Output "SCANNED=$amdWmiScanned"',
        )
    )
    assert source in kv["NAME"] and kv["SCANNED"] == "True"


@requires_pwsh
def test_a_forced_mixed_host_falls_back_to_a_mapped_peer_when_adapter_0_is_unknown():
    kv = _ps_kv(
        _script(
            COLLECT,
            INSTALL_HELPERS,
            INSTALL_WHEEL_MAPS,
            NAME_ARCH_TABLE,
            "$_hipAllArches = @(); $_smiGfxTokens = @(); $_asicGfxTokens = @()",
            "$wmiAmdNames = @('AMD Radeon(TM) Graphics', 'AMD Radeon RX 9070 XT')",
            _env(UNSLOTH_FORCE_ROCM_TORCH = "1"),
            "$nvMaskHidesAll = Test-CudaMaskHidesNvidia",
            "$HasNvidiaSmi = $true; $ProbeAmdBesideNvidia = $true",
            "$ROCmGfxArch = $null; $AmdHasGpuWheels = $false",
            "$HasROCm = $false; $HipSdkInstalled = $false; $ROCmGpuLabel = 'AMD Radeon(TM) Graphics'; $ROCmGpuName = 'x'",
            "$ROCmUnsupportedGfxArch = $null",
            INSTALL_DECISION,
            'Write-Output "ROUTE=$MixedHostRoute"',
            'Write-Output "ARCH=$ROCmGfxArch"',
        )
    )
    assert kv["ROUTE"] == "rocm" and kv["ARCH"] == "gfx1201"
