# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Routing tests for ROCm targets absent from PyTorch's generic wheels.

Torch, hardware probes, and pip are mocked; the generic wheel architecture list was measured.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_9396", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

_CPU_TORCH = stack_mod._TORCH_PROBE_MARKER + "2.10.0+cpu||\n"
# Generic pytorch.org ROCm torch: links HIP, so has_hip_torch is True and every
# "already installed" gate sees a ROCm build -- the state the repair must still fix.
_ROCM_GENERIC_TORCH = stack_mod._TORCH_PROBE_MARKER + "2.10.0+rocm7.1|7.1|\n"
# AMD per-arch torch, as repo.amd.com serves it.
_ROCM_ARCH_TORCH = stack_mod._TORCH_PROBE_MARKER + "2.11.0+rocm7.13.0|7.13|\n"
# Generic wheels whose own rocm tag disagrees with the host: pinned once, or outlived by an
# /opt/rocm upgrade.
_ROCM_GENERIC_TORCH_63 = stack_mod._TORCH_PROBE_MARKER + "2.9.1+rocm6.3|6.3|\n"
_ROCM_GENERIC_TORCH_72 = stack_mod._TORCH_PROBE_MARKER + "2.11.0+rocm7.2|7.2|\n"

_GENERIC = "download.pytorch.org/whl/rocm"
_AMD = "repo.amd.com/rocm/whl"


@pytest.fixture(autouse = True)
def _reset_torch_runtime_probe():
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


def _run_install(
    *,
    gfx_devices = ("gfx1103",),
    inferred = None,
    rocm_version = (7, 1),
    env = None,
    kfd = (),
    torch_probe = None,
    installed_family = None,
    rocm_gpu_visible = True,
    torch_owns_rocm = True,
    probe_source = "kfd",
    unmasked_gfx_devices = None,
):
    """Drive _ensure_rocm_torch() over a host with ``gfx_devices`` and return the pip calls."""
    # The torch probe is cached per process and the autouse fixture clears it once per TEST,
    # so a case driving two hosts in a row would judge the second by the first's torch.
    stack_mod._invalidate_torch_runtime_probe()
    probe = MagicMock()
    probe.returncode = 0
    probe.stdout = torch_probe or _CPU_TORCH

    probes = []

    def _fake_detect(
        dedup = True,
        ignore_hsa_override = False,
        ignore_visible_masks = False,
    ):
        probes.append(dedup)
        # rocminfo runs on the ROCr stack, so its answer is already mask-filtered; only a
        # re-probe that strips the masks sees the whole machine. A caller that sets
        # unmasked_gfx_devices is modelling exactly that gap.
        codes = list(
            unmasked_gfx_devices
            if (ignore_visible_masks and unmasked_gfx_devices is not None)
            else gfx_devices
        )
        # Callers read the recorded probe to decide whether the list is ROCr-filtered and in
        # HIP order. This one hands back device order, which is KFD sysfs; leaving the global
        # as the previous test left it makes those decisions test-order dependent.
        stack_mod._LAST_AMD_GFX_PROBE = probe_source if codes else None
        return list(dict.fromkeys(codes)) if dedup else codes

    _env = dict(env or {})
    if _env.get("UNSLOTH_ROCM_GFX_ARCH"):
        inferred = _env["UNSLOTH_ROCM_GFX_ARCH"]

    with (
        patch.object(stack_mod, "IS_WINDOWS", False),
        patch.object(stack_mod, "pip_install_try", return_value = True) as pip_try,
        patch.object(stack_mod, "pip_install") as pip,
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack_mod, "_has_rocm_gpu", return_value = rocm_gpu_visible),
        patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = inferred),
        patch.object(stack_mod, "_detect_amd_gfx_codes", side_effect = _fake_detect),
        patch.object(stack_mod, "_detect_rocm_version", return_value = rocm_version),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(kfd), create = True),
        patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = installed_family),
        patch.object(stack_mod, "_torch_requires_rocm_sdk", return_value = torch_owns_rocm),
        patch.dict(os.environ, _env, clear = False),
    ):
        for _stale in (
            "UNSLOTH_ROCM_GFX_ARCH",
            "UNSLOTH_AMD_ROCM_MIRROR",
            "UNSLOTH_TORCH_INDEX_URL",
            "UNSLOTH_TORCH_INDEX_FAMILY",
            "HSA_OVERRIDE_GFX_VERSION",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            if _stale not in _env:
                os.environ.pop(_stale, None)
        _out = io.StringIO()
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = probe):
                with contextlib.redirect_stdout(_out):
                    stack_mod._ensure_rocm_torch()
        _run_install.hsa_override_after = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    _run_install.probes = probes
    _run_install.printed = _out.getvalue()
    return str(pip.call_args_list) + str(pip_try.call_args_list)


# ── the reported host, and its two siblings ──────────────────────────────────


@pytest.mark.parametrize(
    "gfx, leaf",
    [
        ("gfx1103", "gfx110X-all"),  # Radeon 780M / 760M / 740M -- the reported card
        ("gfx1032", "gfx103X-all"),  # RX 6650 / 6600
        ("gfx1034", "gfx103X-all"),  # RX 6500 / 6400 / 6300
    ],
)
def test_an_arch_with_no_generic_kernels_routes_to_the_amd_index(gfx, leaf):
    calls = _run_install(gfx_devices = (gfx,))
    assert f"{_AMD}/{leaf}/" in calls, calls
    assert _GENERIC not in calls, calls
    # The carrier is shared with the Strix reroute, so the label has to name the leaf
    # actually installed; a 780M laptop reading "Strix" is the installer lying to it.
    assert f"AMD per-gfx index, {leaf.lower()}" in calls, calls


def test_no_nameable_arch_is_left_without_kernels():
    """Every named arch has either generic kernels or an AMD per-arch route."""
    nameable = {arch for _pat, arch in stack_mod._WIN_GPU_NAME_ARCH_TABLE}
    stranded = {
        arch
        for arch in nameable
        if arch not in stack_mod._GENERIC_ROCM_WHEEL_GFX
        and not stack_mod._generic_rocm_wheel_lacks_kernels(arch)
    }
    assert stranded == set(), stranded


def test_the_rule_covers_the_rdna2_arches_the_name_table_never_spells():
    """Cover RDNA 2 arches reported by rocminfo but absent from the name table."""
    for arch in ("gfx1031", "gfx1033", "gfx1035", "gfx1036"):
        assert stack_mod._generic_rocm_wheel_lacks_kernels(arch), arch
        assert stack_mod._GFX_TO_AMD_INDEX_ARCH[arch] == "gfx103X-all"


def test_the_leaf_comes_from_the_map_windows_already_routes_by():
    """One arch/leaf pairing for both platforms; a second table here could drift from it."""
    assert stack_mod._GFX_TO_AMD_INDEX_ARCH["gfx1103"] == "gfx110X-all"
    assert stack_mod._GFX_TO_AMD_INDEX_ARCH["gfx1032"] == "gfx103X-all"


@pytest.mark.parametrize("arch", ["gfx1030", "gfx1100", "gfx1151", "gfx942", "gfx1201"])
def test_an_arch_the_generic_wheel_carries_is_never_rerouted(arch):
    assert not stack_mod._generic_rocm_wheel_lacks_kernels(arch)


def test_an_arch_with_no_amd_leaf_keeps_todays_behaviour():
    """Nothing better to route to, so the rule declines rather than inventing a leaf."""
    assert not stack_mod._generic_rocm_wheel_lacks_kernels("gfx803")
    assert not stack_mod._generic_rocm_wheel_lacks_kernels("gfx1010")
    assert not stack_mod._generic_rocm_wheel_lacks_kernels(None)


def test_the_reroute_is_not_gated_on_the_rocm_version_floor():
    """Missing generic kernels require rerouting regardless of the ROCm version."""
    calls = _run_install(gfx_devices = ("gfx1103",), rocm_version = (7, 13))
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_the_companion_pins_are_bounded_but_not_floored_at_211():
    """Bound companion versions without excluding older per-arch mirror builds."""
    calls = _run_install(gfx_devices = ("gfx1103",))
    for spec in stack_mod._ROCM_ARCH_INDEX_TORCH_PKG_SPEC:
        assert spec in calls, (spec, calls)
    assert "torch>=2.11.0" not in calls, calls
    # Every companion is bounded above, so none can drift to a different torch major.
    assert all("<" in spec for spec in stack_mod._ROCM_ARCH_INDEX_TORCH_PKG_SPEC)


def test_the_hardware_is_probed_once_for_both_reroutes():
    """The Strix and missing-kernel reroutes share one per-device hardware probe."""
    _run_install(gfx_devices = ("gfx1103",))
    per_device = [dedup for dedup in _run_install.probes if dedup is False]
    assert len(per_device) == 1, _run_install.probes


def test_the_shared_probe_still_reaches_strix_above_the_rocm_floor():
    """The shared probe still reaches missing-kernel routing above the Strix floor."""
    calls = _run_install(gfx_devices = ("gfx1103",), rocm_version = (7, 13))
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_the_mirror_override_is_honoured():
    calls = _run_install(
        gfx_devices = ("gfx1103",), env = {"UNSLOTH_AMD_ROCM_MIRROR": "https://mirror.test/whl"}
    )
    assert "https://mirror.test/whl/gfx110X-all/" in calls, calls


# ── the neighbours this must not disturb ─────────────────────────────────────


@pytest.mark.parametrize("gfx", ["gfx1100", "gfx1102", "gfx1030", "gfx1201"])
def test_a_supported_arch_keeps_the_generic_index(gfx):
    calls = _run_install(gfx_devices = (gfx,))
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


def test_strix_still_takes_its_own_reroute():
    """#7331's path runs through the extracted detection helper now; it must be unchanged."""
    calls = _run_install(gfx_devices = ("gfx1151",), inferred = "gfx1151")
    assert f"{_AMD}/gfx1151/" in calls, calls
    assert "torch>=2.11.0,<2.12.0" in calls, calls


def test_an_explicit_index_pin_wins():
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        env = {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm6.4"},
    )
    assert _AMD not in calls, calls


def test_the_gfx906_arch_override_wins():
    """A pinned MI50 runtime target must not be rerouted by a gfx1103 sitting in the same box."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx906"), env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx906"})
    assert _AMD not in calls, calls


def test_a_mixed_host_whose_runtime_target_is_supported_keeps_the_generic_index():
    """gfx1103 present, but the visible-device mask selects the dGPU: rerouting would pull the
    wrong wheel for the card that will actually run."""
    calls = _run_install(gfx_devices = ("gfx1100", "gfx1103"), env = {"HIP_VISIBLE_DEVICES": "0"})
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


# ── the probe the routing decision indexes into ──────────────────────────────

# Measured on an AMD DevLab strix-halo host (amd-smi 26.2.2): `list` carries no gfx
# token at all, only `static --asic` does, and both head every device with a
# line-leading "GPU: N" while naming no agent.
_AMD_SMI_LIST = "GPU: 0\n    BDF: 0000:03:00.0\n    KFD_ID: 40251\n    NODE_ID: 1\n"


def _amd_smi_asic(arches):
    return "".join(
        f"GPU: {i}\n"
        f"    ASIC:\n"
        f"        MARKET_NAME: AMD Radeon Graphics\n"
        f"        DEVICE_ID: 0x1586\n"
        f"        TARGET_GRAPHICS_VERSION: {arch}\n"
        for i, arch in enumerate(arches)
    )


def _amd_smi_only_host(arches):
    """Patches making the real probes see an amd-smi-only host with ``arches``."""

    def _which(name):
        return None if name == "rocminfo" else f"/usr/bin/{name}"

    def _run(cmd, **kwargs):
        out = MagicMock()
        out.returncode = 0
        out.stdout = (
            _AMD_SMI_LIST if list(cmd[:2]) == ["amd-smi", "list"] else _amd_smi_asic(arches)
        )
        return out

    return (
        patch("shutil.which", side_effect = _which),
        patch.object(stack_mod, "_amd_smi_allowed", return_value = True),
        patch("subprocess.run", side_effect = _run),
    )


def _detect_over_amd_smi(arches, dedup = False):
    _which_p, _allowed_p, _run_p = _amd_smi_only_host(arches)
    with _which_p, _allowed_p, _run_p:
        return stack_mod._detect_amd_gfx_codes(dedup = dedup)


def test_amd_smi_keeps_one_entry_per_device():
    """dedup=False indexes DEVICES, and amd-smi names no agent: without its GPU: N
    header the output is flat, so two cards of one arch collapse into one entry and
    every device ordinal past them addresses the wrong card."""
    assert _detect_over_amd_smi(("gfx1200", "gfx1200", "gfx1032")) == [
        "gfx1200",
        "gfx1200",
        "gfx1032",
    ]


def test_amd_smi_still_reports_arches_when_asked_to_dedup():
    assert _detect_over_amd_smi(("gfx1200", "gfx1200", "gfx1032"), dedup = True) == [
        "gfx1200",
        "gfx1032",
    ]


def test_a_repeated_arch_on_an_amd_smi_host_does_not_shift_the_mask():
    """HIP_VISIBLE_DEVICES=1 over gfx1200, gfx1200, gfx1032 selects a gfx1200. Reading a
    collapsed list would call it gfx1032 and install gfx103X-all wheels, stranding the
    card that will actually run -- on a host the generic index already served."""
    probe = MagicMock()
    probe.returncode = 0
    probe.stdout = _CPU_TORCH
    _which_p, _allowed_p, _run_p = _amd_smi_only_host(("gfx1200", "gfx1200", "gfx1032"))

    def _run(cmd, **kwargs):
        if str(cmd[0]) == sys.executable:
            return probe
        out = MagicMock()
        out.returncode = 0
        out.stdout = (
            _AMD_SMI_LIST
            if list(cmd[:2]) == ["amd-smi", "list"]
            else _amd_smi_asic(("gfx1200", "gfx1200", "gfx1032"))
        )
        return out

    with (
        patch.object(stack_mod, "IS_WINDOWS", False),
        patch.object(stack_mod, "pip_install_try", return_value = True) as pip_try,
        patch.object(stack_mod, "pip_install") as pip,
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False),
        patch.object(stack_mod, "_has_rocm_gpu", return_value = True),
        patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = None),
        patch.object(stack_mod, "_detect_rocm_version", return_value = (7, 1)),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = []),
        patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = None),
        patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1"}, clear = False),
        _which_p,
        _allowed_p,
    ):
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", side_effect = _run):
                stack_mod._ensure_rocm_torch()
    calls = str(pip.call_args_list) + str(pip_try.call_args_list)
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


# ── the sources the runtime target falls back to ─────────────────────────────


def test_kfd_topology_answers_when_neither_userland_probe_is_installed():
    """A runtime-only ROCm install ships no rocminfo and no amd-smi; the kernel's own
    topology still names the GPU, and _has_rocm_gpu() already reads that same sysfs."""
    calls = _run_install(gfx_devices = (), kfd = ("gfx1103",))
    assert f"{_AMD}/gfx110X-all/" in calls, calls
    assert _GENERIC not in calls, calls


@pytest.mark.parametrize(
    "mask", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
@pytest.mark.parametrize("value", ["", "-1"])
@pytest.mark.parametrize("probe", [(), ("gfx1103",)])
def test_a_mask_that_selects_no_gpu_takes_no_reroute(mask, value, probe):
    """An empty (or -1) visible-device mask selects NO GPU, whichever source answers.

    "" and "-1" are a deliberate choice of no device, and NO source here is filtered the
    way honouring that needs: only ROCR reaches rocminfo, amd-smi reads the driver, KFD
    sysfs. So a probe answering is no evidence the GPU is visible -- hence both the empty and
    populated probe, since CI exports an empty mask over a populated /sys/class/kfd. Generic
    torch still installs."""
    calls = _run_install(gfx_devices = probe, kfd = ("gfx1103",), env = {mask: value})
    assert f"{_AMD}/gfx110X-all/" not in calls, calls
    assert _GENERIC in calls, calls


def test_a_mask_naming_a_device_still_reroutes():
    """The guard is only for a mask that selects nothing: one that names an ordinal leaves
    both the runtime-only and the probed host exactly as they were."""
    calls = _run_install(gfx_devices = (), kfd = ("gfx1103",), env = {"HIP_VISIBLE_DEVICES": "0"})
    assert f"{_AMD}/gfx110X-all/" in calls, calls
    calls = _run_install(gfx_devices = ("gfx1103",), env = {"HIP_VISIBLE_DEVICES": "0"})
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_live_probe_outranks_kfd_topology():
    """Only the userland probe is renumbered by a visible-device mask, so it leads."""
    calls = _run_install(gfx_devices = ("gfx1100",), kfd = ("gfx1103",))
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


def test_the_named_arch_repairs_a_host_no_probe_can_see():
    """UNSLOTH_ROCM_GFX_ARCH over a generic ROCm torch: the inferred-arch install is
    gated on there being no HIP torch, so without this fallback the named GPU keeps a
    wheel that has no kernels for it."""
    calls = _run_install(
        gfx_devices = (),
        kfd = (),
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
        torch_probe = _ROCM_GENERIC_TORCH,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_probed_arch_still_wins_over_the_inferred_one():
    """The product-name fallback is last: a runtime that enumerates a GPU decides (#7305).
    That precedence is about the CPU-model guess, which is the half _infer_linux_amd_gfx_arch
    reaches only after the explicit variable, and the case below is the other half."""
    calls = _run_install(
        gfx_devices = ("gfx1100",),
        inferred = "gfx1103",
        torch_probe = _ROCM_GENERIC_TORCH,
    )
    assert _AMD not in calls, calls


def test_an_explicit_arch_outranks_the_probe():
    """UNSLOTH_ROCM_GFX_ARCH is the documented escape hatch, and install.sh,
    _infer_linux_amd_gfx_arch, _runtime_target_is_gfx906 and _detect_windows_gfx_arch all read
    it before probing. Resolving it after the probes made this the one place the hardware
    could overrule the override."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
        torch_probe = _ROCM_GENERIC_TORCH,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls
    # Hiding every GPU is a statement about this run, and still outranks the arch.
    assert _AMD not in _run_install(
        gfx_devices = ("gfx1200",),
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1103", "HIP_VISIBLE_DEVICES": ""},
        torch_probe = _ROCM_GENERIC_TORCH,
    )


def test_the_inferred_arch_install_is_not_repeated_by_the_reroute():
    """#7305's inferred install resolves its index from the same arch the reroutes
    would, so running both force-reinstalls the multi-GB stack twice in one call."""
    calls = _run_install(
        gfx_devices = (),
        inferred = "gfx1151",
        rocm_gpu_visible = False,
    )
    assert calls.count("--index-url") == 1, calls
    assert f"{_AMD}/gfx1151/" in calls, calls


# ── not re-downloading what is already installed ─────────────────────────────


def test_torch_already_on_the_right_per_arch_wheels_is_not_reinstalled():
    """_ensure_rocm_torch runs twice per install and again on every update, and the
    reroute is a --force-reinstall --no-cache-dir of a multi-GB stack."""
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert _AMD not in calls, calls
    assert _GENERIC not in calls, calls
    # Skipping the torch install must still leave rocm_torch_ready set, or the AMD
    # bitsandbytes repair every `studio update` exists for stops running too.
    assert "bitsandbytes" in calls, calls


@pytest.mark.parametrize("family", [None, "gfx103x-all"])
def test_an_unknown_or_foreign_family_still_reinstalls(family):
    """Act only on a family read back positively, never on a guess."""
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = family,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_stale_runtime_beside_a_non_rocm_torch_does_not_veto_the_repair():
    """A rocm-sdk-libraries left behind by an earlier install names a family while the
    torch on disk is CPU-only; the repair must still run."""
    calls = _run_install(gfx_devices = ("gfx1103",), installed_family = "gfx110x-all")
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_leaf_that_needs_torch_211_keeps_its_floor():
    """An unreadable ROCm version reads as 0.0, which is below the Strix floor, so a
    gfx1152 host lands on this branch instead. Its leaf is one of the four whose <2.11
    builds carry the _grouped_mm bug, and the generic branch already floors those."""
    calls = _run_install(gfx_devices = ("gfx1152",), inferred = "gfx1152", rocm_version = None)
    assert f"{_AMD}/gfx1152/" in calls, calls
    assert "torch>=2.11.0,<2.12.0" in calls, calls
    assert "torch>=2.4,<2.12.0" not in calls, calls


# ── the spoofed runtime the per-arch wheels cannot survive ───────────────────


def test_a_kfd_only_spoofed_host_clears_the_override_before_installing():
    """With no rocminfo and no amd-smi the spoof check has nothing to distrust and declines,
    but amdkfd writes gfx_target_version itself, so a single-arch kernel reading the override
    contradicts IS the corroborated spoof. Left set, the gfx1151-only wheels get a device the
    runtime still calls gfx1100 (#7331)."""
    calls = _run_install(
        gfx_devices = (),
        kfd = ("gfx1151",),
        inferred = "gfx1151",
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert f"{_AMD}/gfx1151/" in calls, calls
    assert _run_install.hsa_override_after is None, _run_install.hsa_override_after


def test_an_override_naming_the_real_arch_is_left_alone_on_the_kfd_path():
    """11.5.1 names gfx1151, which is what the kernel reports: nothing is being masked."""
    _run_install(
        gfx_devices = (),
        kfd = ("gfx1151",),
        inferred = "gfx1151",
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
    )
    assert _run_install.hsa_override_after == "11.5.1"


def test_a_mixed_kernel_reading_never_clears_the_override():
    """Two arches means the single-GPU premise the correction rests on does not hold."""
    _run_install(
        gfx_devices = (),
        kfd = ("gfx1151", "gfx1100"),
        inferred = "gfx1151",
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert _run_install.hsa_override_after == "11.0.0"


# ── what the banners are allowed to print ────────────────────────────────────


_SECRET_MIRROR = "https://user:s3cr3t-token@mirror.example/whl"


def test_the_missing_kernel_banner_redacts_mirror_credentials():
    """UNSLOTH_AMD_ROCM_MIRROR can carry userinfo, and installer output reaches CI logs."""
    calls = _run_install(gfx_devices = ("gfx1103",), env = {"UNSLOTH_AMD_ROCM_MIRROR": _SECRET_MIRROR})
    assert "mirror.example/whl/gfx110X-all/" in calls, calls
    assert "s3cr3t-token" not in _run_install.printed, _run_install.printed


def test_the_strix_banner_redacts_mirror_credentials():
    """Same carrier, same banner rule: the reroute this one shares its variables with."""
    _run_install(
        gfx_devices = ("gfx1151",),
        inferred = "gfx1151",
        env = {"UNSLOTH_AMD_ROCM_MIRROR": _SECRET_MIRROR},
    )
    assert "s3cr3t-token" not in _run_install.printed, _run_install.printed


# ── which card decides the family on a mixed host ────────────────────────────


@pytest.mark.parametrize("apu", ["gfx1103", "gfx1036", "gfx1035", "gfx1033"])
def test_a_leading_integrated_gpu_does_not_pick_the_family(apu):
    """Enumeration order alone puts the APU first on a Ryzen box with a Radeon card. The
    family is chosen for ONE arch, so letting the APU decide strands the discrete card the
    generic index was serving. _SHADOWING_INTEGRATED_GFX is the existing policy (#7776)."""
    calls = _run_install(gfx_devices = (apu, "gfx1200"))
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


def test_the_discrete_card_still_decides_when_it_needs_the_amd_index():
    """Deposing the APU is about which card decides, not about avoiding the reroute."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx1032"))
    assert f"{_AMD}/gfx103X-all/" in calls, calls


def test_a_visible_device_mask_still_wins_over_the_preference():
    """A pin is the user naming a device and is honoured verbatim, exactly as
    _visible_devices_pinned documents and _detect_windows_gfx_arch already behaves."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx1200"), env = {"HIP_VISIBLE_DEVICES": "0"})
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_lone_integrated_gpu_is_never_deposed():
    """Nothing else to run on, so the APU is the runtime target."""
    calls = _run_install(gfx_devices = ("gfx1103",))
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_an_all_integrated_host_is_never_deposed():
    """Every candidate is a shadowing APU, so there is no discrete card to prefer."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx1036"))
    assert f"{_AMD}/gfx110X-all/" in calls, calls


# ── building the index URL ───────────────────────────────────────────────────


def test_the_leaf_lands_on_the_path_not_inside_a_mirror_token():
    """A mirror can carry its token in the query. rstrip + concat buried the arch inside
    it and left the request pointing at the bare mirror path."""
    assert (
        stack_mod._index_url_join("https://m.example/whl?token=x", "gfx110X-all")
        == "https://m.example/whl/gfx110X-all/?token=x"
    )
    assert (
        stack_mod._index_url_join("https://m.example/whl#frag", "gfx110X-all")
        == "https://m.example/whl/gfx110X-all/#frag"
    )
    assert (
        stack_mod._index_url_join("https://m.example/whl?a=1#f", "gfx1152")
        == "https://m.example/whl/gfx1152/?a=1#f"
    )


def test_the_split_lands_on_whichever_delimiter_comes_first():
    """A fragment may itself contain "?". Reaching for "?" first then cuts inside the
    fragment, so the leaf is appended to a fragment and the request still asks for the bare
    path -- the same failure the query case above exists to prevent."""
    assert (
        stack_mod._index_url_join("https://m.example/whl#sha256=abc?download=1", "gfx110X-all")
        == "https://m.example/whl/gfx110X-all/#sha256=abc?download=1"
    )


def test_a_plain_mirror_is_joined_exactly_as_before():
    for base in ("https://repo.amd.com/rocm/whl", "https://repo.amd.com/rocm/whl/"):
        assert (
            stack_mod._index_url_join(base, "gfx103X-all")
            == "https://repo.amd.com/rocm/whl/gfx103X-all/"
        )


def test_a_query_mirror_reaches_the_arch_index_end_to_end():
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        env = {"UNSLOTH_AMD_ROCM_MIRROR": "https://m.example/whl?token=x"},
    )
    assert "https://m.example/whl/gfx110X-all/?token=x" in calls, calls


def test_every_strix_arch_has_a_leaf_in_the_shared_map():
    """Both reroutes now build their URL through _amd_arch_index_url, which returns None
    for an arch the map does not name; the Strix set must stay inside it."""
    for arch in ("gfx1150", "gfx1151", "gfx1152"):
        assert stack_mod._GFX_TO_AMD_INDEX_ARCH.get(arch) == arch, arch


def test_a_generic_torch_beside_a_stale_rocm_meta_package_is_still_repaired():
    """Measured 2026-08-27: forcing generic rocm7.1 torch over a per-arch install leaves
    `rocm` and rocm-sdk-libraries-gfx110X-all behind, so the family still reads gfx110x-all
    while torch is 2.10.0+rocm7.1 with no gfx1103 kernels. torch.version.hip is set on both,
    so only torch's own requires separates them."""
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        torch_probe = _ROCM_GENERIC_TORCH,
        installed_family = "gfx110x-all",
        torch_owns_rocm = False,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


# Verbatim `metadata.requires("torch")` for both builds, read on an AMD DevLab host
# after installing generic rocm7.1 torch over an AMD gfx110X-all one.
_AMD_TORCH_REQUIRES = [
    "filelock",
    "typing-extensions>=4.10.0",
    "setuptools<82",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
    "rocm[libraries]==7.13.0",
    "triton==3.6.0+rocm7.13.0",
]
_GENERIC_TORCH_REQUIRES = [
    "filelock",
    "typing-extensions>=4.10.0",
    'setuptools; python_version >= "3.12"',
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
    'triton-rocm==3.6.0; platform_system == "Linux"',
]


@pytest.mark.parametrize(
    "requires, owns",
    [(_AMD_TORCH_REQUIRES, True), (_GENERIC_TORCH_REQUIRES, False), ([], False)],
)
def test_the_ownership_check_reads_torchs_own_requires(requires, owns):
    """Only AMD's torch declares `rocm[libraries]`. triton-rocm rides along on the generic
    ROCm build and rocm-sdk-core is a sibling distribution, so neither may match."""
    with patch("importlib.metadata.requires", return_value = requires):
        assert stack_mod._torch_requires_rocm_sdk() is owns


def test_an_unroutable_discrete_card_does_not_depose_a_routable_apu():
    """gfx1010 (RDNA 1) is in neither the generic wheel nor the per-arch map, so no index
    can make it work. Handing it the decision trades a repairable 780M for nothing."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx1010"))
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_an_all_unroutable_host_keeps_enumeration_order():
    """Neither card has a route, so there is nothing to prefer and nothing to lose."""
    calls = _run_install(gfx_devices = ("gfx1013", "gfx1010"))
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls


# ── a per-arch install outliving the GPU it was made for ─────────────────────


def test_a_stale_per_arch_family_is_replaced_when_the_target_goes_generic():
    """An earlier gfx1103-only run installs gfx110X-all. Add a dGPU, or point
    HIP_VISIBLE_DEVICES at one, and those wheels carry no kernels for the new target,
    while torch.version.hip keeps rocm_torch_ready true and the fallback from running."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert _GENERIC in calls, calls


def test_a_target_inside_the_installed_family_is_left_alone():
    """gfx1101 is served by the same gfx110X-all build; nothing to repick."""
    calls = _run_install(
        gfx_devices = ("gfx1101",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert _GENERIC not in calls, calls
    assert _AMD not in calls, calls


def test_a_stale_family_is_replaced_by_the_right_per_arch_index_too():
    calls = _run_install(
        gfx_devices = ("gfx1032",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert f"{_AMD}/gfx103X-all/" in calls, calls


@pytest.mark.parametrize("family, owns", [(None, True), ("gfx110x-all", False)])
def test_an_unreadable_family_never_forces_a_reinstall(family, owns):
    """None is "unknowable" and a torch that does not own `rocm` is not a per-arch install;
    neither may churn a working venv."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = family,
        torch_owns_rocm = owns,
    )
    assert _GENERIC not in calls, calls
    assert _AMD not in calls, calls


# ── the three host shapes the reroutes reach only once a second signal is missing ──

_ROCM_ARCH_TORCH_210 = stack_mod._TORCH_PROBE_MARKER + "2.10.0+rocm7.13.0|7.13|\n"


def test_a_gfx906_dgpu_never_deposes_a_routable_apu():
    """gfx906's only route fires solely when it is the ONE detected arch, so naming it the
    target on a mixed host installs a rocm7.x wheel whose BLAS has no gfx906 kernels and
    strands both cards. The APU keeps the selection; gfx906 keeps its env opt-in."""
    calls = _run_install(gfx_devices = ("gfx1103", "gfx906"))
    assert f"{_AMD}/gfx110X-all/" in calls, calls
    assert "rocm6.3" not in calls and _GENERIC not in calls, calls
    # Alone it still takes the legacy route, so the exclusion is scoped to the mixed host.
    assert "rocm6.3" in _run_install(gfx_devices = ("gfx906",), rocm_version = (7, 1))


def test_a_sub_211_build_of_the_right_family_is_still_repaired():
    """A matching family is the right SHAPE, not a working build: on the 2.11-floor leaves a
    sub-2.11 wheel carries the _grouped_mm bug. An unreadable ROCm version routes gfx1152
    through the missing-kernel branch, so this is the only floor such a host meets."""
    calls = _run_install(
        gfx_devices = ("gfx1152",),
        inferred = "gfx1152",
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH_210,
        installed_family = "gfx1152",
    )
    assert f"{_AMD}/gfx1152/" in calls and "torch>=2.11.0,<2.12.0" in calls, calls


def test_at_the_floor_the_matching_family_is_still_skipped():
    """The other half of the same gate: at or above 2.11 the family match still wins, so an
    up-to-date host does not re-download the multi-GB stack on every update."""
    calls = _run_install(
        gfx_devices = ("gfx1152",),
        inferred = "gfx1152",
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx1152",
    )
    assert f"{_AMD}/gfx1152/" not in calls, calls
    assert "torch already runs on the gfx1152 wheels" in _run_install.printed


def test_a_stale_family_is_repaired_when_no_generic_tag_resolves():
    """Demoting hands the job to the generic fallback, which resolves no tag at all when the
    ROCm version is unreadable -- the shape of a bundled-runtime host. The branch announced a
    reinstall, installed nothing, and kept the family it just called incompatible."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        inferred = "gfx1200",
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert f"{_AMD}/gfx120X-all/" in calls, calls


# ── the routes a second, independent reading has to survive ───────────────────


@pytest.mark.parametrize("probe", [(), ("gfx1103",)])
def test_an_unreadable_rocm_version_still_routes_a_missing_kernel_arch(probe):
    """A bundled-runtime host has no /opt/rocm version to read, and the missing-kernel
    reroute has no version floor anyway: whichever index a version would pick ships no kernels
    for these arches. Returning early leaves the GPU on a wheel that faults."""
    calls = _run_install(
        gfx_devices = probe,
        kfd = ("gfx1103",),
        rocm_version = None,
        inferred = None,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


@pytest.mark.parametrize("mask", ["", "-1"])
def test_an_empty_rocr_mask_under_a_named_hip_mask_still_selects_no_gpu(mask):
    """ROCr filters BENEATH HIP, so the masks intersect: HIP naming device 0 over a ROCr mask
    exposing nothing leaves nothing to target. Checking only the first mask that is set
    reinstalls a multi-GB stack for a GPU the user hid."""
    calls = _run_install(
        gfx_devices = ("gfx1103",),
        env = {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": mask},
    )
    assert _AMD not in calls, calls
    # A mask that does name a device is untouched, so the guard is scoped to a hidden GPU.
    assert f"{_AMD}/gfx110X-all/" in _run_install(
        gfx_devices = ("gfx1103",),
        env = {"HIP_VISIBLE_DEVICES": "0"},
    )


def test_keeping_the_matching_wheels_also_clears_a_confirmed_spoof():
    """Keeping the wheels is not keeping the status quo. They carry the PHYSICAL arch alone,
    so a confirmed spoof left set has ROCr keep naming the one arch they have no code for
    (#7331). The reinstall arm clears it for that reason; the skip arm needs it as much."""
    calls = _run_install(
        gfx_devices = ("gfx1100",),
        inferred = "gfx1152",
        kfd = ("gfx1152",),
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx1152",
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert f"{_AMD}/gfx1152/" not in calls, calls
    assert "torch already runs on the gfx1152 wheels" in _run_install.printed
    assert _run_install.hsa_override_after is None, _run_install.hsa_override_after


# ── the two mask layers, and the install they outlive ────────────────────────


def test_the_rocr_layer_is_applied_before_the_hip_index_on_an_unfiltered_list():
    """ROCr is processed first and HIP indexes the SURVIVORS, so HIP's index is relative to
    what ROCr left. Neither amd-smi (driver) nor KFD sysfs (kernel) is ROCr-filtered:
    resolving HIP against those lists names a GPU the runtime does not expose, and installs
    per-arch wheels that fault on first use."""
    calls = _run_install(
        gfx_devices = ("gfx1200", "gfx1103"),
        env = {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"},
    )
    # ROCr leaves [gfx1103]; HIP index 0 is therefore the gfx1103, not the gfx1200.
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_rocr_filtered_probe_is_not_filtered_a_second_time():
    """rocminfo is the one probe ROCr renumbers, and it is renumbered whenever the mask is
    set -- not only when it happens to be the first mask set. Applying the mask again to a
    list it already shaped would index the survivors by a position in the original."""
    with (
        patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = ["gfx1103"]),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = [], create = True),
        patch.object(stack_mod, "_LAST_AMD_GFX_PROBE", "rocminfo"),
        patch.dict(
            os.environ,
            {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"},
            clear = False,
        ),
    ):
        assert stack_mod._runtime_gfx_target(None)[0] == "gfx1103"


def test_a_uuid_rocr_mask_on_one_architecture_still_routes():
    """ROCR_VISIBLE_DEVICES also takes GPU UUIDs, and may mix them with indices. A UUID
    names no position in a probe reporting arches, but on a host of one architecture it
    cannot change which arch is selected, so the reroute proceeds."""
    calls = _run_install(
        gfx_devices = ("gfx1103", "gfx1103"),
        env = {"ROCR_VISIBLE_DEVICES": "GPU-8d1f2e3a4b5c6d7e"},
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_an_unmappable_uuid_declines_the_reroute_on_a_mixed_host():
    """On a mixed host the UUID decides everything and nothing here can read it. The reroute
    installs wheels for ONE arch, so picking the first entry may serve the GPU the mask hid
    and leave the selected card with no kernels at all."""
    calls = _run_install(
        gfx_devices = ("gfx1103", "gfx1200"),
        env = {"ROCR_VISIBLE_DEVICES": "GPU-8d1f2e3a4b5c6d7e"},
    )
    assert _AMD not in calls, calls
    assert "names a GPU by UUID" in _run_install.printed, _run_install.printed
    # UNSLOTH_ROCM_GFX_ARCH is the documented way through, and the message says so.
    assert f"{_AMD}/gfx110X-all/" in _run_install(
        gfx_devices = ("gfx1103", "gfx1200"),
        env = {"ROCR_VISIBLE_DEVICES": "GPU-8d1f2e3a4b5c6d7e", "UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
    )


def test_a_stale_per_arch_family_is_repaired_even_when_the_rocm_version_is_unreadable():
    """A per-arch install outlives the GPU it was made for: swap a gfx1200 card into a box on
    gfx110X-all wheels and the generic index would serve it, so the unreadable-version exit
    stands and the stale-family repair never runs on wheels with no gfx1200 kernels."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert f"{_AMD}/gfx120X-all/" in calls, calls
    # A family that already matches still takes the exit, so an up-to-date host is untouched.
    _run_install(
        gfx_devices = ("gfx1200",),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx120x-all",
    )
    assert "skipping torch reinstall" in _run_install.printed, _run_install.printed


# ── two orders, and the mask that cannot be applied to either ────────────────


def _target(
    gfx_devices,
    probe_source,
    env,
    kfd = (),
):
    """Resolve the runtime target with ``probe_source`` named as the probe that answered.

    ``kfd`` is the kernel topology, empty by default: an unreadable /sys/class/kfd leaves
    only the probe's own ordering to go on."""
    with (
        patch.object(stack_mod, "_detect_amd_gfx_codes", return_value = list(gfx_devices)),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = list(kfd), create = True),
        patch.object(stack_mod, "_LAST_AMD_GFX_PROBE", probe_source),
        patch.dict(os.environ, env, clear = False),
    ):
        for _stale in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            if _stale not in env:
                os.environ.pop(_stale, None)
        return stack_mod._runtime_gfx_target(None)[0]


def test_a_mask_mixing_an_index_and_a_uuid_keeps_the_host_ambiguous():
    """AMD documents "0,GPU-DEADBEEFDEADBEEF" as a valid mask. Judging ambiguity on the list
    left AFTER the resolvable tokens were applied can leave one arch standing and hide the
    very ambiguity the UUID created, so the question has to be asked of the original."""
    assert (
        _target(
            ["gfx1103", "gfx1200"],
            "amd-smi",
            {"ROCR_VISIBLE_DEVICES": "0,GPU-8d1f2e3a4b5c6d7e", "HIP_VISIBLE_DEVICES": "1"},
        )
        is None
    )
    # One arch cannot be made ambiguous by any mask over it.
    assert (
        _target(
            ["gfx1103", "gfx1103"],
            "kfd",
            {"ROCR_VISIBLE_DEVICES": "0,GPU-8d1f2e3a4b5c6d7e"},
        )
        == "gfx1103"
    )


def test_amd_smi_discovery_order_is_not_indexed_as_hip_order():
    """amd-smi enumerates in discovery order while the masks index HIP order, derived from
    the KFD node id. setup.sh translates through `amd-smi list -e`'s HIP_ID map and refuses a
    mask when it is missing or not 1:1; no map is read here, so on unlike adapters an
    untranslated ordinal names another card's arch. No KFD topology either, so the ordering
    the next test substitutes is unavailable."""
    assert _target(["gfx1103", "gfx1200"], "amd-smi", {"HIP_VISIBLE_DEVICES": "1"}) is None
    # rocminfo and KFD sysfs are already in the order the masks use.
    assert _target(["gfx1103", "gfx1200"], "kfd", {"HIP_VISIBLE_DEVICES": "1"}) == "gfx1200"
    # Nothing to mistranslate: like adapters give the same arch at every ordinal.
    assert _target(["gfx1200", "gfx1200"], "amd-smi", {"HIP_VISIBLE_DEVICES": "1"}) == "gfx1200"
    # An unset mask is not safety: the runtime still opens HIP ordinal 0, and discovery-order
    # 0 can be the other card. setup.sh declines unlike adapters with no HIP map whether or
    # not a mask is set, and this has to read the host the same way.
    assert _target(["gfx1103", "gfx1200"], "amd-smi", {}) is None


def test_kfd_order_resolves_a_mask_amd_smi_order_cannot():
    """KFD nodes ARE the order the masks index, which is why the fallback above reads them
    when no userland probe answers. Declining an ordering this file already solves leaves a
    Ryzen APU-plus-dGPU box -- the reported shape -- on a wheel it has no kernels for."""
    assert (
        _target(
            ["gfx1200", "gfx1103"],
            "amd-smi",
            {"HIP_VISIBLE_DEVICES": "0"},
            kfd = ["gfx1103", "gfx1200"],
        )
        == "gfx1103"
    )
    # A KFD list of another length is a different view of the machine, not a translation of
    # this one, so the ordinal still has nothing trustworthy to index.
    assert (
        _target(
            ["gfx1200", "gfx1103"],
            "amd-smi",
            {"HIP_VISIBLE_DEVICES": "0"},
            kfd = ["gfx1103", "gfx1200", "gfx1200"],
        )
        is None
    )
    # A UUID names a device no source can place, so KFD does not rescue that one.
    assert (
        _target(
            ["gfx1200", "gfx1103"],
            "amd-smi",
            {"ROCR_VISIBLE_DEVICES": "GPU-8d1f2e3a4b5c6d7e"},
            kfd = ["gfx1103", "gfx1200"],
        )
        is None
    )


def test_an_explicit_arch_is_authoritative_for_the_repair_gate_too():
    """Stacked masks can leave a device list the override is not in: ROCr keeps the gfx1200,
    the user names the gfx1103. The caller reads that set to decide which arch lacks kernels,
    so a target missing from it is selected and then never repaired."""
    calls = _run_install(
        gfx_devices = ("gfx1200", "gfx1103"),
        probe_source = "amd-smi",
        env = {
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "UNSLOTH_ROCM_GFX_ARCH": "gfx1103",
        },
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_generic_wheel_is_judged_by_its_own_rocm_tag():
    """Which arches a generic wheel carries belongs to THAT wheel. Pin rocm6.3 once, or
    upgrade /opt/rocm afterwards, and a gfx1200 box runs a 2.9.1+rocm6.3 build with no kernels
    for it while the host reads 7.2 -- judged by the host, it looks healthy forever."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_GENERIC_TORCH_63,
        torch_owns_rocm = False,
    )
    assert f"{_AMD}/gfx120X-all/" in calls, calls
    # The wheel that does carry it is left alone, whatever the host version reads.
    assert _AMD not in _run_install(
        gfx_devices = ("gfx1200",),
        rocm_version = (6, 3),
        torch_probe = _ROCM_GENERIC_TORCH_72,
        torch_owns_rocm = False,
    )


def test_a_named_arch_resolves_an_ordering_no_probe_can():
    """UNSLOTH_ROCM_GFX_ARCH is what the decline message tells the user to set, so it has to
    resolve the host it is offered for. Read from the environment, not the product-name
    inference, which is weaker than the probe and must not decide a host it left open."""
    assert (
        _target(
            ["gfx1103", "gfx1200"],
            "amd-smi",
            {"HIP_VISIBLE_DEVICES": "1", "UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
        )
        == "gfx1103"
    )
    assert _target(["gfx1103", "gfx1200"], "amd-smi", {"HIP_VISIBLE_DEVICES": "1"}) is None


@pytest.mark.parametrize(
    "gfx, leaf",
    [
        ("gfx1200", "gfx120X-all"),  # RDNA 4, and the generic wheel lists it
        ("gfx1151", "gfx1151"),  # Strix Halo, reached here when the version is unreadable
    ],
)
def test_the_torch_floor_applies_to_the_family_not_to_the_missing_kernel_gate(gfx, leaf):
    """Several floor leaves serve GPUs the generic wheel DOES carry kernels for, so gating the
    2.11 floor on missing kernels never reaches them: the preflight forces the dependency pass
    and this branch then declines it, and the _grouped_mm bug survives every update."""
    calls = _run_install(
        gfx_devices = (gfx,),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH_210,
        installed_family = leaf.lower(),
    )
    assert f"{_AMD}/{leaf}/" in calls, calls
    assert "torch>=2.11.0,<2.12.0" in calls, calls
    # At the floor the same host keeps its wheels, so this is the floor and not a reinstall loop.
    assert f"{_AMD}/{leaf}/" not in _run_install(
        gfx_devices = (gfx,),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = leaf.lower(),
    )


@pytest.mark.parametrize("gfx", ["gfx942", "gfx950"])
def test_a_generic_only_replacement_gpu_still_gets_a_wheel(gfx):
    """The datacentre parts live only on the generic index, so a stale per-arch install that
    outlives its GPU has no AMD leaf to be repaired onto. With no readable host version there
    is no tag either, so the branch announced the reinstall and installed nothing."""
    calls = _run_install(
        gfx_devices = (gfx,),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert _GENERIC in calls, calls
    assert "carries no" in _run_install.printed, _run_install.printed


def test_a_rocr_mask_index_out_of_range_keeps_todays_fallback():
    """_pick_visible_index warns and falls back to GPU 0 for an out-of-range index, matching
    setup.ps1's Resolve-VisibleGpuIndex. A stricter rule here would split the two and withdraw
    the repair from single-GPU hosts: ROCR_VISIBLE_DEVICES=1 on a one-GPU box is a typo."""
    assert _target(["gfx1103"], "kfd", {"ROCR_VISIBLE_DEVICES": "1"}) == "gfx1103"
    assert _target(["gfx1103"], "kfd", {"ROCR_VISIBLE_DEVICES": "0,9"}) == "gfx1103"
    assert _target(["gfx1103", "gfx1200"], "kfd", {"ROCR_VISIBLE_DEVICES": "1"}) == "gfx1200"


def test_a_target_no_index_can_serve_is_not_worth_a_reinstall():
    """gfx1010 is in neither the generic wheel nor the AMD leaf map, so an empty leaf reads as
    "every family is wrong" and spends a multi-GB reinstall on a wheel that cannot carry
    kernels for it either. Demote only when there is somewhere to go."""
    calls = _run_install(
        gfx_devices = ("gfx1010",),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert _GENERIC not in calls and _AMD not in calls, calls
    # A target the generic wheel does carry is still repaired, so the guard is scoped.
    assert _GENERIC in _run_install(
        gfx_devices = ("gfx942",),
        inferred = None,
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )


@pytest.mark.parametrize("gfx", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize(
    "rocm_version, rerouted",
    [((6, 0), True), ((6, 3), True), ((6, 4), False), ((7, 1), False)],
)
def test_generic_kernel_support_is_keyed_by_the_tag_the_version_selects(
    gfx, rocm_version, rerouted
):
    """Which arches the generic wheel carries belongs to the wheel a version resolves to, not
    the index as a whole. AMD puts production RDNA 4 at ROCm 6.4, so a current amdgpu beside a
    stale /opt/rocm gets a rocm6.3 wheel with no kernels while a gfx120X-all leaf exists."""
    calls = _run_install(gfx_devices = (gfx,), rocm_version = rocm_version)
    if rerouted:
        assert f"{_AMD}/gfx120X-all/" in calls, calls
    else:
        assert _AMD not in calls and _GENERIC in calls, calls


@pytest.mark.parametrize("rocm_version", [(6, 0), (7, 1)])
def test_an_arch_with_no_recorded_minimum_is_unaffected_by_the_version(rocm_version):
    """The measurement covers rocm7.x; only arches known to postdate an older tag are keyed by
    it, so everything else keeps the union reading and the generic index it always had."""
    calls = _run_install(gfx_devices = ("gfx1100",), rocm_version = rocm_version)
    assert _AMD not in calls and _GENERIC in calls, calls


@pytest.mark.parametrize("gfx", ["gfx1150", "gfx1151"])
def test_strix_on_an_old_generic_wheel_is_rerouted_without_a_host_version(gfx):
    """The Strix arm is gated on a ROCm VERSION, and an unreadable one reads as 0.0, so it
    never runs on a bundled-runtime host. The rocm6.3 wheel carries no gfx1150/gfx1151 (AMD
    dates support to 7.1), so omitting them from the version map keeps a build that faults."""
    calls = _run_install(
        gfx_devices = (gfx,),
        rocm_version = None,
        torch_probe = _ROCM_GENERIC_TORCH_63,
        torch_owns_rocm = False,
    )
    assert f"{_AMD}/{gfx}/" in calls, calls
    # Only the build the reroute would fetch is left alone: a generic rocm7.2 wheel does
    # carry these arches, but Strix wants AMD's 7.13 fixes over any index below the floor.
    assert _AMD not in _run_install(
        gfx_devices = (gfx,),
        rocm_version = None,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = gfx,
    )


def test_a_masked_gfx906_on_a_mixed_host_is_not_demoted_to_another_dead_wheel():
    """gfx906's only route is the rocm6.3 tag, unlocked only when it is the ONE detected
    arch. A mask can still select the MI50 on a mixed host, and every tag above rocm6.3
    dropped its BLAS kernels, so demoting there buys a second unusable torch."""
    calls = _run_install(
        gfx_devices = ("gfx1100", "gfx906"),
        env = {"HIP_VISIBLE_DEVICES": "1"},
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert "reinstalling for this GPU" not in _run_install.printed, _run_install.printed
    assert _GENERIC not in calls, calls
    # The same stale family with a routable target is still demoted.
    _run_install(
        gfx_devices = ("gfx1100", "gfx1200"),
        env = {"HIP_VISIBLE_DEVICES": "1"},
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert "reinstalling for this GPU" in _run_install.printed, _run_install.printed


@pytest.mark.parametrize(
    "env",
    [
        {"HIP_VISIBLE_DEVICES": "1"},
        {"CUDA_VISIBLE_DEVICES": "2"},
    ],
)
def test_an_ordinal_past_an_inferred_arch_resolves_nothing(env):
    """One guess about the machine is not a device list, so an ordinal past its only entry
    indexes nothing: the mask names a GPU the guess cannot account for, and the out-of-range
    rule would answer with the guess anyway, then reinstall per-arch wheels for it."""
    calls = _run_install(
        gfx_devices = (),
        kfd = (),
        inferred = "gfx1103",
        rocm_gpu_visible = False,
        env = env,
    )
    assert _AMD not in calls, calls
    assert "past the only architecture" in _run_install.printed, _run_install.printed


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"HIP_VISIBLE_DEVICES": "0"},
        {"ROCR_VISIBLE_DEVICES": "1"},
    ],
)
def test_an_inferred_arch_still_installs_where_no_ordinal_contradicts_it(env):
    """Only the contradiction declines. No mask and index 0 agree with the guess, and an
    out-of-range ROCr index keeps the fallback the ROCr layer already documents."""
    calls = _run_install(
        gfx_devices = (),
        kfd = (),
        inferred = "gfx1103",
        rocm_gpu_visible = False,
        env = env,
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


def test_a_named_arch_outranks_an_ordinal_the_inference_cannot_place():
    """UNSLOTH_ROCM_GFX_ARCH is the user answering the question the ordinal left open, and
    the decline message says to set it, so it has to be honoured here too."""
    calls = _run_install(
        gfx_devices = (),
        kfd = (),
        inferred = "gfx1103",
        rocm_gpu_visible = False,
        env = {"HIP_VISIBLE_DEVICES": "1", "UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
    )
    assert f"{_AMD}/gfx110X-all/" in calls, calls


@pytest.mark.parametrize(
    "env",
    [
        {"ROCR_VISIBLE_DEVICES": "1"},
        {"HIP_VISIBLE_DEVICES": "1"},
    ],
)
def test_a_masked_gfx906_is_judged_by_the_host_the_legacy_route_probes(env):
    """ROCr filters the list this function sees, so a physically mixed host arrives as a lone
    gfx906 while _runtime_target_is_gfx906 re-probes the machine and declines the rocm6.3 tag.
    Deriving "is it alone" twice lets the two disagree, and the demotion then installs a newer
    wheel whose BLAS has no gfx906 kernels."""
    calls = _run_install(
        gfx_devices = ("gfx1100", "gfx906"),
        env = env,
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert "reinstalling for this GPU" not in _run_install.printed, _run_install.printed
    assert _GENERIC not in calls, calls


@pytest.mark.parametrize("gfx", ["gfx1150", "gfx1151"])
def test_a_strix_host_with_no_readable_version_still_gets_rocm_torch(gfx):
    """An unreadable version reads as 0.0, so the Strix arm sat out and the missing-kernel
    fallback declined too (these arches are in the generic list). The generic branch had no
    tag either, so a visible Strix host was left on CPU torch permanently."""
    calls = _run_install(
        gfx_devices = (gfx,),
        rocm_version = None,
        torch_owns_rocm = False,
    )
    assert f"{_AMD}/{gfx}/" in calls, calls


def test_the_strix_reroute_keeps_the_wheels_it_would_have_fetched():
    """The branch force-reinstalls a multi-GB stack and _ensure_rocm_torch runs twice per
    install, so acting on the arch alone re-downloads it every time."""
    calls = _run_install(
        gfx_devices = ("gfx1151",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx1151",
    )
    assert _AMD not in calls, calls
    assert "keeping it" in _run_install.printed, _run_install.printed


def test_an_explicit_arch_keeps_its_feature_suffix_out_of_the_target():
    """rocminfo and HSA spell a target with its feature flags (gfx1151:sramecc-:xnack-), so a
    user copying one into UNSLOTH_ROCM_GFX_ARCH offers a string every lookup misses: the Strix
    set, the AMD leaf map and the generic list are keyed on the bare arch, so the one thing
    the decline message tells the user to set would resolve nothing."""
    assert _target([], "rocminfo", {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151:sramecc-:xnack-"}) == (
        "gfx1151"
    )
    calls = _run_install(
        gfx_devices = (),
        rocm_version = (7, 2),
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151:xnack-"},
        torch_probe = _ROCM_GENERIC_TORCH,
        torch_owns_rocm = False,
    )
    assert f"{_AMD}/gfx1151/" in calls, calls


def test_a_version_below_every_index_is_not_read_as_support():
    """A ROCm older than the oldest tag this installer maps resolves no tag at all, and a
    wheel older than every tag predates the arches those tags were measured to add. Reading
    "no tag" as "carries it" preserves exactly the build that cannot run."""
    assert stack_mod._generic_rocm_wheel_lacks_kernels("gfx1200", (5, 7))
    assert stack_mod._generic_rocm_wheel_lacks_kernels("gfx1200", (6, 3))
    assert not stack_mod._generic_rocm_wheel_lacks_kernels("gfx1200", (6, 4))


def test_a_stale_host_version_does_not_pick_a_generic_tag_without_the_target():
    """gfx950 (MI350X) lives only on the generic index, so the repair picks a TAG rather than
    an index -- and a stale /opt/rocm beside a current kernel offers rocm6.3, which predates
    the part. The reinstall would download multiple GB and land on a wheel with no kernels."""
    calls = _run_install(
        gfx_devices = ("gfx950",),
        rocm_version = (6, 3),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )
    assert f"{_GENERIC}7.2" in calls, calls
    # A version whose tag does carry the target still takes that tag, not the newest one.
    assert f"{_GENERIC}7.0" in _run_install(
        gfx_devices = ("gfx950",),
        rocm_version = (7, 0),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
    )


@pytest.mark.parametrize(
    "mask", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
@pytest.mark.parametrize("value", ["", "-1"])
def test_a_mask_that_selects_no_gpu_takes_no_gfx906_reroute_either(mask, value):
    """The legacy route is a harsher version of what the guard above declines: a
    force-reinstall onto an OLDER tag, plus the loss of bitsandbytes. Its detection asks
    whether gfx906 is the sole arch, which no mask filters, so a CI job hiding every GPU over
    a populated host was downgraded while the same job on any other arch was not."""
    calls = _run_install(gfx_devices = ("gfx906",), rocm_version = (7, 2), env = {mask: value})
    assert "rocm6.3" not in calls, calls
    assert f"{_GENERIC}7.2" in calls, calls
    assert "bitsandbytes" in calls, calls
    # A mask naming a device is untouched: the legacy route and the bnb skip both still fire.
    _named = _run_install(gfx_devices = ("gfx906",), rocm_version = (7, 2), env = {mask: "0"})
    assert f"{_GENERIC}6.3" in _named, _named
    # The mask sits above UNSLOTH_ROCM_GFX_ARCH here too, as it does in _runtime_gfx_target:
    # hiding every GPU is a statement about this run, and the arch is read below it.
    _pinned = _run_install(
        gfx_devices = (),
        rocm_version = (7, 2),
        env = {mask: value, "UNSLOTH_ROCM_GFX_ARCH": "gfx906"},
    )
    assert "rocm6.3" not in _pinned, _pinned


def test_an_explicit_arch_still_reconciles_a_contradicting_spoof():
    """UNSLOTH_ROCM_GFX_ARCH returns before the spoof detection runs, and
    HSA_OVERRIDE_GFX_VERSION=11.0.0 is the workaround half these hosts carry from before
    per-arch wheels existed. Naming the physical arch bought the right wheels and left ROCr
    handing torch a gfx1100 agent none of them match: #7331 with the fix downloaded."""
    calls = _run_install(
        gfx_devices = (),
        kfd = ("gfx1151",),
        inferred = "gfx1151",
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert f"{_AMD}/gfx1151/" in calls, calls
    assert _run_install.hsa_override_after is None, _run_install.hsa_override_after


def test_an_explicit_arch_reconciles_the_spoof_on_the_reroute_path_too():
    """The same host with ROCm torch installed takes the Strix reroute instead, reaching the
    clear through _runtime_gfx_target's third return value. Both paths install gfx1151-only
    wheels, so both have to answer the override."""
    calls = _run_install(
        gfx_devices = (),
        kfd = ("gfx1151",),
        inferred = "gfx1151",
        rocm_version = (7, 2),
        torch_probe = _ROCM_GENERIC_TORCH,
        torch_owns_rocm = False,
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert f"{_AMD}/gfx1151/" in calls, calls
    assert _run_install.hsa_override_after is None, _run_install.hsa_override_after


def test_an_arch_naming_what_the_override_spoofs_to_keeps_the_spoof():
    """Setting both to gfx1100 is a deliberate pairing -- run this card as an RDNA 3 dGPU --
    not a leftover. Clearing it would strand the wheels the user just asked for."""
    _run_install(
        gfx_devices = (),
        kfd = ("gfx1103",),
        inferred = "gfx1103",
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1100", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert _run_install.hsa_override_after == "11.0.0"


def test_a_repin_between_per_arch_leaves_is_applied():
    """Swap the card, edit UNSLOTH_TORCH_INDEX_URL to the new leaf, and every shape the
    version string carries stays put: gfx1151 and gfx120X-all are both torch 2.11 with a
    three-part +rocm7.13.0 tag. Judged on that alone the edited pin reads as satisfied, so the
    old wheels survive the update in silence."""
    calls = _run_install(
        gfx_devices = ("gfx1200",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": f"https://{_AMD}/gfx120X-all/"},
    )
    assert f"{_AMD}/gfx120X-all" in calls, calls
    # A pin naming the family already installed is still not a reinstall: this runs on every
    # update and the stack is multi-GB.
    assert _AMD not in _run_install(
        gfx_devices = ("gfx1200",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx120x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": f"https://{_AMD}/gfx120X-all/"},
    )


def test_a_pin_at_a_non_floor_leaf_keeps_the_build_from_that_leaf():
    """The version heuristic reads any 2.11 build as a mismatch, because that is what a build
    from some OTHER index looks like. The leaf serves 2.11 too, so a correctly pinned host
    force-reinstalled under the legacy torch<2.11 cap on every update. A readable family says
    which index the build came from, and is decisive both ways."""
    assert _AMD not in _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": f"https://{_AMD}/gfx110X-all/"},
    )
    # The other family under the same pin is still repinned.
    assert f"{_AMD}/gfx110X-all" in _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx120x-all",
        env = {"UNSLOTH_TORCH_INDEX_URL": f"https://{_AMD}/gfx110X-all/"},
    )


def test_a_floor_leaf_still_repairs_a_sub_211_build_of_its_own_family():
    """gfx1151 and the other floor leaves carry the _grouped_mm bug below torch 2.11, which is
    the one thing their pin has to keep repairing, so a matching family must NOT satisfy it."""
    assert f"{_AMD}/gfx1151" in _run_install(
        gfx_devices = ("gfx1151",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH_210,
        installed_family = "gfx1151",
        env = {"UNSLOTH_TORCH_INDEX_URL": f"https://{_AMD}/gfx1151/"},
    )


def test_a_generic_only_target_gets_its_tag_floor_on_a_fresh_install_too():
    """gfx950 has no AMD per-arch leaf, so the only way to give it kernels is a generic tag
    that carries it (rocm7.0+). That floor was applied solely to a host already on ROCm wheels
    that had to be demoted, so a fresh install or a CPU/CUDA torch walked past it and a stale
    /opt/rocm put the card on rocm6.3. Which tag carries an arch is a fact about the arch."""
    for _probe in (_CPU_TORCH, None):
        calls = _run_install(
            gfx_devices = ("gfx950",),
            rocm_version = (6, 3),
            torch_probe = _probe,
            torch_owns_rocm = False,
        )
        assert f"{_GENERIC}7.2" in calls, calls
    # A host version whose tag does carry the target keeps that tag, not the newest one.
    assert f"{_GENERIC}7.0" in _run_install(
        gfx_devices = ("gfx950",),
        rocm_version = (7, 0),
        torch_probe = _CPU_TORCH,
        torch_owns_rocm = False,
    )
    # An arch the generic wheel serves at every tag is untouched: no floor to apply.
    assert f"{_GENERIC}6.3" in _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (6, 3),
        torch_probe = _CPU_TORCH,
        torch_owns_rocm = False,
    )


def test_matching_per_arch_wheels_clear_a_confirmed_spoof_with_nothing_to_reroute():
    """The spoof clear lived inside the missing-kernel / below-floor branch, so a host whose
    per-arch wheels ALREADY match reached no arm at all -- exactly when the wheels are right
    and the spoof is the only thing wrong. They carry the physical arch alone, so the runtime
    keeps asking them for code they do not have (#7331)."""
    calls = _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_ARCH_TORCH,
        installed_family = "gfx110x-all",
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1100", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"},
    )
    assert _run_install.hsa_override_after is None, _run_install.hsa_override_after
    # Clearing it is not a licence to reinstall: the wheels were already the right ones.
    assert "torch" not in calls, calls


def test_a_spoof_survives_wheels_that_are_not_the_targets_family():
    """The clear above is owned by a family read back positively from a torch that owns it.
    Generic wheels, or another family's, are the paths where the override may be the only
    source of usable kernels, so it must stay set there."""
    _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_GENERIC_TORCH,
        installed_family = None,
        torch_owns_rocm = False,
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1100", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"},
    )
    assert _run_install.hsa_override_after == "10.3.0", _run_install.hsa_override_after


def test_a_rocr_masked_mi50_beside_a_dgpu_keeps_the_generic_wheels():
    """The mixed-host rule withholds the rocm6.3 tag from a gfx906 sharing a machine: the
    downgrade is persistent and every tag above rocm6.3 dropped the dGPU's kernels, while the
    mask lasts one session. rocminfo runs on the ROCr stack, so with ROCR naming the MI50 it
    reports that card alone and the machine read as single-architecture."""
    calls = _run_install(
        gfx_devices = ("gfx906",),
        unmasked_gfx_devices = ("gfx1100", "gfx906"),
        rocm_version = (7, 2),
        probe_source = "rocminfo",
        env = {"ROCR_VISIBLE_DEVICES": "1"},
    )
    assert "rocm6.3" not in calls, calls
    assert f"{_GENERIC}7.2" in calls, calls
    # The dGPU's bitsandbytes is part of what the downgrade cost.
    assert "bitsandbytes" in calls, calls
    # A machine that really is gfx906 alone still takes the legacy tag under the same mask,
    # including a mask whose ordinal names no device: the sole-arch question is about the
    # machine, and the answer decides which wheels carry kernels for the card that is there.
    for _ordinal in ("0", "1"):
        _sole = _run_install(
            gfx_devices = ("gfx906",),
            unmasked_gfx_devices = ("gfx906",),
            rocm_version = (7, 2),
            probe_source = "rocminfo",
            env = {"ROCR_VISIBLE_DEVICES": _ordinal},
        )
        assert f"{_GENERIC}6.3" in _sole, (_ordinal, _sole)


def test_a_generic_only_target_on_a_stale_tag_is_repaired_on_update_too():
    """gfx950 has no per-arch leaf, so _generic_rocm_wheel_lacks_kernels answers False for it
    by design and the preflight read a rocm6.3 build as healthy. The install path applies a
    floor those wheels fail, so the card was repaired once and never from `studio update`."""
    for _tag, _hip, _repair in (
        ("2.9.1+rocm6.3", "6.3.0", True),
        ("2.11.0+rocm7.2", "7.2.0", False),
    ):
        with (
            patch.object(
                stack_mod, "_probe_torch_runtime", return_value = (True, True, _tag, _hip, "")
            ),
            patch.object(stack_mod, "_torch_requires_rocm_sdk", return_value = False),
            patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = None),
        ):
            assert (
                stack_mod._rocm_torch_family_needs_repair("gfx950", (7, 2), ["gfx950"]) is _repair
            ), _tag
    # An arch with a leaf is still judged by the reroute question, not by this one.
    with (
        patch.object(
            stack_mod,
            "_probe_torch_runtime",
            return_value = (True, True, "2.11.0+rocm7.2", "7.2.0", ""),
        ),
        patch.object(stack_mod, "_torch_requires_rocm_sdk", return_value = False),
        patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = None),
    ):
        assert stack_mod._rocm_torch_family_needs_repair("gfx1103", (7, 2), ["gfx1103"]) is True


def test_a_generic_only_target_installs_when_the_rocm_version_is_unreadable():
    """The unreadable-version exit asks the reroute question, which gfx950 answers False to
    for want of a leaf, so a runtime-only host was left on CPU torch on every retry. An
    unreadable version is the absence of a reading, not evidence the tag clears the floor."""
    assert f"{_GENERIC}7.2" in _run_install(
        gfx_devices = ("gfx950",),
        rocm_version = None,
        torch_probe = _CPU_TORCH,
        torch_owns_rocm = False,
    )
    # An arch the generic wheel serves at every tag still exits there.
    _served = _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = None,
        torch_probe = _CPU_TORCH,
        torch_owns_rocm = False,
    )
    assert "torch" not in _served, _served


def test_a_generic_only_target_pinned_to_a_stale_tag_is_reinstalled():
    """Forcing the pass is half a repair: a generic build names no family, so the family arm
    declines, there is no per-arch index to move to, and torch.version.hip keeps the fallback
    from running. A gfx950 pinned to rocm6.3 survived every update, and the preflight asked
    for a pass the install refused."""
    for _ver in ((7, 2), None):
        assert f"{_GENERIC}7.2" in _run_install(
            gfx_devices = ("gfx950",),
            rocm_version = _ver,
            torch_probe = _ROCM_GENERIC_TORCH_63,
            torch_owns_rocm = False,
        ), _ver
    # A generic build whose tag DOES carry the target is left alone, at either version.
    for _ver in ((7, 2), None):
        _kept = _run_install(
            gfx_devices = ("gfx950",),
            rocm_version = _ver,
            torch_probe = _ROCM_GENERIC_TORCH,
            torch_owns_rocm = False,
        )
        assert "torch" not in _kept, (_ver, _kept)
    # So is an arch the generic wheel serves at every tag.
    _served = _run_install(
        gfx_devices = ("gfx1100",),
        rocm_version = (7, 2),
        torch_probe = _ROCM_GENERIC_TORCH_63,
        torch_owns_rocm = False,
    )
    assert "torch" not in _served, _served
