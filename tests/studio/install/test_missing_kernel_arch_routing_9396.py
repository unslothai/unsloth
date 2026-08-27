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
):
    """Drive _ensure_rocm_torch() over a host with ``gfx_devices`` and return the pip calls."""
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
        codes = list(gfx_devices)
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


def test_a_probed_arch_still_wins_over_the_named_one():
    """The fallback is last: a runtime that enumerates a GPU still decides (#7305)."""
    calls = _run_install(
        gfx_devices = ("gfx1100",),
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx1103"},
        torch_probe = _ROCM_GENERIC_TORCH,
    )
    assert _AMD not in calls, calls


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
    """HSA_OVERRIDE_GFX_VERSION=11.0.0 with no rocminfo and no amd-smi: the spoof check
    has no userland reading to distrust and declines, but amdkfd writes
    gfx_target_version itself, so a single-arch kernel reading the override contradicts
    is the corroborated spoof. Leaving the variable set hands the gfx1151-only wheels a
    device the runtime still calls gfx1100, which faults exactly as #7331 did."""
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
    """Enumeration order alone puts the APU first on a Ryzen desktop or laptop with a
    Radeon card in it. The family is chosen for ONE arch, so letting the APU decide
    installs wheels with no kernels for the discrete card the generic index was serving.
    _SHADOWING_INTEGRATED_GFX is the existing policy (#7776) and lists every one of these."""
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
