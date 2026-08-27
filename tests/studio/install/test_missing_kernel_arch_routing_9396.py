# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Routing tests for ROCm targets absent from PyTorch's generic wheels.

Torch, hardware probes, and pip are mocked; the generic wheel architecture list was measured.
"""

from __future__ import annotations

import importlib.util
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

_CPU_TORCH = "2.10.0+cpu||\n"

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
):
    """Drive _ensure_rocm_torch() over a host with ``gfx_devices`` and return the pip calls."""
    probe = MagicMock()
    probe.returncode = 0
    probe.stdout = _CPU_TORCH

    probes = []

    def _fake_detect(dedup = True, ignore_hsa_override = False, ignore_visible_masks = False):
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
        patch.object(stack_mod, "_has_rocm_gpu", return_value = True),
        patch.object(stack_mod, "_infer_linux_amd_gfx_arch", return_value = inferred),
        patch.object(stack_mod, "_detect_amd_gfx_codes", side_effect = _fake_detect),
        patch.object(stack_mod, "_detect_rocm_version", return_value = rocm_version),
        patch.object(stack_mod, "_kfd_gfx_targets", return_value = [], create = True),
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
        with patch("os.path.isdir", return_value = True):
            with patch("subprocess.run", return_value = probe):
                stack_mod._ensure_rocm_torch()
    _run_install.probes = probes
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
    calls = _run_install(
        gfx_devices = ("gfx1103", "gfx906"), env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx906"}
    )
    assert _AMD not in calls, calls


def test_a_mixed_host_whose_runtime_target_is_supported_keeps_the_generic_index():
    """gfx1103 present, but the visible-device mask selects the dGPU: rerouting would pull the
    wrong wheel for the card that will actually run."""
    calls = _run_install(
        gfx_devices = ("gfx1100", "gfx1103"), env = {"HIP_VISIBLE_DEVICES": "0"}
    )
    assert _GENERIC in calls, calls
    assert _AMD not in calls, calls
