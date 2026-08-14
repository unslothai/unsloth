# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training-side ROCm arch gate: automatic GPU selection must not hand a model
to a device the installed torch wheel has no kernels for.

The llama.cpp gate (#7624 / #7669) covers llama-server placement only. Training
goes through ``auto_select_gpu_ids`` -> ``apply_gpu_ids`` -> ``get_device_map``,
which ranks by free VRAM and never asks what the wheel was built for, so on the
#7669 host ``device_map="balanced"`` shards layers onto the gfx1036 iGPU and the
load dies with ``hipErrorInvalidKernelFile``.

Mock-based throughout: there is no AMD hardware or ROCm CI here. torch, its arch
attributes and ``get_arch_list``, and the visibility masks are all faked, in the
shapes the field logs on #7669 document.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from utils.hardware.hardware import (
    DeviceType,
    auto_select_gpu_ids,
    rocm_gpu_ids_without_torch_kernels,
)

# What the published gfx110X torch/llama bundles cover. gfx1036 (the Raphael
# iGPU in #7669) is deliberately absent: that is the whole bug.
GFX110X = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]


def _props(
    arch = "",
    *,
    attr = "gcnArchName",
    name = "AMD Radeon RX 7700 XT",
):
    """A hipDeviceProp_t stand-in exposing exactly one arch spelling, so the
    AMD SDK / Radeon wheels that populate none of the canonical ones can be
    reproduced."""
    p = types.SimpleNamespace(name = name)
    if arch:
        setattr(p, attr, arch)
    return p


def _fake_torch(
    devices,
    *,
    arch_list = GFX110X,
    vendor = "amd",
    available = True,
):
    """A fake torch. ``devices`` is a list of props objects (or exceptions to
    raise for that ordinal); ``arch_list`` is what the wheel was built for."""
    torch = types.ModuleType("torch")
    if vendor == "amd":
        torch.version = types.SimpleNamespace(hip = "7.13.99004", cuda = None)
        torch.__version__ = "2.11.0+rocm7.13.0"
    elif vendor == "amd_sdk":
        # AMD SDK / Radeon wheel: version.hip unset, "rocm" only in __version__.
        torch.version = types.SimpleNamespace()
        torch.__version__ = "2.6.0+rocm6.4"
    else:
        torch.version = types.SimpleNamespace(hip = None, cuda = "12.4")
        torch.__version__ = "2.6.0+cu124"

    def _get_device_properties(ordinal):
        entry = devices[ordinal]
        if isinstance(entry, Exception):
            raise entry
        return entry

    def _get_arch_list():
        if isinstance(arch_list, Exception):
            raise arch_list
        return list(arch_list)

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: len(devices),
        get_arch_list = _get_arch_list,
        get_device_properties = _get_device_properties,
    )
    return torch


@pytest.fixture
def no_mask(monkeypatch):
    """No visibility mask set, so physical id == torch ordinal."""
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 2)


def _install(monkeypatch, torch):
    monkeypatch.setitem(sys.modules, "torch", torch)


class TestTheReportedHost:
    """#7669's machine: RX 7700 XT (gfx1101) + Raphael iGPU (gfx1036), a wheel
    built for the gfx110X family."""

    def test_the_uncovered_igpu_is_dropped(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_a_fully_covered_host_drops_nothing(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1100")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_the_xnack_suffix_does_not_defeat_the_match(self, monkeypatch, no_mask):
        # ROCm advertises feature flags on the arch string.
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101:sramecc-:xnack-"), _props("gfx1036:xnack-")]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}


class TestHsaOverrideKeepsWorking:
    """HSA_OVERRIDE_GFX_VERSION is the standard workaround for an unsupported
    card, and it works by making the device PRESENT a supported arch. The gate
    compares what the device presents, so the override survives it. Comparing
    the real silicon instead would break every user of that workaround."""

    def test_a_spoofed_device_is_kept(self, monkeypatch, no_mask):
        # gfx1103 silicon presenting as gfx1100 under the override.
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1100")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestArchSpellings:
    """AMD SDK / Radeon wheels populate none of the canonical attributes, so
    reading only gcnArchName would leave every device unreadable and the gate a
    no-op on exactly the Windows hosts this targets."""

    @pytest.mark.parametrize("attr", ["gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"])
    def test_every_spelling_is_read(self, monkeypatch, no_mask, attr):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101", attr = attr), _props("gfx1036", attr = attr)]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}


class TestFailsOpen:
    """Every uncertainty keeps the pre-gate selection. A gate that guesses wrong
    here sends a working machine to CPU, which is worse than the bug."""

    def test_a_cuda_wheel_is_inert(self, monkeypatch, no_mask):
        # sm_/compute_ archs, and PTX JIT covers ones not listed, so filtering
        # on NVIDIA would drop working cards.
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1036")], arch_list = ["sm_80", "sm_90"], vendor = "nvidia"),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_an_amd_sdk_wheel_still_gates(self, monkeypatch, no_mask):
        # version.hip unset but "rocm" in __version__: this class of host must
        # NOT be mistaken for CUDA and left ungated.
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101"), _props("gfx1036")], vendor = "amd_sdk"),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_every_device_uncovered_keeps_them_all(self, monkeypatch, no_mask):
        # The wheel covers nothing on this host. Dropping all of them would hand
        # the caller an empty selection and silently force CPU.
        _install(monkeypatch, _fake_torch([_props("gfx900"), _props("gfx906")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_an_unreadable_arch_list_is_unknown(self, monkeypatch, no_mask):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1036")], arch_list = RuntimeError("no arch list")),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()

    @pytest.mark.parametrize(
        "arch_list",
        [[], ["gfx11-generic"], ["gfx110X"], ["", "  "], ["garbage"], ["sm_90"]],
        ids = ["empty", "generic", "family_label", "blank", "garbage", "cuda_tokens"],
    )
    def test_a_non_concrete_arch_list_is_unknown(self, monkeypatch, no_mask, arch_list):
        # No device reports these as its own arch, so the set would match nothing
        # and drop every GPU.
        _install(monkeypatch, _fake_torch([_props("gfx1101")], arch_list = arch_list))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_a_device_with_no_readable_arch_is_kept(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_properties_raising_is_not_fatal(self, monkeypatch, no_mask):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101"), RuntimeError("cannot describe device")]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_no_cuda_runtime_is_inert(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1036")], available = False))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_a_uuid_mask_cannot_be_named_back(self, monkeypatch):
        # ROCR/CUDA both accept UUID tokens; the ordinals cannot be mapped to
        # physical ids, so pinning would address the wrong card.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-DEADBEEFDEADBEEF,0")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestIdSpace:
    """The result is consumed as physical ids (they become CUDA_VISIBLE_DEVICES),
    so an ordinal must be mapped through the active mask."""

    def test_ordinals_map_through_the_mask(self, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
        # Ordinal 1 is physical 3 here, not 1.
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == {3}


class TestSelectorWiring:
    """Every exit of auto_select_gpu_ids, because three of the four hand back
    the WHOLE visible set and would keep feeding the iGPU to balanced sharding."""

    DEVICES = {
        "devices": [
            {"index": 0, "vram_total_gb": 12.0, "vram_used_gb": 0.0},
            {"index": 1, "vram_total_gb": 12.0, "vram_used_gb": 0.0},
        ]
    }

    def _run(
        self,
        *,
        uncovered,
        devices = None,
        required = (14.0, {"required_gb": 14.0}),
    ):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.rocm_gpu_ids_without_torch_kernels",
                return_value = uncovered,
            ),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = required,
            ),
            patch(
                "utils.hardware.hardware.get_visible_gpu_utilization",
                return_value = self.DEVICES if devices is None else devices,
            ),
            patch(
                "utils.hardware.hardware.get_parent_visible_gpu_ids",
                return_value = [0, 1],
            ),
        ):
            return auto_select_gpu_ids("unsloth/test")

    def test_the_ranked_path_never_offers_the_uncovered_gpu(self):
        selected, _meta = self._run(uncovered = {1})
        assert selected == [0]

    def test_the_unestimatable_fallback_is_filtered(self):
        # required_gb None takes the "use everything visible" arm, which is what
        # sent both cards to the child on the reported host.
        selected, metadata = self._run(uncovered = {1}, required = (None, {}))
        assert metadata["selection_mode"] == "fallback_all"
        assert selected == [0]

    def test_the_no_telemetry_fallback_is_filtered(self):
        selected, metadata = self._run(uncovered = {1}, devices = {"devices": []})
        assert metadata["selection_mode"] == "fallback_all"
        assert selected == [0]

    def test_a_covered_host_selects_exactly_as_before(self):
        # The control: with nothing to drop, selection is untouched.
        assert self._run(uncovered = set())[0] == self._run_ungated()

    def _run_ungated(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (14.0, {"required_gb": 14.0}),
            ),
            patch(
                "utils.hardware.hardware.get_visible_gpu_utilization",
                return_value = self.DEVICES,
            ),
            patch(
                "utils.hardware.hardware.get_parent_visible_gpu_ids",
                return_value = [0, 1],
            ),
        ):
            return auto_select_gpu_ids("unsloth/test")[0]
