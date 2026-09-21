# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm arch gate: selection must not pick a device the wheel lacks kernels for (#7669)."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch

import pytest

import utils.hardware.hardware as _hw_module
from utils.hardware.hardware import (
    DeviceType,
    apply_gpu_ids,
    auto_select_gpu_ids,
    rocm_gpu_ids_without_torch_kernels,
)

# gfx1036 (the Raphael iGPU in #7669) is deliberately absent: that is the bug.
GFX110X = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]

# Cleared wholesale so a developer's own ROCm env cannot decide what these assert.
_MASK_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ZE_AFFINITY_MASK",
)


def _props(
    arch = "",
    *,
    attr = "gcnArchName",
    name = "AMD Radeon RX 7700 XT",
):
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
    device_count = None,
):
    torch = types.ModuleType("torch")
    if vendor == "amd":
        torch.version = types.SimpleNamespace(hip = "7.13.99004", cuda = None)
        torch.__version__ = "2.11.0+rocm7.13.0"
    elif vendor == "amd_sdk":
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
        device_count = lambda: len(devices) if device_count is None else device_count,
        get_arch_list = _get_arch_list,
        get_device_properties = _get_device_properties,
    )
    return torch


@pytest.fixture
def no_mask(monkeypatch):
    for var in _MASK_VARS:
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 2)


@pytest.fixture(autouse = True)
def _no_device_ordinal(monkeypatch):
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising = False)


@pytest.fixture(autouse = True)
def _detection_is_declared_not_detected(monkeypatch):
    """Keeps detect_hardware() from latching IS_ROCM off the fake AMD torch session-wide."""
    monkeypatch.setattr(_hw_module, "DEVICE", DeviceType.CUDA)
    monkeypatch.setattr(_hw_module, "IS_ROCM", _hw_module.IS_ROCM)


def _install(monkeypatch, torch):
    monkeypatch.setitem(sys.modules, "torch", torch)


class TestTheReportedHost:
    """#7669: RX 7700 XT (gfx1101) + Raphael iGPU (gfx1036), gfx110X wheel."""

    def test_the_uncovered_igpu_is_dropped(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_a_fully_covered_host_drops_nothing(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1100")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_the_xnack_suffix_does_not_defeat_the_match(self, monkeypatch, no_mask):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101:sramecc-:xnack-"), _props("gfx1036:xnack-")]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}


class TestHsaOverrideKeepsWorking:
    """The override makes a device PRESENT a supported arch; reading silicon breaks it."""

    def test_a_spoofed_device_is_kept(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1100")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestArchSpellings:
    """Reading only gcnArchName makes the gate a no-op on AMD SDK wheels."""

    @pytest.mark.parametrize("attr", ["gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"])
    def test_every_spelling_is_read(self, monkeypatch, no_mask, attr):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101", attr = attr), _props("gfx1036", attr = attr)]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}


class TestFailsOpen:
    """Uncertainty keeps the pre-gate selection: a working machine on CPU is worse."""

    def test_a_cuda_wheel_is_inert(self, monkeypatch, no_mask):
        # PTX JIT covers archs not listed, so filtering on NVIDIA drops working cards.
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1036")], arch_list = ["sm_80", "sm_90"], vendor = "nvidia"),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_an_amd_sdk_wheel_still_gates(self, monkeypatch, no_mask):
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1101"), _props("gfx1036")], vendor = "amd_sdk"),
        )
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_every_device_uncovered_keeps_them_all(self, monkeypatch, no_mask):
        # Dropping all of them hands the caller an empty selection: silent CPU.
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
        _install(monkeypatch, _fake_torch([_props("gfx1101")], arch_list = arch_list))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    @pytest.mark.parametrize(
        "arch_list",
        [
            ["gfx1100", "gfx11-generic"],
            ["gfx11-generic", "gfx1100"],
            ["gfx900", "gfx110X", "gfx1100"],
        ],
        ids = ["generic_last", "generic_first", "family_label_between"],
    )
    def test_one_non_concrete_token_disables_the_whole_list(self, monkeypatch, no_mask, arch_list):
        # The concrete subset alone marks gfx1101 uncovered and drops it (#7624).
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1100"), _props("gfx1101")], arch_list = arch_list),
        )
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

    @pytest.mark.parametrize(
        "third",
        [_props(""), RuntimeError("cannot describe device")],
        ids = ["no_arch_attribute", "properties_raise"],
    )
    def test_one_unreadable_device_does_not_spare_a_known_uncovered_one(
        self, monkeypatch, no_mask, third
    ):
        # Discarding the whole answer would put the uncovered GPU 1 back (#8792).
        monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 3)
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036"), third]))
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    @pytest.mark.parametrize(
        "second",
        [_props(""), RuntimeError("cannot describe device")],
        ids = ["no_arch_attribute", "properties_raise"],
    )
    def test_an_unreadable_device_is_not_an_all_uncovered_host(self, monkeypatch, no_mask, second):
        # The unread device is still selectable, so this is not "every GPU" (#8792).
        _install(monkeypatch, _fake_torch([_props("gfx1036"), second]))
        assert rocm_gpu_ids_without_torch_kernels() == {0}

    def test_no_cuda_runtime_is_inert(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1036")], available = False))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_a_uuid_mask_cannot_be_named_back(self, monkeypatch, no_mask):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-DEADBEEFDEADBEEF,0")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_gpu_device_ordinal_renumbers_the_map_away(self, monkeypatch, no_mask):
        # No visibility spec reads it, so ordinal 1 is not physical 1.
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1,0")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_stacked_rocr_and_cuda_masks_renumber_the_map_away(self, monkeypatch, no_mask):
        # Ordinal 1 is phys1, but the spec sees only ROCr and would name it physical 0.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", True)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,0,1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestIdSpace:
    """Results are consumed as physical ids, so ordinals map through the mask."""

    def test_ordinals_map_through_the_mask(self, monkeypatch, no_mask):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == {3}

    def _rocr_only(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", True)
        monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 2)
        for var in _MASK_VARS:
            monkeypatch.delenv(var, raising = False)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))

    def test_a_rocr_mask_maps_the_ordinals_on_linux(self, monkeypatch):
        self._rocr_only(monkeypatch, "linux")
        assert rocm_gpu_ids_without_torch_kernels() == {3}

    def test_windows_ignores_a_stray_rocr_mask(self, monkeypatch):
        # Windows HIP has no ROCr layer, so reading this mask excludes a nonexistent card.
        self._rocr_only(monkeypatch, "win32")
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_windows_still_honours_a_hip_mask(self, monkeypatch):
        self._rocr_only(monkeypatch, "win32")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
        assert rocm_gpu_ids_without_torch_kernels() == {3}


class TestTheOrdinalToIdMapMustBeTotal:
    """device_count() freezes at torch init while the visible spec re-reads the env, so
    naming an overflow ordinal into the physical namespace collides with a real id."""

    def test_more_ordinals_than_ids_gates_nothing(self, monkeypatch, no_mask):
        # Ordinal 2 has no id; reusing it as one would drop physical 2, the good card.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,0")
        _install(
            monkeypatch,
            _fake_torch(
                [_props("gfx1101"), _props("gfx1036"), _props("gfx1036")],
                device_count = 3,
            ),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_the_selector_keeps_the_covered_card(self, monkeypatch, no_mask):
        # The regression this guards: an empty list is "no GPU", not "inherit".
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,0")
        _install(
            monkeypatch,
            _fake_torch(
                [_props("gfx1101"), _props("gfx1036"), _props("gfx1036")],
                device_count = 3,
            ),
        )
        monkeypatch.setattr(_hw_module, "get_device", lambda: DeviceType.CUDA)
        monkeypatch.setattr(
            _hw_module,
            "get_visible_gpu_utilization",
            lambda: {
                "devices": [
                    {"index": 0, "vram_total_gb": 16.0, "vram_used_gb": 1.0},
                    {"index": 2, "vram_total_gb": 32.0, "vram_used_gb": 1.0},
                ]
            },
        )
        gpu_ids, _ = auto_select_gpu_ids("m", required_override_gb = 8.0)
        assert gpu_ids == [2]

    def test_amd_smi_undercounting_still_gates(self, monkeypatch, no_mask):
        # No mask, so the short list is only amd-smi missing the iGPU; bailing out here
        # would disable the fix on the very host #8792 reports.
        monkeypatch.setattr(_hw_module, "get_physical_gpu_count", lambda: 1)
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_an_id_named_twice_still_trips_the_all_uncovered_guard(self, monkeypatch, no_mask):
        # Both ordinals are physical 0, so a deduplicated set would read as a partial drop.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,0")
        _install(
            monkeypatch,
            _fake_torch([_props("gfx1036"), _props("gfx1036")]),
        )
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestSelectorWiring:
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
        selected, metadata = self._run(uncovered = {1}, required = (None, {}))
        assert metadata["selection_mode"] == "fallback_all"
        assert selected == [0]

    def test_the_no_telemetry_fallback_is_filtered(self):
        selected, metadata = self._run(uncovered = {1}, devices = {"devices": []})
        assert metadata["selection_mode"] == "fallback_all"
        assert selected == [0]

    def test_a_covered_host_selects_exactly_as_before(self):
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


class TestThePinLandsOnTheKeptCard:
    """HIP indexes the agents an inherited ROCr mask left, so a raw physical id misses."""

    def _rocr(
        self,
        monkeypatch,
        mask,
        *,
        devices = ("gfx1101", "gfx1036"),
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", True)
        monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 2)
        _install(monkeypatch, _fake_torch([_props(arch) for arch in devices]))
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["ROCR_VISIBLE_DEVICES"] = mask

    def test_a_reordered_mask_pins_the_covered_card(self, monkeypatch):
        # Physical id 1 verbatim picks ROCr agent 1, physical 0: the card just excluded.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            assert rocm_gpu_ids_without_torch_kernels() == {0}
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
            # Clearing it would hand the child every agent, the gfx1036 among them.
            assert os.environ["ROCR_VISIBLE_DEVICES"] == "1,0"

    def test_a_nonzero_mask_pins_in_range(self, monkeypatch):
        # ROCR=2,3 leaves two agents: an untranslated HIP="3" is out of range.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "2,3")
            assert rocm_gpu_ids_without_torch_kernels() == {3}
            apply_gpu_ids([2], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "0"

    def test_an_identity_mask_is_written_unchanged(self, monkeypatch):
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "0,1")
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_an_inherited_hip_mask_is_already_relative(self, monkeypatch):
        # HIP produced the ids, so translating again would map them twice.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            os.environ["HIP_VISIBLE_DEVICES"] = "1,0"
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"

    def test_an_inherited_cuda_mask_is_already_relative(self, monkeypatch):
        # rocclr already read this as the HIP mask, so translating again writes "0":
        # ROCr agent 0, physical 1, the card the parent hid.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_a_uuid_rocr_mask_is_left_alone(self, monkeypatch):
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "GPU-DEADBEEFDEADBEEF,1")
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"

    def test_windows_writes_physical_ids(self, monkeypatch):
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            monkeypatch.setattr(sys, "platform", "win32")
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"

    def test_a_cuda_host_is_untouched(self, monkeypatch):
        with patch.dict(os.environ):
            monkeypatch.setattr(sys, "platform", "linux")
            monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", False)
            _install(monkeypatch, _fake_torch([_props("gfx1101")], vendor = "nvidia"))
            for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                os.environ.pop(var, None)
            apply_gpu_ids([1, 3], backend = DeviceType.CUDA.value)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,3"
            assert "HIP_VISIBLE_DEVICES" not in os.environ

    def test_a_stale_rocr_var_on_nvidia_does_not_move_the_pin(self, monkeypatch):
        # Reading the var's presence as "this is ROCm" writes CUDA="0", not GPU 1.
        with patch.dict(os.environ):
            monkeypatch.setattr(sys, "platform", "linux")
            monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", False)
            _install(monkeypatch, _fake_torch([_props("")], vendor = "nvidia"))
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            os.environ["ROCR_VISIBLE_DEVICES"] = "1,0"
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
