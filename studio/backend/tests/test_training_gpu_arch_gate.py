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

# What the published gfx110X torch/llama bundles cover. gfx1036 (the Raphael
# iGPU in #7669) is deliberately absent: that is the whole bug.
GFX110X = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]

# Every variable that can renumber or filter the device set. Cleared wholesale so
# a developer's own ROCm env cannot decide what these assert.
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
    for var in _MASK_VARS:
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 2)


@pytest.fixture(autouse = True)
def _no_device_ordinal(monkeypatch):
    """GPU_DEVICE_ORDINAL renumbers torch ordinals and the gate refuses to work
    behind it. Nothing here sets it deliberately, so clear it rather than let a
    developer's own ROCm shell decide what these assert."""
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising = False)


@pytest.fixture(autouse = True)
def _detection_is_declared_not_detected(monkeypatch):
    """Pin the hardware-detection globals for the duration of each test.

    The gate reads ``_get_parent_visible_gpu_spec()``, which calls ``get_device()``
    and so lazily runs ``detect_hardware()`` against whatever sits in
    ``sys.modules["torch"]``. With the fake AMD torch these tests install that
    latches DEVICE / IS_ROCM / CHAT_ONLY for the whole pytest process, and every
    later test in the session then reads ``IS_ROCM = True`` on a host that has no
    AMD GPU. Declaring DEVICE means detection is never entered, and monkeypatch
    restores both names at teardown, so nothing escapes this file."""
    monkeypatch.setattr(_hw_module, "DEVICE", DeviceType.CUDA)
    monkeypatch.setattr(_hw_module, "IS_ROCM", _hw_module.IS_ROCM)


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
        # ROCm 6.4+ ships generic code objects alongside concrete ones, and a
        # generic target covers devices no exact token names. Keeping the concrete
        # subset and calling it the build's whole coverage marks gfx1101 uncovered
        # here and drops a card the wheel does run, so the list is unknown as a
        # whole. Same all-or-nothing rule as the llama.cpp gate (#7624).
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
        # The boundary of the two above, and deliberate. An unreadable device is
        # skipped, not fatal to the probe: skipping leaves GPU 2 exactly as
        # eligible as it was before this gate existed, where discarding the whole
        # answer would put the known-uncovered GPU 1 back alongside it and re-break
        # the #8792 host. A partial answer is never worse than no answer.
        monkeypatch.setattr("utils.hardware.hardware.get_physical_gpu_count", lambda: 3)
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036"), third]))
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_no_cuda_runtime_is_inert(self, monkeypatch, no_mask):
        _install(monkeypatch, _fake_torch([_props("gfx1036")], available = False))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_a_uuid_mask_cannot_be_named_back(self, monkeypatch, no_mask):
        # ROCR/CUDA both accept UUID tokens; the ordinals cannot be mapped to
        # physical ids, so pinning would address the wrong card.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-DEADBEEFDEADBEEF,0")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_gpu_device_ordinal_renumbers_the_map_away(self, monkeypatch, no_mask):
        # ROCclr-layer renumbering that no visibility spec here reads, so ordinal 1
        # is not physical 1 and the id this would exclude names another card.
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1,0")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()

    def test_stacked_rocr_and_cuda_masks_renumber_the_map_away(self, monkeypatch, no_mask):
        # ROCr leaves agents [phys2, phys0, phys1]; CUDA then keeps ROCr-relative
        # 1 and 2, so torch ordinal 0 is phys0 and ordinal 1 is phys1. The visible
        # spec only ever sees the ROCr mask and reports [2, 0, 1], so the uncovered
        # ordinal 1 would be excluded under physical 0's name -- dropping the
        # covered card and leaving the uncovered one selectable, the exact
        # inversion of what the gate is for.
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", True)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,0,1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
        _install(monkeypatch, _fake_torch([_props("gfx1101"), _props("gfx1036")]))
        assert rocm_gpu_ids_without_torch_kernels() == set()


class TestIdSpace:
    """The result is consumed as physical ids (they become CUDA_VISIBLE_DEVICES),
    so an ordinal must be mapped through the active mask."""

    def test_ordinals_map_through_the_mask(self, monkeypatch, no_mask):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
        # Ordinal 1 is physical 3 here, not 1.
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
        # ROCr is the layer that actually masks here, so ordinal 1 is physical 3.
        self._rocr_only(monkeypatch, "linux")
        assert rocm_gpu_ids_without_torch_kernels() == {3}

    def test_windows_ignores_a_stray_rocr_mask(self, monkeypatch):
        # Windows HIP has no ROCr layer, so this mask hides nothing and torch still
        # enumerates both physical cards. Reading it as the ordinal->physical map
        # would drop the iGPU under GPU 3's name: the uncovered card stays in the
        # selection and a card that does not exist gets excluded instead.
        self._rocr_only(monkeypatch, "win32")
        assert rocm_gpu_ids_without_torch_kernels() == {1}

    def test_windows_still_honours_a_hip_mask(self, monkeypatch):
        # HIP_VISIBLE_DEVICES is the Windows mask, so the mapping stands there.
        self._rocr_only(monkeypatch, "win32")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
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


class TestThePinLandsOnTheKeptCard:
    """Dropping an id only helps if the pin built from what survives selects the
    same card. ``apply_gpu_ids`` writes HIP_VISIBLE_DEVICES and leaves an inherited
    ROCr mask in place, and HIP indexes the agents ROCr left -- so a physical id
    written straight through addresses a different device, or none at all."""

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
        # ROCR=1,0 -> torch ordinal 1 is physical 0, the uncovered iGPU. Writing
        # the surviving physical id 1 into HIP verbatim picks ROCr agent #1,
        # which IS physical 0: the card the gate just excluded.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            assert rocm_gpu_ids_without_torch_kernels() == {0}
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
            # ROCr keeps hiding what it was hiding; clearing it would hand the
            # child every agent, the gfx1036 among them.
            assert os.environ["ROCR_VISIBLE_DEVICES"] == "1,0"

    def test_a_nonzero_mask_pins_in_range(self, monkeypatch):
        # ROCR=2,3 leaves two agents, so HIP="3" is out of range and the worker
        # sees no GPU at all rather than the covered one.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "2,3")
            assert rocm_gpu_ids_without_torch_kernels() == {3}
            apply_gpu_ids([2], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "0"

    def test_an_identity_mask_is_written_unchanged(self, monkeypatch):
        # The ordinary case, and every host that never set ROCR by hand.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "0,1")
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_an_inherited_hip_mask_is_already_relative(self, monkeypatch):
        # HIP is then the layer that produced the ids, so translating again
        # would map them a second time.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "1,0")
            os.environ["HIP_VISIBLE_DEVICES"] = "1,0"
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"

    def test_a_uuid_rocr_mask_is_left_alone(self, monkeypatch):
        # No ordinal to name a UUID back to; the gate is inert here too.
        with patch.dict(os.environ):
            self._rocr(monkeypatch, "GPU-DEADBEEFDEADBEEF,1")
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["HIP_VISIBLE_DEVICES"] == "1"

    def test_windows_writes_physical_ids(self, monkeypatch):
        # No ROCr layer there, so a stray ROCR var is not the mapping's source
        # and HIP ids are physical.
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
        # Schedulers export both families of visibility variable, and a Linux
        # NVIDIA node can inherit a ROCR mask that means nothing there. Reading the
        # var's presence as "this is ROCm" would translate against it and write
        # CUDA="0": the worker trains on NVIDIA GPU 0 while the caller asked for
        # GPU 1. The ROCm answer has to come from the build.
        with patch.dict(os.environ):
            monkeypatch.setattr(sys, "platform", "linux")
            monkeypatch.setattr("utils.hardware.hardware.IS_ROCM", False)
            _install(monkeypatch, _fake_torch([_props("")], vendor = "nvidia"))
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            os.environ["ROCR_VISIBLE_DEVICES"] = "1,0"
            apply_gpu_ids([1], backend = DeviceType.CUDA.value)
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
