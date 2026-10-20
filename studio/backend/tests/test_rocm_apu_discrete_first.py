# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic GPU selection on a ROCm host with an APU beside a discrete card: the APU's free
reading is the shared host pool, so it must not outrank a discrete card that fits alone."""

from unittest.mock import patch

import utils.hardware.hardware as hw
from utils.hardware import DeviceType, auto_select_gpu_ids


def _auto_select(
    devices,
    unified,
    required_gb,
    *,
    rocm = True,
):
    inventory = [
        {"index": d["index"], "_rocm_known_unified": d["index"] in unified} for d in devices
    ]
    with (
        patch.object(hw, "get_device", return_value = DeviceType.CUDA),
        patch.object(hw, "IS_ROCM", rocm),
        patch.object(
            hw,
            "estimate_required_model_memory_gb",
            return_value = (required_gb, {"required_gb": required_gb, "model_size_source": "config"}),
        ),
        patch.object(
            hw,
            "_get_parent_visible_gpu_spec",
            return_value = {
                "raw": None,
                "numeric_ids": [d["index"] for d in devices],
                "supports_explicit_gpu_ids": True,
            },
        ),
        patch.object(hw, "rocm_gpu_ids_without_torch_kernels", return_value = set()),
        patch.object(hw, "get_parent_visible_gpu_ids", return_value = [d["index"] for d in devices]),
        patch.object(hw, "get_visible_gpu_utilization", return_value = {"devices": devices}),
        patch.object(hw, "_torch_get_device_inventory", return_value = inventory) as inv,
    ):
        selected, metadata = auto_select_gpu_ids("unsloth/test")
    return selected, metadata, inv


APU_ROW = {"index": 0, "vram_total_gb": 96.0, "vram_used_gb": 8.0}  # GTT pool
DGPU_ROW = {"index": 1, "vram_total_gb": 24.0, "vram_used_gb": 1.0}


def test_auto_select_picks_the_discrete_card_that_fits():
    selected, metadata, _ = _auto_select([APU_ROW, DGPU_ROW], {0}, 16.0)
    assert selected == [1]
    assert metadata["selection_mode"] == "auto"


def test_auto_select_keeps_the_apu_when_no_discrete_card_fits():
    selected, _, _ = _auto_select([APU_ROW, DGPU_ROW], {0}, 40.0)
    assert selected == [0]


def test_auto_select_unchanged_without_an_apu():
    other = {"index": 0, "vram_total_gb": 48.0, "vram_used_gb": 0.0}
    selected, _, _ = _auto_select([other, DGPU_ROW], set(), 16.0)
    assert selected == [0]


def test_auto_select_unchanged_off_rocm():
    selected, _, inv = _auto_select([APU_ROW, DGPU_ROW], {0}, 16.0, rocm = False)
    assert selected == [0]
    inv.assert_not_called()


def test_auto_select_single_gpu_skips_the_inventory():
    selected, _, inv = _auto_select([APU_ROW], {0}, 16.0)
    assert selected == [0]
    inv.assert_not_called()
