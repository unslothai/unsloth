# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm live telemetry must key each card's amd-smi row by its HIP id.

The parent ids are HIP ordinals, while amd-smi filters ``metric`` rows by its own
gpu id. ``amd-smi list -e`` maps one to the other, and the two disagree on hybrid
iGPU hosts. These tests mock amd-smi, torch and the device; no AMD GPU is needed.
"""

from __future__ import annotations

import pytest

from utils.hardware import amd
from utils.hardware import hardware as hw
from utils.hardware.hardware import DeviceType


def _metric(*gpus):
    """amd-smi ``metric`` rows for (amd-smi gpu id, used MiB, total MiB)."""
    return [
        {
            "gpu": idx,
            "usage": {"gfx_activity": util},
            "mem_usage": {
                "used_vram": {"value": used, "unit": "MB"},
                "total_vram": {"value": total, "unit": "MB"},
            },
        }
        for idx, used, total, util in gpus
    ]


# amd-smi gpu 0 is a 16 GiB card that HIP calls 1; amd-smi gpu 1 is a 96 GiB card HIP calls 0.
_TWO_CARDS = _metric((0, 1024, 16384, 5), (1, 2048, 98304, 90))
_REVERSED = [{"gpu": 0, "hip_id": 1}, {"gpu": 1, "hip_id": 0}]


@pytest.fixture
def rocm(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising = False)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(hw, "_reconcile_rocm_unified_memory", lambda *a, **k: None)
    monkeypatch.setattr(hw, "_apply_system_wide_vram", lambda *a, **k: None, raising = False)

    def _host(physical, hip_visible, metric, enumeration):
        monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: physical)
        monkeypatch.setattr(amd, "get_physical_gpu_count", lambda: physical)
        monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: hip_visible)

        def _run(*args, **kwargs):
            return enumeration if args and args[0] == "list" else metric

        monkeypatch.setattr(amd, "_run_amd_smi", _run)

    return _host


def _by_index(result):
    return {d["index"]: (d["vram_total_gb"], d["gpu_utilization_pct"]) for d in result["devices"]}


@pytest.mark.parametrize("entry", ["visible", "payload"])
def test_reordered_host_reports_each_card_under_its_hip_id(rocm, entry):
    rocm(2, 2, _TWO_CARDS, _REVERSED)
    if entry == "visible":
        result = hw.get_visible_gpu_utilization()
    else:
        result = {"devices": hw.get_gpu_utilization()["devices"]}
    assert _by_index(result) == {0: (96.0, 90), 1: (16.0, 5)}


def test_hip_mask_names_the_card_amd_smi_calls_by_another_id(rocm, monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    rocm(2, 1, _TWO_CARDS, _REVERSED)
    result = hw.get_visible_gpu_utilization()
    assert _by_index(result) == {0: (96.0, 90)}
    assert result["parent_visible_gpu_ids"] == [0]
    assert result["devices"][0]["visible_ordinal"] == 0


def test_identity_mapping_is_unchanged(rocm):
    rocm(2, 2, _TWO_CARDS, [{"gpu": 0, "hip_id": 0}, {"gpu": 1, "hip_id": 1}])
    assert _by_index(hw.get_visible_gpu_utilization()) == {0: (16.0, 5), 1: (96.0, 90)}


def test_without_list_e_the_query_is_unchanged(rocm):
    # amd-smi before ROCm 6.4.0 has no `list -e`: keep today's untranslated rows.
    rocm(2, 2, _TWO_CARDS, None)
    assert _by_index(hw.get_visible_gpu_utilization()) == {0: (16.0, 5), 1: (96.0, 90)}


def test_stacked_masks_keep_the_untranslated_query(rocm, monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    rocm(2, 1, _TWO_CARDS, _REVERSED)
    assert _by_index(hw.get_visible_gpu_utilization()) == {1: (96.0, 90)}


def test_single_gpu_is_unchanged(rocm):
    rocm(1, 1, _metric((0, 1024, 16384, 5)), None)
    assert _by_index(hw.get_visible_gpu_utilization()) == {0: (16.0, 5)}
