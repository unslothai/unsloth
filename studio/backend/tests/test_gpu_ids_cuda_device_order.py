# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU ids handed to a worker select the card they were ranked on, whatever
CUDA_DEVICE_ORDER the user exported.

RTX 3060 (PCI 0) + RTX 4090 (PCI 1) with CUDA_DEVICE_ORDER=FASTEST_FIRST: auto
select ranks nvidia-smi rows (PCI order) and picks 1, and the worker, reading
CUDA_VISIBLE_DEVICES=1 fastest-first, got the 3060. Without an inherited mask the
worker now pins PCI_BUS_ID, as every other Studio child does. With a numeric mask
the ids are CUDA ordinals in the user's order, so telemetry comes from torch."""

import os
import sys
from unittest.mock import patch

import pytest

from utils.hardware import hardware as hw
from utils.hardware.hardware import DeviceType


@pytest.fixture(autouse = True)
def _nvidia_host(monkeypatch):
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setitem(sys.modules, "torch", None)


def test_unmasked_worker_pins_pci_order_under_fastest_first():
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_an_inherited_mask_keeps_the_users_order():
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0,1"}
    with patch.dict(os.environ, env, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_default_pci_order_is_unchanged():
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, clear = True):
        hw.apply_gpu_ids([0], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_xpu_and_rocm_targets_do_not_touch_cuda_order(monkeypatch):
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.XPU.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocr_relative_visibility", lambda value: None)
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def _torch_rows(indices):
    return [
        {"index": idx, "visible_ordinal": o, "total_gb": 24.0, "used_gb": 1.0 + o}
        for o, idx in enumerate(indices)
    ]


def _smi_rows(*_a, **_k):
    # nvidia-smi rows filtered by PCI index: the WRONG cards under FASTEST_FIRST.
    return {
        "available": True,
        "devices": [{"index": 1, "visible_ordinal": 0, "vram_total_gb": 12.0, "vram_used_gb": 11.0}],
        "index_kind": "physical",
        "parent_visible_gpu_ids": [1, 0],
    }


def _host(monkeypatch, physical = 2):
    monkeypatch.setattr(hw, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: physical)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", True)
    monkeypatch.setattr(hw, "_smi_visible_utilization", _smi_rows)
    monkeypatch.setattr(hw, "_torch_get_per_device_info", _torch_rows)


def test_numeric_mask_under_fastest_first_reads_torch_not_smi(monkeypatch):
    _host(monkeypatch)
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "1,0"}
    with patch.dict(os.environ, env, clear = True):
        result = hw.get_visible_gpu_utilization()
    assert [d["index"] for d in result["devices"]] == [1, 0]
    assert [d["vram_total_gb"] for d in result["devices"]] == [24.0, 24.0]


@pytest.mark.parametrize(
    "env",
    [
        {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1,0"},
        {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},  # no mask: the worker pins PCI order
        {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"},
    ],
)
def test_smi_stays_where_its_ids_are_valid(monkeypatch, env):
    _host(monkeypatch)
    with patch.dict(os.environ, env, clear = True):
        result = hw.get_visible_gpu_utilization()
    assert result["devices"][0]["vram_total_gb"] == 12.0


def test_single_gpu_mask_is_unchanged(monkeypatch):
    _host(monkeypatch, physical = 1)
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0"}
    with patch.dict(os.environ, env, clear = True):
        result = hw.get_visible_gpu_utilization()
    assert result["devices"][0]["vram_total_gb"] == 12.0
