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
    monkeypatch.setattr(hw, "_unmasked_ids_are_smi_rows", lambda: True)


def test_torch_fallback_ids_keep_the_explicit_order(monkeypatch):
    # No nvidia-smi: the picker listed torch ordinals in FASTEST_FIRST order, so pinning PCI
    # order would hand the worker a different card.
    monkeypatch.setattr(hw, "_unmasked_ids_are_smi_rows", lambda: False)
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def test_unmasked_worker_pins_pci_order_under_fastest_first():
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    # The default order is unchanged.
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, clear = True):
        hw.apply_gpu_ids([0], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_only_unmasked_cuda_workers_are_repinned(monkeypatch):
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([0], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    # An inherited mask keeps the user's order: the ids are its entries.
    env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0,1"}
    with patch.dict(os.environ, env, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.XPU.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocr_relative_visibility", lambda value: None)
    with patch.dict(os.environ, {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, clear = True):
        hw.apply_gpu_ids([1], backend = DeviceType.CUDA.value)
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"


def _torch_rows(indices):
    # Mask entry i is torch ordinal i: under FASTEST_FIRST entry 0 is the 4090.
    cards = [(24.0, 0.5), (12.0, 1.0)]
    return [
        {"index": idx, "visible_ordinal": o, "total_gb": cards[o][0], "used_gb": cards[o][1]}
        for o, idx in enumerate(indices)
    ]


def _smi_utilization():
    # nvidia-smi rows by PCI index: 0 is the 3060, 1 the 4090.
    return {
        "devices": [
            {"index": 0, "vram_total_gb": 12.0, "vram_used_gb": 1.0},
            {"index": 1, "vram_total_gb": 24.0, "vram_used_gb": 0.5},
        ]
    }


def _host(monkeypatch, physical = 2):
    monkeypatch.setattr(hw, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: physical)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", True)
    monkeypatch.setattr(hw, "rocm_gpu_ids_without_torch_kernels", lambda: set())
    monkeypatch.setattr(hw, "get_visible_gpu_utilization", _smi_utilization)
    monkeypatch.setattr(hw, "_torch_get_per_device_info", _torch_rows)


def _auto(env):
    with patch.dict(os.environ, env, clear = True):
        ids, meta = hw.auto_select_gpu_ids("m", required_override_gb = 16.0)
    return ids


_MISMATCHED = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST", "CUDA_VISIBLE_DEVICES": "0,1"}


def test_numeric_mask_under_fastest_first_ranks_the_cards_the_ids_select(monkeypatch):
    # The 16 GB load fits only the 4090, mask entry 0 under FASTEST_FIRST. Ranking the
    # SMI rows picked entry 1, which the worker resolves to the 3060.
    _host(monkeypatch)
    assert _auto(_MISMATCHED) == [0]


@pytest.mark.parametrize(
    "env",
    [
        {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1"},
        {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},  # no mask: the worker pins PCI order
    ],
)
def test_smi_ranking_stays_where_its_ids_are_valid(monkeypatch, env):
    _host(monkeypatch)
    assert _auto(env) == [1]
    # The mismatched shape on the same host ranks through torch.
    assert _auto(_MISMATCHED) == [0]
