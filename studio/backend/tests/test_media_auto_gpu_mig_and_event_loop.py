# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic image/video placement: a MIG card never wins by its whole-card row, and the event-loop gate
never waits on hardware detection."""

import threading
import time
import types

import pytest

from core.inference import diffusion_device as dd
from utils.hardware import hardware as hw
from utils.hardware import nvidia
from utils.hardware.hardware import DeviceType


def _dev(index, total, used):
    return {"index": index, "vram_total_gb": total, "vram_used_gb": used}


def _two_cards(monkeypatch, mig):
    """2x H100 under PCI_BUS_ID, card 0 busy, card 1 idle by its PHYSICAL row."""
    monkeypatch.setattr(dd, "_PLANNED_ORDINAL", dd._PLANNED_ORDINAL)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda visible: ([], None))
    monkeypatch.setattr(
        hw, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": [0, 1], "raw": None}
    )
    monkeypatch.setattr(
        hw,
        "_smi_visible_utilization",
        lambda spec: {"devices": [_dev(0, 80.0, 60.0), _dev(1, 80.0, 2.0)]},
    )
    monkeypatch.setattr(nvidia, "get_compute_capabilities", lambda: {0: (9, 0), 1: (9, 0)})
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: mig)


# G2-1: media auto placement ranks a MIG card by its whole-card row and returns a nonzero
# ordinal that CUDA does not map to that card (pre-R570: does not exist at all).
@pytest.mark.parametrize(
    "mig", [frozenset({1}), frozenset({0}), None], ids = ["card1_mig", "card0_mig", "mig_unknown"]
)
def test_media_rank_never_moves_off_card0_on_a_mig_host(monkeypatch, mig):
    _two_cards(monkeypatch, mig)
    assert dd.resolve_auto_cuda_ordinal() == 0


def test_media_rank_still_moves_on_ordinary_two_card_host(monkeypatch):
    _two_cards(monkeypatch, frozenset())
    assert dd.resolve_auto_cuda_ordinal() == 1  # #13928's feature is kept


# G2-2: the event-loop gate must not wait on _DETECT_LOCK while a detection pass runs.
@pytest.mark.parametrize("device", [DeviceType.MLX, DeviceType.CPU, DeviceType.CUDA])
def test_cached_gate_does_not_block_during_detection(monkeypatch, device):
    monkeypatch.setattr(hw, "DEVICE", device)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_physical_gpu_count", 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(hw, "_detect_hardware_locked", lambda: time.sleep(1.5) or device)
    was = hw.DETECTION_COMPLETE.is_set()
    t = threading.Thread(
        target = hw.detect_hardware, daemon = True
    )  # forced re-detect, DEVICE stays set
    t.start()
    time.sleep(0.2)
    try:
        t0 = time.monotonic()
        dd.auto_cuda_ranking_possible_cached()
        waited = time.monotonic() - t0
    finally:
        t.join()
        if was:
            hw.DETECTION_COMPLETE.set()
        else:
            hw.DETECTION_COMPLETE.clear()
    assert waited < 0.3, f"event-loop gate waited {waited:.2f}s on the detection lock"


# A MIG-mode card with no instances has no `-L` MIG row and no CUDA device: its row is no ordinal.
def test_mig_mode_without_instances_counts_as_mig(monkeypatch):
    out = "0, Disabled\n1, Enabled\n"
    monkeypatch.setattr(nvidia, "get_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(nvidia, "_mig_mode_entry", None)
    monkeypatch.setattr(
        nvidia.gpu_query,
        "run_nvidia_smi",
        lambda argv, **kw: types.SimpleNamespace(returncode = 0, stdout = out),
    )
    assert hw._nvidia_mig_gpu_indices_or_none() == {1}


@pytest.mark.parametrize(
    "out, expected",
    [("0, [N/A]\n1, Disabled\n", set()), ("", None), ("Field is not valid\n", None)],
    ids = ["no_mig", "empty", "unsupported"],
)
def test_mig_mode_query_parsing(monkeypatch, out, expected):
    monkeypatch.setattr(nvidia, "_mig_mode_entry", None)
    monkeypatch.setattr(
        nvidia.gpu_query,
        "run_nvidia_smi",
        lambda argv, **kw: types.SimpleNamespace(returncode = 0, stdout = out),
    )
    assert nvidia.get_mig_mode_gpu_indices_or_none() == expected


# The event-loop gate answers from memory and never calls get_device(), which waits on _DETECT_LOCK.
@pytest.mark.parametrize(
    "mask, count, expected",
    [
        (None, 2, True),
        (None, 1, False),
        ("0,1", None, True),
        ("1", None, False),
        ("0,0", None, False),
        ("0,,1", None, False),
        ("", None, False),
    ],
)
def test_cached_gate_never_calls_get_device(monkeypatch, mask, count, expected):
    settled = threading.Event()
    settled.set()
    monkeypatch.setattr(hw, "DETECTION_COMPLETE", settled)
    monkeypatch.setattr(hw, "DEVICE", DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_physical_gpu_count", count)

    def _blocked():
        raise AssertionError("get_device() on the event loop")

    monkeypatch.setattr(hw, "get_device", _blocked)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert dd.auto_cuda_ranking_possible_cached() is expected
