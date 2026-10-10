# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GPU whose driver loads after Studio starts must still appear, and a later probe must never hide one (#9510)."""

from __future__ import annotations

import pytest

import utils.hardware.hardware as hw
from utils.hardware import nvidia


@pytest.fixture
def smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_physical_gpu_count", None)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)
    monkeypatch.setattr(hw, "_physical_gpu_count_checked_at", None, raising = False)
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    clock = {"now": 1000.0}
    monkeypatch.setattr(hw.time, "monotonic", lambda: clock["now"])
    answer = {"count": 1, "calls": 0}

    def count():
        answer["calls"] += 1
        return answer["count"]

    monkeypatch.setattr(nvidia, "get_physical_gpu_count", count)

    def advance(count, seconds = getattr(hw, "_PHYSICAL_GPU_COUNT_TTL_SECONDS", 60.0)):
        answer["count"] = count
        clock["now"] += seconds

    advance.answer = answer
    return advance


def test_a_late_second_gpu_becomes_selectable(smi):
    assert hw.get_parent_visible_gpu_ids() == [0]
    smi(2)
    assert hw.get_physical_gpu_count() == 2
    assert hw.get_parent_visible_gpu_ids() == [0, 1]
    assert hw.resolve_requested_gpu_ids([1]) == [1]


def test_the_count_is_not_reprobed_inside_the_ttl(smi):
    assert hw.get_physical_gpu_count() == 1
    smi(2, seconds = 1.0)
    assert hw.get_physical_gpu_count() == 1
    assert smi.answer["calls"] == 1


@pytest.mark.parametrize("later", [None, 1, 0])
def test_a_failed_or_lower_probe_never_hides_a_gpu(smi, later):
    smi(2, seconds = 0.0)
    assert hw.get_physical_gpu_count() == 2
    smi(later)
    assert hw.get_physical_gpu_count() == 2
    assert hw.get_parent_visible_gpu_ids() == [0, 1]


def test_an_smi_that_answers_later_replaces_the_torch_fallback(smi):
    smi(None, seconds = 0.0)
    assert hw.get_physical_gpu_count() == 1
    assert hw._physical_gpu_count_from_smi is False
    smi(2)
    assert hw.get_physical_gpu_count() == 2
    assert hw._physical_gpu_count_from_smi is True


def test_non_cuda_counts_are_never_reprobed(smi, monkeypatch):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    assert hw.get_physical_gpu_count() == 1
    smi(2)
    assert hw.get_physical_gpu_count() == 1
    assert smi.answer["calls"] == 0


def test_rocm_counts_are_never_reprobed(smi, monkeypatch):
    from utils.hardware import amd

    calls = []
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(amd, "get_physical_gpu_count", lambda: calls.append(1) or 1)
    assert hw.get_physical_gpu_count() == 1
    smi(2)
    assert hw.get_physical_gpu_count() == 1
    assert len(calls) == 1
