# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GPU whose driver loads after Studio starts must still appear, and a later probe must never hide one (#9510)."""

from __future__ import annotations

from types import SimpleNamespace

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

    def count(gpu_rows_only = False):
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


def test_mig_child_rows_are_not_new_gpus(monkeypatch):
    rows = {"out": "GPU 0: NVIDIA A100 (UUID: GPU-a)\n"}
    monkeypatch.setattr(
        nvidia.gpu_query,
        "run_nvidia_smi",
        lambda argv, **kw: SimpleNamespace(returncode = 0, stdout = rows["out"]),
    )
    assert nvidia.get_physical_gpu_count(gpu_rows_only = True) == 1
    rows["out"] += (
        "  MIG 1g.5gb     Device  0: (UUID: MIG-a)\n  MIG 1g.5gb     Device  1: (UUID: MIG-b)\n"
    )
    assert nvidia.get_physical_gpu_count(gpu_rows_only = True) == 1
    rows["out"] += "GPU 1: NVIDIA A100 (UUID: GPU-b)\n"
    assert nvidia.get_physical_gpu_count(gpu_rows_only = True) == 2


def test_the_refresh_asks_for_physical_rows_only(smi, monkeypatch):
    seen = []
    monkeypatch.setattr(
        nvidia,
        "get_physical_gpu_count",
        lambda gpu_rows_only = False: seen.append(gpu_rows_only) or 1,
    )
    hw.get_physical_gpu_count()
    smi(1)
    hw.get_physical_gpu_count()
    assert seen == [False, True]


def test_the_ttl_starts_after_the_probe_returns(smi, monkeypatch):
    def slow_probe(gpu_rows_only = False):
        smi(1, seconds = 5.0)
        return 1

    monkeypatch.setattr(nvidia, "get_physical_gpu_count", slow_probe)
    hw.get_physical_gpu_count()
    # The probe took 5 s; an expiry measured from before it would re-probe 5 s early.
    assert hw.time.monotonic() - hw._physical_gpu_count_checked_at == 0.0


def test_the_memory_table_covers_every_parent_visible_gpu(smi, monkeypatch):
    smi(2, seconds = 0.0)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(
        hw, "estimate_fp16_model_size_bytes", lambda *a, **k: (8 * (1024**3), "config")
    )
    monkeypatch.setattr(
        hw, "_resolve_model_identifier_for_gpu_estimate", lambda *a, **k: "unsloth/test"
    )
    monkeypatch.setattr(
        hw,
        "_load_config_for_gpu_estimate",
        lambda *a, **k: SimpleNamespace(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 128256,
            tie_word_embeddings = False,
        ),
    )
    _, metadata = hw.estimate_required_model_memory_gb(
        "unsloth/test", training_type = "LoRA/QLoRA", load_in_4bit = True
    )
    assert metadata["estimation_mode"] == "detailed"
    assert "min_per_gpu_2" in metadata["vram_breakdown"]
