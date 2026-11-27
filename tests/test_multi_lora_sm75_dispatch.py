# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""sm75 has no tensor-core tl.dot: the multi-LoRA segment kernels run only where a T4 measured them faster."""

import types

import pytest
import torch

from unsloth.kernels import multi_lora as ml


def _x(device = "cuda:0"):
    return types.SimpleNamespace(device = torch.device(device))


def _a(adapters, rank):
    return types.SimpleNamespace(shape = (adapters, rank, 4096))


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(ml, "_TRITON_ENABLED", True)
    monkeypatch.setattr(ml, "_is_rocm_torch", lambda torch: False)
    monkeypatch.setattr(ml, "_CUDA_MAJOR", {})

    def set_major(major):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (major, 5))

    return set_major


@pytest.mark.parametrize(
    "adapters, rank, want",
    [(4, 16, True), (8, 8, True), (8, 16, True), (2, 16, False), (8, 32, False), (4, 64, False)],
)
def test_sm75_uses_the_kernels_only_where_measured_faster(gpu, adapters, rank, want):
    gpu(7)
    assert ml._use_segment_kernels(_x(), _a(adapters, rank)) is want


@pytest.mark.parametrize("major", [8, 9, 10, 12])
def test_tensor_core_gpus_always_use_the_kernels(gpu, major):
    gpu(major)
    assert ml._use_segment_kernels(_x(), _a(2, 128)) is True


def test_rocm_and_xpu_keep_the_kernels(gpu, monkeypatch):
    gpu(7)
    assert ml._use_segment_kernels(_x("xpu:0"), _a(2, 128)) is True
    monkeypatch.setattr(ml, "_is_rocm_torch", lambda torch: True)
    assert ml._use_segment_kernels(_x(), _a(2, 128)) is True


def test_kill_switch_and_cpu_use_the_fallback(gpu, monkeypatch):
    gpu(10)
    assert ml._use_segment_kernels(_x("cpu"), _a(8, 16)) is False
    monkeypatch.setattr(ml, "_TRITON_ENABLED", False)
    assert ml._use_segment_kernels(_x(), _a(8, 16)) is False


def test_multi_lora_linear_routes_through_the_gate(monkeypatch):
    seen = []
    monkeypatch.setattr(ml, "_use_segment_kernels", lambda X, A: seen.append(A.shape) or False)
    X = torch.randn(6, 8)
    W = torch.randn(4, 8)
    A = torch.randn(2, 3, 8)
    B = torch.randn(2, 4, 3)
    batch = ml.MultiLoraBatch.from_adapter_ids(torch.tensor([0, 1, 0, 1, -1, 0]), 2)
    Y = ml.multi_lora_linear(X, W, None, None, A, B, torch.ones(2), batch)
    assert seen == [A.shape] and Y.shape == (6, 4)
