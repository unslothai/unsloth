# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FrozenFp8Linear hands _scaled_mm a row-major first operand for any input layout: CUDA's _scaled_mm refuses a
column-major mat1 ("self must be row_major"), so a transposed activation or grad_y must be made contiguous before
it is quantised. CPU emulation of that contract, since the fp8 GEMM itself is CUDA-only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import core.training.diffusion_frozen_fp8 as f8


def _strict_scaled_mm(a, b):
    # The CUDA _scaled_mm layout contract: mat1 row-major, mat2 column-major.
    if not a.is_contiguous():
        raise RuntimeError("self must be row_major")
    if not b.t().is_contiguous():
        raise RuntimeError("mat2 must be col_major")
    return (a.float() @ b.float()).to(torch.bfloat16)


@pytest.fixture(autouse = True)
def _strict(monkeypatch):
    monkeypatch.setattr(f8, "_unit_scaled_mm", _strict_scaled_mm)
    f8.BACKWARD_WEIGHTS.active = False
    yield


def _layer(k = 64, n = 32):
    torch.manual_seed(0)
    lin = nn.Linear(k, n, bias = True).to(torch.bfloat16).requires_grad_(False)
    return f8.FrozenFp8Linear(lin)


def _run(layer, x, gy):
    x = x.detach().requires_grad_(True)
    y = layer(x)
    y.backward(gy)
    return y.detach(), x.grad.detach()


def test_column_major_input_and_grad_match_contiguous():
    layer = _layer()
    torch.manual_seed(1)
    x_cm = torch.randn(64, 48, dtype = torch.bfloat16).t()
    gy_cm = torch.randn(32, 48, dtype = torch.bfloat16).t()
    assert not x_cm.is_contiguous() and not gy_cm.is_contiguous()
    y_cm, gx_cm = _run(layer, x_cm, gy_cm)
    y_c, gx_c = _run(layer, x_cm.contiguous(), gy_cm.contiguous())
    assert torch.equal(y_cm, y_c)
    assert torch.equal(gx_cm, gx_c)


def test_batch_one_transposed_activation():
    # (1, K, T).transpose(1, 2) reshapes to a column-major (T, K) view: the batch-1 case the audit names.
    layer = _layer()
    torch.manual_seed(2)
    x = torch.randn(1, 64, 40, dtype = torch.bfloat16).transpose(1, 2)
    assert not x.reshape(-1, 64).is_contiguous()
    gy = torch.randn(1, 40, 32, dtype = torch.bfloat16)
    y_t, gx_t = _run(layer, x, gy)
    y_c, gx_c = _run(layer, x.contiguous(), gy)
    assert torch.equal(y_t, y_c) and torch.equal(gx_t, gx_c)


def test_column_major_grad_alone():
    layer = _layer()
    torch.manual_seed(3)
    x = torch.randn(48, 64, dtype = torch.bfloat16)
    gy_cm = torch.randn(32, 48, dtype = torch.bfloat16).t()
    _, gx_cm = _run(layer, x, gy_cm)
    _, gx_c = _run(layer, x, gy_cm.contiguous())
    assert torch.equal(gx_cm, gx_c)
