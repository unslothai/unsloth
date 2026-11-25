# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_qat_round._checksum must hash a weight's bits for every dtype.

It converted 8-byte dtypes by value, so every float64 tensor below 1 in magnitude hashed to 0 and matched any
other. torchao fake quantization rejects float64 inputs today, so no QAT model reaches it yet. CPU only."""

import os

import pytest
import torch

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_round as qr


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(qr.__file__).startswith(here), qr.__file__


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_checksum_sees_different_values(dtype):
    a = (torch.randn(64, 64, generator = torch.Generator().manual_seed(0)) * 0.02).to(dtype)
    b = (torch.randn(64, 64, generator = torch.Generator().manual_seed(1)) * 0.02).to(dtype)
    assert qr._checksum(a) != qr._checksum(b)
    assert qr._checksum(a) == qr._checksum(a.clone())
