# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A 4-bit out_proj inside nn.MultiheadAttention (granite-vision's SigLIP pooling head, #2672) must run."""

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from unsloth.models.loader_utils import _dequantize_multihead_attention_out_proj

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _pooling_head():
    torch.manual_seed(0)
    head = torch.nn.Module()
    head.attention = torch.nn.MultiheadAttention(64, 4, batch_first = True)
    head.mlp = torch.nn.Linear(64, 64)
    head.to(DEVICE, DTYPE)
    reference = head.attention.out_proj
    packed = bnb.nn.Linear4bit(64, 64, compute_dtype = DTYPE, quant_type = "nf4")
    packed.weight = bnb.nn.Params4bit(
        reference.weight.data.cpu(), requires_grad = False, quant_type = "nf4"
    )
    packed.bias = reference.bias
    try:
        packed = packed.to(DEVICE)
    except Exception as e:
        pytest.skip(f"bitsandbytes cannot quantize on {DEVICE}: {e}")
    if getattr(packed.weight, "quant_state", None) is None:
        pytest.skip(f"bitsandbytes did not quantize on {DEVICE}")
    head.attention.out_proj = packed
    return head


def _attend(attention):
    q = torch.randn(2, 1, 64, device = DEVICE, dtype = DTYPE)
    kv = torch.randn(2, 5, 64, device = DEVICE, dtype = DTYPE)
    return attention(q, kv, kv)[0]


def test_packed_out_proj_fails_then_runs_after_dequant():
    head = _pooling_head()
    packed = head.attention.out_proj
    with pytest.raises(RuntimeError, match = "same dtype"):
        _attend(head.attention)

    assert _dequantize_multihead_attention_out_proj(head) == 1
    out_proj = head.attention.out_proj
    assert type(out_proj) is torch.nn.modules.linear.NonDynamicallyQuantizableLinear
    assert out_proj.weight.dtype == DTYPE and not out_proj.weight.requires_grad
    expected = bnb.functional.dequantize_4bit(packed.weight.data, packed.weight.quant_state).to(
        DTYPE
    )
    assert torch.equal(out_proj.weight, expected)
    assert out_proj.bias is packed.bias
    assert torch.isfinite(_attend(head.attention)).all()
    # The sibling Linear is untouched and a second pass is a no-op.
    assert type(head.mlp) is torch.nn.Linear
    assert _dequantize_multihead_attention_out_proj(head) == 0


def test_float_out_proj_untouched():
    head = torch.nn.Module()
    head.attention = torch.nn.MultiheadAttention(64, 4, batch_first = True)
    before = head.attention.out_proj
    assert _dequantize_multihead_attention_out_proj(head) == 0
    assert head.attention.out_proj is before
