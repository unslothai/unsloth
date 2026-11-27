# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Ulysses Inkling 2 whole sequence bias equals the gather over an int64 [T, T] distance (the formulation it
replaces) bit for bit, values and gradients, without building any [T, T] index."""

import itertools

import pytest
import torch

from unsloth.utils.ulysses_inkling2 import _whole_sequence_bias


def _gather_bias(rel_logits, total, rel_extent):
    idx = torch.arange(total, device = rel_logits.device)
    distance = idx[:, None] - idx[None, :]
    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    bias = rel_logits.gather(-1, gather_index)
    return bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_whole_sequence_bias_matches_the_gather_bitwise(dtype):
    g = torch.Generator().manual_seed(0)
    for batch, heads, total, extent in itertools.product(
        [1, 2], [1, 3], [1, 2, 3, 7, 64, 129], [1, 2, 5, 64, 300]
    ):
        rel = torch.randn(batch, heads, total, extent, generator = g).to(dtype)
        if total > 2 and extent > 1:
            rel[..., 2, 1] = -0.0
        a = rel.clone().requires_grad_(True)
        b = rel.clone().requires_grad_(True)
        want, got = _gather_bias(a, total, extent), _whole_sequence_bias(b, total, extent)
        case = (batch, heads, total, extent)
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous(), case
        assert torch.equal(got.view(-1).view(torch.uint8), want.view(-1).view(torch.uint8)), case
        grad_out = torch.randn(want.shape, generator = g).to(dtype)
        (ga,) = torch.autograd.grad(want, a, grad_out)
        (gb,) = torch.autograd.grad(got, b, grad_out)
        assert torch.equal(ga, gb), case


def test_whole_sequence_bias_builds_no_square_index():
    # Only the [B, H, T, T] bias itself is T x T: no int64 distance / index and no bool mask of that size.
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    total, extent = 96, 8
    rel = torch.randn(1, 2, total, extent, requires_grad = True)
    square = []

    class _Record(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if isinstance(t, torch.Tensor) and t.dtype in (
                    torch.int64,
                    torch.int32,
                    torch.bool,
                ):
                    if t.numel() >= total * total:
                        square.append((str(func), t.dtype, tuple(t.shape)))
            return out

    with _Record():
        bias = _whole_sequence_bias(rel, total, extent)
        bias.sum().backward()
    assert square == [], square
