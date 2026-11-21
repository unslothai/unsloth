# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 LoRA training drops the masked text padding rows: the transformer output is unchanged (fp64) while the
joint sequence shrinks from 512 text rows to the valid ones."""

from __future__ import annotations

import pytest
import torch

from core.training import diffusion_dit_trainer as dit


def _entry(
    valid,
    total = 32,
    layers = 3,
    dim = 16,
    hole = True,
):
    pe = torch.randn(1, total, layers, dim)
    mask = torch.zeros(1, total, dtype = torch.bool)
    if hole:
        # The pipeline's template puts the prompt's own padding between the prompt and a suffix.
        mask[0, : valid - 2] = True
        mask[0, total - 2 :] = True
    else:
        mask[0, :valid] = True
    return pe, mask


def test_text_rows_keeps_only_valid_rows_in_order():
    pe, mask = _entry(10)
    out, m = dit._krea2_text_rows(pe, mask)
    assert out.shape == (1, 10, 3, 16) and m.all()
    assert torch.equal(out, pe[:, mask[0]])
    full = torch.ones(1, 32, dtype = torch.bool)
    assert dit._krea2_text_rows(pe, full)[0] is pe


def test_collate_pads_to_the_longest_or_the_pin_and_drops_the_mask_when_unpadded():
    a, b = _entry(10), _entry(6)
    pe_b, mask_b = dit._krea2_collate([a, b], "cpu", torch.float32)
    assert pe_b.shape == (2, 10, 3, 16)
    assert mask_b.tolist()[1] == [True] * 6 + [False] * 4
    pe_1, mask_1 = dit._krea2_collate([a], "cpu", torch.float32)
    assert pe_1.shape[1] == 10 and mask_1 is None
    pe_p, mask_p = dit._krea2_collate([a], "cpu", torch.float32, pad_to = 14)
    assert pe_p.shape[1] == 14 and mask_p[0].sum() == 10


def _tiny_krea2():
    diffusers = pytest.importorskip("diffusers")
    model_cls = getattr(diffusers, "Krea2Transformer2DModel", None)
    if model_cls is None:
        pytest.skip("diffusers without Krea2Transformer2DModel")
    torch.manual_seed(0)
    return (
        model_cls(
            in_channels = 16,
            num_layers = 2,
            attention_head_dim = 16,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            intermediate_size = 64,
            timestep_embed_dim = 32,
            text_hidden_dim = 16,
            num_text_layers = 3,
            text_num_attention_heads = 2,
            text_num_key_value_heads = 2,
            text_intermediate_size = 32,
            axes_dims_rope = (4, 6, 6),
        )
        .double()
        .eval()
    )


def test_trimmed_forward_matches_the_padded_masked_forward():
    model = _tiny_krea2()
    spec = dit._SPECS["krea-2"]
    pe, mask = _entry(10)
    pe = pe.double()
    noisy = torch.randn(1, 4, 1, 8, 8, dtype = torch.float64)
    t = torch.tensor([400.0], dtype = torch.float64)
    with torch.no_grad():
        # main's path: the full padded sequence with its mask.
        ref = spec.forward(model, noisy, t, None, (pe, mask), None, "cpu", torch.float64)
        got = spec.forward(
            model,
            noisy,
            t,
            None,
            dit._krea2_collate([(pe, mask)], "cpu", torch.float64),
            None,
            "cpu",
            torch.float64,
        )
    assert got.shape == ref.shape
    assert (got - ref).abs().max().item() < 1e-10
