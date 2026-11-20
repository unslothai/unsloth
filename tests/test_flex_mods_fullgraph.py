# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Unsloth's own flex_attention mask_mods trace under torch.compile(fullgraph = True).

torch 2.15 (pytorch/pytorch#185738) rejects flex_attention bodies that mutate their inputs or whose mask_mod returns
an alias of a captured tensor (fullgraph raises; otherwise a graph break into eager dense attention). Checked with
backend = "aot_eager" (Dynamo's HOP checks, no Inductor codegen) against eager flex_attention. The zoo families are
covered by unsloth_zoo's tests/test_flex_mods_fullgraph.py."""

import pytest
import torch

flex = pytest.importorskip("torch.nn.attention.flex_attention")
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _check(
    mask_mod,
    batch,
    q_len,
    kv_len,
    heads = 4,
    dim = 16,
):
    block_mask = create_block_mask(mask_mod, batch, None, q_len, kv_len, device = DEVICE)
    g = torch.Generator(device = "cpu").manual_seed(0)
    q, k, v = (
        torch.randn(batch or 1, heads, n, dim, generator = g).to(DEVICE)
        for n in (q_len, kv_len, kv_len)
    )

    def attend(q, k, v):
        return flex_attention(q, k, v, block_mask = block_mask)

    torch._dynamo.reset()
    compiled = torch.compile(attend, fullgraph = True, backend = "aot_eager")(q, k, v)
    torch.testing.assert_close(compiled, attend(q, k, v), atol = 1e-5, rtol = 1e-5)


def test_blockwise_overlay_mask_mod():
    from unsloth.import_fixes import _bounded_blockwise_overlay

    ids = torch.full((2, 100), -1, dtype = torch.long, device = DEVICE)
    ids[:, 10:60] = 0
    ids[1, 60:] = 1
    _check(_bounded_blockwise_overlay(ids), 2, 128, 128)


def test_prefix_grouper_mask_mod():
    from unsloth.utils.prefix_grouper_kernel import _make_mask_mod

    total = 256
    group_of_kv = torch.arange(total, device = DEVICE) // 128
    is_prefix = (torch.arange(total, device = DEVICE) % 128) < 32
    suffix_of_kv = torch.where(is_prefix, -1, (torch.arange(total, device = DEVICE) % 128 - 32) // 24)
    _check(_make_mask_mod(group_of_kv, is_prefix, suffix_of_kv), None, total, total)
