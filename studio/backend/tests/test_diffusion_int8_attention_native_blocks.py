# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NATIVE_BLOCKS: Auto keeps Qwen-Image-2.1 block 0 native; forced or the env switch =0 routes it. CPU, fakes as in
test_diffusion_int8_attention_dispatch."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D

pytest.importorskip("diffusers.models.attention_dispatch")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import rocm_bf16

    for env in (
        I.INT8_ATTENTION_ENV,
        I.IMPL_ENV,
        D.DISPATCH_ENV,
        "UNSLOTH_DIFFUSION_INT8_ATTENTION_NATIVE_BLOCKS",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(D, "MIN_TOKENS", 0)
    monkeypatch.setattr(D, "_on_card", lambda x, route: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: False)
    monkeypatch.setattr(I, "verdict", lambda *a, **k: {"impl": "cuda", "ok": True, "min_tokens": 0})
    D._DECISIONS.clear()
    yield
    D._DECISIONS.clear()


def _attn_modules(n):
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    wan = WanTransformer3DModel(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 64,
        in_channels = 4,
        out_channels = 4,
        text_dim = 32,
        freq_dim = 32,
        ffn_dim = 64,
        num_layers = n,
        rope_max_seq_len = 32,
    ).to(torch.bfloat16)
    wan.set_attention_backend("native")
    return [b.attn1 for b in wan.blocks]


def _dit(name, n = 4):
    """A DiT of class ``name`` whose ``transformer_blocks.<i>.attn`` are real dispatch-calling attention modules."""
    cls = type(name, (torch.nn.Module,), {})
    dit = cls()
    blocks = torch.nn.ModuleList()
    for m in _attn_modules(n):
        blk = torch.nn.Module()
        blk.attn = m
        blocks.append(blk)
    dit.transformer_blocks = blocks
    return dit


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def _routed(dit):
    return [
        isinstance(b.attn.processor._attention_backend, D.Int8Route) for b in dit.transformer_blocks
    ]


def test_block_index_reads_the_block_number():
    assert D.block_index("transformer_blocks.0.attn") == 0
    assert D.block_index("transformer_blocks.17.attn") == 17
    assert D.block_index("single_transformer_blocks.0.attn") is None
    assert D.block_index("blocks.0.attn1") is None
    assert D.block_index("attn") is None


def test_auto_keeps_qwen_image_21_block_0_on_its_incumbent():
    dit = _dit("QwenImage21Transformer2DModel")
    assert D.install(_Pipe(dit), _TARGET) == 3
    assert _routed(dit) == [False, True, True, True]
    assert D.routed_count(_Pipe(dit)) == 3


def test_a_forced_request_routes_every_block(monkeypatch):
    monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    dit = _dit("QwenImage21Transformer2DModel")
    assert D.install(_Pipe(dit), _TARGET) == 4
    assert _routed(dit) == [True] * 4


def test_the_kill_switch_routes_block_0_again(monkeypatch):
    monkeypatch.setenv(D.NATIVE_BLOCKS_ENV, "0")
    dit = _dit("QwenImage21Transformer2DModel")
    assert D.install(_Pipe(dit), _TARGET) == 4


def test_other_families_route_every_block():
    dit = _dit("ZImageTransformer2DModel")
    assert D.install(_Pipe(dit), _TARGET) == 4
    assert _routed(dit) == [True] * 4
