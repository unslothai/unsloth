# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A layer split on two real CUDA cards against one card, bitwise. Skipped below two cards."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_layer_split as ls

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)

_NL, _NS = 3, 5


def _hidream():
    hidream = pytest.importorskip("diffusers.models.transformers.transformer_hidream_image")
    torch.manual_seed(0)
    model = hidream.HiDreamImageTransformer2DModel(
        patch_size = 2,
        in_channels = 16,
        num_layers = _NL,
        num_single_layers = _NS,
        attention_head_dim = 32,
        num_attention_heads = 4,
        caption_channels = [64, 48],
        text_emb_dim = 64,
        num_routed_experts = 4,
        num_activated_experts = 2,
        axes_dims_rope = (8, 12, 12),
        max_resolution = (64, 64),
        llama_layers = list(range(_NL + _NS)),
    ).to(torch.bfloat16)
    with torch.no_grad():  # weights large enough that every block moves the result
        for p in model.parameters():
            p.normal_(0, 0.05)
    return model.eval()


def _run(model):
    g = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states = torch.randn(1, 16, 32, 32, generator = g),
        encoder_hidden_states_t5 = torch.randn(1, 16, 64, generator = g),
        encoder_hidden_states_llama3 = torch.randn(_NL + _NS, 1, 12, 48, generator = g),
        pooled_embeds = torch.randn(1, 64, generator = g),
    )
    kw = {k: v.to("cuda:0", torch.bfloat16) for k, v in inputs.items()}
    with torch.inference_mode():
        return (
            model(**kw, timesteps = torch.tensor([500], device = "cuda:0"), return_dict = False)[0]
            .float()
            .cpu()
        )


class _Pipe:
    pass


def test_a_lazy_hidream_split_matches_one_card_bitwise():
    assert "HiDreamImageTransformer2DModel" in ls.LAZY_CLASSES
    reference = _run(_hidream().to("cuda:0"))
    pipe = _Pipe()
    pipe.transformer = _hidream().to("cuda:0")
    blocks = [qual for qual, _, _ in ls.find_blocks(pipe.transformer)]
    ls.place_layer_split(
        pipe,
        primary = 0,
        assignment = {q: "cuda:0" if i == 0 else "cuda:1" for i, q in enumerate(blocks)},
    )
    try:
        out = _run(pipe.transformer)
        stats = ls.layer_split_stats(pipe.transformer)
        assert {str(p.device) for p in pipe.transformer.parameters()} == {"cuda:0", "cuda:1"}
        assert stats["lazy_calls"] == 1 and stats["eager_calls"] == 0
        assert torch.equal(out, reference)
    finally:
        ls.undo_layer_split(pipe)


def test_a_hidream_split_hops_once_per_card_boundary_and_records_per_card_segments(monkeypatch):
    from core.inference import diffusion_block_graph as bg

    monkeypatch.setitem(globals(), "_NL", 4)
    monkeypatch.setitem(globals(), "_NS", 6)
    reference = _run(_hidream().to("cuda:0"))
    pipe = _Pipe()
    pipe.transformer = _hidream().to("cuda:0")
    on_second = {"double_stream_blocks.2.block", "double_stream_blocks.3.block"} | {
        f"single_stream_blocks.{i}.block" for i in (3, 4, 5)
    }
    blocks = [qual for qual, _, _ in ls.find_blocks(pipe.transformer)]
    assert on_second <= set(blocks) and len(blocks) == 10
    ls.place_layer_split(
        pipe, primary = 0, assignment = {q: "cuda:1" if q in on_second else "cuda:0" for q in blocks}
    )
    hops = []
    real = ls.cross_outputs
    monkeypatch.setattr(
        ls, "cross_outputs", lambda out, device: hops.append(str(device)) or real(out, device)
    )
    try:
        out = _run(pipe.transformer)
        stats = ls.layer_split_stats(pipe.transformer)
        assert hops == ["cuda:1", "cuda:0", "cuda:1", "cuda:0"]
        assert stats["input_copies"] == 2 and stats["lazy_calls"] == 1
        assert torch.equal(out, reference)
        handle, why = bg.install_block_graphs(pipe.transformer, slots = False)
        assert handle is not None, why
        try:
            assert [len(seg.graphs) for seg in handle.segments] == [2, 2, 3, 3]
        finally:
            handle.free()
    finally:
        ls.undo_layer_split(pipe)
