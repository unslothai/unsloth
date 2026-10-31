# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU checks for beam search on the fast decode path (issue #1099), against unsloth/models/llama.py."""

import types

import torch
from transformers.cache_utils import DynamicCache

BEAMS, HEADS, DIM, SLACK = 3, 2, 4, 5


def _llama():
    from unsloth.models import llama
    return llama


def _attention_with_buffer(seq_len):
    # Same layout LlamaAttention_fast_forward_inference allocates: (seq, 2, bsz, kv_heads, head_dim).
    attn = types.SimpleNamespace()
    attn.paged_attention = torch.randn(seq_len + SLACK, 2, BEAMS, HEADS, DIM)
    attn.paged_attention_K = attn.paged_attention[:, 0]
    attn.paged_attention_V = attn.paged_attention[:, 1]
    K = attn.paged_attention_K[:seq_len].permute(1, 2, 0, 3)
    V = attn.paged_attention_V[:seq_len].permute(1, 2, 0, 3)
    return attn, (K, V)


def _model(attentions):
    layers = [types.SimpleNamespace(self_attn = attn) for attn in attentions]
    return types.SimpleNamespace(model = types.SimpleNamespace(layers = layers))


def test_fast_forward_classes_get_reorder_cache():
    llama = _llama()

    class Fake:
        def prepare_inputs_for_generation(self):
            pass

    llama.fix_prepare_inputs_for_generation(Fake)
    assert Fake._reorder_cache is llama._fast_reorder_cache


def test_decode_buffers_are_reordered_in_place():
    seq_len = 6
    attentions, past = zip(*(_attention_with_buffer(seq_len) for _ in range(2)))
    before = [a.paged_attention.clone() for a in attentions]
    beam_idx = torch.tensor([2, 2, 0])

    out = _llama()._fast_reorder_cache(_model(attentions), list(past), beam_idx)

    for attn, old, (K, V), (K_out, V_out) in zip(attentions, before, past, out):
        # The next decode step reads the buffer, not the returned tuple.
        assert torch.equal(attn.paged_attention[:seq_len], old[:seq_len, :, beam_idx])
        assert torch.equal(attn.paged_attention[seq_len:], old[seq_len:])
        assert K_out is K and V_out is V
        assert torch.equal(K_out, old[:seq_len, 0, beam_idx].permute(1, 2, 0, 3))


def test_prefill_tuples_without_buffers_are_index_selected():
    past = [(torch.randn(BEAMS, HEADS, 4, DIM), torch.randn(BEAMS, HEADS, 4, DIM))]
    attn = types.SimpleNamespace()  # Buffers are dropped by the prefill forward.
    beam_idx = torch.tensor([1, 0, 0])

    out = _llama()._fast_reorder_cache(_model([attn]), past, beam_idx)

    assert torch.equal(out[0][0], past[0][0][beam_idx])
    assert torch.equal(out[0][1], past[0][1][beam_idx])


def test_cache_objects_use_their_own_reorder():
    cache = DynamicCache()
    keys = torch.randn(BEAMS, HEADS, 4, DIM)
    cache.update(keys, keys.clone(), 0)
    beam_idx = torch.tensor([2, 1, 1])

    out = _llama()._fast_reorder_cache(_model([]), cache, beam_idx)

    assert out is cache
    assert torch.equal(cache.layers[0].keys, keys[beam_idx])
