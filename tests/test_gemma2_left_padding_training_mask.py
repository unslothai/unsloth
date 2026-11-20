# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Gemma2 builds its own sliding and global training masks; a left pad query must still attend only itself
(the shared branch's repair, #18272), else the softcapping kernels softmax it over every key."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models.llama as llama_module

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


class _Captured(Exception):
    pass


class _CaptureLayer(torch.nn.Module):
    def __init__(self, seen, n_layers):
        super().__init__()
        self.seen = seen
        self.n_layers = n_layers
        # Skips the shared rotary build: these layers never read position embeddings.
        self.self_attn = types.SimpleNamespace(rotary_emb = None)

    def forward(
        self,
        hidden_states,
        causal_mask = None,
        **kwargs,
    ):
        self.seen.append((kwargs.get("use_sliding_window"), causal_mask))
        if len(self.seen) == self.n_layers:
            raise _Captured
        return (hidden_states,)


def _gemma2_masks(
    attention_mask,
    training,
    dtype = torch.float32,
    window = 3,
):
    config = transformers.Gemma2Config(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
        sliding_window = window,
    )
    model = transformers.Gemma2Model(config).to(dtype)
    model.train(training)
    del model.rotary_emb
    seen = []
    model.layers = torch.nn.ModuleList([_CaptureLayer(seen, 2), _CaptureLayer(seen, 2)])
    model.max_seq_length = 64
    input_ids = torch.full(attention_mask.shape, 5).masked_fill(attention_mask == 0, 0)
    with pytest.raises(_Captured):
        llama_module.LlamaModel_fast_forward(
            model, input_ids = input_ids, attention_mask = attention_mask, use_cache = False
        )
    assert [sliding for sliding, _ in seen] == [True, False]
    return seen


def _reference(attention_mask, dtype, window):
    n = attention_mask.shape[-1]
    swa = llama_module.Gemma2AttentionMaskConverter(is_causal = True, sliding_window = window)
    ga = AttentionMaskConverter(is_causal = True)
    return [
        swa.to_4d(attention_mask, n, dtype, n),
        ga.to_4d(attention_mask, n, dtype, n),
    ]


def _softcap_attention(
    Q,
    K,
    V,
    mask,
    cap = 50.0,
):
    # The eager math of kernels/flex_attention.py _compiled_slow_attention_softcapping.
    A = Q @ K.transpose(-1, -2)
    A = cap * torch.tanh(A / cap) + mask
    return torch.softmax(A, dim = -1, dtype = torch.float32).to(Q.dtype) @ V


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_left_pad_queries_attend_only_themselves(dtype):
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    n = attention_mask.shape[-1]
    seen = _gemma2_masks(attention_mask, training = True, dtype = dtype)
    pad = attention_mask == 0
    eye = torch.eye(n, dtype = torch.bool)
    torch.manual_seed(0)
    Q, K, V = (torch.randn(3, 2, n, 8).to(dtype) for _ in range(3))
    for (_, mask), reference in zip(seen, _reference(attention_mask, dtype, 3)):
        assert mask.shape == (3, 1, n, n) and mask.dtype == dtype
        allowed = mask == 0
        for b in range(3):
            for i in range(n):
                if pad[b, i]:
                    assert torch.equal(allowed[b, 0, i], eye[i]), (b, i, allowed[b, 0, i])
                else:
                    # Real-token rows are the converter's, bit for bit.
                    assert torch.equal(mask[b, 0, i], reference[b, 0, i]), (b, i)
        # The last pad query predicts the first real token: changing later values must not move it.
        later = V.clone()
        later[0, :, 2:] += 3.0
        out = _softcap_attention(Q, K, V, mask)
        moved = _softcap_attention(Q, K, later, mask)
        assert torch.equal(out[0, :, :2], moved[0, :, :2])
        # Real-token outputs are unchanged bitwise against the converter's mask.
        real = ~pad[:, None, :, None].expand_as(out)
        expected = _softcap_attention(Q, K, V, reference)
        assert torch.equal(out[real], expected[real])


def test_eval_masks_are_untouched():
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    seen = _gemma2_masks(attention_mask, training = False)
    for (_, mask), reference in zip(seen, _reference(attention_mask, torch.float32, 3)):
        assert torch.equal(mask, reference)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.bfloat16])
def test_repair_matches_masked_fill_form(dtype):
    # The single-output rewrite equals the two masked_fill form it replaced, bit for bit.
    torch.manual_seed(0)
    mask_2d = (torch.rand(4, 7) > 0.4).long()
    keep = torch.rand(4, 1, 7, 7) > 0.5
    pad_q = (mask_2d == 0)[:, None, :, None]
    diag = torch.eye(7, dtype = torch.bool)
    if dtype == torch.bool:
        mask_4d = keep
        expected = (mask_4d & ~pad_q) | (pad_q & diag)
    else:
        blocked = torch.finfo(dtype).min
        mask_4d = torch.randn(4, 1, 7, 7).to(dtype).masked_fill(~keep, blocked)
        expected = mask_4d.masked_fill(pad_q & ~diag, blocked).masked_fill(pad_q & diag, 0)
    before = mask_4d.clone()
    out = llama_module._pad_queries_attend_self(mask_4d, mask_2d)
    assert out.dtype == dtype and torch.equal(out, expected)
    assert torch.equal(mask_4d, before)
