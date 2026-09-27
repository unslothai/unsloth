# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched Gemma2 decode must pass tensor padding masks even when flash-attn softcapping is
installed: the decode attention is manual matmul and ignores anything but a tensor mask."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models.gemma2 as g2

torch = pytest.importorskip("torch")


class _Captured(Exception):
    pass


def _masks_reaching_attention(monkeypatch, flash):
    seen = []

    def attention(self, hidden_states, past_key_value, position_ids, attention_mask, **kw):
        seen.append((kw["use_sliding_window"], attention_mask))
        if len(seen) == 2:
            raise _Captured
        return hidden_states, past_key_value

    monkeypatch.setattr(g2, "HAS_FLASH_ATTENTION_SOFTCAPPING", flash)
    monkeypatch.setattr(g2, "DEVICE_COUNT", 0)
    monkeypatch.setattr(g2, "Gemma2Attention_fast_forward_inference", attention)
    monkeypatch.setattr(g2, "fast_rms_layernorm_inference_gemma", lambda ln, x, w: x)
    monkeypatch.setattr(g2, "fast_geglu_inference", lambda mlp, x: x)
    # No per-device fp32 buffers on CPU; the stubbed norms never read them.
    monkeypatch.setattr(g2, "per_layer_device", lambda layer: (torch.device("cpu"), slice(None)))

    hidden, bsz, cached = 8, 2, 5
    layer = types.SimpleNamespace(
        input_layernorm = types.SimpleNamespace(weight = torch.ones(hidden)),
        post_attention_layernorm = None,
        pre_feedforward_layernorm = None,
        post_feedforward_layernorm = None,
        self_attn = None,
        mlp = None,
    )
    config = types.SimpleNamespace(
        hidden_size = hidden, sliding_window = 4096, torch_dtype = torch.float32
    )
    model = types.SimpleNamespace(
        layers = [layer, layer],
        embed_tokens = lambda ids: torch.zeros(*ids.shape, hidden),
    )
    self = types.SimpleNamespace(model = model, config = config, max_seq_length = 64)
    past = [(torch.zeros(bsz, 1, cached, 4), torch.zeros(bsz, 1, cached, 4))] * 2
    # Row 1 is left padded by two tokens.
    attention_mask = torch.tensor([[1] * (cached + 1), [0, 0] + [1] * (cached - 1)])
    with pytest.raises(_Captured):
        g2.Gemma2Model_fast_forward_inference(
            self,
            torch.zeros(bsz, 1, dtype = torch.long),
            past,
            torch.full((bsz, 1), cached),
            attention_mask = attention_mask,
        )
    return seen


@pytest.mark.parametrize("flash", [False, True])
def test_batched_decode_masks_left_padding(monkeypatch, flash):
    seen = _masks_reaching_attention(monkeypatch, flash)
    assert [sliding for sliding, _ in seen] == [True, False]
    for _, mask in seen:
        assert isinstance(mask, torch.Tensor)
        assert mask.shape[-1] == 6
        assert torch.all(mask[0] == 0)
        assert torch.all(mask[1, ..., :2] < -1e30)
        assert torch.all(mask[1, ..., 2:] == 0)
