# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma2 cached decode through flash_attn_with_kvcache: routing, left padding and numerics."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models.gemma2 as g2
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")


class _Captured(Exception):
    pass


def _decode_kwargs(
    monkeypatch,
    bsz,
    flash,
    second_device = "cpu",
    mask_rows = None,
    mask = None,
):
    seen = []

    def attention(self, hidden_states, past_key_value, position_ids, attention_mask, **kw):
        seen.append(dict(kw, attention_mask = attention_mask))
        if len(seen) == 2:
            raise _Captured
        return hidden_states, past_key_value

    monkeypatch.setattr(g2, "_flash_decode_usable", lambda config, hidden, mask: flash)
    monkeypatch.setattr(g2, "DEVICE_COUNT", 0)
    monkeypatch.setattr(g2, "Gemma2Attention_fast_forward_inference", attention)
    monkeypatch.setattr(g2, "fast_rms_layernorm_inference_gemma", lambda ln, x, w: x)
    monkeypatch.setattr(g2, "fast_geglu_inference", lambda mlp, x: x)
    # No per-device fp32 buffers on CPU; the stubbed norms never read them.
    monkeypatch.setattr(
        g2, "per_layer_device", lambda layer: (torch.device(layer.device), slice(None))
    )

    hidden, cached = 8, 5
    layer = types.SimpleNamespace(
        device = "cpu",
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
        layers = [layer, types.SimpleNamespace(**{**vars(layer), "device": second_device})],
        embed_tokens = lambda ids: torch.zeros(*ids.shape, hidden),
    )
    self = types.SimpleNamespace(model = model, config = config, max_seq_length = 64)
    past = [(torch.zeros(bsz, 1, cached, 4), torch.zeros(bsz, 1, cached, 4))] * 2
    attention_mask = torch.tensor(
        [[1] * (cached + 1), [0, 0] + [1] * (cached - 1), [0] * 4 + [1] * 2]
    )[mask_rows if mask_rows is not None else slice(0, bsz)]
    if mask is not None:
        attention_mask = torch.tensor(mask)
    with pytest.raises(_Captured):
        g2.Gemma2Model_fast_forward_inference(
            self,
            torch.zeros(bsz, 1, dtype = torch.long),
            past,
            torch.full((bsz, 1), cached),
            attention_mask = attention_mask,
        )
    return seen


def test_flash_decode_passes_left_padding_not_masks(monkeypatch):
    seen = _decode_kwargs(monkeypatch, bsz = 3, flash = True)
    for kw in seen:
        assert kw["flash_decode"] is True
        assert kw["attention_mask"] is None
        assert kw["leftpad"].dtype == torch.int32
        assert kw["leftpad"].tolist() == [0, 2, 4]


def test_flash_decode_single_padded_row_keeps_leftpad(monkeypatch):
    seen = _decode_kwargs(monkeypatch, bsz = 1, flash = True, mask_rows = [1])
    assert [(kw["attention_mask"], kw["leftpad"].tolist()) for kw in seen] == [(None, [2])] * 2


def test_flash_decode_leftpad_follows_layer_device(monkeypatch):
    # Pipeline-parallel device maps: flash-attn rejects a cache_leftpad on another GPU.
    seen = _decode_kwargs(monkeypatch, bsz = 3, flash = True, second_device = "meta")
    assert [kw["leftpad"].device.type for kw in seen] == ["cpu", "meta"]


@pytest.mark.parametrize("rows", [[[0, 1, 1, 1, 1, 1]], []])
@pytest.mark.parametrize("gap_row", [[1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
def test_flash_decode_falls_back_for_non_left_padding(monkeypatch, gap_row, rows):
    mask = rows + [gap_row]
    seen = _decode_kwargs(monkeypatch, bsz = len(mask), flash = True, mask = mask)
    for kw in seen:
        assert kw["flash_decode"] is False
        assert kw["leftpad"] is None
        assert isinstance(kw["attention_mask"], torch.Tensor)


def test_manual_path_keeps_masks(monkeypatch):
    seen = _decode_kwargs(monkeypatch, bsz = 2, flash = False)
    for kw in seen:
        assert kw["flash_decode"] is False
        assert kw["leftpad"] is None
        assert isinstance(kw["attention_mask"], torch.Tensor)


def test_usable_refuses_unsupported_inputs(monkeypatch):
    config = types.SimpleNamespace(head_dim = 256, hidden_size = 2304, num_attention_heads = 8)
    monkeypatch.setattr(g2, "_FLASH_DECODE", True)
    monkeypatch.setattr(g2, "_FLASH_DECODE_PROBED", {})
    mask = torch.ones(2, 6, dtype = torch.long)
    assert not g2._flash_decode_usable(config, torch.zeros(2, 1, 8, dtype = torch.float32), mask)
    assert not g2._flash_decode_usable(
        config, torch.zeros(2, 1, 8, dtype = torch.bfloat16), mask[:, None, None]
    )
    odd = types.SimpleNamespace(head_dim = 100, hidden_size = 800, num_attention_heads = 8)
    assert not g2._flash_decode_usable(odd, torch.zeros(2, 1, 8, dtype = torch.bfloat16), mask)
    monkeypatch.setattr(g2, "_FLASH_DECODE", False)
    assert not g2._flash_decode_usable(config, torch.zeros(2, 1, 8, dtype = torch.bfloat16), mask)


def _reference(Q, K_cache, V_cache, kv, leftpad, scale, softcap, window):
    K = K_cache[:kv].permute(1, 2, 0, 3).float()
    V = V_cache[:kv].permute(1, 2, 0, 3).float()
    groups = Q.shape[1] // K.shape[1]
    K, V = K.repeat_interleave(groups, 1), V.repeat_interleave(groups, 1)
    A = (Q.float() * scale) @ K.transpose(2, 3)
    A = softcap * torch.tanh(A / softcap)
    j = torch.arange(kv, device = Q.device)
    keep = j[None] >= leftpad[:, None].long()
    if window is not None:
        keep &= j[None] >= kv - window
    A = A.masked_fill(~keep[:, None, None], float("-inf"))
    return (torch.softmax(A, -1) @ V).transpose(1, 2)


@pytest.mark.skipif(not has_real_cuda(), reason = "runs the flash_attn_with_kvcache CUDA kernel")
@pytest.mark.parametrize(
    "n_heads,n_kv_heads,head_dim,scalar", [(8, 4, 256, 256), (32, 16, 128, 144)]
)
@pytest.mark.parametrize("window", [None, 64])
def test_flash_decode_matches_reference(n_heads, n_kv_heads, head_dim, scalar, window):
    pytest.importorskip("flash_attn")
    if not g2._FLASH_DECODE:
        pytest.skip("flash-attn softcapping unavailable")
    torch.manual_seed(0)
    bsz, kv, capacity = 3, 200, 456
    cache = torch.randn(capacity, 2, bsz, n_kv_heads, head_dim, device = "cuda", dtype = torch.bfloat16)
    K_cache, V_cache = cache[:, 0], cache[:, 1]
    Q = torch.randn(bsz, n_heads, 1, head_dim, device = "cuda", dtype = torch.bfloat16)
    leftpad = torch.tensor([0, 17, 150], device = "cuda", dtype = torch.int32)
    scale = scalar**-0.5
    out = g2._gemma2_flash_decode(Q, K_cache, V_cache, kv, leftpad, scale, 50.0, window)
    ref = _reference(Q, K_cache, V_cache, kv, leftpad, scale, 50.0, window)
    assert out.shape == (bsz, 1, n_heads, head_dim)
    torch.testing.assert_close(out.float(), ref, atol = 2e-2, rtol = 2e-2)
