# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native decode writes K / V projections straight into the step's KV-cache slot (K rotated there in place).
Checked against the copy-based path it replaces."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import torch

from unsloth.models import llama

N_HEADS, N_KV, HEAD_DIM, HIDDEN = 8, 2, 64, 256

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")


def _attention(dtype, device):
    torch.manual_seed(0)

    def linear(i, o):
        layer = torch.nn.Linear(i, o, bias = False).to(device = device, dtype = dtype)
        layer.weight.requires_grad_(False)
        return layer

    pos = torch.arange(4096, device = device, dtype = torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, device = device).float() / HEAD_DIM))
    freqs = torch.cat([pos * inv, pos * inv], dim = -1)
    cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
    config = types.SimpleNamespace(
        num_attention_heads = N_HEADS,
        num_key_value_heads = N_KV,
        hidden_size = HIDDEN,
        sliding_window = None,
    )
    rotary = types.SimpleNamespace(
        extend_rope_embedding = lambda x, n: None, get_cached = lambda n, i: (cos, sin)
    )
    return (
        types.SimpleNamespace(
            config = config,
            num_key_value_groups = N_HEADS // N_KV,
            head_dim = HEAD_DIM,
            q_proj = linear(HIDDEN, N_HEADS * HEAD_DIM),
            k_proj = linear(HIDDEN, N_KV * HEAD_DIM),
            v_proj = linear(HIDDEN, N_KV * HEAD_DIM),
            o_proj = linear(N_HEADS * HEAD_DIM, HIDDEN),
            rotary_emb = rotary,
        ),
        cos,
        sin,
    )


def _rotate(x, cos, sin, position):
    h = x.shape[-1] // 2
    rh = torch.cat((-x[..., h:], x[..., :h]), dim = -1)
    return x * cos[position] + rh * sin[position]


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bsz", [1, 3])
@pytest.mark.parametrize("ignore_out", [False, True])
def test_kv_slot_holds_rotated_k_and_raw_v(dtype, bsz, ignore_out, monkeypatch):
    if ignore_out:
        # Quantized projections (NVFP4 / FP8) return a fresh tensor: the slot must still be filled.
        real = llama.fast_linear_forward
        monkeypatch.setattr(
            llama,
            "fast_linear_forward",
            lambda proj, X, temp_lora = None, out = None: real(proj, X).clone(),
        )
    attn, cos, sin = _attention(dtype, "cuda")
    cached = 5
    K1 = torch.randn(bsz, N_KV, cached, HEAD_DIM, dtype = dtype, device = "cuda")
    V1 = torch.randn(bsz, N_KV, cached, HEAD_DIM, dtype = dtype, device = "cuda")
    past = (K1, V1)
    for step in range(4):
        seq_len = cached + step
        X = torch.randn(bsz, 1, HIDDEN, dtype = dtype, device = "cuda")
        out, past = llama.LlamaAttention_fast_forward_inference(
            attn,
            X,
            past,
            torch.full((bsz, 1), seq_len, device = "cuda"),
            attention_mask = None,
            do_prefill = step == 0,
        )
        k = (X @ attn.k_proj.weight.t()).view(bsz, N_KV, HEAD_DIM)
        v = (X @ attn.v_proj.weight.t()).view(bsz, N_KV, HEAD_DIM)
        torch.testing.assert_close(
            attn.paged_attention_K[seq_len], _rotate(k, cos, sin, seq_len), atol = 2e-2, rtol = 2e-2
        )
        torch.testing.assert_close(attn.paged_attention_V[seq_len], v, atol = 0, rtol = 0)
        assert past[0].shape[-2] == seq_len + 1
        torch.testing.assert_close(
            past[0][:, :, -1], attn.paged_attention_K[seq_len], atol = 0, rtol = 0
        )
        # Earlier slots are untouched by later steps.
        torch.testing.assert_close(
            attn.paged_attention_K[:cached].permute(1, 2, 0, 3), K1, atol = 0, rtol = 0
        )
