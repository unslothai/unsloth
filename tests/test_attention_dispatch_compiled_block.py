# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Attention dispatch traced fullgraph with a dynamic sequence length (CPU)."""

import pytest
import torch

ad = pytest.importorskip("unsloth.utils.attention_dispatch")


def _run(backend, causal_mask = None):
    def attend(Q, K, V):
        bsz, n_heads, q_len, head_dim = Q.shape
        config = ad.AttentionConfig(backend = backend, n_kv_heads = K.shape[1], n_groups = n_heads // K.shape[1])
        context = ad.AttentionContext(
            bsz = bsz,
            q_len = q_len,
            kv_seq_len = K.shape[-2],
            n_heads = n_heads,
            head_dim = head_dim,
            requires_grad = True,
            seq_info = None,
            attention_mask = None,
            causal_mask = causal_mask,
        )
        return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)

    return attend


def _qkv(q_len):
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(1, 4, q_len, 16, generator = g)
    K = torch.randn(1, 2, q_len, 16, generator = g)
    V = torch.randn(1, 2, q_len, 16, generator = g)
    return Q, K, V


def _reference(Q, K, V):
    K = K.repeat_interleave(2, dim = 1)
    V = V.repeat_interleave(2, dim = 1)
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal = True).transpose(1, 2)


@pytest.mark.parametrize("backend", ["sdpa", "xformers"])
def test_plain_causal_attention_traces_with_a_dynamic_length(backend):
    backend = {"sdpa": ad.SDPA, "xformers": ad.XFORMERS}[backend]
    torch._dynamo.reset()
    compiled = torch.compile(_run(backend), fullgraph = True, backend = "eager", dynamic = True)
    for q_len in (8, 12):
        Q, K, V = _qkv(q_len)
        out = compiled(Q, K, V)
        torch.testing.assert_close(out.reshape(1, q_len, 4, 16), _reference(Q, K, V))


def test_xformers_eager_choice_is_unchanged(monkeypatch):
    # Outside a trace the backend is the caller's: the SDPA route is only taken while compiling.
    seen = []
    monkeypatch.setattr(ad, "build_xformers_block_causal_mask", lambda *a, **k: seen.append(1) or None)
    monkeypatch.setattr(ad, "move_xformers_attention_bias", lambda bias, device: bias)
    monkeypatch.setattr(ad, "xformers_attention", lambda Q, K, V, attn_bias = None, **kw: Q)
    Q, K, V = _qkv(8)
    _run(ad.XFORMERS)(Q, K, V)
    assert seen
