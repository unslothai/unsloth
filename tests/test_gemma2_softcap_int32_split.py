# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma2 softcap attention keeps each compiled call under 2**31 score elements, else int32 wraps."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.kernels.flex_attention as fk
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")


def _spy(monkeypatch):
    calls = []

    def fake(Q, K, V, causal_mask, self, bsz, q_len):
        calls.append((bsz, Q.shape[0], tuple(causal_mask.shape)))
        return (
            Q[:, 0]
            + causal_mask[..., :q_len, :q_len]
            .sum(-1)
            .reshape(bsz if causal_mask.dim() == 4 else 1, -1)[..., None]
        )

    monkeypatch.setattr(fk, "_compiled_slow_attention_softcapping", fake)
    return calls


def _inputs(bsz, q_len, mask_dims):
    Q = torch.arange(bsz, dtype = torch.float32)[:, None, None, None].expand(bsz, 1, q_len, 2).clone()
    mask = torch.arange(bsz * q_len * q_len, dtype = torch.float32).reshape(bsz, 1, q_len, q_len)
    return Q, (mask if mask_dims == 4 else mask[0, 0])


@pytest.mark.parametrize("mask_dims", [4, 2])
def test_batch_split_when_scores_pass_int32(monkeypatch, mask_dims):
    calls = _spy(monkeypatch)
    q_len = 8
    # 2**25 heads x 64 = 2**31 score elements per row, so every row is its own call.
    config = types.SimpleNamespace(num_attention_heads = 2**25)
    Q, mask = _inputs(3, q_len, mask_dims)
    out = fk.slow_attention_softcapping(
        Q, Q, Q, mask, types.SimpleNamespace(config = config), 3, q_len
    )
    assert [c[:2] for c in calls] == [(1, 1)] * 3
    assert all(c[2] == ((1, 1, q_len, q_len) if mask_dims == 4 else (q_len, q_len)) for c in calls)
    reference = fk._compiled_slow_attention_softcapping(
        Q, Q, Q, mask, types.SimpleNamespace(config = config), 3, q_len
    )
    torch.testing.assert_close(out, reference)


def test_small_batches_take_one_call(monkeypatch):
    calls = _spy(monkeypatch)
    config = types.SimpleNamespace(num_attention_heads = 16)
    Q, mask = _inputs(4, 8, 4)
    fk.slow_attention_softcapping(Q, Q, Q, mask, types.SimpleNamespace(config = config), 4, 8)
    assert [c[:2] for c in calls] == [(4, 4)]


@pytest.mark.skipif(
    not has_real_cuda(), reason = "runs the compiled kernel past 2**31 score elements on CUDA"
)
def test_compiled_kernel_past_int32_matches_eager():
    if torch.cuda.mem_get_info()[0] < 40 * 2**30:
        pytest.skip("needs about 40 GB free for the 2.4e9 element score tensor")
    bsz, heads, kv_heads, q_len, head_dim = 8, 16, 8, 4305, 16
    config = types.SimpleNamespace(
        num_attention_heads = heads,
        num_key_value_heads = kv_heads,
        query_pre_attn_scalar = 256,
        attn_logit_softcapping = 50.0,
    )
    layer = types.SimpleNamespace(
        config = config, head_dim = head_dim, num_key_value_groups = heads // kv_heads
    )
    g = torch.Generator(device = "cuda").manual_seed(0)
    Q = torch.randn(bsz, heads, q_len, head_dim, device = "cuda", dtype = torch.bfloat16, generator = g)
    K = torch.randn(
        bsz, kv_heads, q_len, head_dim, device = "cuda", dtype = torch.bfloat16, generator = g
    )
    V = torch.randn(
        bsz, kv_heads, q_len, head_dim, device = "cuda", dtype = torch.bfloat16, generator = g
    )
    keep = torch.ones(bsz, q_len, dtype = torch.bool, device = "cuda")
    keep[-1, :3950] = False  # the last row is mostly left padding, as in the failing batch
    i = torch.arange(q_len, device = "cuda")
    allowed = (i[None, :] <= i[:, None])[None, None] & keep[:, None, None, :]
    mask = torch.zeros(bsz, 1, q_len, q_len, device = "cuda", dtype = torch.bfloat16)
    mask.masked_fill_(~allowed, torch.finfo(torch.bfloat16).min)
    out = fk.slow_attention_softcapping(Q, K, V, mask, layer, bsz, q_len)
    assert torch.isfinite(out).all()
    eager = fk._compiled_slow_attention_softcapping._torchdynamo_orig_callable
    ref = eager(Q[-1:], K[-1:], V[-1:], mask[-1:], layer, 1, q_len)
    torch.testing.assert_close(out[-1:], ref, atol = 2e-2, rtol = 2e-2)


def _raising_compiled(monkeypatch, error):
    calls = {"compiled": 0, "eager": 0}

    def compiled(*a):
        calls["compiled"] += 1
        raise error

    def eager(Q, K, V, causal_mask, self, bsz, q_len):
        calls["eager"] += 1
        return Q[:, 0]

    compiled._torchdynamo_orig_callable = eager
    monkeypatch.setattr(fk, "_compiled_slow_attention_softcapping", compiled)
    monkeypatch.setattr(fk, "_SOFTCAP_EAGER", {})
    return calls


def test_compile_failure_falls_back_to_eager_once(monkeypatch):
    calls = _raising_compiled(monkeypatch, RuntimeError("inductor::_alloc_from_pool ..."))
    config = types.SimpleNamespace(num_attention_heads = 16)
    Q, mask = _inputs(2, 8, 4)
    layer = types.SimpleNamespace(config = config)
    for _ in range(3):
        out = fk.slow_attention_softcapping(Q, Q, Q, mask, layer, 2, 8)
    assert out.shape == (2, 8, 2)
    assert calls == {"compiled": 1, "eager": 3}


def test_out_of_memory_is_not_swallowed(monkeypatch):
    calls = _raising_compiled(monkeypatch, torch.OutOfMemoryError("out of memory"))
    Q, mask = _inputs(2, 8, 4)
    with pytest.raises(torch.OutOfMemoryError):
        fk.slow_attention_softcapping(
            Q,
            Q,
            Q,
            mask,
            types.SimpleNamespace(config = types.SimpleNamespace(num_attention_heads = 16)),
            2,
            8,
        )
    assert calls["eager"] == 0
