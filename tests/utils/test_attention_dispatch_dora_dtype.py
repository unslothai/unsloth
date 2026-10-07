"""Regression test for unslothai/unsloth#1013: run_attention must downcast DoRA's
fp32 q/k/v for FlashAttention while leaving already-16bit tensors untouched."""

import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


def _run(monkeypatch, qkv_dtype, backend):
    captured = {}

    def _check(Q):
        captured["dtype"] = Q.dtype
        if Q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("FlashAttention only support fp16 and bf16 data type")

    def fake_flash_dense(Q, K, V, **kwargs):
        _check(Q)
        return torch.zeros_like(Q)

    def fake_flash_varlen(Q, K, V, *args, **kwargs):
        _check(Q)
        return torch.zeros_like(Q)

    monkeypatch.setattr(ad, "flash_attn_func", fake_flash_dense, raising = False)
    monkeypatch.setattr(ad, "flash_attn_varlen_func", fake_flash_varlen, raising = False)

    bsz, n_heads, q_len, head_dim = 1, 2, 4, 8
    Q = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)
    K = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)
    V = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)

    seq_info = None
    if backend == ad.FLASH_VARLEN:
        cu = torch.tensor([0, q_len], dtype = torch.int32)
        seq_info = (None, cu, q_len)

    config = ad.AttentionConfig(
        backend = backend,
        n_kv_heads = n_heads,
        n_groups = 1,
        flash_dense_kwargs = {"causal": True},
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True},
    )
    context = ad.AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = False,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    return captured["dtype"]


def test_dense_flash_downcasts_fp32_qkv(monkeypatch):
    assert _run(monkeypatch, torch.float32, ad.FLASH_DENSE) in (torch.bfloat16, torch.float16)


def test_varlen_flash_downcasts_fp32_qkv(monkeypatch):
    assert _run(monkeypatch, torch.float32, ad.FLASH_VARLEN) in (torch.bfloat16, torch.float16)


def test_bf16_qkv_left_untouched(monkeypatch):
    assert _run(monkeypatch, torch.bfloat16, ad.FLASH_DENSE) == torch.bfloat16


def _run_xformers(monkeypatch, qkv_dtype, fp32_unsupported):
    # On sm_100+ only xformers' fp16/bf16 flash-2 op runs, so fp32 must be downcast.
    captured = {}

    def fake_xformers_attention(
        Q,
        K,
        V,
        attn_bias = None,
        **kwargs,
    ):
        captured["dtype"] = Q.dtype
        if fp32_unsupported and Q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("no operator found for memory_efficient_attention with fp32")
        return torch.zeros_like(Q)

    monkeypatch.setattr(ad, "_XFORMERS_FP32_UNSUPPORTED", fp32_unsupported, raising = False)
    monkeypatch.setattr(ad, "xformers_attention", fake_xformers_attention, raising = False)
    monkeypatch.setattr(
        ad, "build_xformers_block_causal_mask", lambda *a, **k: object(), raising = False
    )

    bsz, n_heads, q_len, head_dim = 1, 2, 4, 8
    Q = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)
    K = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)
    V = torch.randn(bsz, n_heads, q_len, head_dim, dtype = qkv_dtype)

    config = ad.AttentionConfig(backend = ad.XFORMERS, n_kv_heads = n_heads, n_groups = 1)
    context = ad.AttentionContext(
        bsz = bsz,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = False,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    return captured["dtype"]


def test_xformers_downcasts_fp32_qkv_on_sm100_plus(monkeypatch):
    assert _run_xformers(monkeypatch, torch.float32, True) in (torch.bfloat16, torch.float16)


def test_xformers_leaves_fp32_qkv_below_sm100(monkeypatch):
    assert _run_xformers(monkeypatch, torch.float32, False) == torch.float32
