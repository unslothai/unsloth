"""Regression test for unslothai/unsloth#1013: run_attention must downcast DoRA's
fp32 q/k/v for FlashAttention while leaving already-16bit tensors untouched."""

import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


def _run(monkeypatch, qkv_dtype, backend):
    captured = {}

    def _check(Q):
        captured["dtype"] = Q.dtype
        # Mirror the real kernel constraint so an unfixed dispatch fails loudly.
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
    # fp32 DoRA output must be downcast to a flash-compatible dtype.
    assert _run(monkeypatch, torch.float32, ad.FLASH_DENSE) in (torch.bfloat16, torch.float16)


def test_varlen_flash_downcasts_fp32_qkv(monkeypatch):
    assert _run(monkeypatch, torch.float32, ad.FLASH_VARLEN) in (torch.bfloat16, torch.float16)


def test_bf16_qkv_left_untouched(monkeypatch):
    # Standard LoRA path (already bf16) must not be altered.
    assert _run(monkeypatch, torch.bfloat16, ad.FLASH_DENSE) == torch.bfloat16
