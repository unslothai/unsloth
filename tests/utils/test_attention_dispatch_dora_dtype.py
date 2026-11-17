"""Regression test for unslothai/unsloth#1013: run_attention must downcast DoRA's
fp32 q/k/v for FlashAttention (under autocast) while leaving already-16bit tensors
untouched, and must keep an fp32 model's attention (no autocast) in fp32 via SDPA."""

import torch
import unsloth  # noqa: F401

from unsloth.utils import attention_dispatch as ad


def _autocast_bf16(monkeypatch):
    # DoRA trains under autocast; CPU CI cannot enter CUDA autocast, so fake its state.
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda *a: torch.bfloat16, raising = False)


def _spy_sdpa(monkeypatch, captured):
    real = ad.scaled_dot_product_attention

    def spy(Q, K, V, *args, **kwargs):
        captured["dtype"] = Q.dtype
        captured["sdpa"] = True
        return real(Q, K, V, *args, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", spy)


def _run(
    monkeypatch,
    qkv_dtype,
    backend,
    autocast = True,
):
    captured = {}
    if autocast:
        _autocast_bf16(monkeypatch)
    _spy_sdpa(monkeypatch, captured)

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
        seq_info = (torch.tensor([q_len], dtype = torch.int32), cu, q_len)

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


def _run_xformers(
    monkeypatch,
    qkv_dtype,
    fp32_unsupported,
    autocast = True,
):
    # On sm_100+ only xformers' fp16/bf16 flash-2 op runs, so fp32 must be downcast.
    captured = {}
    if autocast:
        _autocast_bf16(monkeypatch)
    _spy_sdpa(monkeypatch, captured)

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

    monkeypatch.setattr(ad, "_xformers_rejects_fp32", lambda device: fp32_unsupported)
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


# dtype = torch.float32 model, no autocast: attention must stay fp32 (SDPA), not silently bf16.
def test_fp32_model_dense_flash_keeps_fp32_via_sdpa(monkeypatch):
    assert _run(monkeypatch, torch.float32, ad.FLASH_DENSE, autocast = False) == torch.float32


def test_fp32_model_varlen_flash_keeps_fp32_via_sdpa(monkeypatch):
    assert _run(monkeypatch, torch.float32, ad.FLASH_VARLEN, autocast = False) == torch.float32


def test_fp32_model_xformers_sm100_keeps_fp32_via_sdpa(monkeypatch):
    assert _run_xformers(monkeypatch, torch.float32, True, autocast = False) == torch.float32


def test_fp32_model_sdpa_matches_fp32_reference(monkeypatch):
    # Value check, not just dtype: output equals an fp32 causal SDPA reference.
    monkeypatch.setattr(ad, "_xformers_rejects_fp32", lambda device: True)
    torch.manual_seed(0)
    Q, K, V = (torch.randn(1, 2, 16, 8) for _ in range(3))
    config = ad.AttentionConfig(backend = ad.XFORMERS, n_kv_heads = 2, n_groups = 1)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = 16,
        kv_seq_len = 16,
        n_heads = 2,
        head_dim = 8,
        requires_grad = False,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
    )
    out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal = True).transpose(1, 2)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.reshape(ref.shape), ref, atol = 1e-6, rtol = 1e-6)
