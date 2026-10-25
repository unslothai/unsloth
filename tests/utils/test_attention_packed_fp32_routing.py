# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Packed fp32 keeps one xformers launch (#14158); the fp32 downcast follows Q's device."""

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

ad = pytest.importorskip("unsloth.utils.attention_dispatch")

H, KV, D = 4, 2, 16


class _CudaLooking(torch.Tensor):
    # Routing reads Q.is_cuda; the math still runs on CPU.
    @property
    def is_cuda(self):
        return True


def _seq_info(lengths, device = "cpu"):
    lens = torch.tensor(lengths, dtype = torch.int32, device = device)
    cu = torch.cat([torch.zeros(1, dtype = torch.int32, device = device), lens.cumsum(0).int()])
    return lens, cu, max(lengths)


def _attend(Q, K, V, seq_info, backend):
    T = Q.shape[-2]
    config = ad.AttentionConfig(backend = backend, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = T,
        kv_seq_len = T,
        n_heads = H,
        head_dim = Q.shape[-1],
        requires_grad = True,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)


@pytest.fixture
def native_packed_host(monkeypatch):
    """An sm_80+ host without flash-attn: xformers installed, torch varlen preferred for packing."""
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(ad, "HAS_XFORMERS", True)
    monkeypatch.setattr(ad, "_NATIVE_BEATS_XFORMERS", True)
    monkeypatch.setattr(ad, "_SDPA_PACKED_SEGMENTS", True)
    monkeypatch.setattr(ad, "_DEVICE_BACKENDS", None)
    monkeypatch.setattr(ad, "_ring_attention_active", lambda: False)
    calls = {"xformers": 0, "sdpa": 0}

    def fake_xformers(
        Q,
        K,
        V,
        attn_bias = None,
        **kwargs,
    ):
        calls["xformers"] += 1
        return torch.zeros_like(Q)

    def counted_sdpa(*args, **kwargs):
        calls["sdpa"] += 1
        return torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)

    monkeypatch.setattr(ad, "xformers_attention", fake_xformers)
    monkeypatch.setattr(ad, "build_xformers_block_causal_mask", lambda *a, **k: None)
    monkeypatch.setattr(ad, "move_xformers_attention_bias", lambda bias, device: bias)
    monkeypatch.setattr(ad, "scaled_dot_product_attention", counted_sdpa)
    monkeypatch.setattr(ad, "_torch_varlen_takes", lambda Q, *a, **k: Q.dtype != torch.float32)
    monkeypatch.setattr(ad, "_torch_varlen_beats_segments", lambda *a, **k: True)
    monkeypatch.setattr(ad, "_torch_varlen_packed", lambda *a, **k: calls.__setitem__("sdpa", -1))
    return calls


def _packed_qkv(dtype, lengths):
    T = sum(lengths)
    g = torch.Generator().manual_seed(0)
    return [
        torch.randn(1, h, T, D, generator = g).to(dtype).as_subclass(_CudaLooking)
        for h in (H, KV, KV)
    ]


def test_packed_fp32_takes_one_xformers_call(native_packed_host):
    lengths = list(
        range(2, 18)
    )  # 16 distinct lengths: 16 SDPA launches per call on the segment path
    assert ad.select_attention_backend(use_varlen = True) == ad.SDPA
    Q, K, V = _packed_qkv(torch.float32, lengths)
    _attend(Q, K, V, _seq_info(lengths), ad.select_attention_backend(use_varlen = True))
    assert native_packed_host == {"xformers": 1, "sdpa": 0}


def test_packed_bf16_keeps_torch_varlen(native_packed_host):
    lengths = list(range(2, 18))
    Q, K, V = _packed_qkv(torch.bfloat16, lengths)
    _attend(Q, K, V, _seq_info(lengths), ad.select_attention_backend(use_varlen = True))
    assert native_packed_host == {"xformers": 0, "sdpa": -1}


def test_packed_fp32_without_xformers_stays_on_sdpa(native_packed_host, monkeypatch):
    monkeypatch.setattr(ad, "HAS_XFORMERS", False)
    lengths = [3, 5]
    Q, K, V = _packed_qkv(torch.float32, lengths)
    _attend(Q, K, V, _seq_info(lengths), ad.SDPA)
    assert native_packed_host["xformers"] == 0 and native_packed_host["sdpa"] == 2


def test_packed_fp32_dropout_stays_on_sdpa(native_packed_host):
    lengths = [3, 5]
    Q, K, V = _packed_qkv(torch.float32, lengths)
    config = ad.AttentionConfig(
        backend = ad.SDPA, n_kv_heads = KV, n_groups = H // KV, sdpa_kwargs = {"dropout_p": 0.1}
    )
    assert not ad._packed_fp32_prefers_xformers(
        config, ad.AttentionContext(1, 8, 8, H, D, True, _seq_info(lengths), None, None), Q, K, V
    )


def test_packed_fp32_deterministic_stays_on_sdpa(native_packed_host):
    # xformers' flash backward is non-deterministic; the per-segment SDPA path is the deterministic one.
    lengths = [3, 5]
    Q, K, V = _packed_qkv(torch.float32, lengths)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _attend(Q, K, V, _seq_info(lengths), ad.SDPA)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert native_packed_host["xformers"] == 0 and native_packed_host["sdpa"] == 2


def test_packed_fp32_large_head_dim_stays_on_sdpa_on_sm100(native_packed_host, monkeypatch):
    # sm_100+ downcasts fp32 for xformers' flash op, which has no kernel above head_dim 256.
    monkeypatch.setattr(ad, "_xformers_rejects_fp32", lambda device: True)
    lengths = [3, 5]
    T = sum(lengths)
    Q, K, V = [torch.randn(1, h, T, 512).as_subclass(_CudaLooking) for h in (H, KV, KV)]
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(1, T, T, H, 512, True, _seq_info(lengths), None, None)
    assert not ad._packed_fp32_prefers_xformers(config, context, Q, K, V)
    monkeypatch.setattr(ad, "_xformers_rejects_fp32", lambda device: False)
    assert ad._packed_fp32_prefers_xformers(config, context, Q, K, V)


def test_xformers_fp32_downcast_follows_q_device(monkeypatch):
    # A B200 host (device 0 rejects fp32 in xformers) must not downcast CPU tensors.
    monkeypatch.setattr(ad, "_XFORMERS_FP32_UNSUPPORTED", True)
    monkeypatch.setattr(ad, "_XFORMERS_FP32_UNSUPPORTED_BY_INDEX", (True, False), raising = False)
    seen = {}

    def fake_xformers(
        Q,
        K,
        V,
        attn_bias = None,
        **kwargs,
    ):
        seen["dtype"] = Q.dtype
        return torch.zeros_like(Q)

    monkeypatch.setattr(ad, "xformers_attention", fake_xformers)
    monkeypatch.setattr(ad, "build_xformers_block_causal_mask", lambda *a, **k: None)
    monkeypatch.setattr(ad, "move_xformers_attention_bias", lambda bias, device: bias)
    Q = torch.randn(1, H, 8, D)
    K = torch.randn(1, KV, 8, D)
    config = ad.AttentionConfig(backend = ad.XFORMERS, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(1, 8, 8, H, D, True, None, None, None)
    ad.run_attention(config = config, context = context, Q = Q, K = K, V = K.clone())
    assert seen["dtype"] == torch.float32
    assert ad._xformers_rejects_fp32(torch.device("cuda", 0))
    assert not ad._xformers_rejects_fp32(torch.device("cuda", 1))
    assert not ad._xformers_rejects_fp32(torch.device("cpu"))


def _reference(Q, K, V, lengths):
    out, start = [], 0
    for n in lengths:
        sl = slice(start, start + n)
        k = K[:, :, sl].repeat_interleave(H // KV, dim = 1)
        v = V[:, :, sl].repeat_interleave(H // KV, dim = 1)
        out.append(
            torch.nn.functional.scaled_dot_product_attention(Q[:, :, sl], k, v, is_causal = True)
        )
        start += n
    return torch.cat(out, dim = 2).transpose(1, 2)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a real CUDA device")
def test_packed_fp32_cuda_parity(monkeypatch):
    if not ad.HAS_XFORMERS:
        pytest.skip("needs xformers")
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(ad, "_NATIVE_BEATS_XFORMERS", True)
    calls = {"xformers": 0, "sdpa": 0}
    real_xf, real_sdpa = ad.xformers_attention, ad.scaled_dot_product_attention

    def xf(*a, **k):
        calls["xformers"] += 1
        return real_xf(*a, **k)

    def sdpa(*a, **k):
        calls["sdpa"] += 1
        return real_sdpa(*a, **k)

    monkeypatch.setattr(ad, "xformers_attention", xf)
    monkeypatch.setattr(ad, "scaled_dot_product_attention", sdpa)
    lengths = list(range(2, 66))
    T = sum(lengths)
    g = torch.Generator(device = "cuda").manual_seed(0)
    leaves = [
        torch.randn(1, h, T, 64, device = "cuda", generator = g, requires_grad = True)
        for h in (H, KV, KV)
    ]
    out = _attend(*leaves, _seq_info(lengths, "cuda"), ad.select_attention_backend(True))
    assert calls == {"xformers": 1, "sdpa": 0}
    grad = torch.randn_like(out.float())
    got = torch.autograd.grad(out.float(), leaves, grad)
    ref_leaves = [t.detach().double().requires_grad_() for t in leaves]
    ref = _reference(*ref_leaves, lengths)
    want = torch.autograd.grad(ref, ref_leaves, grad.double())
    # sm_100+ runs xformers in bf16 (fp32 downcast); older cards run fp32 cutlass.
    tol = 3e-2 if ad._xformers_rejects_fp32(leaves[0].device) else 2e-3
    torch.testing.assert_close(out.double().reshape(ref.shape), ref, atol = tol, rtol = tol)
    for a, b in zip(got, want):
        torch.testing.assert_close(a.double(), b, atol = tol, rtol = tol)
