# SPDX-License-Identifier: AGPL-3.0-only
"""Masked training SDPA on cuDNN attention (GRPO / DPO padding) pads Q / K / V and the mask to a
length bucket: outputs and gradients match the unpadded call against an fp32 oracle."""

import types

import pytest
import torch
import torch.nn.functional as F

from unsloth.utils import attention_dispatch as ad


def _cudnn_sdpa_available():
    if not torch.cuda.is_available() or ad._CUDNN_CHOICE is None:
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(2, 2, 64, 64, device = "cuda", dtype = torch.bfloat16)
        m = torch.ones(2, 1, 64, 64, device = "cuda", dtype = torch.bool)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            F.scaled_dot_product_attention(q, q, q, attn_mask = m)
        return torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        return False


needs_cudnn = pytest.mark.skipif(not _cudnn_sdpa_available(), reason = "no masked cuDNN SDPA here")


def _qkv(
    bsz,
    n_heads,
    n_kv,
    q_len,
    head_dim,
    gqa,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    mk = lambda h: torch.randn(
        bsz, h, q_len, head_dim, device = "cuda", dtype = torch.bfloat16, generator = g
    )
    q, k, v = mk(n_heads), mk(n_kv), mk(n_kv)
    if not gqa:
        k = k.repeat_interleave(n_heads // n_kv, 1)
        v = v.repeat_interleave(n_heads // n_kv, 1)
    return [t.requires_grad_() for t in (q, k, v)]


def _mask(
    kind,
    bsz,
    q_len,
    seed = 0,
):
    """The masks run_attention hands SDPA: (bsz, 1, q, k) bool keep masks with fully masked rows
    opened up, a (bsz, 1, 1, k) key-padding mask, or an HF-style additive bf16 mask."""
    g = torch.Generator().manual_seed(seed)
    real = [max(1, int(q_len * f)) for f in torch.rand(bsz, generator = g).clamp(0.3, 1.0)]
    real[0] = q_len  # one unpadded row
    pos = torch.arange(q_len)
    keep = torch.stack(
        [(pos >= q_len - n) if kind.startswith("left") else (pos < n) for n in real]
    ).cuda()
    if kind == "keys_only":
        return keep[:, None, None, :]
    causal = torch.ones(q_len, q_len, dtype = torch.bool, device = "cuda").tril()
    mask = causal[None, None] & keep[:, None, None, :]
    mask = mask | ~mask.any(dim = -1, keepdim = True)
    if kind.endswith("additive"):
        return torch.zeros(mask.shape, dtype = torch.bfloat16, device = "cuda").masked_fill(
            ~mask, torch.finfo(torch.bfloat16).min
        )
    return mask


def _run(fn, q, k, v, grad, **kw):
    for t in (q, k, v):
        t.grad = None
    out = fn(q, k, v, **kw)
    out.backward(grad)
    return out.detach().float(), [t.grad.float() for t in (q, k, v)]


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def test_gate_masked(monkeypatch):
    if ad._CUDNN_CHOICE is None or not torch.cuda.is_available():
        pytest.skip("needs CUDA and torch's cuDNN SDPA backend")
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: ad._CUDNN_CHOICE)
    q = torch.zeros(2, 4, 8, 64, device = "cuda", dtype = torch.bfloat16)
    m4 = torch.ones(2, 1, 8, 8, dtype = torch.bool, device = "cuda")
    applies = ad._cudnn_length_bucket_applies
    assert applies(q, q, q, attn_mask = m4)
    assert applies(q, q, q, attn_mask = m4[:, :, :1])  # key-padding broadcast over queries
    assert applies(q, q, q, attn_mask = torch.zeros(2, 4, 8, 8, dtype = q.dtype, device = "cuda"))
    # Batch-shared (cached) masks and single-row batches keep the unpadded call.
    assert not applies(q, q, q, attn_mask = m4[:1])
    assert not applies(q[:1], q[:1], q[:1], attn_mask = m4[:1])
    assert not applies(q, q, q, attn_mask = m4, is_causal = True)  # SDPA rejects both
    assert not applies(q, q, q, attn_mask = m4, dropout_p = 0.1)
    assert not applies(q, q, q, attn_mask = m4[0, 0])  # 2D: not a shape run_attention builds
    assert not applies(q, q, q, attn_mask = m4.float())  # neither bool nor Q's dtype
    assert not applies(q, q, q, attn_mask = m4[..., :4])
    grad_mask = torch.zeros(2, 1, 8, 8, dtype = q.dtype, device = "cuda", requires_grad = True)
    assert not applies(q, q, q, attn_mask = grad_mask)
    assert not applies(q, q, q)  # not causal, no mask
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: 1)  # flash
    assert not applies(q, q, q, attn_mask = m4)
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: ad._CUDNN_CHOICE)
    monkeypatch.setattr(ad, "_CUDNN_LENGTH_BUCKET", 0)
    assert not applies(q, q, q, attn_mask = m4)


@pytest.mark.parametrize("backend", ["ring", "ulysses"])
def test_masked_not_applied_under_context_parallelism(monkeypatch, backend):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: ad._CUDNN_CHOICE)
    monkeypatch.setattr(ad, "get_cp_manager", lambda: types.SimpleNamespace(backend = backend))
    q = torch.randn(2, 2, 200, 64, device = "cuda", dtype = torch.bfloat16)
    m = torch.ones(2, 1, 200, 200, dtype = torch.bool, device = "cuda")
    assert not ad._cudnn_length_bucket_applies(q, q, q, attn_mask = m)


@needs_cudnn
@pytest.mark.parametrize("q_len", [2, 3, 127, 128, 129, 300, 513, 1000])
@pytest.mark.parametrize("kind", ["left", "right", "keys_only", "left_additive"])
@pytest.mark.parametrize("gqa", [True, False])
def test_masked_bucketed_matches_unpadded(q_len, kind, gqa):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    kw = dict(enable_gqa = True) if gqa else {}
    bsz = 3
    q, k, v = _qkv(bsz, 8, 2, q_len, 128, gqa, seed = q_len)
    mask = _mask(kind, bsz, q_len, seed = q_len)
    grad = torch.randn(bsz, 8, q_len, 128, device = "cuda", dtype = torch.bfloat16)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        ref, ref_g = _run(F.scaled_dot_product_attention, q, k, v, grad, attn_mask = mask, **kw)
        got, got_g = _run(ad._cudnn_length_bucketed_sdpa, q, k, v, grad, attn_mask = mask, **kw)
    # fp32 oracle on the same bf16 inputs.
    q32, k32, v32 = (t.detach().float().requires_grad_() for t in (q, k, v))
    m32 = mask if mask.dtype == torch.bool else mask.float()
    with sdpa_kernel([SDPBackend.MATH]):
        orc, orc_g = _run(
            F.scaled_dot_product_attention, q32, k32, v32, grad.float(), attn_mask = m32, **kw
        )
    assert got.shape == ref.shape
    for a, b, o in [(got, ref, orc)] + list(zip(got_g, ref_g, orc_g)):
        assert torch.isfinite(a).all()
        # The bucketed call is as close to fp32 as the unpadded cuDNN call (bf16 rounding).
        assert _rel(a, o) <= 1.25 * _rel(b, o) + 2e-3, (_rel(a, o), _rel(b, o))


@needs_cudnn
def test_masked_training_sdpa_buckets_lengths(monkeypatch):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    seen = []
    real = ad.scaled_dot_product_attention

    def record(Q, K, V, **kwargs):
        seen.append((Q.shape[-2], kwargs["attn_mask"].shape[-1]))
        return real(Q, K, V, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", record)
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: ad._CUDNN_CHOICE)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        for q_len in (5, 130, 256, 700):
            q, k, v = _qkv(2, 4, 4, q_len, 64, False)
            out = ad._training_sdpa(q, k, v, attn_mask = _mask("left", 2, q_len))
            assert out.shape[-2] == q_len
    if ad._avoids_cudnn(q, True):
        assert seen == [(5, 5), (130, 130), (256, 256), (700, 700)]  # SM100 pin: unchanged
    else:
        assert seen == [(128, 128), (256, 256), (384, 384), (768, 768)]


@needs_cudnn
def test_masked_compiled_dynamic_lengths_one_graph():
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch._dynamo.reset()
    counters.clear()

    def f(q, k, v, m):
        return ad._cudnn_length_bucketed_sdpa(q, k, v, attn_mask = m, enable_gqa = True)

    compiled = torch.compile(f, dynamic = True, fullgraph = True)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        for q_len in (300, 517, 1000, 1024):
            q, k, v = _qkv(2, 8, 2, q_len, 128, True)
            m = _mask("left", 2, q_len)
            torch.testing.assert_close(compiled(q, k, v, m), f(q, k, v, m), atol = 2e-2, rtol = 2e-2)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])


@needs_cudnn
def test_masked_gate_compiles_without_graph_breaks(monkeypatch):
    """The real gate (constant probe on the mask shape) traces fullgraph across lengths."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    if ad._avoids_cudnn(torch.empty(1, device = "cuda"), True):
        pytest.skip("SM100 training stays off cuDNN")
    torch._dynamo.reset()
    counters.clear()

    def f(q, k, v, m):
        return ad._training_sdpa(q, k, v, attn_mask = m, enable_gqa = True)

    compiled = torch.compile(f, dynamic = True, fullgraph = True)
    for q_len in (300, 517, 1000):
        q, k, v = _qkv(2, 8, 2, q_len, 128, True)
        m = _mask("left", 2, q_len)
        torch.testing.assert_close(compiled(q, k, v, m), f(q, k, v, m), atol = 2e-2, rtol = 2e-2)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])


@needs_cudnn
@pytest.mark.parametrize("masked", [True, False])
def test_inference_prefill_bucketed_decode_untouched(monkeypatch, masked):
    """No-grad prefill (q_len == k_len) is bucketed like training; a decode step (q_len 1) is not."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    seen = []
    real = ad.scaled_dot_product_attention

    def record(Q, K, V, **kwargs):
        seen.append((Q.shape[-2], K.shape[-2]))
        return real(Q, K, V, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", record)
    monkeypatch.setattr(ad, "_sdpa_masked_choice", lambda *a, **k: ad._CUDNN_CHOICE)
    monkeypatch.setattr(ad, "_sdpa_backend_state", lambda *a, **k: (True, ad._CUDNN_CHOICE))
    L = 300
    q, k, v = (t.detach() for t in _qkv(2, 8, 8, L, 128, False))
    kw = dict(attn_mask = _mask("left", 2, L)) if masked else dict(is_causal = True)
    with torch.no_grad(), sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        got = ad._inference_sdpa(q, k, v, **kw)
        want = real(q, k, v, **kw)
    # cuDNN takes no single-query call: the decode step runs on torch's default kernels.
    step_mask = dict(attn_mask = torch.ones(2, 1, 1, L, dtype = torch.bool, device = "cuda"))
    with torch.no_grad():
        ad._inference_sdpa(q[:, :, -1:], k, v, **(step_mask if masked else {}))
    assert seen[0] == (384, 384) and seen[-1] == (1, L), seen
    assert got.shape == want.shape
    # fp32 oracle: the bucketed prefill is as accurate as the unpadded one.
    m32 = kw.get("attn_mask")
    with sdpa_kernel([SDPBackend.MATH]):
        orc = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            **({"attn_mask": m32} if masked else {"is_causal": True}),
        )
    assert _rel(got.float(), orc) <= 1.25 * _rel(want.float(), orc) + 2e-3


@pytest.mark.parametrize("masked", [True, False])
def test_decode_sdpa_avoids_cudnn_where_flagged(monkeypatch, masked):
    """Decode (one query against a growing cache) runs off cuDNN on flagged devices: cuDNN would
    build a plan for every new cache length, i.e. every generated token."""
    if not torch.cuda.is_available() or ad._CUDNN_CHOICE is None:
        pytest.skip("needs CUDA and torch's cuDNN SDPA backend")
    seen = []
    real = F.scaled_dot_product_attention

    def record(*a, **k):
        seen.append(torch.backends.cuda.cudnn_sdp_enabled())
        return real(*a, **k)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    monkeypatch.setattr(ad, "_decode_picks_cudnn", lambda *a: True)
    q = torch.randn(4, 8, 1, 64, device = "cuda", dtype = torch.bfloat16)
    kv = torch.randn(4, 8, 300, 64, device = "cuda", dtype = torch.bfloat16)
    kw = dict(attn_mask = torch.ones(4, 1, 1, 300, dtype = torch.bool, device = "cuda")) if masked else {}
    monkeypatch.setattr(ad, "_NO_CUDNN_DECODE", frozenset({q.device.index}))
    out = ad.decode_scaled_dot_product_attention(q, kv, kv, **kw)
    monkeypatch.setattr(ad, "_NO_CUDNN_DECODE", frozenset())
    ref = ad.decode_scaled_dot_product_attention(q, kv, kv, **kw)
    assert seen[0] is False and seen[1] == torch.backends.cuda.cudnn_sdp_enabled()
    torch.testing.assert_close(out, ref, atol = 2e-2, rtol = 2e-2)
    # Prefill-shaped calls are never pinned.
    monkeypatch.setattr(ad, "_NO_CUDNN_DECODE", frozenset({q.device.index}))
    ad.decode_scaled_dot_product_attention(kv, kv, kv, is_causal = True)
    assert seen[-1] == torch.backends.cuda.cudnn_sdp_enabled()


@pytest.mark.parametrize("masked", [True, False])
def test_decode_sdpa_pins_only_where_torch_picks_cudnn(monkeypatch, masked):
    """The pin is entered only when torch's own pick for the call is cuDNN: elsewhere (torch 2.10 on
    H100) sdpa_kernel is pure overhead per decode call."""
    if not torch.cuda.is_available() or ad._CUDNN_CHOICE is None:
        pytest.skip("needs CUDA and torch's cuDNN SDPA backend")
    q = torch.randn(4, 8, 1, 64, device = "cuda", dtype = torch.bfloat16)
    kv = torch.randn(4, 8, 300, 64, device = "cuda", dtype = torch.bfloat16)
    mask = torch.ones(4, 1, 1, 300, dtype = torch.bool, device = "cuda") if masked else None
    torch_pick = int(torch._fused_sdp_choice(q, kv, kv, attn_mask = mask))
    ad._DECODE_PICKS_CUDNN.clear()
    assert ad._decode_picks_cudnn(q, kv, mask, False) == (torch_pick == ad._CUDNN_CHOICE)
    entered = []
    real = ad._sdpa_kernel
    monkeypatch.setattr(ad, "_sdpa_kernel", lambda b: entered.append(1) or real(b))
    monkeypatch.setattr(ad, "_NO_CUDNN_DECODE", frozenset({q.device.index}))
    for picks in (False, True):
        monkeypatch.setattr(ad, "_decode_picks_cudnn", lambda *a, p = picks: p)
        ad.decode_scaled_dot_product_attention(q, kv, kv, attn_mask = mask)
    assert entered == [1]


def test_decode_no_cudnn_devices(monkeypatch):
    monkeypatch.setattr(ad, "_visible_capabilities", lambda: ((8, 0), (9, 0), (10, 0), (12, 0)))
    monkeypatch.delenv("UNSLOTH_SDPA_DECODE_CUDNN", raising = False)
    if torch.version.hip is None:
        assert ad._decode_no_cudnn_devices() == frozenset({1, 2})
    monkeypatch.setenv("UNSLOTH_SDPA_DECODE_CUDNN", "1")
    assert ad._decode_no_cudnn_devices() == frozenset()
