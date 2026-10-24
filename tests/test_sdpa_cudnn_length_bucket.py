# SPDX-License-Identifier: AGPL-3.0-only
"""Training SDPA on cuDNN attention pads the sequence to a length bucket (bounded cuDNN plans)."""

import types

import pytest
import torch

from unsloth.utils import attention_dispatch as ad


def _cudnn_sdpa_available():
    if not torch.cuda.is_available() or ad._CUDNN_CHOICE is None:
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(1, 2, 64, 64, device = "cuda", dtype = torch.bfloat16)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = True)
        return torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        return False


needs_cudnn = pytest.mark.skipif(not _cudnn_sdpa_available(), reason = "no cuDNN SDPA on this device")


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
    q = mk(n_heads)
    k, v = mk(n_kv), mk(n_kv)
    if not gqa:
        k = k.repeat_interleave(n_heads // n_kv, 1)
        v = v.repeat_interleave(n_heads // n_kv, 1)
    return [t.requires_grad_() for t in (q, k, v)]


def test_applies_only_to_causal_unmasked_square_cuda_calls(monkeypatch):
    monkeypatch.setattr(ad, "_sdpa_backend_state", lambda *a, **k: (True, ad._CUDNN_CHOICE))
    if ad._CUDNN_CHOICE is None:
        pytest.skip("torch has no cuDNN SDPA backend")
    cpu = torch.zeros(1, 2, 8, 16, dtype = torch.bfloat16)
    assert not ad._cudnn_length_bucket_applies(cpu, cpu, cpu, is_causal = True)
    if not torch.cuda.is_available():
        return
    q = cpu.cuda()
    k_short = torch.zeros(1, 2, 4, 16, device = "cuda", dtype = torch.bfloat16)
    assert ad._cudnn_length_bucket_applies(q, q, q, is_causal = True)
    assert not ad._cudnn_length_bucket_applies(q, q, q, is_causal = False)
    assert not ad._cudnn_length_bucket_applies(
        q, q, q, attn_mask = torch.ones(8, 8, dtype = torch.bool, device = "cuda"), is_causal = True
    )
    assert not ad._cudnn_length_bucket_applies(q, q, q, dropout_p = 0.1, is_causal = True)
    assert not ad._cudnn_length_bucket_applies(q, k_short, k_short, is_causal = True)
    assert not ad._cudnn_length_bucket_applies(q.float(), q.float(), q.float(), is_causal = True)
    monkeypatch.setattr(ad, "_CUDNN_LENGTH_BUCKET", 0)
    assert not ad._cudnn_length_bucket_applies(q, q, q, is_causal = True)


@pytest.mark.parametrize("backend", ["ring", "ulysses"])
def test_not_applied_under_context_parallelism(monkeypatch, backend):
    # Sequence shards: padding a shard would add unmasked zero keys inside the global sequence.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(ad, "_sdpa_backend_state", lambda *a, **k: (True, ad._CUDNN_CHOICE))
    monkeypatch.setattr(ad, "get_cp_manager", lambda: types.SimpleNamespace(backend = backend))
    q = torch.randn(1, 2, 200, 64, device = "cuda", dtype = torch.bfloat16)
    assert not ad._cudnn_length_bucket_applies(q, q, q, is_causal = True)


def test_not_applied_when_sdpa_would_not_pick_cudnn(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(ad, "_sdpa_backend_state", lambda *a, **k: (True, 1))  # flash
    q = torch.zeros(1, 2, 8, 64, device = "cuda", dtype = torch.bfloat16)
    assert not ad._cudnn_length_bucket_applies(q, q, q, is_causal = True)


@needs_cudnn
@pytest.mark.parametrize("q_len", [2, 127, 128, 129, 300, 1000])
@pytest.mark.parametrize("gqa", [True, False])
def test_bucketed_matches_unpadded_cudnn(q_len, gqa):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    kw = dict(enable_gqa = True) if gqa else {}
    q, k, v = _qkv(2, 8, 2, q_len, 128, gqa)
    grad = torch.randn(2, 8, q_len, 128, device = "cuda", dtype = torch.bfloat16)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = True, **kw)
        ref.backward(grad)
        ref_grads = [t.grad.clone() for t in (q, k, v)]
        for t in (q, k, v):
            t.grad = None
        out = ad._cudnn_length_bucketed_sdpa(q, k, v, is_causal = True, **kw)
        out.backward(grad)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # Same kernel family on a longer (padded) sequence: tiles differ, so compare within bf16 rounding.
    torch.testing.assert_close(out, ref, atol = 2e-2, rtol = 2e-2)
    for got, want in zip((q.grad, k.grad, v.grad), ref_grads):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, atol = 3e-2, rtol = 3e-2)


@needs_cudnn
def test_padded_lengths_are_bucket_multiples(monkeypatch):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    seen = []
    real = ad.scaled_dot_product_attention

    def record(Q, K, V, **kwargs):
        seen.append(Q.shape[-2])
        return real(Q, K, V, **kwargs)

    monkeypatch.setattr(ad, "scaled_dot_product_attention", record)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        for q_len in (5, 130, 255, 256, 700):
            q, k, v = _qkv(1, 4, 4, q_len, 64, False)
            out = ad._training_sdpa(q, k, v, is_causal = True)
            assert out.shape[-2] == q_len
    if ad._avoids_cudnn(q, True):
        assert seen == [5, 130, 255, 256, 700]  # the SM100 pin keeps these calls off cuDNN
    else:
        assert seen == [128, 256, 256, 384, 768]


@needs_cudnn
def test_compiled_dynamic_lengths_one_graph():
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch._dynamo.reset()
    counters.clear()

    def f(q, k, v):
        return ad._cudnn_length_bucketed_sdpa(q, k, v, is_causal = True, enable_gqa = True)

    compiled = torch.compile(f, dynamic = True, fullgraph = True)
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        for q_len in (300, 517, 1000, 1024):
            q, k, v = _qkv(2, 8, 2, q_len, 128, True)
            got = compiled(q, k, v)
            want = f(q, k, v)
            torch.testing.assert_close(got, want, atol = 2e-2, rtol = 2e-2)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
