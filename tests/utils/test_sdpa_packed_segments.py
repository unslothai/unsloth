# SPDX-License-Identifier: AGPL-3.0-only
"""Packed rows on SDPA run per segment: same result as the dense block-diagonal mask, without the (T, T) scores."""

import pytest
import torch
from real_accelerator import has_real_cuda

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import packing


def _seq_info(lengths):
    lengths = torch.tensor(lengths, dtype = torch.int32)
    cu = torch.cat((torch.zeros(1, dtype = torch.int32), lengths.cumsum(0).int()))
    return lengths, cu, int(lengths.max())


def _run(
    monkeypatch,
    segments,
    lengths,
    total,
    n_heads,
    n_kv,
    qkv,
    is_causal = True,
    window = None,
):
    monkeypatch.setattr(ad, "_SDPA_PACKED_SEGMENTS", segments)
    packing.clear_packed_caches()
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = n_kv, n_groups = n_heads // n_kv)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = total,
        kv_seq_len = total,
        n_heads = n_heads,
        head_dim = qkv[0].shape[-1],
        requires_grad = True,
        seq_info = _seq_info(lengths),
        attention_mask = None,
        causal_mask = None,
        is_causal = is_causal,
        sliding_window = window,
    )
    Q, K, V = (x.detach().clone().requires_grad_() for x in qkv)
    out = ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)
    out = out.reshape(1, total, n_heads, -1)
    (
        out.float() * torch.linspace(-1, 1, out.numel(), device = out.device).view_as(out)
    ).sum().backward()
    return out.detach(), Q.grad, K.grad, V.grad


CASES = [
    ((5, 3, 4), 12, 4, 4, True, None),
    ((5, 3, 4), 12, 4, 2, True, None),
    ((4, 4, 4), 12, 4, 1, True, None),
    ((7,), 10, 4, 2, True, None),
    ((6, 2, 5), 13, 4, 2, True, 3),
    ((6, 2, 5), 13, 4, 2, False, None),
    ((6, 2, 5), 13, 4, 4, False, 3),
    ((1, 1, 9), 11, 2, 1, True, 4),
]


@pytest.mark.parametrize(
    "device, dtype", [("cpu", torch.float32), ("cuda", torch.bfloat16), ("cuda", torch.float16)]
)
@pytest.mark.parametrize("lengths, total, n_heads, n_kv, is_causal, window", CASES)
def test_segments_match_dense_mask(
    monkeypatch, device, dtype, lengths, total, n_heads, n_kv, is_causal, window
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    g = torch.Generator().manual_seed(3407)
    shapes = ((1, n_heads, total, 64), (1, n_kv, total, 64), (1, n_kv, total, 64))
    qkv = [torch.randn(s, generator = g).to(device, dtype) for s in shapes]
    args = (lengths, total, n_heads, n_kv, qkv, is_causal, window)
    dense = _run(monkeypatch, False, *args)
    segs = _run(monkeypatch, True, *args)
    tol = dict(atol = 1e-5, rtol = 1e-5) if dtype == torch.float32 else dict(atol = 2e-2, rtol = 2e-2)
    for name, a, b in zip(("out", "dQ", "dK", "dV"), dense, segs):
        assert torch.isfinite(b).all(), name
        torch.testing.assert_close(b.float(), a.float(), **tol, msg = name)


def test_segments_never_build_the_dense_mask(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("dense packed mask built")

    monkeypatch.setattr(ad, "build_sdpa_packed_attention_mask", refuse)
    qkv = [torch.randn(1, 2, 9, 8) for _ in range(3)]
    out = _run(monkeypatch, True, (4, 5), 9, 2, 2, qkv)[0]
    assert out.shape == (1, 9, 2, 8)


# Not torch.cuda.is_available(): the zoo CUDA spoof patches it to True process-wide.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_peak_memory_is_linear_in_tokens(monkeypatch):
    total, n_heads, n_kv = 4096, 16, 8
    g = torch.Generator().manual_seed(0)
    shapes = ((1, n_heads, total, 128), (1, n_kv, total, 128), (1, n_kv, total, 128))
    qkv = [torch.randn(s, generator = g).cuda().bfloat16() for s in shapes]
    peaks = {}
    for segments in (False, True):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _run(monkeypatch, segments, (2048, 2048), total, n_heads, n_kv, qkv)
        torch.cuda.synchronize()
        peaks[segments] = (torch.cuda.max_memory_allocated() - base) / 2**20
    # The dense path holds (16, 4096, 4096) fp32 scores (1 GiB).
    assert peaks[True] < 256, peaks
    assert peaks[False] > 4 * peaks[True], peaks
