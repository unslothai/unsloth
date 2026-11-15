# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Mid-generate recompiles, counted on real Triton binaries: once the first decode / prefill shapes are
built, other batch sizes (GEMV rows up to the default cap of 4) and prompt lengths, multiples of 16
included, build nothing new; outputs still match a reference."""

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

if not has_real_cuda():
    pytest.skip(reason = "needs a CUDA device", allow_module_level = True)
pytest.importorskip("triton")

import unsloth  # noqa: F401


def _binaries(*kernels):
    total = 0
    for k in kernels:
        caches = getattr(k, "device_caches", None)
        if caches is None:
            pytest.skip(reason = "this Triton keeps no device_caches to count binaries in")
        total += sum(len(c[0]) for c in caches.values())
    return total


@pytest.mark.gpu
def test_gemv_rows_no_new_binary_per_batch_size():
    F = pytest.importorskip("bitsandbytes.functional")
    from unsloth.kernels import nf4_gemv as G

    g = torch.Generator(device = "cuda").manual_seed(0)
    W = torch.randn(1024, 4096, dtype = torch.float16, device = "cuda", generator = g)
    q, s = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
    W_dq = F.dequantize_4bit(q, s).float()

    def run(m):
        X = torch.randn(m, 1, 4096, dtype = torch.float16, device = "cuda", generator = g)
        out = torch.empty(m, 1, 1024, dtype = torch.float16, device = "cuda")
        G.gemv_nf4_rows(X, q, s.absmax, s.state2.code, s.state2.absmax, s.offset, s.code, s.blocksize, s.state2.blocksize, s.shape, out)  # fmt: skip
        ref = X.float().view(m, -1) @ W_dq.t()
        err = (out.float().view(m, -1) - ref).abs().max() / ref.abs().max()
        assert err < 1e-2, (m, err.item())

    run(1)
    run(2)
    built = _binaries(*G._KERNELS)
    for m in (3, 4):
        run(m)
    assert _binaries(*G._KERNELS) == built


def _rope_ref(x, cos, sin):
    half = x.shape[-1] // 2
    seq = x.shape[2]
    c, s = cos[:seq].float(), sin[:seq].float()
    xf = x.float()
    return xf * c + torch.cat((-xf[..., half:], xf[..., :half]), -1) * s


@pytest.mark.gpu
def test_rope_no_new_binary_per_seqlen():
    from unsloth.kernels import rope_embedding as RE

    H, KV, D = 8, 4, 64
    emb = torch.outer(torch.arange(512.0, device = "cuda"), torch.rand(D // 2, device = "cuda"))
    emb = torch.cat((emb, emb), -1)
    cos, sin = emb.cos().half(), emb.sin().half()

    def run(seq):
        qkv = torch.randn(3, seq, (H + 2 * KV) * D, device = "cuda", dtype = torch.float16)
        qkv.requires_grad_(True)
        Q = qkv[..., : H * D].view(3, seq, H, D).transpose(1, 2)
        K = qkv[..., H * D : (H + KV) * D].view(3, seq, KV, D).transpose(1, 2)
        q_ref, k_ref = _rope_ref(Q.detach(), cos, sin), _rope_ref(K.detach(), cos, sin)
        q, k = RE.fast_rope_embedding(Q, K, cos, sin)
        torch.testing.assert_close(q.float(), q_ref, rtol = 2e-3, atol = 2e-3)
        torch.testing.assert_close(k.float(), k_ref, rtol = 2e-3, atol = 2e-3)
        (q.float().sum() + k.float().sum()).backward()

    run(1)
    run(7)
    built = _binaries(RE._rope_embedding_QK)
    for seq in (16, 32, 48, 300):
        run(seq)
    assert _binaries(RE._rope_embedding_QK) == built


@pytest.mark.gpu
def test_qk_norm_rope_no_new_binary_per_seqlen():
    from unsloth.kernels import qk_norm_rope as QN

    H, KV, D = 8, 4, 128
    norms = [torch.nn.RMSNorm(D, eps = 1e-6, device = "cuda", dtype = torch.float16) for _ in "qk"]
    for n in norms:
        n.weight.requires_grad_(False)
        n.variance_epsilon = 1e-6
    emb = torch.outer(torch.arange(512.0, device = "cuda"), torch.rand(D // 2, device = "cuda"))
    emb = torch.cat((emb, emb), -1)
    cos, sin = emb.cos().half(), emb.sin().half()

    def run(seq):
        Q = torch.randn(2, seq, H, D, device = "cuda", dtype = torch.float16, requires_grad = True)
        K = torch.randn(2, seq, KV, D, device = "cuda", dtype = torch.float16, requires_grad = True)
        q, k = QN.fast_qk_norm_rope(norms[0], norms[1], Q, K, cos, sin)
        x = Q.detach().float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim = True) + 1e-6) * norms[0].weight.float()
        q_ref = _rope_ref(x.transpose(1, 2), cos, sin)
        torch.testing.assert_close(q.float(), q_ref, rtol = 1e-2, atol = 1e-2)
        (q.float().sum() + k.float().sum()).backward()

    run(1)
    run(7)
    kernels = QN._qk_norm_rope_forward, QN._qk_norm_rope_backward
    built = _binaries(*kernels)
    for seq in (16, 32, 48, 300):
        run(seq)
    assert _binaries(*kernels) == built
