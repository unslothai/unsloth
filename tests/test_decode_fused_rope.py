# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Single-token decode RoPE (llama path) and q/k norm + RoPE (qwen3 path) in one launch vs the
unfused decode op sequence they replace, both against an fp64 reference, plus the fallbacks."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")

import unsloth  # noqa: F401
from unsloth.kernels import qk_norm_rope as QK
from unsloth.kernels import rope_embedding as RE

SHAPES = [(1, 16, 8, 128), (4, 16, 8, 128), (8, 32, 8, 64), (3, 14, 2, 64), (5, 8, 8, 256)]


class _Norm(torch.nn.Module):
    def __init__(
        self,
        dim,
        dtype,
        eps = 1e-6,
    ):
        super().__init__()
        w = 1 + 0.1 * torch.randn(dim, device = "cuda", dtype = dtype)
        self.weight = torch.nn.Parameter(w, requires_grad = False)
        self.variance_epsilon = eps


def _table(positions, dim, dtype):
    inv = 1.0 / (1e6 ** (torch.arange(0, dim, 2, device = "cuda").float() / dim))
    f = torch.outer(torch.arange(positions, device = "cuda").float(), inv)
    e = torch.cat((f, f), -1)
    return e.cos().to(dtype), e.sin().to(dtype)


def _unfused_norm(norm, X):
    # fast_rms_layernorm_inference
    XX = X.to(torch.float32)
    var = XX.square().mean(-1, keepdim = True)
    var += norm.variance_epsilon
    XX *= var.rsqrt_()
    X = XX.to(X.dtype)
    X *= norm.weight
    return X


def _unfused_rope(Qn, Kn, cos, sin, pos):
    # The decode block of LlamaAttention_fast_forward_inference, in place.
    cos = cos[pos].unsqueeze(1).to(Qn.dtype)
    sin = sin[pos].unsqueeze(1).to(Qn.dtype)
    h = Qn.shape[-1] // 2
    for X in (Qn, Kn):
        RH = torch.empty_like(X)
        RH[..., :h] = X[..., h:]
        RH[..., h:] = X[..., :h]
        RH[..., :h].neg_()
        X *= cos
        X.addcmul_(RH, sin)
    return Qn, Kn


def _oracle_rope(X, cos, sin, pos):
    c, s = cos[pos].unsqueeze(1).double(), sin[pos].unsqueeze(1).double()
    h = X.shape[-1] // 2
    return X * c + torch.cat((-X[..., h:], X[..., :h]), -1) * s


def _oracle_norm(norm, X):
    X = X.double()
    return (
        X
        * torch.rsqrt(X.square().mean(-1, keepdim = True) + norm.variance_epsilon)
        * norm.weight.double()
    )


def _err(a, b):
    return (a.double() - b).abs().mean().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_decode_rope_inplace_matches_unfused(dtype, shape):
    torch.manual_seed(0)
    bsz, nq, nkv, d = shape
    cos, sin = _table(4096, d, dtype)
    pos = torch.randint(0, 4000, (bsz, 1), device = "cuda")
    qbuf = 3 * torch.randn(bsz, 1, nq * d, device = "cuda", dtype = dtype)
    kbuf = 3 * torch.randn(bsz, 1, nkv * d, device = "cuda", dtype = dtype)
    view = lambda b, n: b.view(bsz, 1, n, d).transpose(1, 2)  # noqa: E731
    Qo, Ko = (
        _oracle_rope(view(qbuf, nq).double(), cos, sin, pos),
        _oracle_rope(view(kbuf, nkv).double(), cos, sin, pos),
    )
    Qu, Ku = _unfused_rope(view(qbuf, nq).clone(), view(kbuf, nkv).clone(), cos, sin, pos)
    qf, kf = qbuf.clone(), kbuf.clone()
    assert RE.decode_rope_inplace(view(qf, nq), view(kf, nkv), cos, sin, pos)
    Qf, Kf = view(qf, nq), view(kf, nkv)
    ulp = torch.finfo(dtype).eps * max(Qo.abs().max().item(), Ko.abs().max().item())
    assert (Qf - Qu).abs().max().item() <= 2 * ulp and (Kf - Ku).abs().max().item() <= 2 * ulp
    # mean error vs fp64 does not grow (fewer roundings, not more)
    assert _err(Qf, Qo) <= _err(Qu, Qo) * 1.05 and _err(Kf, Ko) <= _err(Ku, Ko) * 1.05


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_decode_qk_norm_rope_matches_unfused(dtype, shape):
    torch.manual_seed(0)
    bsz, nq, nkv, d = shape
    cos, sin = _table(4096, d, dtype)
    pos = torch.randint(0, 4000, (bsz, 1), device = "cuda")
    Q = 3 * torch.randn(bsz, 1, nq, d, device = "cuda", dtype = dtype)
    K = 3 * torch.randn(bsz, 1, nkv, d, device = "cuda", dtype = dtype)
    qn, kn = _Norm(d, dtype), _Norm(d, dtype)
    Qo = _oracle_rope(_oracle_norm(qn, Q).transpose(1, 2), cos, sin, pos)
    Ko = _oracle_rope(_oracle_norm(kn, K).transpose(1, 2), cos, sin, pos)
    Qu, Ku = _unfused_rope(
        _unfused_norm(qn, Q.clone()).transpose(1, 2),
        _unfused_norm(kn, K.clone()).transpose(1, 2),
        cos,
        sin,
        pos,
    )
    out = QK.decode_qk_norm_rope(qn, kn, Q, K, cos, sin, pos)
    assert out is not None
    Qf, Kf = out
    assert Qf.shape == Qu.shape and Kf.shape == Ku.shape
    ulp = torch.finfo(dtype).eps * max(Qo.abs().max().item(), Ko.abs().max().item())
    assert (Qf - Qu).abs().max().item() <= 2 * ulp and (Kf - Ku).abs().max().item() <= 2 * ulp
    assert _err(Qf, Qo) <= _err(Qu, Qo) * 1.05 and _err(Kf, Ko) <= _err(Ku, Ko) * 1.05


def test_decode_fallbacks():
    cos, sin = _table(64, 128, torch.bfloat16)
    q = torch.randn(2, 4, 1, 128, device = "cuda", dtype = torch.bfloat16)
    k = torch.randn(2, 2, 1, 128, device = "cuda", dtype = torch.bfloat16)
    p = torch.zeros(2, 1, dtype = torch.long, device = "cuda")
    before = q.clone()
    # a float32 table: the unfused path multiplies in Q's dtype, so it keeps that path
    assert not RE.decode_rope_inplace(q, k, cos.float(), sin.float(), p)
    assert torch.equal(q, before)
    assert not RE.decode_rope_inplace(
        torch.randn(2, 4, 3, 128, device = "cuda", dtype = torch.bfloat16), k, cos, sin, p
    )
    assert not RE.decode_rope_inplace(q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), p.cpu())
    n32 = _Norm(128, torch.float32)
    assert (
        QK.decode_qk_norm_rope(n32, n32, q.transpose(1, 2), k.transpose(1, 2), cos, sin, p) is None
    )


def test_decode_kill_switch(monkeypatch):
    cos, sin = _table(64, 128, torch.bfloat16)
    q = torch.randn(2, 4, 1, 128, device = "cuda", dtype = torch.bfloat16)
    k = torch.randn(2, 2, 1, 128, device = "cuda", dtype = torch.bfloat16)
    p = torch.zeros(2, 1, dtype = torch.long, device = "cuda")
    n = _Norm(128, torch.bfloat16)
    monkeypatch.setattr(RE, "DECODE_FUSED_ROPE", False)
    monkeypatch.setattr(QK, "DECODE_FUSED_ROPE", False)
    assert not RE.decode_rope_inplace(q, k, cos, sin, p)
    assert QK.decode_qk_norm_rope(n, n, q.transpose(1, 2), k.transpose(1, 2), cos, sin, p) is None
