# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""fast_qk_norm_rope (per-head RMSNorm + RoPE, one launch each way) vs the unfused
fast_rms_layernorm + fast_rope_embedding path and an fp64 reference."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")

import unsloth  # noqa: F401
import torch._dynamo.utils as dynamo_utils
from unsloth.kernels import rms_layernorm
from unsloth.kernels.qk_norm_rope import _unfused, fast_qk_norm_rope

TRACEABLE = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")
BF16 = pytest.param(
    torch.bfloat16,
    marks = pytest.mark.skipif(
        not rms_layernorm._BF16_TRACEABLE, reason = "no native bf16: bf16 stays untraced"
    ),
)
CDTYPE = torch.bfloat16 if rms_layernorm._BF16_TRACEABLE else torch.float16
EPS = 1e-6


class _Norm(torch.nn.Module):
    def __init__(
        self,
        dim,
        dtype,
        seed,
        trainable = False,
    ):
        super().__init__()
        g = torch.Generator(device = "cuda").manual_seed(seed)
        w = torch.randn(dim, device = "cuda", generator = g) * 0.3 + 1
        self.weight = torch.nn.Parameter(w.to(dtype), requires_grad = trainable)
        self.variance_epsilon = EPS


def _case(
    bsz,
    seq,
    n_heads,
    n_kv,
    head_dim,
    dtype,
    packed,
    trainable = False,
):
    g = torch.Generator(device = "cuda").manual_seed(0)
    q_lin = (torch.randn(bsz, seq, n_heads * head_dim, device = "cuda", generator = g) * 2 + 0.3).to(
        dtype
    )
    k_lin = (torch.randn(bsz, seq, n_kv * head_dim, device = "cuda", generator = g) * 0.5).to(dtype)
    inv = 1.0 / (1e6 ** (torch.arange(0, head_dim, 2, device = "cuda").float() / head_dim))
    emb = torch.outer(torch.arange(seq + 16, device = "cuda").float(), inv)
    emb = torch.cat((emb, emb), -1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    if packed:
        lens = [seq // 3, seq // 3, seq - 2 * (seq // 3)]
        pos = torch.cat([torch.arange(n) for n in lens]).repeat(bsz, 1).cuda()
    else:
        pos = None
    grads = (
        torch.randn(bsz, n_heads, seq, head_dim, device = "cuda", generator = g).to(dtype),
        torch.randn(bsz, n_kv, seq, head_dim, device = "cuda", generator = g).to(dtype),
    )
    norms = _Norm(head_dim, dtype, 1, trainable), _Norm(head_dim, dtype, 2, trainable)
    return q_lin, k_lin, cos, sin, pos, grads, norms


def _run(fn, q_lin, k_lin, cos, sin, pos, grads, norms, head_dim):
    bsz, seq = q_lin.shape[:2]
    a = q_lin.detach().clone().requires_grad_(True)
    b = k_lin.detach().clone().requires_grad_(True)
    Q = a.view(bsz, seq, -1, head_dim)
    K = b.view(bsz, seq, -1, head_dim)
    q, k = fn(norms[0], norms[1], Q, K, cos, sin, pos)
    torch.autograd.backward((q, k), grads)
    return q.detach(), k.detach(), a.grad, b.grad


def _ref64(q_lin, k_lin, cos, sin, pos, grads, norms, head_dim):
    bsz, seq = q_lin.shape[:2]
    a = q_lin.double().requires_grad_(True)
    b = k_lin.double().requires_grad_(True)
    if pos is None:
        pos = torch.arange(seq, device = "cuda").repeat(bsz, 1)
    c, s = cos[pos].double()[:, None], sin[pos].double()[:, None]

    def f(x, w):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + EPS) * w.double()
        y = y.transpose(1, 2)
        half = head_dim // 2
        return y * c + torch.cat((-y[..., half:], y[..., :half]), -1) * s

    q = f(a.view(bsz, seq, -1, head_dim), norms[0].weight)
    k = f(b.view(bsz, seq, -1, head_dim), norms[1].weight)
    torch.autograd.backward((q, k), [x.double() for x in grads])
    return q.detach(), k.detach(), a.grad, b.grad


def _err(x, ref):
    return (x.double() - ref).abs().max().item()


@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 130, 32, 8, 128),
        (1, 77, 12, 4, 64),
        (2, 64, 8, 1, 256),
        (1, 33, 16, 16, 128),
        (1, 50, 20, 5, 128),
    ],
)
def test_matches_unfused_within_rounding(shape, packed, dtype):
    bsz, seq, n_heads, n_kv, head_dim = shape
    case = _case(bsz, seq, n_heads, n_kv, head_dim, dtype, packed)
    fused = _run(fast_qk_norm_rope, *case, head_dim)
    main = _run(_unfused, *case, head_dim)
    ref = _ref64(*case, head_dim)
    ulp = torch.finfo(dtype).eps
    for f, m, r in zip(fused, main, ref):
        scale = r.abs().max().item()
        assert _err(f, r) <= 1.5 * _err(m, r) + 2 * ulp * scale


def test_fused_inputs_and_grads_untouched():
    case = _case(2, 64, 16, 4, 128, CDTYPE, True)
    q_lin, k_lin, cos, sin, pos, grads, norms = case
    a = q_lin.clone().requires_grad_(True)
    b = k_lin.clone().requires_grad_(True)
    g0, g1 = grads[0].clone(), grads[1].clone()
    q, k = fast_qk_norm_rope(
        norms[0], norms[1], a.view(2, 64, 16, 128), b.view(2, 64, 4, 128), cos, sin, pos
    )
    torch.autograd.backward((q, k), grads)
    assert torch.equal(a.detach(), q_lin) and torch.equal(b.detach(), k_lin)
    assert torch.equal(grads[0], g0) and torch.equal(grads[1], g1)
    assert q.data_ptr() != a.data_ptr() and k.data_ptr() != b.data_ptr()


def test_trainable_norm_weights_fall_back_to_unfused():
    case = _case(1, 40, 8, 2, 64, torch.float32, False, trainable = True)
    q_lin, k_lin, cos, sin, pos, grads, norms = case
    q, k = fast_qk_norm_rope(
        norms[0], norms[1], q_lin.view(1, 40, 8, 64), k_lin.view(1, 40, 2, 64), cos, sin, pos
    )
    torch.autograd.backward((q, k), grads)
    assert norms[0].weight.grad is not None and norms[1].weight.grad is not None


def test_broadcast_position_ids():
    bsz, seq = 3, 50
    case = _case(bsz, seq, 8, 2, 128, CDTYPE, False)
    q_lin, k_lin, cos, sin, _, grads, norms = case
    one_row = torch.arange(seq, device = "cuda")[None]
    full = one_row.expand(bsz, seq)
    a = _run(fast_qk_norm_rope, q_lin, k_lin, cos, sin, one_row, grads, norms, 128)
    b = _run(fast_qk_norm_rope, q_lin, k_lin, cos, sin, full, grads, norms, 128)
    c = _run(fast_qk_norm_rope, q_lin, k_lin, cos, sin, None, grads, norms, 128)
    for x, y, z in zip(a, b, c):
        assert torch.equal(x, y) and torch.equal(x, z)


@pytest.mark.skipif(not TRACEABLE, reason = "needs torch.library.triton_op")
@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_compiled_no_graph_breaks_same_bytes(dtype, packed):
    torch._dynamo.reset()
    dynamo_utils.counters.clear()
    case = _case(2, 96, 16, 4, 128, dtype, packed)
    eager = _run(fast_qk_norm_rope, *case, 128)
    compiled = _run(torch.compile(fast_qk_norm_rope, fullgraph = True), *case, 128)
    assert sum(dynamo_utils.counters["graph_break"].values()) == 0
    for x, y in zip(eager, compiled):
        assert torch.equal(x, y)
    torch._dynamo.reset()
