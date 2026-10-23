# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""fast_rope_embedding against an fp64 oracle over head dims, head counts, GQA, input layouts,
packed position ids, dtypes and gradient strides, plus the in-place contract callers see."""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "Triton RoPE kernel needs CUDA"),
]

BF16 = torch.bfloat16
# (batch, seq, heads, kv heads, head_dim): pow2 and not (80, 96), 1 to 40 heads, GQA, decode.
SHAPES = [
    (2, 37, 8, 8, 64),
    (1, 64, 32, 8, 128),
    (3, 20, 4, 1, 64),
    (2, 33, 40, 8, 128),
    (1, 17, 6, 2, 80),
    (2, 5, 12, 12, 96),
    (2, 9, 16, 4, 256),
    (1, 1, 32, 8, 128),
    (4, 3, 1, 1, 32),
]


def _cos_sin(rows, head_dim, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device = "cuda").float() / head_dim))
    freqs = torch.outer(torch.arange(rows, device = "cuda").float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _packed_ids(batch, seq):
    # Packed sequences restart at 0; rows differ per batch.
    lens = [max(1, seq // 3), max(1, seq // 3)]
    if seq - sum(lens) > 0:
        lens.append(seq - sum(lens))
    row = torch.cat([torch.arange(n, dtype = torch.int32) for n in lens])[:seq]
    return torch.stack([row + b for b in range(batch)]).cuda()


def _oracle(
    x,
    cos,
    sin,
    ids,
    backward = False,
):
    # x [b, h, s, d]; fp64 rotation of the exact low-precision inputs.
    x, cos, sin = x.double(), cos.double(), sin.double()
    seq = x.shape[2]
    if ids is None:
        c, s = cos[:seq], sin[:seq]
    else:
        c, s = cos[ids.long()].unsqueeze(1), sin[ids.long()].unsqueeze(1)
    half = x.shape[-1] // 2
    x0, x1 = x[..., :half], x[..., half:]
    c, s = c[..., :half], s[..., :half]
    if backward:
        s = -s
    out = torch.cat((x0 * c - x1 * s, x1 * c + x0 * s), -1)
    # Magnitude of the terms, for a rounding bound that survives cancellation.
    scale = torch.cat(((x0 * c).abs() + (x1 * s).abs(), (x1 * c).abs() + (x0 * s).abs()), -1)
    return out, scale


def _assert_rounding_close(got, ref, scale, dtype):
    # fma(a, b, -round(c * d)) then a store: at most ~1 ulp of the terms involved.
    eps = torch.finfo(dtype).eps
    err = (got.double() - ref).abs()
    bound = eps * (scale + ref.abs()) + torch.finfo(dtype).tiny
    worst = (err / bound).max().item()
    assert worst <= 1.0, f"error {worst:.3f}x the rounding bound"


def _inputs(shape, dtype, layout, g):
    b, s, h, kv, d = shape
    if layout == "projection":
        # As the attention modules build them: [b, h, s, d] views of [b, s, h * d] projections.
        q_lin = torch.randn(b, s, h * d, device = "cuda", generator = g).to(dtype)
        k_lin = torch.randn(b, s, kv * d, device = "cuda", generator = g).to(dtype)
        q_lin.requires_grad_(True)
        k_lin.requires_grad_(True)
        Q = (q_lin * 1).view(b, s, h, d).transpose(1, 2)
        K = (k_lin * 1).view(b, s, kv, d).transpose(1, 2)
        to_bhsd = lambda t, n: t.view(b, s, n, d).transpose(1, 2)
    else:
        q_lin = torch.randn(b, h, s, d, device = "cuda", generator = g).to(dtype).requires_grad_(True)
        k_lin = torch.randn(b, kv, s, d, device = "cuda", generator = g).to(dtype).requires_grad_(True)
        Q, K = q_lin * 1, k_lin * 1
        to_bhsd = lambda t, n: t
    return q_lin, k_lin, Q, K, to_bhsd


@pytest.mark.parametrize("grad_kind", ["dense", "transposed", "expanded"])
@pytest.mark.parametrize("layout", ["projection", "contiguous"])
@pytest.mark.parametrize("with_ids", [False, True], ids = ["positions", "position_ids"])
@pytest.mark.parametrize("dtype", [torch.float16, BF16, torch.float32], ids = str)
@pytest.mark.parametrize("shape", SHAPES, ids = lambda s: "b{}_s{}_h{}_kv{}_d{}".format(*s))
def test_rope_matches_fp64_oracle(shape, dtype, with_ids, layout, grad_kind):
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    b, s, h, kv, d = shape
    g = torch.Generator(device = "cuda").manual_seed(sum(shape))
    cos, sin = _cos_sin(s + 11, d, dtype)
    ids = _packed_ids(b, s) if with_ids else None
    q_lin, k_lin, Q, K, to_bhsd = _inputs(shape, dtype, layout, g)
    q_in, k_in = Q.detach().clone(), K.detach().clone()

    q, k = fast_rope_embedding(Q, K, cos, sin, ids)
    assert q.shape == (b, h, s, d) and k.shape == (b, kv, s, d)
    for got, x in ((q, q_in), (k, k_in)):
        ref, scale = _oracle(x, cos, sin, ids)
        _assert_rounding_close(got, ref, scale, dtype)

    if grad_kind == "dense":
        dq = torch.randn(q.shape, device = "cuda", generator = g).to(dtype)
        dk = torch.randn(k.shape, device = "cuda", generator = g).to(dtype)
    elif grad_kind == "transposed":
        dq = torch.randn(b, s, h, d, device = "cuda", generator = g).to(dtype).transpose(1, 2)
        dk = torch.randn(b, s, kv, d, device = "cuda", generator = g).to(dtype).transpose(1, 2)
    else:
        dq = torch.randn(1, 1, 1, d, device = "cuda", generator = g).to(dtype).expand(q.shape)
        dk = torch.randn(1, 1, 1, d, device = "cuda", generator = g).to(dtype).expand(k.shape)
    dq_before, dk_before = dq.clone(), dk.clone()
    torch.autograd.backward((q, k), (dq, dk))
    assert torch.equal(dq, dq_before) and torch.equal(dk, dk_before), "upstream grad was mutated"
    for grad, n, up in ((q_lin.grad, h, dq), (k_lin.grad, kv, dk)):
        ref, scale = _oracle(up, cos, sin, ids, backward = True)
        _assert_rounding_close(to_bhsd(grad, n), ref, scale, dtype)


def test_rope_float32_tables_with_half_inputs():
    """Gemma-style fp32 cos / sin with bf16 Q / K computes in fp32 and stores bf16."""
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    shape = (2, 19, 8, 2, 128)
    g = torch.Generator(device = "cuda").manual_seed(3)
    cos, sin = _cos_sin(32, 128, torch.float32)
    for ids in (None, _packed_ids(2, 19)):
        _, _, Q, K, _ = _inputs(shape, BF16, "projection", g)
        q_in, k_in = Q.detach().clone(), K.detach().clone()
        q, k = fast_rope_embedding(Q, K, cos, sin, ids)
        assert q.dtype == BF16 and k.dtype == BF16
        for got, x in ((q, q_in), (k, k_in)):
            ref, scale = _oracle(x, cos, sin, ids)
            _assert_rounding_close(got, ref, scale, BF16)


@pytest.mark.parametrize("with_ids", [False, True], ids = ["positions", "position_ids"])
def test_rope_output_layout_and_aliasing(with_ids):
    """Layouts and aliasing the attention modules see: outputs keep the projection layout, a
    strided position-id input is not mutated, gradients come back contiguous."""
    from unsloth.kernels.rope_embedding import fast_rope_embedding

    b, s, h, kv, d = 2, 24, 8, 2, 64
    g = torch.Generator(device = "cuda").manual_seed(5)
    cos, sin = _cos_sin(32, d, torch.float16)
    ids = _packed_ids(b, s) if with_ids else None
    q_lin, k_lin, Q, K, _ = _inputs((b, s, h, kv, d), torch.float16, "projection", g)
    Q_before = Q.detach().clone()
    q, k = fast_rope_embedding(Q, K, cos, sin, ids)
    assert q.stride() == Q.stride() and k.stride() == K.stride()
    if with_ids:
        assert torch.equal(Q.detach(), Q_before), "a strided input was rotated in place"
    else:
        # The position path rotates the [b, s, h, d] buffer in place, as it always has.
        assert q.data_ptr() == Q.data_ptr()
    torch.autograd.backward((q, k), (torch.ones_like(q), torch.ones_like(k)))
    assert q_lin.grad.is_contiguous() and k_lin.grad.is_contiguous()


@pytest.mark.parametrize("with_ids", [False, True], ids = ["positions", "position_ids"])
def test_rope_contiguous_qk_input_rotates_in_place(with_ids):
    """[b, h, s, d]-contiguous Q / K on the position-id path are rotated in place (no copy)."""
    from unsloth.kernels.rope_embedding import Fast_RoPE_Embedding_QK

    b, s, h, kv, d = 1, 8, 4, 2, 64
    cos, sin = _cos_sin(16, d, torch.float32)
    Q = torch.randn(b, h, s, d, device = "cuda")
    K = torch.randn(b, kv, s, d, device = "cuda")
    ids = _packed_ids(b, s)
    with torch.no_grad():
        q, k = Fast_RoPE_Embedding_QK.apply(Q, K, cos, sin, ids)
    assert q.data_ptr() == Q.data_ptr() and k.data_ptr() == K.data_ptr()


def test_rope_heads_per_program_bounds():
    from unsloth.kernels.rope_embedding import ROPE_TILE_ELEMENTS, _heads_per_program
    for n_heads in (1, 2, 3, 7, 8, 32, 40, 64, 128):
        for block in (16, 32, 64, 128, 256, 1024, 2048):
            heads = _heads_per_program(n_heads, block)
            assert heads >= 1 and heads & (heads - 1) == 0
            assert heads <= max(1, ROPE_TILE_ELEMENTS // block)
            assert heads <= triton_next_pow2(n_heads)


def triton_next_pow2(n):
    return 1 << (n - 1).bit_length()


@pytest.mark.parametrize("layout", ["projection", "contiguous"])
def test_rope_positions_pair_matches_two_single_calls(layout):
    """The fused Q + K position path keeps the bytes, strides, aliasing and gradient layout of
    Fast_RoPE_Embedding applied to Q and K separately."""
    from unsloth.kernels.rope_embedding import Fast_RoPE_Embedding, fast_rope_embedding

    shape = (2, 21, 8, 2, 64)
    cos, sin = _cos_sin(32, 64, BF16)

    def run(fn):
        g = torch.Generator(device = "cuda").manual_seed(11)
        q_lin, k_lin, Q, K, _ = _inputs(shape, BF16, layout, g)
        q, k = fn(Q, K)
        aliased = (q.data_ptr() == Q.data_ptr(), k.data_ptr() == K.data_ptr())
        dq = torch.randn(q.shape, device = "cuda", generator = g).to(BF16)
        dk = torch.randn(k.shape, device = "cuda", generator = g).to(BF16)
        torch.autograd.backward((q, k), (dq, dk))
        return q, k, q_lin.grad, k_lin.grad, aliased

    def single(Q, K):
        q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        k = Fast_RoPE_Embedding.apply(K.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        return q, k

    ref = run(single)
    got = run(lambda Q, K: fast_rope_embedding(Q, K, cos, sin, None))
    assert ref[4] == got[4]
    for r, x in zip(ref[:4], got[:4]):
        assert r.stride() == x.stride()
        assert torch.equal(r.view(torch.int16), x.view(torch.int16))
