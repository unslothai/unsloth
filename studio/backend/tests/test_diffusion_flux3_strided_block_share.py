# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 blocks whose attention runs the strided cuDNN / INT8 processor (diffusion_flux3_attn) share one rotation +
act quant of the block input across Q / K / V and the MLP input (diffusion_flux_convrot_fused._f3_attn_ff), as the
stock and QK-RoPE processors do. CPU: the int8 pieces run the plain modules, the INT8 attention kernel a plain SDPA,
so the shared path must match the unshared one bit for bit."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
f3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")

from core.inference import diffusion_flux3_attn as fa  # noqa: E402
from core.inference import diffusion_flux_convrot_fused as fused  # noqa: E402
from core.inference import diffusion_int8_gemm as gemm  # noqa: E402

F = torch.nn.functional
DIM, HEADS, HD = 64, 2, 32


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.delenv(fa.STRIDED_ATTN_ENV, raising = False)
    monkeypatch.delenv(fused.FLUX_CONVROT_FUSED_ENV, raising = False)
    fa.strided_processor_class.cache_clear()
    fused.uninstall()
    calls = {"widths": [], "kernel": 0, "producer": 0}
    monkeypatch.setattr(fused, "_rotated", lambda lin: True)

    def shared(
        x,
        linears,
        merged = None,
    ):
        calls["widths"].append(len(linears))
        return [m(x) for m in linears]

    def producer(down, prefix, gate, value):
        calls["producer"] += 1
        return down(F.silu(gate) * value)

    def kernel(
        q,
        k,
        v,
        mode = 0,
    ):
        calls["kernel"] += 1
        return F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)

    monkeypatch.setattr(gemm, "shared_rotq_linears", shared)
    monkeypatch.setattr(gemm, "producer_linear", producer)
    monkeypatch.setattr(fa._i8, "int8_attention_call", kernel)
    yield calls
    fused.uninstall()
    fa.strided_processor_class.cache_clear()


def _strided(route):
    proc = fa.strided_processor_class()()
    if route == "cudnn":
        proc._attention_backend = "_native_cudnn"
    else:
        proc._attention_backend = None
        proc._unsloth_int8 = True
        proc._unsloth_int8_min_tokens = 4
    return proc


def _rope(seq):
    g = torch.Generator().manual_seed(1)
    ang = torch.rand(seq, HD, generator = g) * 3
    return ang.cos(), ang.sin()


def _case(kind):
    g = torch.Generator().manual_seed(2)
    hs = torch.randn(2, 12, DIM, generator = g)
    enc = torch.randn(2, 5, DIM, generator = g)
    mod = torch.randn(2, 1, 3 * DIM, generator = g) * 0.1
    cmod = torch.randn(2, 1, 3 * DIM, generator = g) * 0.1
    if kind == "mode":
        rope = _rope(12)
        return f3.Flux3ModeTransformerBlock, lambda b: b(hs, mod, rope)
    rope = _rope(17)
    return f3.Flux3SingleTransformerBlock, lambda b: b(hs, enc, mod, cmod, rope)


def _block(cls, route):
    torch.manual_seed(0)
    block = cls(DIM, HEADS, HD).eval()
    block.attn.set_processor(_strided(route))
    return block


def test_gate_accepts_the_strided_processor(env):
    assert fused._install_patches()  # noqa: SLF001 - fills the class table the gate reads
    for route in ("cudnn", "int8"):
        block = _block(f3.Flux3SingleTransformerBlock, route)
        assert fused._f3_block_ready(block)  # noqa: SLF001
    block = _block(f3.Flux3SingleTransformerBlock, "cudnn")

    class Other(f3.Flux3AttnProcessor):
        pass

    block.attn.set_processor(Other())
    assert not fused._f3_block_ready(block)  # noqa: SLF001


@pytest.mark.parametrize("route", ["cudnn", "int8"])
@pytest.mark.parametrize("kind", ["mode", "single"])
@pytest.mark.parametrize("compiled", [False, True])
def test_strided_block_shares_one_quant_and_matches(env, kind, route, compiled):
    """One quant of the block input for Q / K / V and the MLP input (before: Q / K / V, then the MLP input alone),
    attention through the strided processor's own method (the INT8 kernel on the INT8 route), output bit for bit the
    unpatched block's."""
    calls = env
    cls, call = _case(kind)
    block = _block(cls, route)
    with torch.no_grad():
        ref = call(block)
        kernel_ref = calls["kernel"]
        before = fused.call_counts().get("block", 0)
        assert fused._install_patches()  # noqa: SLF001
        if compiled:
            torch._dynamo.reset()
            out = call(torch.compile(block, backend = "eager", fullgraph = True))
            torch._dynamo.reset()
        else:
            out = call(block)
    if isinstance(ref, tuple):
        assert all(torch.equal(a, b) for a, b in zip(ref, out))
    else:
        assert torch.equal(ref, out)
    assert calls["widths"] == [4]
    assert calls["producer"] == 1
    assert calls["kernel"] - kernel_ref == (1 if route == "int8" else 0)
    if not compiled:
        assert fused.call_counts()["block"] == before + 1


def test_below_the_int8_floor_the_block_runs_the_stock_attention(env):
    """A call the strided processor hands to the stock processor keeps the stock mirror, still on the shared quant."""
    calls = env
    cls, call = _case("mode")
    block = _block(cls, "int8")
    block.attn.processor._unsloth_int8_min_tokens = 10_000
    with torch.no_grad():
        ref = call(block)
        assert fused._install_patches()  # noqa: SLF001
        out = call(block)
    assert torch.equal(ref, out)
    assert calls["widths"] == [4] and calls["kernel"] == 0


@pytest.mark.parametrize("route", ["cudnn", "int8", "stock"])
def test_strided_call_is_its_projections_then_the_projected_method(env, route):
    """``__call__`` and ``unsloth_projected_attention`` run the same ops below the projections; the method declines
    exactly the calls ``__call__`` hands to the stock processor."""
    torch.manual_seed(1)
    attn = f3.Flux3Attention(query_dim = DIM, heads = HEADS, dim_head = HD).eval()
    proc = _strided("int8" if route == "stock" else route)
    if route == "stock":
        proc._unsloth_int8_min_tokens = 10_000
    attn.set_processor(proc)
    x = torch.randn(2, 12, DIM, generator = torch.Generator().manual_seed(3))
    rope = _rope(12)
    with torch.no_grad():
        got = proc(attn, x, None, rope)
        q, k, v = f3._get_qkv_projections(attn, x)
        via = proc.unsloth_projected_attention(attn, q, k, v, x.shape[1], rope)
        stock = f3.Flux3AttnProcessor()
        stock._attention_backend = proc._attention_backend if route == "cudnn" else None
        want = stock(attn, x, None, rope)
    if route == "stock":
        assert via is None
    else:
        assert torch.equal(got, via)
    assert torch.equal(got, want)
