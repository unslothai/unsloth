# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 SwiGLU MLP feeding the down projection's act quant from the projection halves (video_minimax_h3_ffn).
The int8 piece (``producer_linear``) is replaced by the plain modules, so the patched forward has to match the stock
diffusers ``FeedForward`` bit for bit on CPU, eager and traced; the GPU kernel is checked by its own probe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture()
def env(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.attention import FeedForward

    from core.inference import diffusion_int8_gemm as gemm
    from core.inference import video_minimax_h3_ffn as ffn

    monkeypatch.delenv(ffn.FUSED_SWIGLU_ENV, raising = False)
    calls = {"producer": 0}

    def producer(down, prefix, gate, value):
        assert prefix is None
        calls["producer"] += 1
        calls["halves"] = (gate, value)
        return down(value * torch.nn.functional.silu(gate))

    monkeypatch.setattr(gemm, "producer_linear", producer)
    downs = set()
    # only the down projections registered here count as swapped rotated int8 Linears
    monkeypatch.setattr(
        gemm,
        "_swapped_rec",
        lambda m: ("v2", 256, True, m.weight, True) if id(m) in downs else None,
    )

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ff = FeedForward(64, inner_dim = 96, activation_fn = "swiglu", bias = False)

        def forward(self, x):
            return self.ff(x) + x

    def build(swapped = True):
        torch.manual_seed(0)
        h = Holder().eval()
        if swapped:
            downs.add(id(h.ff.net[2]))
        return h

    yield torch, ffn, build, calls


def _x(torch):
    return torch.randn(2, 33, 64, generator = torch.Generator().manual_seed(1))


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_patched_mlp_matches_stock(env, mode):
    torch, ffn, build, calls = env
    h = build()
    x = _x(torch)
    with torch.no_grad():
        ref = h(x)
        assert ffn.install(h) == 1
        assert "forward" in h.ff.__dict__
        if mode == "compiled":
            torch._dynamo.reset()
            out = torch.compile(h, backend = "eager", fullgraph = True)(x)
            torch._dynamo.reset()
        else:
            out = h(x)
    assert calls["producer"] >= 1
    assert torch.equal(out, ref)
    if mode == "eager":
        gate, value = calls["halves"]
        # halves of the one projection output, no copy: value first, gate second (diffusers' SwiGLU.chunk order)
        assert gate.stride() == value.stride() and gate.stride(-2) == 2 * 96
        assert gate.data_ptr() - value.data_ptr() == 96 * gate.element_size()
    assert ffn.uninstall(h) == 1 and "forward" not in h.ff.__dict__
    with torch.no_grad():
        assert torch.equal(h(x), ref)


def test_kill_switch_keeps_stock(env, monkeypatch):
    torch, ffn, build, calls = env
    monkeypatch.setenv(ffn.FUSED_SWIGLU_ENV, "0")
    h = build()
    assert ffn.install(h) == 0
    assert "forward" not in h.ff.__dict__
    with torch.no_grad():
        h(_x(torch))
    assert calls["producer"] == 0


def test_per_call_gates_run_stock(env):
    torch, ffn, build, calls = env
    x = _x(torch)
    h = build(swapped = False)
    with torch.no_grad():
        ref = h(x)
        ffn.install(h)
        assert torch.equal(h(x), ref)
    assert calls["producer"] == 0
    h = build()
    h.ff.net[1].p = 0.5
    h.ff.net[1].train()
    ffn.install(h)
    assert not ffn._ready(h.ff)
    h = build()
    with torch.no_grad():
        ref = h(x)
    h.ff.net[0].register_forward_hook(lambda m, a, o: o)
    ffn.install(h)
    with torch.no_grad():
        assert torch.equal(h(x), ref)
    assert calls["producer"] == 0


def test_h3_blocks_are_eligible(env):
    torch, ffn, _build, _calls = env
    h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    block_cls = getattr(h3, "MiniMaxH3TransformerBlock", None)
    if block_cls is None:
        pytest.skip("this diffusers has no MiniMaxH3TransformerBlock")
    try:
        block = block_cls(
            hidden_size = 64,
            num_attention_heads = 2,
            attention_head_dim = 32,
            ffn_dim = 96,
            time_embed_dim = 32,
            norm_eps = 1e-6,
            qk_norm_eps = 1e-6,
        )
    except TypeError:
        pytest.skip("MiniMaxH3TransformerBlock signature changed")
    assert ffn._eligible(block.ff)
    assert ffn.install(block) == 1


def test_video_installs_it_before_the_first_forward():
    src = (_BACKEND / "core" / "inference" / "video.py").read_text(encoding = "utf-8")
    i = src.index("from .video_minimax_h3_ffn import install as install_h3_ffn")
    assert src.index("compile_blocks_below_offload_hooks(", src.index("install_int8_gemm(")) < i
    assert '"h3_fused_swiglu"' in src[i : i + 600]
