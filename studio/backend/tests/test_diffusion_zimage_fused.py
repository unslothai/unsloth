# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_zimage_fused.py``; the bit-identity tests need CUDA and inductor's fma addcmul lowering."""

from __future__ import annotations

import pytest

from core.inference import diffusion_zimage_fused as zf

torch = pytest.importorskip("torch")
zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")


def _ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    from core.inference.diffusion_qwenimage21_rope import inductor_addcmul_is_fma

    return inductor_addcmul_is_fma()


needs_cuda = pytest.mark.skipif(not _ready(), reason = "needs CUDA (not ROCm) and inductor's fma addcmul lowering")


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(zf.ZIMAGE_FUSED_ENV, raising = False)
    yield
    zf.uninstall()


def test_stock_processor_fingerprint_matches_installed_diffusers():
    assert zf._stock_digest(zmod.ZSingleStreamAttnProcessor) in zf._FINGERPRINTS["ZSingleStreamAttnProcessor.__call__"]


def test_disabled_and_wrong_model_are_noops(monkeypatch):
    assert zf.install(torch.nn.Linear(4, 4)) == {"real_rope": False, "fused_qkv": 0}
    monkeypatch.setenv(zf.ZIMAGE_FUSED_ENV, "0")
    assert zf.install_modules(torch.nn.Linear(4, 4)) == {"real_rope": False, "fused_qkv": 0}


def test_fuse_linears_rejects_mismatched_parts():
    a, b = torch.nn.Linear(8, 8, bias = False), torch.nn.Linear(8, 8, bias = True)
    assert zf._fuse_linears([a, b]) is None
    assert zf._fuse_linears([a, torch.nn.Linear(4, 8, bias = False)]) is None


def test_fuse_linears_plain_bf16_matches_and_shares_storage():
    parts = [torch.nn.Linear(16, 8, bias = False) for _ in range(3)]
    x = torch.randn(5, 16)
    ref = torch.cat([p(x) for p in parts], dim = -1)
    fused = zf._fuse_linears(parts)
    assert zf._share_storage(fused, parts)
    assert torch.equal(fused(x), ref)
    assert parts[1].weight.data_ptr() == fused.weight.data_ptr() + 8 * 16 * fused.weight.element_size()


def _block(quant: str):
    torch.manual_seed(0)
    blk = zmod.ZImageTransformerBlock(0, 384, 4, 4, 1e-5, True, modulation = True).cuda().to(torch.bfloat16).eval()
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    if quant == "int8":
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        quantize_(blk, Int8DynamicActivationInt8WeightConfig(), filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "adaLN" not in fqn)
    return blk


def _inputs(seq = 96):
    g = torch.Generator(device = "cpu").manual_seed(1)
    x = torch.randn(1, seq, 384, generator = g).cuda().to(torch.bfloat16)
    mask = torch.ones(1, seq, dtype = torch.bool).cuda()
    ang = torch.rand(1, seq, 48, generator = g) * 6.28
    freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64).cuda()
    adaln = torch.randn(1, 256, generator = g).cuda().to(torch.bfloat16)
    return x, mask, freqs, adaln


@needs_cuda
@pytest.mark.parametrize("quant", ["bf16", "int8"])
def test_compiled_block_stays_within_the_compile_floor(quant):
    # Not bit-identical to the stock compiled block: moving RoPE and the QKV split into Inductor moves its fusion
    # boundaries, and with emulate_precision_casts off those decide where bf16 rounding happens. The bar is the
    # stock compile's own distance from eager.
    if quant == "int8":
        pytest.importorskip("torchao.quantization")
    blk = _block(quant)
    x, mask, freqs, adaln = _inputs()
    with torch.no_grad():
        eager = blk(x, mask, freqs, adaln)
        torch._dynamo.reset()
        ref = torch.compile(blk)(x, mask, freqs, adaln)
        res = zf.install_modules(blk)
        assert res["fused_qkv"] == 1
        torch._dynamo.reset()
        out = torch.compile(blk)(x, mask, freqs, adaln)
    floor = (ref != eager).sum().item()
    assert (out != eager).sum().item() <= 1.1 * floor + 16
    assert (out.float() - eager.float()).abs().max().item() <= 2 * (ref.float() - eager.float()).abs().max().item() + 1e-6


@needs_cuda
def test_eager_calls_keep_the_stock_processor():
    blk = _block("bf16")
    x, mask, freqs, adaln = _inputs()
    with torch.no_grad():
        ref = blk(x, mask, freqs, adaln)
        zf.install_modules(blk)
        out = blk(x, mask, freqs, adaln)
    assert torch.equal(out, ref)


@needs_cuda
def test_wrapped_projection_falls_back_to_stock_projections():
    blk = _block("bf16")
    zf.install_modules(blk)
    attn = blk.attention
    assert zf._qkv_intact(attn) is not None
    attn.to_k = torch.nn.Sequential(attn.to_k)  # stand-in for a LoRA wrapper
    assert zf._qkv_intact(attn) is None
