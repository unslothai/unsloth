# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image speed-layer fusions (diffusion_moss_fused): one q/k/v GEMM, the int8 runner hook and the fused q/k
RMSNorm + RoPE kernel. The block reads them only inside a compiled forward, so the tests force that branch."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(scope = "module")
def mods():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("diffusers without AutoencoderKLFlux2")
    from core.inference import diffusion_moss_fused as fused
    from core.inference import diffusion_moss_image as moss

    return torch, moss, fused


def _tiny(torch, moss):
    torch.manual_seed(0)
    return moss.MOSSImageTransformer(
        dim = 256, ffn_dim = 512, num_heads = 2, num_layers = 3, text_dim = 64
    ).eval()


def _inputs(torch):
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, 128, 4, 4, generator = g),
        torch.tensor([500.0]),
        torch.randn(1, 7, 64, generator = g),
    )


def _compiling(torch, moss):
    return mock.patch.object(torch.compiler, "is_compiling", lambda: True), mock.patch.object(
        moss, "_length_dynamic", lambda *a: None
    )


def _fuse_all(
    moss,
    fused,
    transformer,
    runner = None,
):
    from core.inference.diffusion_zimage_fused import _fuse_linears, _share_storage
    for block in fused._blocks(transformer):
        parts = [block.proj[n] for n in ("q", "k", "v")]
        linear = _fuse_linears(parts)
        assert linear is not None and _share_storage(linear, parts)
        block.__dict__[moss.FUSED_QKV_ATTR] = (linear, *parts, runner)


def test_fused_qkv_matches_the_three_projections(mods):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    x, t, e = _inputs(torch)
    with torch.no_grad():
        ref = transformer(x, t, e, return_dict = False)[0]
        _fuse_all(moss, fused, transformer)
        a, b = _compiling(torch, moss)
        with a, b:
            out = transformer(x, t, e, return_dict = False)[0]
        eager = transformer(x, t, e, return_dict = False)[0]
    assert torch.equal(out, ref) and torch.equal(eager, ref)
    # the parts are views of the fused weight: no second copy
    block = fused._blocks(transformer)[0]
    linear = block.__dict__[moss.FUSED_QKV_ATTR][0]
    assert (
        block.proj["q"].weight.untyped_storage().data_ptr()
        == linear.weight.untyped_storage().data_ptr()
    )


def test_a_replaced_projection_drops_back_to_stock(mods):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    x, t, e = _inputs(torch)
    with torch.no_grad():
        _fuse_all(moss, fused, transformer)
        block = fused._blocks(transformer)[0]
        replacement = torch.nn.Linear(256, 256, bias = False)
        replacement.weight.copy_(block.proj["q"].weight * 2)
        block.proj["q"] = replacement
        a, b = _compiling(torch, moss)
        with a, b:
            out = transformer(x, t, e, return_dict = False)[0]
        ref = transformer(x, t, e, return_dict = False)[0]
    assert torch.equal(out, ref)


def test_the_runner_reaches_every_block_linear(mods):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    x, t, e = _inputs(torch)
    seen = []

    def runner(linear, value):
        seen.append(linear)
        return linear(value)

    with torch.no_grad():
        ref = transformer(x, t, e, return_dict = False)[0]
        for block in fused._blocks(transformer):
            block.__dict__[moss.LINEAR_RUNNER_ATTR] = runner
        a, b = _compiling(torch, moss)
        with a, b:
            out = transformer(x, t, e, return_dict = False)[0]
    assert torch.equal(out, ref)
    assert len(seen) == 6 * 3


def test_uninstall_and_kill_switch_clear_the_blocks(mods, monkeypatch):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    _fuse_all(moss, fused, transformer)
    for block in fused._blocks(transformer):
        block.__dict__[moss.QK_ROPE_ATTR] = object()
    monkeypatch.setenv(fused.MOSS_FUSED_ENV, "0")
    assert fused.install(transformer) == {"fused_qkv": 0, "int8_runner": 0, "qk_rope": 0}
    for block in fused._blocks(transformer):
        assert not {moss.FUSED_QKV_ATTR, moss.LINEAR_RUNNER_ATTR, moss.QK_ROPE_ATTR} & set(
            block.__dict__
        )


def test_other_denoisers_and_offload_are_left_alone(mods):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    assert fused.install(torch.nn.Linear(2, 2)) == {"fused_qkv": 0, "int8_runner": 0, "qk_rope": 0}
    assert fused.install(transformer, offload_active = True) == {
        "fused_qkv": 0,
        "int8_runner": 0,
        "qk_rope": 0,
    }


def test_bf16_keeps_three_gemms_and_rope_is_opt_in(mods, monkeypatch):
    torch, moss, fused = mods
    transformer = _tiny(torch, moss)
    from core.inference import diffusion_int8_fused as i8

    # resident on a (pretend) CUDA device with plain bf16 weights: no runner, so no fused q/k/v and no RoPE kernel
    monkeypatch.setattr(i8, "resident_cuda_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(fused, "qk_rope_device_ok", lambda index: True)
    monkeypatch.setattr(fused, "qk_rope_op", lambda: object())
    monkeypatch.delenv(fused.MOSS_QK_ROPE_ENV, raising = False)
    assert fused.install_modules(transformer) == {"fused_qkv": 0, "int8_runner": 0, "qk_rope": 0}
    monkeypatch.setenv(fused.MOSS_QK_ROPE_ENV, "1")
    assert (
        fused.install_modules(transformer)["qk_rope"] == 0
    )  # fp32 tiny norms: not the bf16 kernel's input
    for block in fused._blocks(transformer):
        for n in (block.q_norm, block.k_norm):
            n.weight.data = n.weight.data.bfloat16()
    assert fused.install_modules(transformer)["qk_rope"] == 3


def test_the_patch_is_registered_for_the_moss_class():
    from core.inference.diffusion_autoopt import patches
    spec = next(s for s in patches.specs() if s.name == "moss_fused")
    assert spec.classes == ("MOSSImageTransformer",) and spec.env == "UNSLOTH_DIFFUSION_MOSS_FUSED"


def test_qk_rope_kernel_is_the_eager_ops(mods):
    torch, moss, fused = mods
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if fused.qk_rope_op() is None:
        pytest.skip("no Triton")
    assert fused.qk_rope_device_ok(torch.cuda.current_device())
    # block level: the fused path through a real (tiny, bf16) transformer equals the stock eager block
    transformer = _tiny(torch, moss).to("cuda", torch.bfloat16)
    x, t, e = (v.to("cuda") for v in _inputs(torch))
    with torch.no_grad():
        ref = transformer(x.bfloat16(), t, e.bfloat16(), return_dict = False)[0]
        for block in fused._blocks(transformer):
            block.__dict__[moss.QK_ROPE_ATTR] = fused.qk_rope_op()
        a, b = _compiling(torch, moss)
        with a, b:
            out = transformer(x.bfloat16(), t, e.bfloat16(), return_dict = False)[0]
    assert (out.float() - ref.float()).abs().max() <= 2**-5 * ref.float().abs().max()
