# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``diffusion_qwenimage21_fused``; exactness tests need CUDA sm90 / sm100."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_fused as i8f
from core.inference import diffusion_qwenimage21_fused as qf

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

DIM, HEADS, HEAD_DIM, GROUP = 256, 2, 128, 64


def _ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False)
    except Exception:  # noqa: BLE001
        return False
    return qf.arch_ok()


needs_cuda = pytest.mark.skipif(
    not _ready(),
    reason = "needs CUDA (not ROCm), torchao Int8Tensor, and an arch the int8 GEMM swap leaves stock",
)


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(qf.Q21_CONVROT_FUSED_ENV, raising = False)
    yield
    qf.uninstall()


def test_stock_prepare_qkv_fingerprint_matches_installed_diffusers():
    from core.inference.diffusion_qwenimage21_rope import _FINGERPRINTS, _digest
    assert _digest(getattr(qmod, qf._PREPARE)) in _FINGERPRINTS[qf._PREPARE]


def test_kill_switch_offload_and_wrong_model_are_noops(monkeypatch):
    class Other(torch.nn.Module):
        pass

    assert qf.install(Other()) == {"fused_qkv": 0, "out": 0}
    blk_model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    assert qf.install(blk_model, offload_active = True) == {"fused_qkv": 0, "out": 0}
    monkeypatch.setenv(qf.Q21_CONVROT_FUSED_ENV, "0")
    assert qf.install(blk_model) == {"fused_qkv": 0, "out": 0}
    assert not getattr(getattr(qmod, qf._PREPARE), "__unsloth_q21_fused__", False)


def _block(rotate: bool = True):
    torch.manual_seed(0)
    blk = qmod.QwenImage21TransformerBlock(DIM, HEADS, HEAD_DIM).cuda().to(torch.bfloat16).eval()
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    names = [n for n, m in blk.named_modules() if isinstance(m, torch.nn.Linear)]
    if rotate:
        from core.inference.diffusion_convrot import rotate_linears_, warm_rotation_cache
        rotate_linears_(blk, names, GROUP)
        warm_rotation_cache(blk, "cuda", torch.bfloat16)
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    quantize_(
        blk,
        Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear),
    )
    return blk


def _x(seq = 96, scale = 1.0):
    g = torch.Generator(device = "cpu").manual_seed(1)
    x = torch.randn(1, seq, DIM, generator = g) * scale
    x[..., :3] *= 40
    return x.cuda().to(torch.bfloat16)


def _engage(blk):
    with torch.inference_mode(False), torch.no_grad():
        assert i8f.install(blk) == 1
    res = qf.install_modules(blk)
    assert res == {"fused_qkv": 1, "out": 1}
    return res


@needs_cuda
def test_shared_projections_equal_the_separate_rotated_ones():
    blk = _block()
    x = _x()
    attn, ff = blk.attn, blk.img_mlp
    with torch.no_grad():
        ref_q, ref_k, ref_v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        ref_out = attn.to_out[0](x)
        ref_ff = type(ff).forward(ff, x)
        _engage(blk)
        fused = qf._qkv_intact(attn)
        from core.inference.diffusion_convrot import is_rotated_linear

        assert is_rotated_linear(fused) and fused.out_features == 3 * DIM
        q, k, v = i8f.int8_linear(fused, x).chunk(3, dim = -1)
        out = i8f.int8_linear_core(attn.to_out[0], x)
        y_ff = ff(x)
    assert torch.equal(q, ref_q) and torch.equal(k, ref_k) and torch.equal(v, ref_v)
    assert torch.equal(out, ref_out)
    assert torch.equal(y_ff, ref_ff)
    assert attn.to_q.weight.qdata.data_ptr() == fused.weight.qdata.data_ptr()


def mod_params(x):
    g = torch.Generator(device = "cpu").manual_seed(2)
    return (torch.randn(1, 4 * DIM, generator = g) * 0.1).cuda().to(torch.bfloat16)


def _rope(seq):
    g = torch.Generator(device = "cpu").manual_seed(3)
    ang = torch.rand(seq, HEAD_DIM // 2, generator = g) * 6.28
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).cuda()


@needs_cuda
def test_compiled_block_stays_within_the_compile_floor(monkeypatch):
    # Inductor's own act quant is not eager-exact: bar = the stock compile's distance from eager
    traced = []
    real_linear = i8f.int8_linear

    def counting(module, x):
        traced.append(
            module.out_features
        )  # runs at trace time: proves the compiled graph took the fused QKV
        return real_linear(module, x)

    monkeypatch.setattr(i8f, "int8_linear", counting)
    blk = _block()
    x = _x()
    rope = _rope(x.shape[1])
    with torch.no_grad(), torch._inductor.config.patch(emulate_precision_casts = True):
        eager = blk(x, mod_params(x), rotary_emb = rope)
        torch._dynamo.reset()
        ref = torch.compile(blk)(x, mod_params(x), rotary_emb = rope)
        _engage(blk)
        assert getattr(getattr(qmod, qf._PREPARE), "__unsloth_q21_fused__", False)
        torch._dynamo.reset()
        out = torch.compile(blk)(x, mod_params(x), rotary_emb = rope)
        again = blk(x, mod_params(x), rotary_emb = rope)
    assert torch.equal(again, eager)
    assert 3 * DIM in traced
    floor = (ref != eager).sum().item()
    assert (out != eager).sum().item() <= 1.1 * floor + 16
    assert (out.float() - eager.float()).abs().max().item() <= 2 * (
        ref.float() - eager.float()
    ).abs().max().item() + 1e-6


@needs_cuda
def test_plain_int8_block_is_left_alone():
    blk = _block(rotate = False)
    assert not qf.rotated_ff(blk.img_mlp)
    assert qf.install_modules(blk) == {"fused_qkv": 0, "out": 0}
    assert not qf.is_installed(blk.attn) and "forward" not in blk.attn.to_out[0].__dict__


@needs_cuda
def test_arch_covered_by_the_int8_gemm_swap_is_left_alone(monkeypatch):
    blk = _block()
    monkeypatch.setattr(qf, "arch_ok", lambda device = None: False)
    assert not qf.rotated_ff(blk.img_mlp)
    assert qf.install_modules(blk) == {"fused_qkv": 0, "out": 0}


@needs_cuda
def test_replaced_projection_falls_back_and_uninstall_restores(monkeypatch):
    blk = _block()
    _engage(blk)
    attn = blk.attn
    assert qf._qkv_intact(attn) is not None
    original = attn.to_k
    attn.to_k = torch.nn.Identity()  # a LoRA wrapper / later swap
    assert qf._qkv_intact(attn) is None
    attn.to_k = original
    qf.uninstall(blk)
    assert not getattr(getattr(qmod, qf._PREPARE), "__unsloth_q21_fused__", False)
    assert not qf.is_installed(attn) and "forward" not in attn.to_out[0].__dict__
    i8f.uninstall(blk)


@needs_cuda
def test_deferred_install_is_cancelled_by_uninstall():
    blk = _block()
    model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    model.blk = blk.cpu()
    assert qf.install(model) == {"fused_qkv": None, "out": None}
    assert "q21_convrot_fused" in model.__dict__["_unsloth_first_call_hooks"]
    qf.uninstall(model)
    assert "q21_convrot_fused" not in model.__dict__["_unsloth_first_call_hooks"]


def test_model_unload_restores_the_process_global_patch(monkeypatch):
    from core.inference import diffusion as d

    sentinel = getattr(qmod, qf._PREPARE)
    fake = lambda *a, **k: None  # noqa: E731
    fake.__unsloth_q21_fused__ = True
    qf._STATE["stock"] = (qmod, sentinel)
    monkeypatch.setattr(qmod, qf._PREPARE, fake)
    d._uninstall_fused_dit_patches()
    assert getattr(qmod, qf._PREPARE) is sentinel
