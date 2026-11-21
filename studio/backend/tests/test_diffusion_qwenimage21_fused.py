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
    # pins the stock rotation; rotq_h16 dispatch: test_diffusion_convrot_unlisted.py
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_ROTQUANT_UNLISTED", "0")
    monkeypatch.setattr(i8f, "_RQ_DEVICES", frozenset())
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


def _gemm_path_ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False)
    except Exception:  # noqa: BLE001
        return False
    return torch.cuda.get_device_capability() >= (8, 0)


@pytest.mark.skipif(not _gemm_path_ready(), reason = "needs NVIDIA sm80+ CUDA and torchao Int8Tensor")
def test_int8_gemm_path_rotates_each_shared_input_once(monkeypatch):
    """Where diffusion_int8_gemm swaps the rotated Linears (forced here on any sm80+ part): Q / K / V and the SwiGLU
    gate / value each quantize their shared input ONCE (4 fused rotation calls per block instead of 7), Q / K / V
    bit-identical to the stock helper on the same swapped Linears, the block within the int8 rounding of it."""
    from core.inference import diffusion_convrot_quant as cq
    from core.inference import diffusion_int8_gemm as g8
    from core.inference.diffusion_convrot import rotate_linears_, warm_rotation_cache
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    index = torch.cuda.current_device()
    g8._DEVICE_CFG.pop(index, None)
    cq._ROTQ_DEVICE.pop(index, None)
    if g8.device_config(index) is None or cq.rotquant_device_config(index) is None:
        pytest.skip("int8 GEMM / fused rotation probe refused this device")
    assert not qf.arch_ok()
    torch.manual_seed(0)
    blk = qmod.QwenImage21TransformerBlock(DIM, HEADS, HEAD_DIM).cuda().to(torch.bfloat16).eval()
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    names = [n for n, m in blk.named_modules() if isinstance(m, torch.nn.Linear)]
    rotate_linears_(blk, names, 256)
    warm_rotation_cache(blk, "cuda", torch.bfloat16)
    quantize_(
        blk,
        Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear),
    )
    assert g8.install(blk) == 7
    x = _x()
    stock_prepare = getattr(qmod, qf._PREPARE)
    with torch.no_grad():
        before = cq.rotquant_call_count()
        ref_qkv = stock_prepare(blk.attn, x, None, None, None, None)
        assert cq.rotquant_call_count() == before + 3
        ref = blk(x, mod_params(x), rotary_emb = _rope(x.shape[1]))
        assert qf.install_shared_qkv(blk) == 1
        before = cq.rotquant_call_count()
        qkv = getattr(qmod, qf._PREPARE)(blk.attn, x, None, None, None, None)
        assert cq.rotquant_call_count() == before + 1
        assert all(torch.equal(a, b) for a, b in zip(qkv[:3], ref_qkv[:3])) and qkv[3] == ref_qkv[3]
        before = cq.rotquant_call_count()
        out = blk(x, mod_params(x), rotary_emb = _rope(x.shape[1]))
        assert cq.rotquant_call_count() == before + 4
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim = 0)
    assert cos > 0.9999
    g8.uninstall(blk)


def test_shared_qkv_reported_and_gated_on_the_transformer_cards(monkeypatch):
    """Shared-QKV installs on the transformer's card (not the current one) if int8 GEMM covers it;
    its count is reported even where fused-QKV declines."""
    import core.inference.diffusion_qwenimage21_fused as mod

    model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    calls = []
    swap_cards = {
        1
    }  # cuda:1 holds the transformer and is swap-covered; the current card cuda:0 is not
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mod, "_swap_covers", lambda index: index in swap_cards)
    monkeypatch.setattr(mod, "install_shared_qkv", lambda t, logger = None: calls.append(t) or 2)
    monkeypatch.setattr(i8f, "resident_cuda_cards", lambda m: ())
    assert mod.install(model, offload_active = True) == {"fused_qkv": 2, "out": 0}
    monkeypatch.setattr(i8f, "resident_cuda_cards", lambda m: (1,))
    monkeypatch.setattr(mod, "install_modules", lambda r, logger = None: {"fused_qkv": 0, "out": 0})
    assert mod.install(model) == {"fused_qkv": 2, "out": 0}
    assert calls == [model, model]
    swap_cards.clear()  # every card on the stock path: the helper is not installed
    assert mod.install(model) == {"fused_qkv": 0, "out": 0}
    assert calls == [model, model]


@pytest.mark.parametrize("rocm", ["hip", "tag"])
def test_rocm_cards_are_not_swap_covered(monkeypatch, rocm):
    """ROCm: ``torch.cuda.is_available()`` and a CUDA-like capability, but the int8 GEMM swap declines it, so no card
    counts as swap-covered and the process-global QKV helper is not installed or reported."""
    import core.inference.diffusion_int8_gemm as g8
    import core.inference.diffusion_qwenimage21_fused as mod

    model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (9, 0))
    monkeypatch.setattr(g8, "_gates", lambda cap: (True, True))
    monkeypatch.setattr(g8, "int8_gemm_mode", lambda: "auto")
    monkeypatch.setattr(mod, "install_shared_qkv", lambda t, logger = None: calls.append(t) or 2)
    monkeypatch.setattr(i8f, "resident_cuda_cards", lambda m: ())
    assert mod._swap_card_present(model)  # NVIDIA with the swap gated in: covered
    if rocm == "hip":
        monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    else:
        monkeypatch.setattr(torch, "__version__", "2.9.0+rocm6.4")
    assert not mod._swap_card_present(model)
    assert mod.install(model, offload_active = True) == {"fused_qkv": 0, "out": 0}
    assert calls == []


def test_swap_probe_failure_is_not_swap_covered(monkeypatch):
    import core.inference.diffusion_int8_gemm as g8
    import core.inference.diffusion_qwenimage21_fused as mod

    model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(i8f, "resident_cuda_cards", lambda m: ())

    def boom(index = None):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    monkeypatch.setattr(g8, "int8_gemm_mode", lambda: "auto")
    assert not mod._swap_card_present(model)


@pytest.mark.parametrize(
    "env", ["UNSLOTH_DIFFUSION_INT8_ROTQUANT", "UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT", "probe"]
)
def test_shared_qkv_needs_the_rotation_kernel_not_just_the_fused_gemm(monkeypatch, env):
    """A fused-GEMM arch with ``rotq_i8`` switched off (or failed its per-device probe): ``shared_rotq_linears``
    declines every record (``not r[2]``), so the helper is neither installed nor counted."""
    import core.inference.diffusion_convrot_quant as cq
    import core.inference.diffusion_int8_gemm as g8
    import core.inference.diffusion_qwenimage21_fused as mod

    model = type("QwenImage21Transformer2DModel", (torch.nn.Module,), {})()
    calls = []
    for name in (
        "UNSLOTH_DIFFUSION_INT8_ROTQUANT",
        "UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT",
        "UNSLOTH_DIFFUSION_INT8_GEMM",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(g8, "int8_gemm_mode", lambda: "auto")
    monkeypatch.setattr(mod, "install_shared_qkv", lambda t, logger = None: calls.append(t) or 2)
    monkeypatch.setattr(i8f, "resident_cuda_cards", lambda m: ())
    monkeypatch.delitem(cq._ROTQ_DEVICE, 0, raising = False)
    assert g8._gates((8, 9)) == (True, True)
    assert mod.install(model, offload_active = True) == {"fused_qkv": 2, "out": 0}  # control: engaged
    assert calls == [model]
    calls.clear()
    qf.uninstall()
    if env == "probe":
        monkeypatch.setitem(
            cq._ROTQ_DEVICE, 0, None
        )  # rotq_i8 failed its bit-exact probe on cuda:0
    else:
        monkeypatch.setenv(env, "0")
        assert g8._gates((8, 9)) == (True, False)
    assert not mod._swap_card_present(model)
    assert mod.install(model, offload_active = True) == {"fused_qkv": 0, "out": 0}
    assert calls == []
