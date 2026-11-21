# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1 int8 ConvRot producer fusion (diffusion_flux_convrot_fused): shared Q / K / V quant, and the FF / single-block
down projections reading GELU(tanh) (and the attention output) from their producers. The int8 pieces are replaced by
the plain modules here, so a forced-on patch has to match stock bit for bit on CPU (eager and dynamo fullgraph). The
Triton kernel itself is checked by its own probe (exhaustive bf16 GELU + ``codes_close``) on the GPU."""

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
    f1 = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    from core.inference import diffusion_flux_convrot_fused as fused
    from core.inference import diffusion_int8_gemm as gemm

    fused.uninstall()
    monkeypatch.delenv(fused.FLUX_CONVROT_FUSED_ENV, raising = False)
    yield torch, f1, fused, gemm
    fused.uninstall()


def _force_on(monkeypatch, torch, fused, gemm, calls):
    """Every Linear counts as a swapped rotated int8 Linear; the int8 pieces run the plain modules."""
    import torch.nn.functional as F

    monkeypatch.setattr(fused, "_rotated", lambda lin: True)

    def shared(x, linears):
        calls["shared"] += 1
        return [m(x) for m in linears]

    def gelu(down, prefix, gate):
        calls["gelu"] += 1
        h = F.gelu(gate, approximate = "tanh")
        if prefix is not None:
            assert prefix.dim() == 4 and prefix.stride(-1) == 1
            h = torch.cat([prefix.flatten(2, 3), h], dim = -1)
        return down(h)

    monkeypatch.setattr(gemm, "shared_rotq_linears", shared)
    monkeypatch.setattr(gemm, "gelu_linear", gelu)


def _tiny(torch, f1):
    torch.manual_seed(0)
    return f1.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 8,
        num_layers = 1,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        pooled_projection_dim = 12,
        axes_dims_rope = (4, 6, 6),
    ).eval()


def _inputs(torch, dtype = None):
    g = torch.Generator().manual_seed(3)
    b, s, t = 2, 12, 5
    x = {
        "hidden_states": torch.randn(b, s, 8, generator = g),
        "encoder_hidden_states": torch.randn(b, t, 24, generator = g),
        "pooled_projections": torch.randn(b, 12, generator = g),
        "timestep": torch.tensor([0.3, 0.7]),
        "img_ids": torch.randn(s, 3, generator = g),
        "txt_ids": torch.zeros(t, 3),
        "return_dict": False,
    }
    if dtype is not None:
        x = {
            k: v.to(dtype) if hasattr(v, "is_floating_point") and v.is_floating_point() else v
            for k, v in x.items()
        }
    return x


def _install(fused, model):
    done = fused._install_f1_patches()  # noqa: SLF001 - the install minus the int8 / device gates
    assert done == {"f1_projections": True, "f1_ff": True, "f1_single": True}, done
    assert fused._mark_f1(model) == 2 + 2  # noqa: SLF001 - the double block's two FeedForwards, two single blocks


@pytest.mark.parametrize("mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_flux1_patched_model_matches_stock(env, monkeypatch, mode, dtype):
    torch, f1, fused, gemm = env
    dt = getattr(torch, dtype)
    calls = {"shared": 0, "gelu": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    model = _tiny(torch, f1).to(dt)
    x = _inputs(torch, dt)
    with torch.no_grad():
        ref = model(**x)[0]
        _install(fused, model)
        if mode == "compiled":
            torch._dynamo.reset()
            out = torch.compile(model, backend = "eager", fullgraph = True)(**x)[0]
            torch._dynamo.reset()
        else:
            out = model(**x)[0]
    fused.uninstall()
    with torch.no_grad():
        back = model(**x)[0]
    assert torch.equal(ref, out) and torch.equal(ref, back)
    # double block: image and context Q / K / V; single blocks: Q / K / V each
    assert calls["shared"] == 2 + 2
    # double block: ff and ff_context; single blocks: [attention | GELU] into proj_out each
    assert calls["gelu"] == 2 + 2


def test_flux1_arch_patched_single_block_keeps_addcmul(env, monkeypatch):
    """Over Studio's addcmul arch patch the wrapper keeps that gated residual (bitwise the arch forward)."""
    torch, f1, fused, gemm = env
    from core.inference import diffusion_arch_patches as ap

    calls = {"shared": 0, "gelu": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    cls = f1.FluxSingleTransformerBlock
    monkeypatch.setattr(cls, "forward", ap._flux_single_forward)  # noqa: SLF001
    model = _tiny(torch, f1).to(torch.bfloat16)
    x = _inputs(torch, torch.bfloat16)
    with torch.no_grad():
        ref = model(**x)[0]
        _install(fused, model)
        assert cls.forward._unsloth_stock is ap._flux_single_forward  # noqa: SLF001
        out = model(**x)[0]
    from core.inference import diffusion_int8_fused as i8

    assert i8._flux_single_class_is_arch_patched(model.single_transformer_blocks[0])  # noqa: SLF001
    fused.uninstall()
    assert cls.forward is ap._flux_single_forward  # noqa: SLF001
    assert torch.equal(ref, out)
    assert calls["gelu"] == 2 + 2


def test_unmarked_or_unrotated_modules_keep_the_stock_path(env, monkeypatch):
    """The GELU patches sit on shared diffusers classes: an unmarked FeedForward (another model) or an unrotated down
    projection never reaches the fused path."""
    torch, f1, fused, gemm = env
    from diffusers.models.attention import FeedForward

    calls = {"shared": 0, "gelu": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    model = _tiny(torch, f1)
    _install(fused, model)
    other = FeedForward(16, 16, activation_fn = "gelu-approximate").eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        ref = torch.nn.Sequential(*other.net)(x)
        assert torch.equal(other(x), ref)
    assert calls["gelu"] == 0
    monkeypatch.setattr(fused, "_rotated", lambda lin: False)
    with torch.no_grad():
        model(**_inputs(torch))
    assert calls == {"shared": 0, "gelu": 0}


def test_install_scopes_to_flux1_with_rotated_linears(env, monkeypatch):
    torch, f1, fused, _gemm = env
    model = _tiny(torch, f1)
    # no ConvRot Linears: nothing installed, nothing marked
    res = fused.install(model)
    assert res["rotated"] == 0 and not fused.is_installed()
    assert not any(fused._F1_MARK in m.__dict__ for m in model.modules())  # noqa: SLF001
    monkeypatch.setattr(fused, "_rotated_linears", lambda t: 7)
    res = fused.install(model)
    assert res["patches"] == {"f1_projections": True, "f1_ff": True, "f1_single": True}
    assert res["marked"] == 4 and fused.is_installed()
    fused.uninstall()
    assert not fused.is_installed()
    monkeypatch.setenv(fused.FLUX_CONVROT_FUSED_ENV, "0")
    assert fused.install(model)["patches"] == {}


def test_patch_spec_covers_flux1():
    pytest.importorskip("torch")
    from core.inference.diffusion_autoopt import patches

    spec = next(s for s in patches.specs() if s.name == "flux_convrot_fused")
    assert "FluxTransformer2DModel" in spec.classes


def test_gelu_reference_and_producer_reference(env):
    torch, _f1, _fused, _gemm = env
    import torch.nn.functional as F
    from core.inference import diffusion_convrot_quant as cq

    g = torch.randn(6, 512).to(torch.bfloat16)
    pre = torch.randn(2, 3, 2, 128).to(torch.bfloat16)
    want = torch.cat([pre.reshape(6, -1), F.gelu(g, approximate = "tanh")], dim = -1)
    assert torch.equal(cq.producer_reference(pre, g, g, 2), want)
    assert torch.equal(cq.producer_reference(None, g, None, 2), F.gelu(g, approximate = "tanh"))


def test_gelu_switch_and_arch_gate(monkeypatch):
    pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    monkeypatch.delenv(cq.GELU_FUSED_ENV, raising = False)
    assert cq.gelu_arch_ok((12, 0)) and cq.gelu_arch_ok((8, 9))
    assert not cq.gelu_arch_ok((8, 0)) and not cq.gelu_arch_ok(None)
    monkeypatch.setenv(cq.GELU_FUSED_ENV, "1")
    assert cq.gelu_arch_ok((8, 0))
    monkeypatch.setenv(cq.GELU_FUSED_ENV, "0")
    assert not cq.gelu_arch_ok((12, 0))
    # off: arming refuses before any probe, and a call returns None (the caller keeps the stock path)
    monkeypatch.setitem(cq._H16_DEVICE, 0, "ieee")  # noqa: SLF001
    assert cq._arm_gelu(0) is False and not cq.gelu_armed(0)  # noqa: SLF001
    monkeypatch.delenv(cq.GELU_FUSED_ENV, raising = False)
    monkeypatch.setenv(cq.INT8_ROTQUANT_PRODUCER_ENV, "0")
    assert cq._arm_gelu(0) is False  # noqa: SLF001


def test_gelu_op_fake_shapes(env):
    torch, _f1, _fused, _gemm = env
    from torch._subclasses.fake_tensor import FakeTensorMode
    from core.inference import diffusion_convrot_quant as cq

    op = cq._gelu_op()  # noqa: SLF001
    assert op is not None
    with FakeTensorMode():
        gate = torch.empty(10, 512, dtype = torch.bfloat16)
        pre = torch.empty(2, 5, 4, 64, dtype = torch.bfloat16)
        q, s = op(pre, gate, 256, True)
        assert q.shape == (10, 512 + 256) and q.dtype == torch.int8 and s.shape == (10,)
        q, s = op(None, gate, 256, False)
        assert q.shape == (10, 512) and s.shape == (10,)


def test_gelu_op_cpu_body_is_the_stock_reference(env):
    """No verdict (CPU): the op body is the stock GELU, concat, rotation and act quant, bit for bit."""
    torch, _f1, _fused, _gemm = env
    pytest.importorskip("torchao")
    from core.inference import diffusion_convrot_quant as cq

    g = (torch.randn(9, 512) * 3).to(torch.bfloat16)
    pre = torch.randn(3, 3, 2, 128).to(torch.bfloat16)
    before = cq.gelu_call_count()
    q, s = cq._producer_run(pre, g, g, 256, True, 2)  # noqa: SLF001
    rq, rs = cq.rotquant_reference(cq.producer_reference(pre, g, g, 2).contiguous(), 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)
    assert cq.gelu_call_count() == before + 1


def test_rotq_h16x_gelu_source_contract():
    """ACT 2 reads only the gate (no value load) and the CUDA helper pins its rounding; ROCm gets the portable one."""
    import inspect

    from core.inference import diffusion_convrot_quant as cq

    src = inspect.getsource(cq._kernels)  # noqa: SLF001
    assert (
        "if ACT == 2:\n                prod = _gelu_bf16(g)\n            else:\n                v = tl.load("
        in src
    )
    assert "if _hip_backend():" in src and "libdevice.fma_rn(0.044715, cube, y)" in src
    assert "tl.extra.cuda" not in src


def test_gelu_entry_keeps_the_stock_path_below_the_card_min_rows(monkeypatch):
    """Short activations (FLUX.1's text-stream MLP, small renders) measured slower fused: the caller keeps stock."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    calls = []
    monkeypatch.setattr(cq, "_GELU_HANDLE", lambda *a: calls.append(a) or ("q", "s"))
    monkeypatch.setitem(cq._GELU_DEVICE, None, True)  # noqa: SLF001 - CPU tensors carry device index None
    monkeypatch.setitem(cq._GELU_DEVICE_MIN_ROWS, None, 2048)  # noqa: SLF001
    monkeypatch.setattr(cq, "producer_layout_ok", lambda *a: True)
    small, big = (
        torch.empty(256, 512, dtype = torch.bfloat16),
        torch.empty(4096, 512, dtype = torch.bfloat16),
    )
    assert cq.convrot_gelu_act_quant(None, small, 256, True) is None and not calls
    assert cq.convrot_gelu_act_quant(None, big, 256, True) == ("q", "s") and len(calls) == 1
    assert cq._GELU_MIN_ROWS[(12, 0)] == 2048 and cq._GELU_MIN_ROWS[(8, 9)] == 1024  # noqa: SLF001
