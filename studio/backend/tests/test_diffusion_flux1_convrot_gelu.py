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
    # another test module may have imported the arch patches against a stand-in torch
    monkeypatch.setattr(ap, "torch", torch)
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
    """GELU patches hit shared diffusers classes; unmarked or unrotated modules stay stock."""
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
    """Short activations (FLUX.1's text-stream MLP, small renders) and, on RTX PRO 6000, the prefix-free FF measured
    slower fused: the caller keeps stock there."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    calls = []
    monkeypatch.setattr(cq, "_GELU_HANDLE", lambda *a: calls.append(a) or ("q", "s"))
    monkeypatch.setitem(cq._GELU_DEVICE, None, True)  # noqa: SLF001 - CPU tensors carry device index None
    monkeypatch.setattr(cq, "producer_layout_ok", lambda *a: True)
    small, big = (
        torch.empty(256, 512, dtype = torch.bfloat16),
        torch.empty(4096, 512, dtype = torch.bfloat16),
    )
    pre = torch.empty(1, 4096, 2, 128, dtype = torch.bfloat16)
    monkeypatch.setitem(cq._GELU_DEVICE_MIN_ROWS, None, cq._GELU_MIN_ROWS[(8, 9)])  # noqa: SLF001
    assert cq.convrot_gelu_act_quant(None, small, 256, True) is None and not calls
    assert cq.convrot_gelu_act_quant(None, big, 256, True) == ("q", "s") and len(calls) == 1
    monkeypatch.setitem(cq._GELU_DEVICE_MIN_ROWS, None, cq._GELU_MIN_ROWS[(12, 0)])  # noqa: SLF001
    assert cq.convrot_gelu_act_quant(None, big, 256, True) is None and len(calls) == 1
    assert cq.convrot_gelu_act_quant(pre, big, 256, True) == ("q", "s") and len(calls) == 2
    assert cq.convrot_gelu_act_quant(pre[:, :256], small, 256, True) is None


def test_gelu_verdict_is_persisted_and_a_later_load_skips_the_probe(monkeypatch):
    """The probe's cold compile (4.8 s RTX PRO 6000, 16 s L4) is paid once per card and source: the verdict joins the
    ConvRot verdict file and a later process adopts it without probing."""
    pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    store = {}
    probes = []
    monkeypatch.setattr(cq, "_verdict_load", lambda index: dict(store) if store else None)
    monkeypatch.setattr(cq, "_verdict_store", lambda index, upd: store.update(upd))
    monkeypatch.setattr(cq, "_gelu_probe", lambda index: probes.append(index) or True)
    monkeypatch.setattr(cq, "_device_cap", lambda index: (12, 0))
    monkeypatch.setattr(cq, "_GELU_DEVICE", {})
    monkeypatch.setattr(cq, "_GELU_OFF", set())
    monkeypatch.setitem(cq._H16_DEVICE, 5, "ieee")  # noqa: SLF001
    monkeypatch.setitem(cq._ROTQ_DEVICE, 5, (128, 32, 8, 4))  # noqa: SLF001
    monkeypatch.delenv(cq.GELU_FUSED_ENV, raising = False)
    assert cq.gelu_probe_pending(5)
    assert cq._arm_gelu(5) and probes == [5] and store == {"gelu": True}  # noqa: SLF001
    cq._GELU_DEVICE.clear()  # noqa: SLF001 - a new process
    assert not cq.gelu_probe_pending(5)
    assert cq._arm_gelu(5) and probes == [5]  # noqa: SLF001 - adopted, not probed again
    # a persisted verdict also reaches the device table through the shared adoption path
    cq._GELU_DEVICE.clear()  # noqa: SLF001
    cq._adopt_verdict(5, {"cfg": [128, 32, 8, 4], "h16": "ieee", "gelu": False})  # noqa: SLF001
    assert cq._GELU_DEVICE[5] is False  # noqa: SLF001


def test_flux1_loads_put_the_gelu_probe_in_the_prewarm_pool(monkeypatch):
    torch = pytest.importorskip("torch")
    import types

    import core.inference.diffusion as D
    from core.inference import diffusion_convrot_quant as cq
    from core.inference import diffusion_flux_convrot_fused as fused
    from core.inference import diffusion_int8_gemm as gemm

    seen = []
    monkeypatch.setattr(
        gemm,
        "start_early_probe_prewarm",
        lambda index, rotated, producer, logger = None, gelu = False: seen.append(
            (rotated, producer, gelu)
        )
        or True,
    )
    from core.inference.diffusion_families import detect_family

    cuda = types.SimpleNamespace(device = "cuda", ordinal = 0)
    assert D._kick_early_probe_prewarm(detect_family("black-forest-labs/FLUX.1-schnell"), cuda)
    assert D._kick_early_probe_prewarm(detect_family("black-forest-labs/flux-3-image-klein"), cuda)
    assert seen == [(True, False, True), (True, True, False)]
    # the swap-time warm-up asks for the GELU jobs on a FLUX.1 denoiser whose verdict is pending
    calls = []
    monkeypatch.setattr(gemm, "join_early_probe_prewarm", lambda index = None: None)
    monkeypatch.setattr(gemm, "_EARLY_DONE", {})
    monkeypatch.setattr(gemm, "_gemm_probe_jobs", lambda index: [])
    monkeypatch.setattr(cq, "rotq_probes_pending", lambda index: False)
    monkeypatch.setattr(cq, "producer_probe_pending", lambda index: True)
    monkeypatch.setattr(cq, "gelu_probe_pending", lambda index: True)
    monkeypatch.setattr(cq, "prewarm_probe_kernels", lambda index, **kw: calls.append(kw) or 3)
    flux1 = type("FluxTransformer2DModel", (), {})()
    assert gemm._prewarm_probes(0, flux1, True, None) == 3  # noqa: SLF001
    assert calls[-1]["gelu"] is True and calls[-1]["producer"] is False
    assert fused.gelu_probe_expected(flux1) and not fused.producer_probe_expected(flux1)


def _tiny_chroma(torch):
    torch.manual_seed(0)
    return diffusers_chroma()(
        patch_size = 1,
        in_channels = 8,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        axes_dims_rope = (4, 6, 6),
        approximator_num_channels = 8,
        approximator_hidden_dim = 16,
        approximator_layers = 1,
    ).eval()


def diffusers_chroma():
    import diffusers
    return getattr(diffusers, "ChromaTransformer2DModel", None)


def test_chroma_after_flux1_keeps_its_stock_projections(env, monkeypatch):
    """transformer_flux._get_projections is module-wide and Chroma's attention calls it: only FLUX.1 attentions
    install marked take the shared quant (load order no longer matters)."""
    torch, f1, fused, gemm = env
    if diffusers_chroma() is None:
        pytest.skip("diffusers without Chroma")
    calls = {"shared": 0, "gelu": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    flux = _tiny(torch, f1)
    _install(fused, flux)
    chroma = _tiny_chroma(torch)
    g = torch.Generator().manual_seed(3)
    x = {
        "hidden_states": torch.randn(1, 12, 8, generator = g),
        "encoder_hidden_states": torch.randn(1, 5, 24, generator = g),
        "timestep": torch.tensor([0.4]),
        "img_ids": torch.randn(12, 3, generator = g),
        "txt_ids": torch.zeros(5, 3),
        "return_dict": False,
    }
    with torch.no_grad():
        out = chroma(**x)[0]
    assert calls == {"shared": 0, "gelu": 0}
    fused.uninstall()
    with torch.no_grad():
        assert torch.equal(out, chroma(**x)[0])


def test_unmarked_modules_pay_one_dict_lookup(env):
    """The FLUX.1 class patches see every FeedForward / single block in the process: an unmarked one is rejected on
    its instance __dict__, before any nn.Module attribute lookup (no __getattr__ miss per call)."""
    torch, f1, fused, _gemm = env
    from diffusers.models.attention import FeedForward

    misses = []

    class Counting(FeedForward):
        def __getattr__(self, name):
            misses.append(name)
            return super().__getattr__(name)

    ff = Counting(16, 16, activation_fn = "gelu-approximate").eval()
    misses.clear()
    assert not fused._gelu_ff_ready(ff)  # noqa: SLF001
    assert misses == []
    block = f1.FluxSingleTransformerBlock(32, 2, 16)
    assert not fused._f1_single_ready(block)  # noqa: SLF001


@pytest.mark.parametrize("how", ["hook", "pre_hook", "forward"])
def test_a_hooked_dropout_keeps_the_stock_ff(env, monkeypatch, how):
    """The fused FF path never calls ``net[1]``: a forward / pre-forward hook or an instance ``forward`` on the Dropout
    must keep the stock path so the callback still runs."""
    torch, f1, fused, gemm = env
    calls = {"shared": 0, "gelu": 0}
    _force_on(monkeypatch, torch, fused, gemm, calls)
    model = _tiny(torch, f1)
    _install(fused, model)
    with torch.no_grad():
        model(**_inputs(torch))
    assert calls["gelu"] > 0
    first = calls["gelu"]
    fired = []
    from diffusers.models.attention import FeedForward

    ffs = [m for m in model.modules() if type(m) is FeedForward and fused._gelu_ff_ready(m)]  # noqa: SLF001
    assert ffs
    for ff in ffs:
        drop = ff.net[1]
        if how == "hook":
            drop.register_forward_hook(lambda m, i, o: fired.append(1))
        elif how == "pre_hook":
            drop.register_forward_pre_hook(lambda m, i: fired.append(1))
        else:
            stock = drop.forward
            drop.forward = lambda x, _s = stock: (fired.append(1), _s(x))[1]
        assert not fused._gelu_ff_ready(ff)  # noqa: SLF001
    calls.update(gelu = 0)
    with torch.no_grad():
        model(**_inputs(torch))
    # Only the FF calls leave the fused path (the single-block GELU producer has no Dropout), and every hook runs.
    assert calls["gelu"] == first - len(ffs)
    assert len(fired) == len(ffs)


def test_kill_switches_and_what_they_restore(env, monkeypatch):
    """``UNSLOTH_DIFFUSION_FLUX_CONVROT_FUSED=0`` restores main exactly (no FLUX.1 patch installed, stock functions
    back); ``UNSLOTH_DIFFUSION_FLUX1_GELU_FUSED=0`` / ``..._INT8_ROTQUANT_PRODUCER=0`` only turn the GELU producer off
    (the shared Q / K / V quant stays: bit-identical to main, documented)."""
    torch, f1, fused, _gemm = env
    from core.inference import diffusion_convrot_quant as cq

    stock_proj = f1._get_projections  # noqa: SLF001
    stock_single = f1.FluxSingleTransformerBlock.forward
    model = _tiny(torch, f1)
    monkeypatch.setattr(fused, "_rotated_linears", lambda t: 7)
    assert fused.install(model)["patches"]["f1_projections"]
    assert f1._get_projections is not stock_proj  # noqa: SLF001
    monkeypatch.setenv(fused.FLUX_CONVROT_FUSED_ENV, "0")
    assert fused.install(model)["patches"] == {}
    assert (
        f1._get_projections is stock_proj and f1.FluxSingleTransformerBlock.forward is stock_single
    )  # noqa: SLF001
    assert not fused.is_installed()
    monkeypatch.delenv(fused.FLUX_CONVROT_FUSED_ENV)
    for env_name in (cq.GELU_FUSED_ENV, cq.INT8_ROTQUANT_PRODUCER_ENV):
        monkeypatch.setenv(env_name, "0")
        monkeypatch.setitem(cq._H16_DEVICE, 3, "ieee")  # noqa: SLF001
        monkeypatch.setattr(cq, "_device_cap", lambda index: (12, 0))
        assert cq._arm_gelu(3) is False  # noqa: SLF001 - the GELU producer is off
        assert fused.install(model)["patches"][
            "f1_projections"
        ]  # the shared quant stays, as documented
        monkeypatch.delenv(env_name)


def test_a_failed_gelu_probe_is_logged(monkeypatch):
    pytest.importorskip("torch")
    from core.inference import diffusion_convrot_quant as cq

    msgs = []
    log = type("L", (), {"info": lambda self, fmt, *a: msgs.append(fmt % a)})()
    monkeypatch.setattr(cq, "_verdict_load", lambda index: None)
    monkeypatch.setattr(cq, "_verdict_store", lambda index, upd: None)
    monkeypatch.setattr(cq, "_gelu_probe", lambda index: False)
    monkeypatch.setattr(cq, "_device_cap", lambda index: (8, 9))
    monkeypatch.setattr(cq, "_GELU_DEVICE", {})
    monkeypatch.setattr(cq, "_GELU_OFF", set())
    monkeypatch.setitem(cq._H16_DEVICE, 7, "ieee")  # noqa: SLF001
    monkeypatch.delenv(cq.GELU_FUSED_ENV, raising = False)
    assert cq._arm_gelu(7, log) is False  # noqa: SLF001
    assert len(msgs) == 1 and "failed its probe on cuda:7" in msgs[0]
