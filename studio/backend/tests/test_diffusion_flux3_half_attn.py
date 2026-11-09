# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 float16 attention on fp16-only GPUs (float32 compute): scaling, gating, kill switch, registry."""

from __future__ import annotations

import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_flux3_half_attn as ha  # noqa: E402

MOD = ha._MODULE


def _sdpa(
    query,
    key,
    value,
    attn_mask = None,
    **kwargs,
):
    # diffusers' dispatch layout: (batch, seq, heads, head_dim)
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask = attn_mask
    )
    return out.transpose(1, 2)


_sdpa.__name__ = ha._ATTR


class Flux3Attention(torch.nn.Module):
    def __init__(
        self,
        head_dim = 128,
        wmax = 6.8125,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.norm_q = torch.nn.RMSNorm(head_dim)
        self.norm_k = torch.nn.RMSNorm(head_dim)
        with torch.no_grad():
            self.norm_q.weight.fill_(1.0)
            self.norm_q.weight[3] = wmax
            self.norm_k.weight.fill_(0.5)


class Transformer(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Flux3Attention(**kw), Flux3Attention()])


Transformer.__module__ = MOD


@pytest.fixture
def fake_module(monkeypatch):
    mod = types.ModuleType(MOD)
    mod.dispatch_attention_fn = _sdpa
    monkeypatch.setitem(sys.modules, MOD, mod)
    monkeypatch.delenv(ha.FLUX3_FP16_ATTN_ENV, raising = False)
    monkeypatch.setattr(ha, "fp16_only_tensor_core_gpu", lambda device = "cuda": True)
    yield mod
    ha.uninstall()


def _qkv(
    v_scale,
    seed = 0,
    b = 1,
    s = 64,
    h = 4,
    d = 128,
):
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.rms_norm(torch.randn(b, s, h, d, generator = g), (d,)) * 3
    k = torch.nn.functional.rms_norm(torch.randn(b, s, h, d, generator = g), (d,)) * 3
    v = torch.randn(b, s, h, d, generator = g) * v_scale
    # one head far louder than the rest: the per-head scale keeps the quiet heads' precision
    v[:, :, 0] *= 1000.0
    return q, k, v


@pytest.mark.parametrize("v_scale", (1e-6, 1.0, 1e3, 1e7))
def test_scaled_half_attention_matches_float32(monkeypatch, fake_module, v_scale):
    q, k, v = _qkv(v_scale)
    want = _sdpa(q.double(), k.double(), v.double())
    ha._STOCK[ha._ATTR] = _sdpa
    monkeypatch.setattr(ha, "_eligible", lambda *a: True)
    got = ha._half_dispatch(q, k, v, attn_mask = None)
    assert got.dtype == torch.float32 and torch.isfinite(got).all()
    for head in range(q.shape[2]):
        ref = want[:, :, head]
        err = (got[:, :, head].double() - ref).abs().max() / ref.abs().max()
        assert err < 5e-3, (head, float(err))
    # an unscaled float16 cast would overflow (or flush) where the scaled one does not
    if v_scale >= 1e3:
        assert not torch.isfinite(v.half()).all()


def test_ineligible_calls_run_stock(fake_module):
    calls = []

    def stock(
        query,
        key,
        value,
        attn_mask = None,
        **kwargs,
    ):
        calls.append(query.dtype)
        return value

    stock.__name__ = ha._ATTR
    ha._STOCK[ha._ATTR] = stock
    q, k, v = _qkv(1.0)
    assert ha._half_dispatch(q, k, v) is v
    assert ha._half_dispatch(q.half(), k.half(), v.half()) is not None
    assert calls == [torch.float32, torch.float16]


def test_qk_bound_reads_the_norm_weights():
    assert math.isclose(ha.qk_bound(Transformer()), math.sqrt(128) * 6.8125)
    assert ha.qk_bound(torch.nn.Linear(2, 2)) == math.inf
    t = Transformer()
    t.blocks[0].norm_k = torch.nn.LayerNorm(128)
    assert ha.qk_bound(t) == math.inf


def test_install_gates_and_uninstall_restores(monkeypatch, fake_module):
    pipe = types.SimpleNamespace(transformer = Transformer())
    assert ha.install_for_pipe(pipe, "cuda", "default") is True
    assert fake_module.dispatch_attention_fn is ha._half_dispatch and ha.is_installed()
    assert ha.install_for_pipe(pipe, "cuda", "default") is True  # idempotent
    ha.uninstall()
    assert fake_module.dispatch_attention_fn is _sdpa and not ha.is_installed()
    # an explicit speed off is exact: stock; an unset request that resolved to off (dense loads) still engages
    assert ha.install_for_pipe(pipe, "cuda", "off") is False
    assert ha.install_for_pipe(pipe, "cuda", "off", explicit_off = True) is False
    assert ha.install_for_pipe(pipe, "cuda", "default", explicit_off = True) is False
    assert fake_module.dispatch_attention_fn is _sdpa
    assert ha.install_for_pipe(pipe, "cuda", "off", explicit_off = False) is True
    ha.uninstall()
    # a norm weight that could push Q / K near the float16 range: stock
    loud = types.SimpleNamespace(transformer = Transformer(wmax = 2000.0))
    assert ha.install_for_pipe(loud, "cuda", "default") is False
    assert (
        ha.install_for_pipe(
            types.SimpleNamespace(transformer = torch.nn.Linear(2, 2)), "cuda", "default"
        )
        is False
    )
    monkeypatch.setattr(ha, "fp16_only_tensor_core_gpu", lambda device = "cuda": False)
    assert ha.install_for_pipe(pipe, "cuda", "default") is False
    assert fake_module.dispatch_attention_fn is _sdpa


def test_kill_switch(monkeypatch, fake_module):
    monkeypatch.setenv(ha.FLUX3_FP16_ATTN_ENV, "0")
    pipe = types.SimpleNamespace(transformer = Transformer())
    assert ha.install_for_pipe(pipe, "cuda", "default") is False
    assert fake_module.dispatch_attention_fn is _sdpa


def test_unexpected_symbol_is_left_alone(fake_module):
    def other(*a, **k):
        return None

    fake_module.dispatch_attention_fn = other
    assert (
        ha.install_for_pipe(types.SimpleNamespace(transformer = Transformer()), "cuda", "default")
        is False
    )
    assert fake_module.dispatch_attention_fn is other


@pytest.mark.parametrize(
    "cap, want",
    [
        ((6, 1), False),
        ((7, 0), True),
        ((7, 5), True),
        ((8, 0), False),
        ((8, 9), False),
        ((12, 0), False),
    ],
)
def test_only_volta_and_turing_qualify(monkeypatch, cap, want):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: cap)
    import core.inference.rocm_bf16 as rb

    monkeypatch.setattr(rb, "is_rocm_torch", lambda t: False)
    assert ha.fp16_only_tensor_core_gpu("cuda") is want
    assert ha.fp16_only_tensor_core_gpu("cpu") is False


def test_registered_as_a_load_patch_that_reads_explicit_off(monkeypatch):
    from core.inference.diffusion_autoopt import patches

    spec = {s.name: s for s in patches.specs(patches.STAGE_LOAD)}["flux3_fp16_attn"]
    assert (
        not spec.speed_only and spec.env == ha.FLUX3_FP16_ATTN_ENV and spec.backends == ("image",)
    )
    seen = {}
    monkeypatch.setattr(ha, "install_for_pipe", lambda *a, **k: seen.update(k) or True)
    assert (
        spec.install(patches.PatchContext(pipe = object(), speed_mode = "off", explicit_off = False))
        is True
    )
    assert seen == {"explicit_off": False}


def test_real_flux3_transformer_engages_and_matches_float32(monkeypatch):
    """diffusers' own FLUX.3 transformer: the bound is read from its Flux3Attention RMSNorms, every attention call of a
    forward takes the float16 path, and the output stays at float16 rounding of the float32 forward."""
    import inspect

    fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    monkeypatch.delenv(ha.FLUX3_FP16_ATTN_ENV, raising = False)
    monkeypatch.setattr(ha, "fp16_only_tensor_core_gpu", lambda device = "cuda": True)
    # CPU stands in for the CUDA device: float32 inputs and no mask, as on the GPU
    monkeypatch.setattr(ha, "_eligible", lambda q, k, v, m: q.dtype == torch.float32 and m is None)
    torch.manual_seed(0)
    model = fmod.Flux3Transformer2DModel(
        in_channels = 16,
        cond_in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 32,
        num_attention_heads = 4,
        joint_attention_dim = 32,
        timestep_channels = 32,
        axes_dims_rope = (8, 8, 8, 8),
    ).eval()
    g = torch.Generator().manual_seed(1)
    kw = {
        "hidden_states": torch.randn(1, 64, 16, generator = g),
        "encoder_hidden_states": torch.randn(1, 9, 32, generator = g) * 50,
        "timestep": torch.tensor([0.7]),
        "img_ids": torch.randint(0, 16, (64, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 16, (9, 4), generator = g).float(),
        "return_dict": False,
    }
    if "target_timestep" in inspect.signature(model.forward).parameters and model.config.get(
        "flow_map", False
    ):
        kw["target_timestep"] = torch.tensor([0.5])
    stock = fmod.dispatch_attention_fn
    stock_calls = [0]

    def counted(*a, **k):
        stock_calls[0] += 1
        return stock(*a, **k)

    monkeypatch.setattr(fmod, "dispatch_attention_fn", counted)
    with torch.no_grad():
        want = model(**kw)[0]
    monkeypatch.setattr(fmod, "dispatch_attention_fn", stock)
    assert math.isfinite(ha.qk_bound(model)) and ha.qk_bound(model) < ha.QK_BOUND
    before = ha.call_count()
    try:
        assert (
            ha.install_for_pipe(types.SimpleNamespace(transformer = model), "cuda", "default") is True
        )
        with torch.no_grad():
            got = model(**kw)[0]
    finally:
        ha.uninstall()
    assert ha.call_count() - before == stock_calls[0] > 0
    assert sys.modules[MOD].dispatch_attention_fn is ha._ORIGINAL[0]
    err = (got - want).abs().max() / want.abs().max()
    assert torch.isfinite(got).all() and err < 1e-2, float(err)
