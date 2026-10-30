# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image streamed int8 weight-only modulations: scope, cache key, the outlier-column split and the GEMV
module's torch reference path. CPU-only (the Triton kernel itself is checked on a GPU by the bench scripts)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import core.inference.diffusion_int8_gemv as gv
import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import TQ_FP8, TQ_INT8

_MODS = ("img_mod", "txt_mod")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for var in (
        tq.INT8_MODULATION_ENV,
        tq.INT8_MOD_OUTLIERS_ENV,
        tq.INT8_TEXT_STREAM_ENV,
        gv.INT8_GEMV_ENV,
    ):
        monkeypatch.delenv(var, raising = False)


def test_tokens_only_inside_the_streamed_scope():
    assert tq.streamed_weight_only_tokens_for_scheme(TQ_INT8, "qwen-image") == ()
    assert tq.streamed_weight_only_layout(TQ_INT8, "qwen-image") == {}
    with tq.int8_text_stream_scope(True):
        assert tq.streamed_weight_only_tokens_for_scheme(TQ_INT8, "qwen-image") == _MODS
        assert tq.streamed_weight_only_tokens_for_scheme(TQ_FP8, "qwen-image") == ()
        assert tq.streamed_weight_only_tokens_for_scheme(TQ_INT8, "flux.1") == ()
        assert tq.streamed_weight_only_layout(TQ_INT8, "qwen-image") == {
            "weight_only_name_tokens": sorted(_MODS),
            "weight_only_mode": "wo",
            "weight_only_outliers": tq.DEFAULT_MOD_OUTLIERS,
        }
    with tq.int8_text_stream_scope(False):
        assert tq.streamed_weight_only_tokens_for_scheme(TQ_INT8, "qwen-image") == ()


@pytest.mark.parametrize(
    "raw, mode",
    [
        ("", "wo"),
        ("1", "wo"),
        ("on", "wo"),
        ("0", "off"),
        ("off", "off"),
        ("w8a8", "w8a8"),
        ("w8a8rot", "w8a8rot"),
    ],
)
def test_modulation_switch(monkeypatch, raw, mode):
    monkeypatch.setenv(tq.INT8_MODULATION_ENV, raw)
    assert tq.int8_modulation_mode() == mode
    with tq.int8_text_stream_scope(True):
        tokens = tq.streamed_weight_only_tokens_for_scheme(TQ_INT8, "qwen-image")
    assert tokens == (() if mode == "off" else _MODS)


@pytest.mark.parametrize("raw, n", [("", 8), ("0", 0), ("4", 4), ("99", 16), ("junk", 8)])
def test_outlier_switch(monkeypatch, raw, n):
    monkeypatch.setenv(tq.INT8_MOD_OUTLIERS_ENV, raw)
    assert tq.int8_mod_outliers() == n


@pytest.mark.parametrize(
    "raw, keyed", [("", None), ("8", None), ("junk", None), ("0", "0"), ("4", "4"), ("99", "16")]
)
def test_compile_cache_switch_keys_on_the_outlier_count_the_reader_uses(monkeypatch, raw, keyed):
    from core.inference import diffusion_compile_cache as cc

    monkeypatch.setenv(tq.INT8_MOD_OUTLIERS_ENV, raw)
    assert cc.graph_switches().get(tq.INT8_MOD_OUTLIERS_ENV) == keyed
    assert keyed is None or int(keyed) == tq.int8_mod_outliers()


def test_cache_key_carries_the_layout(monkeypatch):
    from core.inference.diffusion_prequant_cache import key_parts

    def parts():
        return key_parts(
            family = "qwen-image",
            transformer_class = "QwenImageTransformer2DModel",
            scheme = TQ_INT8,
            component = "transformer",
            source_fp = "fp",
        )

    resident = parts()
    assert "weight_only_name_tokens" not in resident and "weight_only_outliers" not in resident
    with tq.int8_text_stream_scope(True):
        default = parts()
        monkeypatch.setenv(tq.INT8_MOD_OUTLIERS_ENV, "0")
        no_outliers = parts()
        monkeypatch.setenv(tq.INT8_MODULATION_ENV, "w8a8rot")
        rot = parts()
        monkeypatch.setenv(tq.INT8_MODULATION_ENV, "0")
        off = parts()
    assert default["weight_only_outliers"] == 8 and no_outliers["weight_only_outliers"] == 0
    assert rot["weight_only_mode"] == "w8a8rot"
    assert "weight_only_name_tokens" not in off
    keys = [repr(sorted(p.items())) for p in (resident, default, no_outliers, rot, off)]
    assert len(set(keys)) == len(keys)


def _linear(
    n = 48,
    k = 256,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    lin = nn.Linear(k, n, dtype = torch.bfloat16)
    lin.weight.data = (torch.randn(n, k, generator = g) * 0.02).to(torch.bfloat16)
    lin.bias.data = torch.randn(n, generator = g).to(torch.bfloat16)
    return lin


def _outlier_x(
    m = 2,
    k = 256,
    seed = 1,
    channels = (3, 77),
):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator = g)
    for c in channels:
        x[:, c] = 300.0 + c
    return x.to(torch.bfloat16)


def test_split_is_exact_on_the_dense_weight():
    lin = _linear()
    x = _outlier_x()
    want = torch.nn.functional.linear(x.float(), lin.weight.float(), lin.bias.float())
    moved = gv.split_outliers_(lin, (77, 3))
    assert moved == 2 and gv.is_gemv_linear(lin)
    assert lin.outlier_idx.tolist() == [3, 77]
    assert bool((lin.weight[:, [3, 77]] == 0).all())
    got = lin(x.float().to(torch.bfloat16)).float()
    assert torch.allclose(got, want, rtol = 2e-2, atol = 2e-2)


@pytest.mark.parametrize("bad", [(), None, tuple(range(17)), (999,)])
def test_split_refuses_bad_channel_sets(bad):
    lin = _linear()
    before = lin.weight.clone()
    assert gv.split_outliers_(lin, bad) == 0
    assert type(lin) is nn.Linear and torch.equal(lin.weight, before)


def _quantized(lin):
    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    config = tq._weight_only_config()
    if config is None:
        pytest.skip("torchao without Int8Tensor weight-only")
    quantize_(lin, config, filter_fn = lambda _m, fqn = "": True)
    assert gv.weight_only_parts(lin.weight) is not None
    return lin


def test_outlier_columns_cut_the_weight_only_error():
    x = _outlier_x()
    plain, split = _linear(), _linear()
    truth = x.double() @ plain.weight.double().t() + plain.bias.double()
    gv.split_outliers_(split, (3, 77))
    _quantized(plain)
    _quantized(split)
    gv.install(nn.ModuleDict({"a": plain}), ["a"])

    def err(y):
        return float((y.double() - truth).norm() / truth.norm())

    q, s = gv.weight_only_parts(plain.weight)
    e_plain = err(gv.reference(x, q, s, plain.bias))
    q2, s2 = gv.weight_only_parts(split.weight)
    e_split = err(gv.reference(x, q2, s2, split.bias, split.outlier_idx, split.outlier_weight))
    assert e_split < e_plain / 3
    # the module forward (CPU: torchao weight-only + the dense columns) agrees with the fused reference
    y_mod = split(x)
    y_ref = gv.reference(x, q2, s2, split.bias, split.outlier_idx, split.outlier_weight)
    assert torch.allclose(y_mod.float(), y_ref.float(), rtol = 2e-2, atol = 5e-2)


def test_reference_grouped_scales_match_dequant():
    g = torch.Generator().manual_seed(3)
    q = torch.randint(-127, 128, (32, 128), generator = g, dtype = torch.int8)
    s = torch.rand(32, 4, generator = g) * 1e-2
    x = torch.randn(3, 128, generator = g).to(torch.bfloat16)
    deq = (q.float().reshape(32, 4, 32) * s.reshape(32, 4, 1)).reshape(32, 128)
    want = (x.float() @ deq.t()).to(torch.bfloat16)
    assert torch.allclose(gv.reference(x, q, s, None).float(), want.float(), rtol = 1e-2, atol = 1e-3)


class _Tte(nn.Module):
    def __init__(self, hot = (5, 40)):
        super().__init__()
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps

        self.time_proj = Timesteps(256, flip_sin_to_cos = True, downscale_freq_shift = 0, scale = 1000)
        self.timestep_embedder = TimestepEmbedding(256, 64)
        with torch.no_grad():
            self.timestep_embedder.linear_2.bias[list(hot)] = 50.0


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_text_embed = _Tte()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(64, 96))


def test_timestep_outlier_channels_finds_the_hot_ones():
    pytest.importorskip("diffusers")
    toy = _Toy()
    assert set(gv.timestep_outlier_channels(toy, 2)) == {5, 40}
    assert gv.timestep_outlier_channels(toy, 0) == ()
    assert gv.timestep_outlier_channels(nn.Linear(4, 4), 4) == ()


def test_strict_load_round_trip_restores_the_columns():
    src = nn.ModuleDict({"mod": _linear()})
    x = _outlier_x()
    gv.split_outliers_(src["mod"], (3, 77))
    _quantized(src["mod"])
    want = src["mod"](x)
    state = src.state_dict()
    assert "mod.outlier_weight" in state and "mod.outlier_idx" in state
    dst = nn.ModuleDict({"mod": _linear(seed = 9)})
    _quantized(dst["mod"])
    assert gv.prepare_outlier_buffers(dst, state) == 1
    dst.load_state_dict(state, strict = True, assign = True)
    assert gv.is_gemv_linear(dst["mod"])
    assert torch.equal(dst["mod"](x), want)


def test_install_skips_plain_linears_without_the_kernel(monkeypatch):
    monkeypatch.setenv(gv.INT8_GEMV_ENV, "0")
    plain = _quantized(_linear())
    split = _linear()
    gv.split_outliers_(split, (3,))
    _quantized(split)
    tree = nn.ModuleDict({"plain": plain, "split": split, "dense": _linear()})
    assert gv.install(tree, ["plain", "split", "dense"]) == 1
    assert type(tree["plain"]) is nn.Linear and gv.is_gemv_linear(tree["split"])
    assert type(tree["dense"]) is nn.Linear
    assert gv._OP_HANDLE is None


def test_streamed_quantize_builds_the_layout(monkeypatch):
    pytest.importorskip("diffusers")
    if tq._weight_only_config() is None:
        pytest.skip("torchao without Int8Tensor weight-only")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(64, 512, dtype = torch.bfloat16))
            self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(64, 512, dtype = torch.bfloat16))
            self.other = nn.Linear(64, 512, dtype = torch.bfloat16)

    toy = _Toy()
    toy.time_text_embed.to(torch.bfloat16)
    toy.block = Block()
    temb = torch.nn.functional.silu(
        toy.time_text_embed.timestep_embedder(
            toy.time_text_embed.time_proj(torch.tensor([0.5])).to(torch.bfloat16)
        )
    )
    want = toy.block.img_mod(temb.float().to(torch.bfloat16)).float()
    with tq.int8_text_stream_scope(True):
        names = tq.quantize_streamed_weight_only(toy, TQ_INT8, "qwen-image", min_features = 32)
    assert set(names) == {
        "block.img_mod.1",
        "block.txt_mod.1",
    }  # the float32 toy img_mod is not bf16: left alone
    layout = toy._unsloth_weight_only_layout
    assert (
        layout["weight_only_outliers"] == tq.DEFAULT_MOD_OUTLIERS
        and layout["weight_only_mode"] == "wo"
    )
    assert gv.is_gemv_linear(toy.block.img_mod[1]) and type(toy.block.other) is nn.Linear
    assert {5, 40} <= set(toy.block.img_mod[1].outlier_idx.tolist())
    got = toy.block.img_mod(temb).float()
    assert float((got - want).norm() / want.norm()) < 1e-2
    # resident scope: nothing changes
    fresh = _Toy()
    with tq.int8_text_stream_scope(False):
        assert tq.quantize_streamed_weight_only(fresh, TQ_INT8, "qwen-image", min_features = 32) == ()
    assert type(fresh.img_mod[1]) is nn.Linear
