# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp16-card native quant of bf16 Linears as stored matches the float32-widened build."""

from __future__ import annotations

import ast
import copy
import inspect
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq  # noqa: E402
import core.inference.diffusion_transformer_quant as tq  # noqa: E402
from core.inference import diffusion_manual_cast as mc  # noqa: E402

KILL_SWITCH = "UNSLOTH_DIFFUSION_FP16_CARD_QUANT_STORAGE"


@pytest.fixture
def turing(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (7, 5))
    for env in (KILL_SWITCH, "UNSLOTH_DIFFUSION_FP16_CARD_INT8", "UNSLOTH_NATIVE_INT8_ROT"):
        monkeypatch.delenv(env, raising = False)
    # CPU: no ConvRot (its rotation is a CUDA W8A8 detail), so both builds are plain per-row int8
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ROT", "0")


def _tiny_flux3():
    fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
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
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16).to(torch.float32)
    g = torch.Generator().manual_seed(1)
    kw = {
        "hidden_states": torch.randn(1, 64, 16, generator = g),
        "encoder_hidden_states": torch.randn(1, 9, 32, generator = g),
        "timestep": torch.tensor([0.7]),
        "img_ids": torch.randint(0, 16, (64, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 16, (9, 4), generator = g).float(),
        "return_dict": False,
    }
    if "target_timestep" in inspect.signature(model.forward).parameters and model.config.get(
        "flow_map", False
    ):
        kw["target_timestep"] = torch.tensor([0.5])
    return model, kw


def _stored_bf16(model):
    keep = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not any(k in name for k in keep):
                p.data = p.data.to(torch.bfloat16)
    mc.apply_manual_cast_(model, torch.float32)
    return model


def _quantise(model, scheme):
    target = types.SimpleNamespace(device = "cuda", dtype = torch.float32)
    return tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        target,
        mode = scheme,
        family = "flux.3-klein",
        min_features = 16,
    )


def _native(model):
    return {n: m for n, m in model.named_modules() if nq.is_native_linear(m)}


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_manual_cast_source_builds_the_same_layers_as_the_widened_load(turing, scheme):
    if scheme == "fp8" and not hasattr(tq, "fp16_card_fp8_scheme"):
        pytest.skip("fp8 on Turing is a separate change")
    widened, kw = _tiny_flux3()
    stored = _stored_bf16(copy.deepcopy(widened))
    assert _quantise(widened, scheme) == scheme
    assert _quantise(stored, scheme) == scheme
    a, b = _native(widened), _native(stored)
    assert a and set(a) == set(b)
    for name in a:
        assert b[name].compute_dtype == torch.float32 == a[name].compute_dtype, name
        assert torch.equal(a[name].weight_q, b[name].weight_q), name
        assert torch.equal(a[name].weight_scale, b[name].weight_scale), name
        if a[name].bias is not None:
            assert b[name].bias.dtype == torch.float32 and torch.equal(
                a[name].bias, b[name].bias
            ), name
    dense = [m for m in stored.modules() if type(m).__name__ == "ManualCastLinear"]
    assert dense and all(m.weight.dtype == torch.bfloat16 for m in dense)
    with torch.no_grad():
        want = widened(**kw)[0]
        got = stored(**kw)[0]
    assert got.dtype == torch.float32 and torch.equal(got, want)


def test_filter_counts_a_manual_cast_linear_by_its_compute_dtype():
    # the first floating owner is converted up front (module.dtype answers float32); the next one keeps bf16
    wrapper = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64)).to(
        torch.bfloat16
    )
    mc.apply_manual_cast_(wrapper, torch.float32)
    cast = wrapper[1]
    assert cast.weight.dtype == torch.bfloat16
    assert tq.linear_compute_dtype(cast) == torch.float32
    assert tq.make_filter_fn(16, require_dtype = torch.float32)(cast, "x")
    assert not tq.make_filter_fn(16, require_dtype = torch.float32)(
        torch.nn.Linear(64, 64).to(torch.bfloat16), "x"
    )
    assert tq.make_filter_fn(16, require_dtype = torch.float32)(torch.nn.Linear(64, 64), "x")


def test_the_loader_keeps_bf16_storage_only_for_the_fp16_card_route():
    import core.inference.diffusion as d

    src = ast.unparse(ast.parse(inspect.getsource(d.DiffusionBackend)))
    i = src.index("transformer_storage=")
    gate = src[
        i : src.index("load_flux3_pipeline", i) if "load_flux3_pipeline" in src[i:] else i + 600
    ]
    assert "fp16_card_quant_storage_enabled()" in gate
    assert "fp16_card_int8_host(target)" in gate and "native_quant_scheme(" in gate
    assert "native_offload_scheme is None" in gate


@pytest.mark.parametrize("raw", ["0", "off", "false", "no"])
def test_kill_switch(monkeypatch, raw):
    monkeypatch.setenv(KILL_SWITCH, raw)
    assert not tq.fp16_card_quant_storage_enabled()
    monkeypatch.delenv(KILL_SWITCH)
    assert tq.fp16_card_quant_storage_enabled()
    assert tq.FP16_CARD_QUANT_STORAGE_ENV == KILL_SWITCH
