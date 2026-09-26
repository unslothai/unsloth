# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lumina-2 / HiDream-I1 declare no ``_repeated_blocks``; Studio supplies them so the denoiser compiles regionally."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_regional_compile as rc
from core.inference import diffusion_speed as ds


def _tiny_lumina():
    return diffusers.Lumina2Transformer2DModel(
        sample_size = 8,
        patch_size = 2,
        in_channels = 4,
        hidden_size = 32,
        num_layers = 2,
        num_refiner_layers = 1,
        num_attention_heads = 2,
        num_kv_heads = 1,
        multiple_of = 16,
        norm_eps = 1e-5,
        scaling_factor = 1.0,
        axes_dim_rope = (4, 4, 8),
        axes_lens = (32, 32, 32),
        cap_feat_dim = 16,
    ).eval()


def _tiny_hidream():
    return diffusers.HiDreamImageTransformer2DModel(
        patch_size = 2,
        in_channels = 4,
        out_channels = 4,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 8,
        num_attention_heads = 2,
        caption_channels = [8, 8],
        text_emb_dim = 8,
        num_routed_experts = 4,
        num_activated_experts = 2,
        axes_dims_rope = (4, 2, 2),
        max_resolution = (8, 8),
        llama_layers = [0, 1, 2, 3],
    ).eval()


def _hidream_inputs(batch = 2):
    g = torch.Generator().manual_seed(0)
    return dict(
        hidden_states = torch.randn(batch, 4, 8, 8, generator = g),
        timesteps = torch.tensor([500] * batch),
        encoder_hidden_states_t5 = torch.randn(batch, 5, 8, generator = g),
        encoder_hidden_states_llama3 = torch.randn(4, batch, 7, 8, generator = g),
        pooled_embeds = torch.randn(batch, 8, generator = g),
        return_dict = False,
    )


@pytest.mark.parametrize("build", [_tiny_lumina, _tiny_hidream])
def test_released_class_still_declares_no_repeated_blocks(build):
    # Drift guard: once Diffusers declares them, ensure_repeated_blocks returns theirs and changes nothing.
    model = build()
    declared = list(type(model)._repeated_blocks or [])
    if declared:
        assert list(rc.ensure_repeated_blocks(model)) == declared
        return
    with pytest.raises(ValueError):
        model.compile_repeated_blocks()


@pytest.mark.parametrize(
    "build, blocks",
    [
        (_tiny_lumina, ["Lumina2TransformerBlock"]),
        (_tiny_hidream, ["HiDreamImageTransformerBlock", "HiDreamImageSingleTransformerBlock"]),
    ],
)
def test_supplied_blocks_make_compile_repeated_blocks_run(build, blocks):
    model = build()
    assert list(rc.ensure_repeated_blocks(model)) == blocks
    assert model._repeated_blocks == blocks and not type(model)._repeated_blocks
    model.compile_repeated_blocks(fullgraph = True, dynamic = True)
    compiled = [m for m in model.modules() if type(m).__name__ in blocks]
    assert compiled and all(getattr(m, "_compiled_call_impl", None) is not None for m in compiled)


def test_unverified_class_with_empty_blocks_is_left_alone():
    model = _tiny_lumina()
    model.__class__ = type("SomeOtherTransformer", (type(model),), {})
    assert rc.ensure_repeated_blocks(model) == ()
    assert "_repeated_blocks" not in vars(model)


def test_declared_blocks_are_returned_unchanged():
    model = _tiny_lumina()
    model._repeated_blocks = ["Custom"]
    assert rc.ensure_repeated_blocks(model) == ("Custom",)


def test_discovery_matches_the_verified_names():
    assert rc.discover_repeated_blocks(_tiny_lumina()) == rc.verified_repeated_blocks(
        "Lumina2Transformer2DModel"
    )
    assert rc.discover_repeated_blocks(_tiny_hidream()) == rc.verified_repeated_blocks(
        "HiDreamImageTransformer2DModel"
    )


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
def test_hidream_dense_experts_match_the_released_loop(dtype, tol):
    torch.manual_seed(0)
    model = _tiny_hidream().to(dtype)
    inputs = {
        k: (v.to(dtype) if v.is_floating_point() else v)
        for k, v in _hidream_inputs().items()
        if k != "return_dict"
    }
    with torch.no_grad():
        ref = model(**inputs, return_dict = False)[0]
        assert rc.install_traceable_moe(model) == 4
        out = model(**inputs, return_dict = False)[0]
    torch.testing.assert_close(out, ref, rtol = tol, atol = tol)


def test_hidream_unpicked_expert_overflow_adds_zero_not_nan():
    model = _tiny_hidream()
    moe = next(m for m in model.modules() if type(m).__name__ == "MOEFeedForwardSwiGLU")
    rc.install_traceable_moe(model)
    x = torch.randn(6, 16)
    idx = torch.tensor([[0, 1]] * 6).view(-1)
    weights = torch.full((12, 1), 0.5)
    moe.experts[3].forward = lambda t: torch.full_like(t, float("inf"))  # never picked
    with torch.no_grad():
        out = moe.moe_infer(x, idx, weights)
    assert torch.isfinite(out).all()


def _block_calls(model, inputs, name):
    captured = {}

    def grab(mod, args, kwargs):
        captured.setdefault(type(mod).__name__, (mod, args, kwargs))

    hooks = [
        m.register_forward_pre_hook(grab, with_kwargs = True)
        for m in model.modules()
        if type(m).__name__ == name
    ]
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return captured[name]


def _assert_one_graph(mod, args, kwargs):
    torch._dynamo.reset()
    explained = torch._dynamo.explain(mod)(*args, **kwargs)
    assert explained.graph_break_count == 0, [str(r.reason) for r in explained.break_reasons]
    assert explained.graph_count == 1


@pytest.mark.parametrize(
    "name", ["HiDreamImageTransformerBlock", "HiDreamImageSingleTransformerBlock"]
)
def test_hidream_blocks_trace_without_graph_breaks(name):
    model = _tiny_hidream()
    rc.ensure_repeated_blocks(model)
    _assert_one_graph(*_block_calls(model, _hidream_inputs(), name))


def test_hidream_released_moe_loop_breaks_the_graph():
    # Negative control for the test above: without the traceable experts the block does not trace whole.
    model = _tiny_hidream()
    mod, args, kwargs = _block_calls(model, _hidream_inputs(), "HiDreamImageTransformerBlock")
    torch._dynamo.reset()
    explained = torch._dynamo.explain(mod)(*args, **kwargs)
    assert explained.graph_break_count > 0


def test_lumina_block_traces_without_graph_breaks():
    model = _tiny_lumina()
    rc.ensure_repeated_blocks(model)
    g = torch.Generator().manual_seed(0)
    inputs = dict(
        hidden_states = torch.randn(1, 4, 8, 8, generator = g),
        timestep = torch.tensor([0.5]),
        encoder_hidden_states = torch.randn(1, 6, 16, generator = g),
        encoder_attention_mask = torch.ones(1, 6, dtype = torch.bool),
        return_dict = False,
    )
    _assert_one_graph(*_block_calls(model, inputs, "Lumina2TransformerBlock"))


def test_family_compiles_regionally_trusts_verified_classes(monkeypatch):
    fake = types.ModuleType("diffusers")
    for name in ("Lumina2Transformer2DModel", "HiDreamImageTransformer2DModel", "OtherEmpty"):
        setattr(fake, name, type(name, (), {"_repeated_blocks": []}))
    monkeypatch.setitem(sys.modules, "diffusers", fake)

    def fam(cls):
        return types.SimpleNamespace(transformer_class = cls, denoiser_attr = "transformer")

    assert ds.family_compiles_regionally(fam("Lumina2Transformer2DModel")) is True
    assert ds.family_compiles_regionally(fam("HiDreamImageTransformer2DModel")) is True
    assert ds.family_compiles_regionally(fam("OtherEmpty")) is False


def test_studio_compile_path_engages_on_lumina(monkeypatch):
    monkeypatch.setattr(ds, "guard_compiled_blocks", lambda *a, **k: None)
    pipe = types.SimpleNamespace(transformer = _tiny_lumina())
    assert ds._compile_repeated_blocks(pipe, None) is True
    blocks = [
        m for m in pipe.transformer.modules() if type(m).__name__ == "Lumina2TransformerBlock"
    ]
    assert blocks and all(getattr(m, "_compiled_call_impl", None) is not None for m in blocks)
