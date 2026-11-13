# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""generate() decode-step routing to Unsloth's flex decode (unsloth_flex_decode): StaticCache one-token steps
switch to it, prefill and dynamic caches keep their attention, UNSLOTH_FLEX_DECODE=0 restores the old routing."""

import types

import pytest
import torch

import unsloth.models._flex_decode as FD
import unsloth.models.vision as vision
from unsloth.models.vision import _NonFlexGenerate, _generate_routes


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FLEX_ATTENTION_GENERATE", raising = False)
    monkeypatch.delenv(FD.ENV, raising = False)


class _Static:
    is_compileable = True


class _Dynamic:
    is_compileable = False


def _model(model_type, attn):
    from transformers import LlamaConfig

    config = LlamaConfig(
        num_hidden_layers = 1,
        hidden_size = 32,
        intermediate_size = 64,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        vocab_size = 64,
    )
    config.model_type = model_type
    config._attn_implementation = attn
    model = types.SimpleNamespace(config = config, seen = [])

    def prepare_inputs_for_generation(
        input_ids,
        cache_position = None,
        past_key_values = None,
        **kwargs,
    ):
        model.seen.append(config._attn_implementation)
        if cache_position is not None:
            input_ids = input_ids[:, -cache_position.shape[0] :]
        return {"input_ids": input_ids, "cache_position": cache_position}

    model.prepare_inputs_for_generation = prepare_inputs_for_generation
    model.modules = lambda: iter(())
    return model


def _steps(
    model,
    cache,
    n_new = 3,
    prompt = 5,
    batch = 1,
):
    for step in range(n_new):
        n = prompt + step
        ids = torch.zeros(batch, n, dtype = torch.long)
        cp = torch.arange(n) if step == 0 else torch.tensor([n - 1])
        model.prepare_inputs_for_generation(ids, cache_position = cp, past_key_values = cache)


@pytest.fixture
def flex_on(monkeypatch):
    import unsloth.models._decode_graphs as G

    monkeypatch.setattr(FD, "wanted", lambda device: True)
    monkeypatch.setattr(G, "call_plan", lambda: "graph")
    monkeypatch.setattr(
        FD, "routable", lambda model, config: FD._model_type(config) in FD.MODEL_TYPES
    )
    monkeypatch.setattr(vision, "_model_device", lambda model: torch.device("cpu"))


def test_static_decode_steps_use_flex_decode_and_prefill_keeps_sdpa(flex_on):
    model = _model("gemma4_text", "sdpa")
    assert [d for _, _, d in _generate_routes(model)] == [FD.ATTN_KEY]
    with _NonFlexGenerate(model):
        _steps(model, _Static())
    assert model.seen == ["sdpa", FD.ATTN_KEY, FD.ATTN_KEY]
    assert model.config._attn_implementation == "sdpa"


def test_requested_attention_weights_keep_the_model_attention(flex_on, monkeypatch):
    # Flex decode returns no attention weights: output_attentions keeps every step on the model's attention.
    monkeypatch.setenv(FD.ENV, "1")
    for kwargs in (
        {"output_attentions": True},
        {"generation_config": types.SimpleNamespace(output_attentions = True)},
    ):
        model = _model("gemma4_text", "eager")
        assert vision._wants_attentions(model, kwargs)
        with _NonFlexGenerate(model, flex_decode = not vision._wants_attentions(model, kwargs)):
            _steps(model, _Static())
        assert model.seen == ["eager", "eager", "eager"], kwargs
    assert not vision._wants_attentions(
        _model("gemma4_text", "eager"), {"output_attentions": False}
    )


def test_auto_routes_only_graph_planned_calls(flex_on, monkeypatch):
    # Eager flex decode is slower than SDPA: auto switches only calls that decode through CUDA graphs, =1 forces.
    import unsloth.models._decode_graphs as G

    monkeypatch.setattr(G, "call_plan", lambda: "eager")
    model = _model("gemma4_text", "sdpa")
    with _NonFlexGenerate(model):
        _steps(model, _Static())
    assert model.seen == ["sdpa", "sdpa", "sdpa"]
    monkeypatch.setenv(FD.ENV, "1")
    model = _model("gemma4_text", "sdpa")
    with _NonFlexGenerate(model):
        _steps(model, _Static())
    assert model.seen == ["sdpa", FD.ATTN_KEY, FD.ATTN_KEY]


def test_auto_caps_the_batch(flex_on, monkeypatch):
    # auto: batches over UNSLOTH_FLEX_DECODE_MAX_BATCH (default 4) keep SDPA; the env var moves the cap, =1 lifts it.
    for batch, env, expected in (
        (4, None, FD.ATTN_KEY),
        (8, None, "sdpa"),
        (8, "8", FD.ATTN_KEY),
        (8, "0", FD.ATTN_KEY),
    ):
        if env is None:
            monkeypatch.delenv(FD.MAX_BATCH_ENV, raising = False)
        else:
            monkeypatch.setenv(FD.MAX_BATCH_ENV, env)
        model = _model("gemma4_text", "sdpa")
        with _NonFlexGenerate(model):
            _steps(model, _Static(), batch = batch)
        assert model.seen == ["sdpa", expected, expected], (batch, env)
    monkeypatch.delenv(FD.MAX_BATCH_ENV, raising = False)
    monkeypatch.setenv(FD.ENV, "1")
    model = _model("gemma4_text", "sdpa")
    with _NonFlexGenerate(model):
        _steps(model, _Static(), batch = 8)
    assert model.seen == ["sdpa", FD.ATTN_KEY, FD.ATTN_KEY]


def test_dynamic_cache_keeps_the_model_attention(flex_on):
    model = _model("gemma4_text", "sdpa")
    with _NonFlexGenerate(model):
        _steps(model, _Dynamic())
    assert model.seen == ["sdpa", "sdpa", "sdpa"]


def test_off_switch_and_unlisted_models_are_unchanged(flex_on, monkeypatch):
    model = _model("llama", "sdpa")
    assert _generate_routes(model) == []
    monkeypatch.setattr(FD, "wanted", lambda device: False)
    gemma = _model("gemma4_text", "sdpa")
    assert _generate_routes(gemma) == []
    with _NonFlexGenerate(gemma):
        _steps(gemma, _Static())
    assert gemma.seen == ["sdpa", "sdpa", "sdpa"]


def test_flex_preferred_config_decodes_on_flex_decode_when_routable(flex_on, monkeypatch):
    # A flex-preferred config (prefill on flex) keeps the eager decode unless its type is routable.
    model = _model("gemma3_text", "flex_attention")
    routes = _generate_routes(model)
    assert [(p, d) for _, p, d in routes] == [("flex_attention", "eager")]
    monkeypatch.setattr(FD, "routable", lambda model, config: True)
    routes = _generate_routes(model)
    assert [(p, d) for _, p, d in routes] == [("flex_attention", FD.ATTN_KEY)]
    with _NonFlexGenerate(model):
        _steps(model, _Static())
        _steps(model, _Dynamic())
    assert model.seen == [
        "flex_attention",
        FD.ATTN_KEY,
        FD.ATTN_KEY,
        "flex_attention",
        "eager",
        "eager",
    ]


def test_policy_auto_needs_sm80_and_env_overrides(monkeypatch):
    monkeypatch.setattr(FD, "register", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    for major, expected in ((7, False), (8, True), (9, True), (12, True)):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device = None, m = major: (m, 0)
        )
        assert FD.wanted("cuda:0") is expected
        monkeypatch.setenv(FD.ENV, "0")
        assert FD.wanted("cuda:0") is False
        monkeypatch.setenv(FD.ENV, "1")
        assert FD.wanted("cuda:0") is True
        monkeypatch.delenv(FD.ENV)
    assert FD.wanted("cpu") is False
    # ROCm reports gfx11 as (11, x): auto stays off there, =1 still forces it.
    monkeypatch.setattr(torch.version, "hip", "7.2")
    assert FD.wanted("cuda:0") is False
    monkeypatch.setenv(FD.ENV, "1")
    assert FD.wanted("cuda:0") is True
    monkeypatch.delenv(FD.ENV)
    monkeypatch.setattr(FD, "register", lambda: False)
    monkeypatch.setenv(FD.ENV, "1")
    assert FD.wanted("cuda:0") is False


def test_routable_excludes_gpt_oss_and_chunked_layers(monkeypatch):
    monkeypatch.setattr(FD, "_describe", lambda model, config: {"full": {(64, 8, 1)}})
    for model_type, layer_types, expected in (
        ("gemma4_text", ["sliding_attention", "full_attention"], True),
        ("qwen3_5_text", ["linear_attention", "full_attention"], True),
        ("gpt_oss", ["sliding_attention", "full_attention"], False),
        ("gemma4_text", ["chunked_attention", "full_attention"], False),
        ("llama", ["full_attention"], False),
    ):
        config = types.SimpleNamespace(
            model_type = model_type, layer_types = layer_types, _attn_implementation = "sdpa"
        )
        assert FD.routable(None, config) is expected, model_type
    monkeypatch.setenv(FD.ENV, "1")
    config = types.SimpleNamespace(
        model_type = "llama", layer_types = ["full_attention"], _attn_implementation = "sdpa"
    )
    assert FD.routable(None, config) is True
    config.model_type = "gpt_oss"
    assert FD.routable(None, config) is False


def test_unavailable_flex_decode_shape_falls_back_to_sdpa(monkeypatch):
    decode = pytest.importorskip("unsloth_zoo.flex_attention.decode")
    if FD.FlexDecodeUnavailable is None:
        pytest.skip("unsloth_zoo without FlexDecodeUnavailable")

    def unavailable(*args, **kwargs):
        raise decode.FlexDecodeUnavailable("no tile fits")

    monkeypatch.setattr(FD, "flex_decode_attention", unavailable)
    torch.manual_seed(0)
    batch, heads, kv_heads, dim, kv_length, position = 2, 4, 2, 16, 32, 9
    pad = torch.ones(batch, kv_length, dtype = torch.long)
    pad[1, :3] = 0
    mask = decode.decode_block_mask(batch, kv_length, torch.tensor(position), attention_mask = pad)
    q = torch.randn(batch, heads, 1, dim, dtype = torch.float64)
    k = torch.randn(batch, kv_heads, kv_length, dim, dtype = torch.float64)
    v = torch.randn(batch, kv_heads, kv_length, dim, dtype = torch.float64)
    module = torch.nn.Module()
    module.num_key_value_groups = heads // kv_heads
    module.is_causal = True
    out, _ = FD._attention(module, q, k, v, mask, scaling = 0.25)

    visible = (torch.arange(kv_length) <= position).view(1, kv_length) & pad.bool()
    rep = lambda x: x.repeat_interleave(heads // kv_heads, dim = 1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, rep(k), rep(v), attn_mask = visible[:, None, None, :], scale = 0.25
    )
    torch.testing.assert_close(out, ref.transpose(1, 2), rtol = 0, atol = 1e-12)
    assert module._unsloth_flex_decode_stats == {
        "flex": 0,
        "fallback": {"flex_decoding unavailable for this shape": 1},
    }


def test_other_flex_decode_errors_propagate(monkeypatch):
    decode = pytest.importorskip("unsloth_zoo.flex_attention.decode")

    def broken(*args, **kwargs):
        raise ValueError("bad shapes")

    monkeypatch.setattr(FD, "flex_decode_attention", broken)
    mask = decode.decode_block_mask(1, 16, torch.tensor(3))
    q = torch.randn(1, 2, 1, 8)
    k = v = torch.randn(1, 2, 16, 8)
    with pytest.raises(ValueError, match = "bad shapes"):
        FD._attention(torch.nn.Module(), q, k, v, mask)


@pytest.mark.parametrize("path", ["dense_mask", "unavailable", "elided_causal"])
def test_softcapped_fallback_keeps_the_softcap(monkeypatch, path):
    # Fallback must keep the logit softcap (SDPA drops it silently).
    torch.manual_seed(0)
    heads, kv_heads, dim, kv_len = 4, 2, 16, 12
    q_len = 3 if path == "elided_causal" else 1
    q = torch.randn(2, heads, q_len, dim, dtype = torch.float64) * 4
    k = torch.randn(2, kv_heads, kv_len, dim, dtype = torch.float64) * 4
    v = torch.randn(2, kv_heads, kv_len, dim, dtype = torch.float64)
    visible = torch.ones(2, kv_len, dtype = torch.bool)
    visible[1, :4] = False
    module = torch.nn.Module()
    module.is_causal = True
    if path == "unavailable":
        decode = pytest.importorskip("unsloth_zoo.flex_attention.decode")
        if FD.FlexDecodeUnavailable is None:
            pytest.skip("unsloth_zoo without FlexDecodeUnavailable")

        def unavailable(*args, **kwargs):
            raise decode.FlexDecodeUnavailable("no tile fits")

        monkeypatch.setattr(FD, "flex_decode_attention", unavailable)
        pad = visible.long()
        mask = decode.decode_block_mask(2, kv_len, torch.tensor(kv_len - 1), attention_mask = pad)
        dense = visible[:, None, None, :]
    elif path == "dense_mask":
        mask = dense = visible[:, None, None, :]
    else:
        mask, dense = None, torch.ones(q_len, kv_len, dtype = torch.bool).tril()
    out, _ = FD._attention(module, q, k, v, mask, scaling = 0.25, softcap = 5.0)
    rep = lambda x: x.repeat_interleave(heads // kv_heads, dim = 1)
    scores = torch.tanh((q @ rep(k).transpose(2, 3)) * 0.25 / 5.0) * 5.0
    scores = scores.masked_fill(~dense, float("-inf"))
    ref = (torch.softmax(scores, -1) @ rep(v)).transpose(1, 2)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 1e-12)


def test_register_installs_the_block_mask_device_transfer(monkeypatch):
    # accelerate moves a split model's layer arguments with BlockMask.to(): the routed registration must install
    # zoo's transfer (rebuilds mask_mod's offsets / padding on the target card) even if gpt-oss never registered.
    decode = pytest.importorskip("unsloth_zoo.flex_attention.decode")
    if not hasattr(decode, "install_block_mask_transfer") or not FD.available():
        pytest.skip("unsloth_zoo without the shared BlockMask transfer")
    from torch.nn.attention.flex_attention import BlockMask

    plain = getattr(BlockMask.to, "__wrapped__", BlockMask.to)
    monkeypatch.setattr(BlockMask, "to", plain)
    monkeypatch.setattr(FD, "_REGISTERED", False)
    assert FD.register()
    assert getattr(BlockMask.to, "_unsloth_gpt_oss", False)


def test_stats_counter_does_not_recompile_a_compiled_caller():
    from torch._dynamo.utils import counters

    module = torch.nn.Module()
    FD._count(module, None)

    @torch.compile(backend = "eager", fullgraph = True)
    def step(x):
        FD._count(module, None)
        return x + 1

    torch._dynamo.reset()
    before = sum(counters["recompiles"].values())
    for _ in range(12):
        step(torch.ones(2))
    assert sum(counters["recompiles"].values()) == before
    assert module.__dict__["_unsloth_flex_decode_stats"]["flex"] == 1
