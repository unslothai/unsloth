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
):
    for step in range(n_new):
        n = prompt + step
        ids = torch.zeros(1, n, dtype = torch.long)
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
