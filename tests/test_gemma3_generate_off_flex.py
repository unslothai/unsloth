# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""generate() of a Gemma 3 decoder that Unsloth put on flex_attention runs on the non-flex path."""

import inspect
import types

import pytest

import unsloth.models.vision as vision
from unsloth.models._utils import _EXPLICIT_FLEX_ATTR
from unsloth.models.vision import _NonFlexGenerate, _flex_generate_configs


@pytest.fixture(autouse = True)
def _no_opt_out(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FLEX_ATTENTION_GENERATE", raising = False)


def _text_model(model_type = "gemma3_text", attn = "flex_attention"):
    from transformers import Gemma3TextConfig, LlamaConfig

    cls = Gemma3TextConfig if model_type == "gemma3_text" else LlamaConfig
    config = cls(
        num_hidden_layers = 1,
        hidden_size = 32,
        intermediate_size = 64,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        vocab_size = 64,
    )
    config._attn_implementation = attn
    return types.SimpleNamespace(config = config)


def _vlm_model():
    from transformers import Gemma3Config

    config = Gemma3Config()
    config.text_config._attn_implementation = "flex_attention"
    config.vision_config._attn_implementation = "sdpa"
    return types.SimpleNamespace(config = config)


def test_text_model_generates_off_flex_and_restores():
    model = _text_model()
    with _NonFlexGenerate(model):
        assert model.config._attn_implementation == "eager"
    assert model.config._attn_implementation == "flex_attention"
    assert "_unsloth_non_flex_generate" not in model.__dict__


def test_restored_when_generate_raises():
    model = _text_model()
    with pytest.raises(RuntimeError):
        with _NonFlexGenerate(model):
            raise RuntimeError("boom")
    assert model.config._attn_implementation == "flex_attention"


def test_vlm_moves_only_the_decoder():
    model = _vlm_model()
    top = model.config._attn_implementation
    with _NonFlexGenerate(model):
        assert model.config.text_config._attn_implementation == "eager"
        assert model.config.vision_config._attn_implementation == "sdpa"
    assert model.config.text_config._attn_implementation == "flex_attention"
    assert model.config.vision_config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == top


def test_nested_calls_restore_once():
    model = _text_model()
    with _NonFlexGenerate(model):
        with _NonFlexGenerate(model):
            assert model.config._attn_implementation == "eager"
        assert model.config._attn_implementation == "eager"
    assert model.config._attn_implementation == "flex_attention"


@pytest.mark.parametrize(
    "model",
    [
        _text_model(attn = "eager"),
        _text_model(attn = "sdpa"),
        _text_model(model_type = "llama"),
    ],
    ids = ["gemma3-eager", "gemma3-sdpa", "llama-flex"],
)
def test_untouched_unless_unsloth_chose_flex_for_gemma3(model):
    before = model.config._attn_implementation
    assert _flex_generate_configs(model) == []
    with _NonFlexGenerate(model):
        assert model.config._attn_implementation == before
    assert model.config._attn_implementation == before


def test_explicit_flex_request_is_kept():
    model = _text_model()
    setattr(model, _EXPLICIT_FLEX_ATTR, True)
    with _NonFlexGenerate(model):
        assert model.config._attn_implementation == "flex_attention"


def test_opt_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FLEX_ATTENTION_GENERATE", "1")
    model = _text_model()
    with _NonFlexGenerate(model):
        assert model.config._attn_implementation == "flex_attention"


def test_generate_enters_the_scope():
    source = inspect.getsource(vision.unsloth_base_fast_generate)
    assert "_NonFlexGenerate(self)" in source
