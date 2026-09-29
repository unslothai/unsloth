# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which models compile their decode step, and the static cache length they get."""

import types

import pytest

import unsloth.models.vision as vision
import contextlib

from unsloth.models.vision import (
    _CompileDecodeOnRepeat,
    _EagerDecodeSteps,
    _compiles_decode,
    _decode_cache_bucket,
    _eager_decodes,
)


def _model(model_type, text_type = None):
    config = types.SimpleNamespace(model_type = model_type)
    if text_type is not None:
        config.text_config = types.SimpleNamespace(model_type = text_type)
    return types.SimpleNamespace(config = config, generation_config = types.SimpleNamespace())


@pytest.fixture(autouse = True)
def _zoo_supports_it(monkeypatch):
    monkeypatch.setattr(vision, "unsloth_decode_compile", object())
    monkeypatch.setenv("UNSLOTH_COMPILE_DECODE", "1")
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    monkeypatch.delenv("UNSLOTH_EAGER_DECODE", raising = False)


def test_decode_compile_is_opt_in(monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_DECODE")
    assert not _compiles_decode(_model("qwen3_5", "qwen3_5_text"))


@pytest.mark.parametrize(
    "model",
    [
        _model("qwen3_5", "qwen3_5_text"),
        _model("qwen3_5_moe", "qwen3_5_moe_text"),
        _model("qwen3_5_text"),
        _model("qwen3_5_moe_text"),
    ],
)
def test_qwen3_5_compiles_decode(model):
    assert _compiles_decode(model)


@pytest.mark.parametrize(
    "model",
    [
        _model("llama"),
        _model("gemma3", "gemma3_text"),
        _model("qwen3_next"),
        _model("qwen3_vl", "qwen3_vl_text"),
    ],
)
def test_other_models_keep_eager_decode(model):
    assert not _compiles_decode(model)


@pytest.mark.parametrize(
    "env",
    [
        ("UNSLOTH_COMPILE_DECODE", "0"),
        ("UNSLOTH_COMPILE_DISABLE", "1"),
        ("UNSLOTH_COMPILE_DISABLE", "partial"),
    ],
)
def test_opt_outs(monkeypatch, env):
    monkeypatch.setenv(*env)
    assert not _compiles_decode(_model("qwen3_5", "qwen3_5_text"))


def test_uncompileable_quantizer_keeps_eager_decode():
    model = _model("qwen3_5", "qwen3_5_text")
    model.hf_quantizer = types.SimpleNamespace(is_compileable = False)
    assert not _compiles_decode(model)
    model.hf_quantizer = types.SimpleNamespace(is_compileable = True)
    assert _compiles_decode(model)


@pytest.mark.parametrize("device", ["cpu", "disk"])
def test_offloaded_model_keeps_eager_decode(device):
    model = _model("qwen3_5", "qwen3_5_text")
    model.hf_device_map = {"model.layers.0": 0, "model.layers.1": device}
    assert not _compiles_decode(model)
    model.hf_device_map = {"model.layers.0": 0, "model.layers.1": 1}
    assert _compiles_decode(model)


def test_old_zoo_keeps_eager_decode(monkeypatch):
    monkeypatch.setattr(vision, "unsloth_decode_compile", None)
    assert not _compiles_decode(_model("qwen3_5", "qwen3_5_text"))


@pytest.mark.parametrize(
    "length, bucket", [(74, 1024), (1024, 1024), (1025, 2048), (1501, 2048), (3100, 4096)]
)
def test_bucket_rounds_up(length, bucket):
    assert _decode_cache_bucket(length) == bucket


class _FakeGenerating:
    """Stands in for the model: records the length transformers would allocate and whether
    it would compile the decode step."""

    def __init__(self):
        self.allocated = []

    def _prepare_static_cache(
        self, cache_implementation, batch_size, max_cache_len, *rest, **kwargs
    ):
        self.allocated.append(max_cache_len)
        return types.SimpleNamespace(max_cache_len = max_cache_len)

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        return True


def _generate(
    model,
    scopes,
    batch_size,
    length,
    positional = False,
):
    with _CompileDecodeOnRepeat(model):
        if positional:  # transformers 5.2 - 5.5 call shape
            model._prepare_static_cache("static", batch_size, length, {})
        else:  # 5.17 passes keywords
            model._prepare_static_cache(
                cache_implementation = "static",
                batch_size = batch_size,
                max_cache_len = length,
                prefill_chunk_size = None,
                model_kwargs = {},
            )
        compiled = model._valid_auto_compile_criteria({}, None)
        return compiled, len(scopes)


@pytest.fixture
def scopes(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake_scope():
        entered.append(1)
        yield

    monkeypatch.setattr(vision, "unsloth_decode_compile", fake_scope)
    return entered


def test_first_shape_stays_eager_then_compiles_on_repeat(scopes):
    model = _FakeGenerating()
    assert _generate(model, scopes, 1, 200) == (False, 0)
    assert _generate(model, scopes, 1, 300) == (True, 1)  # same 1024 bucket
    assert _generate(model, scopes, 3, 300) == (False, 1)  # new batch size
    assert _generate(model, scopes, 1, 1500) == (False, 1)  # new 2048 bucket
    assert _generate(model, scopes, 3, 90, positional = True) == (True, 2)
    assert model.allocated == [1024, 1024, 1024, 2048, 1024]


def test_batch_dynamic_graph_skips_eager_warm_up(scopes):
    model = _FakeGenerating()
    _generate(model, scopes, 1, 100)
    _generate(model, scopes, 2, 100)
    assert _generate(model, scopes, 4, 100) == (False, 0)  # one compiled batch: still static
    _generate(model, scopes, 1, 100)
    _generate(model, scopes, 2, 100)
    assert len(scopes) == 2
    assert _generate(model, scopes, 3, 100) == (True, 3)  # batch now dynamic: no warm-up
    assert _generate(model, scopes, 5, 1500) == (False, 3)  # other length: its own graphs
    assert _generate(model, scopes, 1, 100) == (True, 4)
    assert _generate(model, scopes, 6, 100) == (True, 5)


def test_hooks_are_removed_after_the_call(scopes):
    model = _FakeGenerating()
    _generate(model, scopes, 1, 10)
    assert "_prepare_static_cache" not in model.__dict__
    assert "_valid_auto_compile_criteria" not in model.__dict__


class _Refusing(_FakeGenerating):
    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        return False  # e.g. bnb 4-bit or CPU offload


def test_transformers_refusal_is_respected(scopes):
    model = _Refusing()
    _generate(model, scopes, 1, 10)
    assert _generate(model, scopes, 1, 10) == (False, 0)


@pytest.fixture
def eager_scope(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake_scope():
        entered.append(1)
        yield

    monkeypatch.setattr(vision, "unsloth_eager_decode", fake_scope)
    return entered


def test_eager_decode_selection(monkeypatch, eager_scope):
    assert _eager_decodes(_model("qwen3_5", "qwen3_5_text"))
    assert _eager_decodes(_model("qwen3_5_moe_text"))
    assert not _eager_decodes(_model("llama"))
    monkeypatch.setenv("UNSLOTH_EAGER_DECODE", "0")
    assert not _eager_decodes(_model("qwen3_5", "qwen3_5_text"))
    monkeypatch.delenv("UNSLOTH_EAGER_DECODE")
    monkeypatch.setattr(vision, "unsloth_eager_decode", None)  # older unsloth_zoo
    assert not _eager_decodes(_model("qwen3_5", "qwen3_5_text"))


class _Forwarding:
    def __init__(self):
        self.seen = []

    def forward(self, input_ids = None, inputs_embeds = None):
        self.seen.append((input_ids if input_ids is not None else inputs_embeds).shape)
        return "out"


def test_only_one_token_steps_run_in_eager_decode(eager_scope):
    import torch
    model = _Forwarding()
    with _EagerDecodeSteps(model):
        assert model.forward(input_ids = torch.zeros(2, 7, dtype = torch.long)) == "out"
        assert eager_scope == []  # prefill keeps its compiled regions
        model.forward(input_ids = torch.zeros(2, 1, dtype = torch.long))
        model.forward(inputs_embeds = torch.zeros(3, 1, 8))
        assert len(eager_scope) == 2
    assert "forward" not in model.__dict__
    assert model.seen == [(2, 7), (2, 1), (3, 1, 8)]
