# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which models compile their decode step, and the static cache length they get."""

import types

import pytest

import unsloth.models.vision as vision
import contextlib

from unsloth.models.vision import _CompileDecodeOnRepeat, _compiles_decode, _decode_cache_bucket


def _model(model_type, text_type = None):
    config = types.SimpleNamespace(model_type = model_type)
    if text_type is not None:
        config.text_config = types.SimpleNamespace(model_type = text_type)
    return types.SimpleNamespace(config = config, generation_config = types.SimpleNamespace())


@pytest.fixture(autouse = True)
def _zoo_supports_it(monkeypatch):
    monkeypatch.setattr(vision, "unsloth_decode_compile", object())
    monkeypatch.delenv("UNSLOTH_COMPILE_DECODE", raising = False)
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)


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

    def _prepare_static_cache(self, cache_implementation, batch_size, max_cache_len, *rest, **kwargs):
        self.allocated.append(max_cache_len)
        return types.SimpleNamespace(max_cache_len = max_cache_len)

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        return True


def _generate(model, scopes, batch_size, length, positional = False):
    with _CompileDecodeOnRepeat(model):
        if positional:  # transformers 5.2 - 5.5 call shape
            model._prepare_static_cache("static", batch_size, length, {})
        else:  # 5.17 passes keywords
            model._prepare_static_cache(cache_implementation = "static", batch_size = batch_size, max_cache_len = length, prefill_chunk_size = None, model_kwargs = {})
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
