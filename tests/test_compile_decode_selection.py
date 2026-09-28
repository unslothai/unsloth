# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which models compile their decode step, and the static cache length they get."""

import types

import pytest

import unsloth.models.vision as vision
from unsloth.models.vision import _bucket_static_cache, _compiles_decode, _decode_cache_bucket


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


@pytest.mark.parametrize("model", [
    _model("qwen3_5", "qwen3_5_text"),
    _model("qwen3_5_moe", "qwen3_5_moe_text"),
    _model("qwen3_5_text"),
    _model("qwen3_5_moe_text"),
])
def test_qwen3_5_compiles_decode(model):
    assert _compiles_decode(model)


@pytest.mark.parametrize("model", [
    _model("llama"),
    _model("gemma3", "gemma3_text"),
    _model("qwen3_next"),
    _model("qwen3_vl", "qwen3_vl_text"),
])
def test_other_models_keep_eager_decode(model):
    assert not _compiles_decode(model)


@pytest.mark.parametrize("env", [("UNSLOTH_COMPILE_DECODE", "0"), ("UNSLOTH_COMPILE_DISABLE", "1"), ("UNSLOTH_COMPILE_DISABLE", "partial")])
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


@pytest.mark.parametrize("length, bucket", [(74, 1024), (1024, 1024), (1025, 2048), (1501, 2048), (3100, 4096)])
def test_bucket_rounds_up(length, bucket):
    assert _decode_cache_bucket(length) == bucket


def _recorder():
    seen = []

    def prepare_static_cache(cache_implementation, batch_size, max_cache_len, *rest, **kwargs):
        seen.append(max_cache_len)

    return seen, prepare_static_cache


def test_wrapper_buckets_keyword_length():
    # transformers 5.17 passes max_cache_len by keyword, after counting prompt / inputs_embeds.
    seen, prepare = _recorder()
    _bucket_static_cache(prepare)(cache_implementation = "static", batch_size = 1, max_cache_len = 1501, prefill_chunk_size = None, model_kwargs = {})
    assert seen == [2048]


def test_wrapper_buckets_positional_length():
    # transformers 5.2 - 5.5 signature: (cache_implementation, batch_size, max_cache_len, model_kwargs).
    seen, prepare = _recorder()
    _bucket_static_cache(prepare)("static", 3, 70, {})
    assert seen == [1024]
