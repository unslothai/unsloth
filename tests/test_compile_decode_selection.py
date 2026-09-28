# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which models compile their decode step, and the static cache length they get."""

import types

import pytest

import unsloth.models.vision as vision
from unsloth.models.vision import _compiles_decode, _decode_cache_bucket


def _model(model_type, text_type = None):
    config = types.SimpleNamespace(model_type = model_type)
    if text_type is not None:
        config.text_config = types.SimpleNamespace(model_type = text_type)
    return types.SimpleNamespace(config = config, generation_config = types.SimpleNamespace())


@pytest.fixture(autouse = True)
def _zoo_supports_it(monkeypatch):
    monkeypatch.setattr(vision, "UNSLOTH_DECODE_COMPILE", [False])
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


def test_old_zoo_keeps_eager_decode(monkeypatch):
    monkeypatch.setattr(vision, "UNSLOTH_DECODE_COMPILE", None)
    assert not _compiles_decode(_model("qwen3_5", "qwen3_5_text"))


needs_max_cache_len = pytest.mark.skipif(not vision._HAS_MAX_CACHE_LEN, reason = "transformers without max_cache_len")


@needs_max_cache_len
@pytest.mark.parametrize("prompt, new, bucket", [(10, 64, 1024), (500, 524, 1024), (500, 525, 2048), (3000, 100, 4096)])
def test_bucket_rounds_up(prompt, new, bucket):
    assert _decode_cache_bucket(_model("qwen3_5"), prompt, {"max_new_tokens": new}) == bucket


@needs_max_cache_len
def test_bucket_from_max_length():
    assert _decode_cache_bucket(_model("qwen3_5"), 10, {"max_length": 1500}) == 2048


@needs_max_cache_len
def test_bucket_respects_caller_ceiling():
    assert _decode_cache_bucket(_model("qwen3_5"), 10, {"max_new_tokens": 5, "max_cache_len": 64}) is None
    config = types.SimpleNamespace(max_cache_len = 64, max_new_tokens = 5)
    assert _decode_cache_bucket(_model("qwen3_5"), 10, {"generation_config": config}) is None


@needs_max_cache_len
def test_bucket_needs_a_length():
    assert _decode_cache_bucket(_model("qwen3_5"), 10, {}) is None
