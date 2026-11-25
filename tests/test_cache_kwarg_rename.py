# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers 5.24 renamed the KV cache keyword past_key_values to `cache` (generate's model_kwargs,
prepare_inputs_for_generation, every model forward). Unsloth's fast generation path read it under the
old name only, so on 5.24 every decode step re-ran the whole sequence without a cache. CPU checks."""

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
BS, PAST_LEN, SEQ = 2, 3, 5


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_cache_kwarg_under_test", ROOT / "unsloth" / "models" / "_cache_kwarg.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    dtype = torch.float32
    config = None


def _kv(length, layers = 2):
    return tuple(
        (torch.randn(BS, 1, length, 4), torch.randn(BS, 1, length, 4)) for _ in range(layers)
    )


def _dynamic(legacy):
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        cache.update(k, v, layer_idx)
    return cache


def _prepare(input_ids, **kwargs):
    from unsloth.models import llama
    return llama._fast_prepare_inputs_for_generation(FakeModel(), input_ids, **kwargs)


@pytest.mark.parametrize("as_dynamic", [False, True])
def test_prepare_inputs_reads_the_cache_keyword(as_dynamic):
    # What transformers 5.24 generate hands prepare_inputs_for_generation.
    legacy = _kv(PAST_LEN)
    cache = _dynamic(legacy) if as_dynamic else legacy
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(input_ids, attention_mask = torch.ones(BS, SEQ, dtype = torch.long), cache = cache)

    assert torch.equal(result["input_ids"], input_ids[:, PAST_LEN:]), "uncached tokens only"
    assert result["position_ids"].tolist() == [[3, 4]] * BS
    assert "cache" not in result, "the fast forwards take past_key_values; cache= would be dropped"
    out = result["past_key_values"]
    assert isinstance(out, tuple) and len(out) == len(legacy)
    for (k, v), (k0, v0) in zip(out, legacy):
        assert torch.equal(k, k0) and torch.equal(v, v0)


def test_decode_step_with_cache_keyword_feeds_only_the_last_token():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(input_ids, cache = _kv(SEQ - 1))
    assert torch.equal(result["input_ids"], input_ids[:, [-1]])
    assert result["past_key_values"] is not None


def test_empty_cache_keyword_still_prefills_everything():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(input_ids, cache = DynamicCache())
    assert torch.equal(result["input_ids"], input_ids)
    assert result.get("past_key_values") is None and "cache" not in result


def test_cache_keyword_helpers():
    helper = _load_helper()
    a, b = object(), object()
    assert helper.get_cache_kwarg({"cache": a, "past_key_values": b}) is a
    assert helper.get_cache_kwarg({"cache": None, "past_key_values": b}) is b
    assert helper.get_cache_kwarg({}, default = b) is b
    kwargs = {"cache": None, "past_key_values": b, "x": 1}
    assert helper.pop_cache_kwarg(kwargs) is b and kwargs == {"x": 1}

    class Old:
        past_key_values = b

    class New:
        cache = a

        @property
        def past_key_values(self):
            raise AssertionError("5.24 logs a deprecation on this property; read cache first")

    assert helper.output_cache(Old()) is b
    assert helper.output_cache(New()) is a
    assert helper.output_cache(torch.zeros(1)) is None


def test_tensor_parallel_guard_sees_the_cache_keyword():
    from unsloth.models import tp_native

    class Layer(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            attention_mask = None,
            cache = None,
            **kwargs,
        ):
            return hidden_states

    layer = Layer()
    tp_native._refuse_cached_forward(layer)
    cache = _dynamic(_kv(PAST_LEN))
    original = tp_native._replica_in
    tp_native._replica_in = lambda module: False
    try:
        with pytest.raises(RuntimeError, match = "cached decoding"):
            layer(torch.zeros(1), cache = cache)
        with pytest.raises(RuntimeError, match = "cached decoding"):
            layer(torch.zeros(1), None, cache)
        assert layer(torch.zeros(1)) is not None
    finally:
        tp_native._replica_in = original
