# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CPU checks for user-supplied past_key_values in generate() (issue #497), against unsloth/models/llama.py."""

import pytest
import torch
from transformers.cache_utils import Cache, DynamicCache

BS, PAST_LEN, SEQ = 2, 3, 5


class FakeModel:
    dtype = torch.float32
    config = None


def _llama():
    from unsloth.models import llama
    return llama


def _kv(length, layers = 2):
    return tuple(
        (torch.randn(BS, 1, length, 4), torch.randn(BS, 1, length, 4)) for _ in range(layers)
    )


def _prepare(input_ids, **kwargs):
    return _llama()._fast_prepare_inputs_for_generation(FakeModel(), input_ids, **kwargs)


def test_tuple_cache_becomes_dynamic_cache_for_generate():
    legacy = _kv(PAST_LEN)
    cache = _llama()._ensure_cache_is_dynamic(legacy)
    assert isinstance(cache, Cache)
    assert cache.get_seq_length() == PAST_LEN
    assert _llama()._ensure_cache_is_dynamic(cache) is cache
    assert _llama()._ensure_cache_is_dynamic(None) is None


@pytest.mark.parametrize("as_dynamic", [False, True])
def test_prefill_onto_partial_cache_feeds_every_uncached_token(as_dynamic):
    legacy = _kv(PAST_LEN)
    past = _llama()._ensure_cache_is_dynamic(legacy) if as_dynamic else legacy
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    mask = torch.ones(BS, SEQ, dtype = torch.long)
    result = _prepare(input_ids, attention_mask = mask, past_key_values = past)

    assert torch.equal(result["input_ids"], input_ids[:, PAST_LEN:])
    assert result["position_ids"].tolist() == [[3, 4]] * BS
    assert result["cache_position"].tolist() == [3, 4]
    # Unsloth's forwards index past_key_values[layer][0|1].
    out = result["past_key_values"]
    assert isinstance(out, tuple) and len(out) == len(legacy)
    for (k, v), (k0, v0) in zip(out, legacy):
        assert torch.equal(k, k0) and torch.equal(v, v0)


def test_decode_step_still_feeds_only_the_last_token():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(input_ids, past_key_values = _kv(SEQ - 1))
    assert torch.equal(result["input_ids"], input_ids[:, [-1]])


def test_full_length_user_position_ids_are_sliced_to_the_fed_tokens():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    pos = torch.arange(SEQ).expand(BS, -1) + 10
    result = _prepare(input_ids, past_key_values = _kv(PAST_LEN), position_ids = pos)
    assert result["position_ids"].tolist() == [[13, 14]] * BS

    result = _prepare(input_ids, position_ids = pos[0])
    assert torch.equal(result["position_ids"], pos[0])


def test_user_position_ids_with_inputs_embeds_and_no_input_ids():
    embeds = torch.randn(BS, SEQ, 4)
    pos = torch.arange(SEQ).expand(BS, -1)
    result = _prepare(None, inputs_embeds = embeds, position_ids = pos)
    assert torch.equal(result["position_ids"], pos)
    assert result["inputs_embeds"] is embeds


def test_empty_dynamic_cache_is_dropped_before_prefill():
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    result = _prepare(input_ids, past_key_values = DynamicCache())
    assert result["past_key_values"] is None
    assert result["input_ids"].shape == (BS, SEQ)


def test_suffix_only_input_with_full_mask_feeds_the_whole_new_turn():
    # transformers 5 generate() accepts only the new tokens when attention_mask spans cache + new.
    new = torch.arange(BS * 2).reshape(BS, 2)
    mask = torch.ones(BS, PAST_LEN + 2, dtype = torch.long)
    result = _prepare(new, attention_mask = mask, past_key_values = _kv(PAST_LEN))
    assert torch.equal(result["input_ids"], new)
    assert result["position_ids"].tolist() == [[3, 4]] * BS


def test_static_cache_is_flattened_to_its_filled_length():
    from transformers import LlamaConfig
    from transformers.cache_utils import StaticCache

    config = LlamaConfig(
        num_hidden_layers = 2, num_attention_heads = 1, num_key_value_heads = 1, hidden_size = 4
    )
    try:
        cache = StaticCache(config = config, max_cache_len = 8)
    except TypeError:
        pytest.skip("StaticCache signature differs on this transformers")
    legacy = _kv(PAST_LEN)
    for i, (k, v) in enumerate(legacy):
        cache.update(k, v, i, {"cache_position": torch.arange(PAST_LEN)})
    out = _llama()._cache_as_legacy_tuple(cache)
    for (k, v), (k0, v0) in zip(out, legacy):
        assert torch.equal(k, k0) and torch.equal(v, v0)


def test_cache_layers_missing_positions_are_rejected():
    cache = _llama()._ensure_cache_is_dynamic(_kv(PAST_LEN))
    # A QuantizedLayer counts every token but keeps only the unquantized tail in .keys.
    cache.layers[0].keys = cache.layers[0].keys[..., 1:, :]
    cache.get_seq_length = lambda layer_idx = 0: PAST_LEN
    with pytest.raises(ValueError, match = "does not keep every cached position"):
        _llama()._cache_as_legacy_tuple(cache)


def test_partial_static_cache_mask_spans_the_flattened_length():
    from transformers import LlamaConfig
    from transformers.cache_utils import StaticCache
    from tests.utils.test_prepare_inputs_leftpad import FakeModelWith4DMask

    config = LlamaConfig(
        num_hidden_layers = 2, num_attention_heads = 1, num_key_value_heads = 1, hidden_size = 4
    )
    try:
        cache = StaticCache(config = config, max_cache_len = 16)
    except TypeError:
        pytest.skip("StaticCache signature differs on this transformers")
    for i, (k, v) in enumerate(_kv(PAST_LEN)):
        cache.update(k, v, i, {"cache_position": torch.arange(PAST_LEN)})
    model = FakeModelWith4DMask()
    input_ids = torch.arange(BS * SEQ).reshape(BS, SEQ)
    mask = torch.ones(BS, SEQ, dtype = torch.long)
    _llama()._fast_prepare_inputs_for_generation(
        model, input_ids, attention_mask = mask, past_key_values = cache
    )
    assert model.mask_calls[0]["target_length"] == SEQ  # past_len + fed tokens, not max_cache_len
