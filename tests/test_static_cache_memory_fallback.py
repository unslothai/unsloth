# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static cache memory fallback for generate (unslothai/unsloth#2590).

FastModel generate forces a static KV cache, sized for prompt + max_new_tokens before the first
token. On a memory-tight or unified-memory device (Jetson) that allocation alone can OOM where a
growing cache would not, so generate must fall back to a dynamic cache when it does not fit.
"""

import types

import pytest
import torch

from unsloth.models import vision
from unsloth.models.vision import _static_cache_bytes, _static_cache_does_not_fit

GB = 1024**3


def _llama_config(**overrides):
    from transformers import LlamaConfig

    kwargs = dict(
        hidden_size = 256,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        num_hidden_layers = 4,
        intermediate_size = 512,
        vocab_size = 1000,
    )
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def _model(
    config,
    dtype = torch.bfloat16,
    device = "cuda",
    device_map = None,
):
    from transformers import GenerationConfig
    return types.SimpleNamespace(
        config = config,
        generation_config = GenerationConfig(),
        dtype = dtype,
        device = torch.device(device),
        hf_device_map = device_map,
    )


def _real_static_cache_bytes(config, batch, max_cache_len, dtype):
    """What transformers itself allocates, the reference the estimate must match."""
    from transformers import StaticCache

    cache = StaticCache(config = config, max_cache_len = max_cache_len)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    cache.early_initialization(
        batch_size = batch,
        num_heads = config.num_key_value_heads,
        head_dim = head_dim,
        dtype = dtype,
        device = "cpu",
    )
    return sum(
        t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)
    )


def _require_early_initialization():
    from transformers import StaticCache
    if not hasattr(StaticCache, "early_initialization"):
        pytest.skip("StaticCache.early_initialization not in this transformers")


@pytest.mark.parametrize("batch", [1, 3])
def test_estimate_matches_transformers_full_attention(batch):
    _require_early_initialization()
    config = _llama_config()
    input_ids = torch.zeros(batch, 37, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 91})
    assert need == _real_static_cache_bytes(config, batch, 37 + 91, torch.bfloat16)


def test_estimate_matches_transformers_sliding_window():
    _require_early_initialization()
    config = _llama_config(
        sliding_window = 16,
        layer_types = ["sliding_attention", "full_attention"] * 2,
    )
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 100})
    assert need == _real_static_cache_bytes(config, 1, 110, torch.bfloat16)


def test_estimate_counts_beams_and_previous_length():
    config = _llama_config()
    model = _model(config)
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    one = _static_cache_bytes(model, input_ids, {"max_new_tokens": 90})
    assert _static_cache_bytes(model, input_ids, {"max_new_tokens": 90, "num_beams": 4}) == 4 * one
    # transformers reuses the largest length an earlier call allocated
    model._previous_max_cache_length = 1000
    assert _static_cache_bytes(model, input_ids, {"max_new_tokens": 90}) == 10 * one


def test_no_estimate_with_caller_cache_or_encoder_decoder():
    config = _llama_config()
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    assert _static_cache_bytes(_model(config), input_ids, {"past_key_values": object()}) is None
    config.is_encoder_decoder = True
    assert _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 5}) is None


@pytest.fixture
def fake_cuda(monkeypatch):
    state = {"free": 0, "reserved": 0, "allocated": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (state["free"], 80 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: state["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: state["allocated"])
    monkeypatch.setattr(vision.logger, "warning_once", lambda *a, **k: None)
    return state


def _cache_bytes_for(tokens):
    config = _llama_config()
    return config, _static_cache_bytes(
        _model(config), torch.zeros(1, 1, dtype = torch.long), {"max_new_tokens": tokens - 1}
    )


def test_small_cache_keeps_static(fake_cuda):
    config, need = _cache_bytes_for(4096)
    fake_cuda["free"] = 4 * need
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert not _static_cache_does_not_fit(_model(config), ids, {"max_new_tokens": 4095})


def test_cache_over_half_of_free_falls_back(fake_cuda):
    config, need = _cache_bytes_for(4096)
    fake_cuda["free"] = need
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert _static_cache_does_not_fit(_model(config), ids, {"max_new_tokens": 4095})


def test_allocator_cached_blocks_count_as_free(fake_cuda):
    config, need = _cache_bytes_for(4096)
    fake_cuda.update(free = need, reserved = 5 * need, allocated = 2 * need)
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert not _static_cache_does_not_fit(_model(config), ids, {"max_new_tokens": 4095})


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"device": "cpu"},
        {"device_map": {"model.layers.0": 0, "model.layers.1": 1}},
    ],
)
def test_unmeasured_placements_keep_static(fake_cuda, model_kwargs):
    config, need = _cache_bytes_for(4096)
    fake_cuda["free"] = 0
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert not _static_cache_does_not_fit(
        _model(config, **model_kwargs), ids, {"max_new_tokens": 4095}
    )


def test_probe_failure_keeps_static(fake_cuda, monkeypatch):
    def boom(device = None):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    config = _llama_config()
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert not _static_cache_does_not_fit(_model(config), ids, {"max_new_tokens": 4095})
