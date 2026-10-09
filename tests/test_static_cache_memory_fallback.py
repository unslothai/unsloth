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
        pytest.skip(
            reason = "the reference, StaticCache.early_initialization, is missing from this "
            "transformers; the remaining tests still check the estimate",
        )


@pytest.mark.parametrize("batch", [1, 3])
def test_estimate_matches_transformers_full_attention(batch):
    _require_early_initialization()
    config = _llama_config()
    input_ids = torch.zeros(batch, 37, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 91})
    assert need == _real_static_cache_bytes(config, batch, 37 + 91 - 1, torch.bfloat16)


def test_estimate_matches_transformers_sliding_window():
    _require_early_initialization()
    config = _llama_config(
        sliding_window = 16,
        layer_types = ["sliding_attention", "full_attention"] * 2,
    )
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 100})
    assert need == _real_static_cache_bytes(config, 1, 109, torch.bfloat16)


def test_estimate_matches_transformers_sliding_window_without_layer_types():
    """Mistral style: a sliding_window and no layer_types makes every layer sliding."""
    _require_early_initialization()
    from transformers import MistralConfig

    config = MistralConfig(
        hidden_size = 256,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        num_hidden_layers = 4,
        intermediate_size = 512,
        vocab_size = 1000,
        sliding_window = 16,
    )
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 100})
    assert need == _real_static_cache_bytes(config, 1, 109, torch.bfloat16)


def test_estimate_uses_mla_key_and_value_dims():
    from transformers import DeepseekV3Config

    config = DeepseekV3Config(
        hidden_size = 256,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        num_hidden_layers = 3,
        qk_nope_head_dim = 48,
        qk_rope_head_dim = 16,
        v_head_dim = 32,
        vocab_size = 1000,
    )
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 90})
    assert need == 1 * 4 * (48 + 16 + 32) * 2 * 99 * 3


def test_estimate_default_length_adds_the_prompt():
    """transformers turns a default max_length of 20 into 20 new tokens after the prompt."""
    config = _llama_config()
    model = _model(config)
    model.generation_config.max_length = None
    input_ids = torch.zeros(1, 500, dtype = torch.long)
    assert _static_cache_bytes(model, input_ids, {}) == _static_cache_bytes(
        model, input_ids, {"max_new_tokens": 20}
    )


def test_estimate_default_length_capped_at_context():
    config = _llama_config(max_position_embeddings = 4096)
    model = _model(config)
    model.generation_config.max_length = 4096
    input_ids = torch.zeros(1, 500, dtype = torch.long)
    assert _static_cache_bytes(model, input_ids, {}) == _static_cache_bytes(
        model, torch.zeros(1, 1, dtype = torch.long), {"max_new_tokens": 4095}
    )


def test_partial_caller_config_inherits_the_model_length():
    """A caller GenerationConfig that only sets sampling still gets the model's max_length."""
    from transformers import GenerationConfig

    config = _llama_config(max_position_embeddings = 131072)
    model = _model(config)
    model.generation_config.max_length = 131072
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    caller = GenerationConfig(do_sample = True, temperature = 0.7)
    caller.max_length = None
    assert _static_cache_bytes(model, input_ids, {"generation_config": caller}) == (
        _static_cache_bytes(model, torch.zeros(1, 1, dtype = torch.long), {"max_new_tokens": 131071})
    )


def test_explicit_max_length_is_the_total():
    config = _llama_config()
    model = _model(config)
    input_ids = torch.zeros(1, 900, dtype = torch.long)
    assert _static_cache_bytes(model, input_ids, {"max_length": 1000}) == _static_cache_bytes(
        model, input_ids, {"max_new_tokens": 100}
    )


def test_legacy_static_cache_is_never_constructed(monkeypatch):
    """Before transformers 4.56 StaticCache allocates in __init__: count every layer as full."""
    import transformers

    def explode(*args, **kwargs):
        raise AssertionError("constructed a legacy StaticCache")

    monkeypatch.setattr(vision, "_static_cache_preallocates", lambda: True)
    monkeypatch.setattr(transformers, "StaticCache", explode)
    config = _llama_config(sliding_window = 16, layer_types = ["sliding_attention"] * 4)
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    need = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 90})
    assert need == 1 * 2 * (32 + 32) * 2 * 99 * 4


def test_estimate_uses_the_compile_decode_bucket(monkeypatch):
    config = _llama_config()
    input_ids = torch.zeros(1, 1, dtype = torch.long)
    plain = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 1026})
    monkeypatch.setattr(vision, "_compiles_decode", lambda model: True)
    bucketed = _static_cache_bytes(_model(config), input_ids, {"max_new_tokens": 1026})
    assert bucketed * 1026 == plain * 2048


def test_estimate_counts_beams_and_previous_length():
    config = _llama_config()
    model = _model(config)
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    one = _static_cache_bytes(model, input_ids, {"max_new_tokens": 90})
    assert _static_cache_bytes(model, input_ids, {"max_new_tokens": 90, "num_beams": 4}) == 4 * one
    # transformers reuses the largest length an earlier call allocated
    model._previous_max_cache_length = 990
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


def test_xpu_device_is_measured(monkeypatch):
    xpu = types.SimpleNamespace(
        mem_get_info = lambda device = None: (0, 80 * GB),
        memory_reserved = lambda device = None: 0,
        memory_allocated = lambda device = None: 0,
    )
    monkeypatch.setattr(torch, "xpu", xpu, raising = False)
    monkeypatch.setattr(vision.logger, "warning_once", lambda *a, **k: None)
    ids = torch.zeros(1, 1, dtype = torch.long)
    model = _model(_llama_config(), device = "xpu")
    assert _static_cache_does_not_fit(model, ids, {"max_new_tokens": 4095})


def test_probe_failure_keeps_static(fake_cuda, monkeypatch):
    def boom(device = None):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    config = _llama_config()
    ids = torch.zeros(1, 1, dtype = torch.long)
    assert not _static_cache_does_not_fit(_model(config), ids, {"max_new_tokens": 4095})


def _tiny_generate(monkeypatch, free_bytes):
    """Real generate through unsloth_base_fast_generate on a random tiny Llama, with the
    device reporting `free_bytes` free. Returns the cache class generate ended with."""
    if not torch.cuda.is_available():
        pytest.skip(reason = "the guard only measures CUDA devices")
    from transformers import GenerationConfig, LlamaForCausalLM

    torch.manual_seed(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = LlamaForCausalLM(_llama_config()).to("cuda", dtype).eval()
    model.generation_config = GenerationConfig(max_length = 4096, pad_token_id = 0)
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (free_bytes, 80 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 0)
    monkeypatch.setattr(vision.logger, "warning_once", lambda *a, **k: None)
    ids = torch.ones(1, 5, dtype = torch.long, device = "cuda")
    out = model.generate(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        max_new_tokens = 3,
        do_sample = False,
        return_dict_in_generate = True,
    )
    return type(out.past_key_values).__name__


def test_generate_uses_dynamic_cache_when_static_does_not_fit(monkeypatch):
    assert _tiny_generate(monkeypatch, free_bytes = 1024) == "DynamicCache"


def test_generate_keeps_static_cache_when_it_fits(monkeypatch):
    assert _tiny_generate(monkeypatch, free_bytes = 40 * GB) == "StaticCache"


def test_upper_bound_is_never_below_the_exact_size():
    config = _llama_config(
        sliding_window = 16, layer_types = ["sliding_attention", "full_attention"] * 2
    )
    model = _model(config)
    input_ids = torch.zeros(1, 10, dtype = torch.long)
    exact = _static_cache_bytes(model, input_ids, {"max_new_tokens": 500})
    bound = _static_cache_bytes(model, input_ids, {"max_new_tokens": 500}, exact = False)
    assert bound >= exact and bound == 4 * 509 * 2 * (32 + 32) * 2


def test_sliding_model_that_fits_exactly_keeps_static(fake_cuda):
    """The upper bound alone would fall back; the exact layer plan says it fits."""
    config = _llama_config(
        sliding_window = 16, layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    )
    model = _model(config)
    ids = torch.zeros(1, 1, dtype = torch.long)
    exact = _static_cache_bytes(model, ids, {"max_new_tokens": 4095})
    bound = _static_cache_bytes(model, ids, {"max_new_tokens": 4095}, exact = False)
    fake_cuda["free"] = 2 * exact + 1
    assert bound > fake_cuda["free"] // 2
    assert not _static_cache_does_not_fit(model, ids, {"max_new_tokens": 4095})
