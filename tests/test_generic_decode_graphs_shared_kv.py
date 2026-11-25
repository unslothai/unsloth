# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Gemma 3n shared-KV layers (transformers 5.x) decode through graphs instead of refusing; tokens match eager."""

import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

cache_utils = pytest.importorskip("transformers.cache_utils")
if not hasattr(cache_utils, "DummyLayer"):
    pytest.skip("transformers without DummyLayer cache layers", allow_module_level = True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_generic_decode_graphs_default import _DefaultHost  # noqa: E402

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


@pytest.fixture
def host(monkeypatch):
    backend = type("Host", (_DefaultHost,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    monkeypatch.setattr(G, "_STUDIO_DEFAULT_OK", True)
    for name in (ENV, "UNSLOTH_DECODE_GENERIC_GRAPHS", "UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP"):
        monkeypatch.delenv(name, raising = False)
    return backend


@pytest.fixture
def gemma3n():
    from transformers import Gemma3nForCausalLM, Gemma3nTextConfig

    config = Gemma3nTextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 32,
        hidden_size_per_layer_input = 8,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        laurel_rank = 4,
        altup_num_inputs = 2,
        num_kv_shared_layers = 2,
        sliding_window = 16,
        layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        activation_sparsity_pattern = [0.0] * 4,
        max_position_embeddings = 64,
    )
    torch.manual_seed(0)
    model = Gemma3nForCausalLM._from_config(config, dtype = torch.float32).eval()
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


def _gen(model, new = 10):
    ids = torch.randint(3, 100, (1, 5), generator = torch.Generator().manual_seed(1))
    return model.generate(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        max_new_tokens = new,
        min_new_tokens = new,
        do_sample = False,
        pad_token_id = 0,
    )


def test_shared_kv_cache_replays(host, gemma3n, monkeypatch):
    from transformers import StaticCache

    cache = StaticCache(config = gemma3n.config, max_cache_len = 16)
    assert any(type(layer) is cache_utils.DummyLayer for layer in cache.layers)
    monkeypatch.setenv(ENV, "0")
    eager = _gen(gemma3n)
    monkeypatch.delenv(ENV)
    graphed = _gen(gemma3n)
    stats = GG.stats(gemma3n) or {"captures": 0, "replays": 0, "refusals": {}}
    assert torch.equal(graphed, eager)
    assert stats["replays"] > 0 and "layer DummyLayer" not in stats["refusals"], stats
