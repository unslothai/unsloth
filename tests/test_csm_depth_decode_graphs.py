# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""CSM depth decoder: per-frame generate() replays decode graphs kept across frames; codes match eager."""

import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

modeling_csm = pytest.importorskip("transformers.models.csm.modeling_csm")

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
def csm():
    from transformers import CsmConfig

    small = dict(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
    )
    config = CsmConfig(
        # More codebooks than prompt + frames: input_ids' last dim must not pass for the sequence length.
        **small,
        num_codebooks = 16,
        vocab_size = 64,
        text_vocab_size = 128,
        depth_decoder_config = dict(
            **small,
            num_codebooks = 16,
            vocab_size = 64,
            backbone_hidden_size = 32,
            max_position_embeddings = 17,
        ),
    )
    torch.manual_seed(0)
    model = modeling_csm.CsmForConditionalGeneration._from_config(
        config, dtype = torch.float32
    ).eval()
    if not hasattr(model.depth_decoder, "_prepare_static_cache"):
        # transformers 4.x: no static cache on the depth decoder, so the nested routing stays off (stock generate).
        pytest.skip("CSM depth decoder has no static cache on this transformers version")
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    # getattr: the merge base has no nested routing, so this test fails there on the counts, not on an import.
    getattr(vision, "_patch_nested_generate", lambda model: None)(model)
    return model


def _gen(model, frames = 6):
    ids = torch.randint(3, 100, (1, 3), generator = torch.Generator().manual_seed(1))
    return model.generate(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        max_new_tokens = frames,
        do_sample = False,
        depth_decoder_do_sample = False,
        output_audio = False,
    )


def test_depth_decoder_frames_replay_one_kept_graph(host, csm, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(csm)
    monkeypatch.delenv(ENV)
    assert GG.stats(csm.depth_decoder) == {} or GG.stats(csm.depth_decoder)["replays"] == 0
    reused = GG.PLANS["reused kept cache"]
    graphed = _gen(csm)
    stats = GG.stats(csm.depth_decoder) or {"captures": 0, "replays": 0, "refusals": {}}
    assert torch.equal(graphed, eager)
    assert graphed.shape[-1] == csm.config.num_codebooks
    assert stats["captures"] == 1 and stats["replays"] > 0, stats
    assert stats["refusals"] == {}, stats
    assert GG.PLANS["reused kept cache"] > reused


def test_depth_decoder_generate_outside_a_parent_call_is_stock(csm):
    depth = csm.depth_decoder
    seen = []
    depth._old_generate = lambda *a, **kw: seen.append(kw) or "stock"
    ids = torch.zeros(1, 2, dtype = torch.long)
    assert depth.generate(input_ids = ids) == "stock"
    assert "cache_implementation" not in seen[0]
    with torch.inference_mode():
        depth.generate(input_ids = ids, past_key_values = object())
    assert "cache_implementation" not in seen[1]


def test_patch_is_idempotent(csm):
    generate = csm.depth_decoder.generate
    vision._patch_nested_generate(csm)
    assert csm.depth_decoder.generate is generate
    assert csm.depth_decoder._old_generate.__name__ != "unsloth_nested_fast_generate"


def test_graphs_off_keeps_the_stock_call(csm, monkeypatch):
    # Kill switch (or a platform without decode graphs): the per-frame call is the stock one, no static cache.
    monkeypatch.setenv(ENV, "0")
    depth = csm.depth_decoder
    real, seen = depth._old_generate, []

    def recording(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    depth._old_generate = recording
    _gen(csm, frames = 2)
    assert seen and all(
        "cache_implementation" not in kw and "compile_config" not in kw for kw in seen
    )


def test_flash_attention_depth_decoder_keeps_the_stock_call(host, csm, monkeypatch):
    # FlashAttention has no mask for unpadded rows: a static cache's unfilled slots would change the codes.
    monkeypatch.setattr(vision, "_uses_flash_attention_for_generation", lambda config: True)
    depth = csm.depth_decoder
    real, seen = depth._old_generate, []

    def recording(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    depth._old_generate = recording
    _gen(csm, frames = 2)
    assert seen and all("cache_implementation" not in kw for kw in seen)
