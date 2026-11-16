# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Whisper generate() through decode graphs: plan counts the decoder prompt; tokens match eager, also on new audio."""

import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

from transformers import WhisperForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_generic_decode_graphs_default import _DefaultHost  # noqa: E402

WHISPER = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
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
def whisper():
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER, dtype = torch.float32).eval()
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


def _features(model, seed):
    g = torch.Generator().manual_seed(seed)
    frames = 2 * model.config.max_source_positions
    return torch.randn(1, model.config.num_mel_bins, frames, generator = g)


def _gen(
    model,
    features,
    new = 12,
):
    # Scores, not only tokens: a random tiny Whisper picks the same tokens for any clip.
    out = model.generate(
        input_features = features,
        max_new_tokens = new,
        do_sample = False,
        output_scores = True,
        return_dict_in_generate = True,
    )
    return out.sequences, torch.stack(out.scores)


def _eager(
    model,
    features,
    monkeypatch,
    new = 12,
):
    monkeypatch.setenv(ENV, "0")
    try:
        return _gen(model, features, new)
    finally:
        monkeypatch.delenv(ENV)


def test_whisper_decode_steps_replay_and_match_eager(host, whisper, monkeypatch):
    a, b = _features(whisper, 1), _features(whisper, 2)
    eager_a, eager_b = _eager(whisper, a, monkeypatch), _eager(whisper, b, monkeypatch)
    before = GG.PLANS["graph"]
    graphed_a = _gen(whisper, a)
    stats = GG.stats(whisper)
    assert GG.PLANS["graph"] == before + 1  # not a "short call" planned on the 80 mel bins
    assert stats["captures"] >= 1 and stats["replays"] > 0, stats
    assert stats["refusals"] == {}, stats
    assert all(torch.equal(x, y) for x, y in zip(graphed_a, eager_a))
    # Another clip: transformers builds a new cross-attention cache, so nothing of the first call replays stale.
    graphed_b = _gen(whisper, b)
    assert all(torch.equal(x, y) for x, y in zip(graphed_b, eager_b))
    assert not torch.equal(eager_a[1], eager_b[1])


def test_encoder_decoder_plan_counts_the_decoder_prompt(whisper):
    features = _features(whisper, 1)
    assert GG._plan_len(whisper, (), {"input_features": features, "max_new_tokens": 12}) == 13
    ids = torch.ones(1, 4, dtype = torch.long)
    assert (
        GG._plan_len(
            whisper,
            (),
            {"input_features": features, "decoder_input_ids": ids, "max_new_tokens": 12},
        )
        == 16
    )


def test_unfilled_cross_attention_cache_is_refused(whisper):
    from transformers import EncoderDecoderCache, StaticCache

    config = whisper.config.get_text_config(decoder = True)
    cache = EncoderDecoderCache(
        StaticCache(config = config, max_cache_len = 8), StaticCache(config = config, max_cache_len = 8)
    )
    assert GG._cache_reason(cache) == "cross-attention cache not filled"
    assert GG._self_attention_cache(cache) is cache.self_attention_cache
