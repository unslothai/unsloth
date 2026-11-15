# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""CSM generate(): the depth decoder's per-frame generate() (the remaining codebooks of every audio frame) decodes
over a static cache through decode CUDA graphs kept from frame to frame (host stand-in backend of
test_generic_decode_graphs). Codes match eager decoding; outside a parent generate() it is the stock generate()."""

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
    # One capture, kept with its static cache: every later frame replays from its first decode step.
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
