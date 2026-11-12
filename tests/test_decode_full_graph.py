# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whole-step CUDA-graph decode (UNSLOTH_DECODE_CUDA_GRAPHS_FULL=1): the attention cores are captured
too, over a power-of-two span of cache rows with a device-side keep mask. Teacher-forced logits
match the eager step to bf16 summation-order noise, every step after the first is one replay,
spans re-capture as the cache fills, and sliding-window models keep the segmented capture."""

import os

os.environ.setdefault("UNSLOTH_DECODE_CUDA_GRAPHS", "1")

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import torch

import unsloth.models._decode_graphs as G
from tests.test_decode_cuda_graphs import MODELS, PROMPTS, _causal, _load

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")


def _forced(model, enc, tokens, enabled, full, monkeypatch):
    """Logits of every decode step with the next tokens forced (so arms see the same inputs)."""
    monkeypatch.setattr(G, "ENABLED", enabled)
    monkeypatch.setattr(G, "FULL", full)
    ids, mask = enc.input_ids, enc.attention_mask
    with torch.inference_mode():
        out = model(input_ids = ids, attention_mask = mask, use_cache = True, return_dict = True)
        past, logits = out.past_key_values, [out.logits[:, -1].float().cpu()]
        pos = mask.long().cumsum(-1)[:, -1:] - 1
        for t in range(tokens.shape[1] - 1):
            nxt = tokens[:, t : t + 1]
            mask = torch.cat([mask, torch.ones_like(nxt)], 1)
            pos = pos + 1
            out = model(
                input_ids = nxt, attention_mask = mask, position_ids = pos,
                past_key_values = past, use_cache = True, return_dict = True,
            )  # fmt: skip
            past = out.past_key_values
            logits.append(out.logits[:, -1].float().cpu())
    G.release(model)
    return torch.stack(logits, 1)


def _kl(p_logits, q_logits):
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


@cuda
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("load_in_4bit", [False, True])
@pytest.mark.parametrize("bsz", [1, 4])
def test_full_graph_matches_eager(name, load_in_4bit, bsz, monkeypatch):
    monkeypatch.setattr(G, "FULL_MAX_BSZ", 8)  # the batched whole-step path too
    model, tok = _load(name, load_in_4bit)
    enc = tok(PROMPTS[:bsz], return_tensors = "pt", padding = True).to("cuda")
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(10, 1000, (bsz, 300), generator = g).cuda()
    eager = _forced(model, enc, tokens, False, False, monkeypatch)
    before = dict(G.stats(_causal(model)))
    full = _forced(model, enc, tokens, True, True, monkeypatch)
    stats = G.stats(_causal(model))
    assert stats["disabled"] is None, stats
    # Every step but the eager ones (the first fills the cache, a cache growth reallocates it)
    # was one whole-step replay.
    replays = stats["replays"] - before.get("replays", 0)
    eager_steps = stats["eager_steps"] - before.get("eager_steps", 0)
    assert replays + eager_steps == tokens.shape[1] - 1, stats
    assert eager_steps <= 3, stats
    # 300 steps from a short prompt cross the 256-row span: at least two spans captured.
    assert stats["captures"] - before.get("captures", 0) >= 2, stats
    assert torch.isfinite(full).all()
    assert _kl(eager, full) < 1e-3
    # Summation-order noise only: a few bf16 ulps on average (tiny random models have tiny logits).
    assert float((eager - full).abs().mean()) < 0.02 * float(eager.abs().mean())


@cuda
def test_full_and_segmented_graphs_coexist(monkeypatch):
    """Toggling FULL between generations: each mode keeps its own graphs; segmented stays bitwise."""
    model, tok = _load(MODELS[0], False)
    enc = tok(PROMPTS[:2], return_tensors = "pt", padding = True).to("cuda")
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(10, 1000, (2, 40), generator = g).cuda()
    monkeypatch.setattr(G, "FULL_MAX_BSZ", 8)
    eager = _forced(model, enc, tokens, False, False, monkeypatch)
    seg = _forced(model, enc, tokens, True, False, monkeypatch)
    full = _forced(model, enc, tokens, True, True, monkeypatch)
    seg2 = _forced(model, enc, tokens, True, False, monkeypatch)
    assert torch.equal(eager, seg)
    assert torch.equal(eager, seg2)
    assert _kl(eager, full) < 1e-3


@cuda
def test_full_graph_generate_and_lora(monkeypatch):
    """generate() with LoRA adapters: one capture per span, replays, finite logits close to eager."""
    model, tok = _load(MODELS[1], False, lora = True)
    enc = tok(PROMPTS[:1], return_tensors = "pt").to("cuda")

    def gen(enabled, full):
        monkeypatch.setattr(G, "ENABLED", enabled)
        monkeypatch.setattr(G, "FULL", full)
        out = model.generate(
            **enc, max_new_tokens = 40, min_new_tokens = 40, do_sample = False,
            pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
        )  # fmt: skip
        return torch.stack(out.logits).float().cpu()

    eager = gen(False, False)
    before = dict(G.stats(_causal(model)))
    full = gen(True, True)
    stats = G.stats(_causal(model))
    assert stats["disabled"] is None, stats
    assert stats["replays"] - before.get("replays", 0) == 40 - 2
    # Greedy tokens may part on a near-tie; compare the first step that diverges onwards only
    # through its logits, which see the same prefix until then.
    assert _kl(eager[:5], full[:5]) < 1e-3


def test_full_span_policy(monkeypatch):
    monkeypatch.setattr(G, "FULL", True)
    model = types.SimpleNamespace(config = types.SimpleNamespace(sliding_window = None))
    assert G._full_span(model, 1, 512) == 256
    assert G._full_span(model, 256, 512) == 256
    assert G._full_span(model, 257, 512) == 512
    assert G._full_span(model, 600, 768) == 768  # capped at the cache rows
    assert G._full_span(model, 1025, 4096) == 2048
    model.config.sliding_window = 4096
    assert G._full_span(model, 10, 2048) == 256
    # A window shorter than the cache: the segmented capture slices it eagerly.
    assert G._full_span(model, 10, 8192) is None
    model.config.sliding_window = None
    # Batched decode keeps the segmented capture unless widened.
    assert G._full_span(model, 10, 512, bsz = 2) is None
    monkeypatch.setattr(G, "FULL_MAX_BSZ", 4)
    assert G._full_span(model, 10, 512, bsz = 4) == 256
    assert G._full_span(model, 10, 512, bsz = 5) is None
    monkeypatch.setattr(G, "FULL", False)
    assert G._full_span(model, 10, 512) is None


@cuda
def test_batched_default_stays_bitwise(monkeypatch):
    """FULL on, default FULL_MAX_BSZ: a batch of 2 runs the segmented capture, bit for bit eager."""
    model, tok = _load(MODELS[0], False)
    enc = tok(PROMPTS[:2], return_tensors = "pt", padding = True).to("cuda")
    tokens = torch.randint(10, 1000, (2, 30), generator = torch.Generator().manual_seed(2)).cuda()
    eager = _forced(model, enc, tokens, False, False, monkeypatch)
    full = _forced(model, enc, tokens, True, True, monkeypatch)
    assert torch.equal(eager, full)
