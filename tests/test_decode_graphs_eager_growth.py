# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An eager KV cache grows by the eager increment even after a graph generation left its store."""

import types

import torch

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._decode_graphs as G


def _alloc(attn, seq_len, increment):
    kv = torch.arange(seq_len * 4, dtype = torch.float32).reshape(1, 1, seq_len, 4)
    G.alloc_decode_buffers(
        attn, kv, -kv, 1, 1, 1, 4, 4, torch.float32, torch.device("cpu"), increment
    )


def _attn():
    return types.SimpleNamespace(config = types.SimpleNamespace(model_type = "llama"))


def test_eager_growth_ignores_a_retained_graph_store(monkeypatch):
    increment = 512
    eager_only = _attn()
    monkeypatch.setattr(G, "ENABLED", False)
    _alloc(eager_only, 1023, increment)
    G.resize_decode_buffers(eager_only, increment, 1, 1, 1, 4)
    expected_rows = eager_only.paged_attention.shape[0]
    assert expected_rows == 1023 + 1 + 2 * increment

    attn = _attn()
    monkeypatch.setattr(G, "ENABLED", True)
    G.start_generation(32768)
    _alloc(attn, 31, increment)
    G.end_generation()
    assert "_unsloth_graph_buffers" in attn.__dict__

    # Graphs off for the next generation; the store from the last one is still on the module.
    monkeypatch.setattr(G, "ENABLED", False)
    _alloc(attn, 1023, increment)
    kv = attn.paged_attention[:1023].clone()
    epoch = G.BUFFER_EPOCH
    G.resize_decode_buffers(attn, increment, 1, 1, 1, 4)
    assert attn.paged_attention.shape[0] == expected_rows
    assert attn.attention.shape[-1] == eager_only.attention.shape[-1]
    assert torch.equal(attn.paged_attention[:1023], kv)
    assert G.BUFFER_EPOCH == epoch


def test_graph_backed_growth_still_doubles(monkeypatch):
    attn = _attn()
    monkeypatch.setattr(G, "ENABLED", True)
    monkeypatch.setattr(G, "PREALLOC", False)
    G.start_generation(0)
    _alloc(attn, 31, 512)
    rows = attn.paged_attention.shape[0]
    epoch = G.BUFFER_EPOCH
    G.resize_decode_buffers(attn, 1, 1, 1, 1, 4)
    G.end_generation()
    assert attn.paged_attention.shape[0] >= 2 * rows
    assert attn.paged_attention.shape[0] % G.BUCKET == 0
    assert G.BUFFER_EPOCH == epoch + 1
