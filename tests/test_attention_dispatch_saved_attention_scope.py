# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Automatic saved attention (UNSLOTH_GC_SAVE_ATTENTION unset) routes a long training call only where a
saved-attention frame can exist: under Unsloth's smart checkpoint, and when traced, inside Unsloth's own
compiled block. Elsewhere saved_attention_call has no frame and runs the plain call behind
torch.compiler.disable, which is a graph break per layer for nothing (Unsupported under fullgraph=True)."""

import pytest
import torch
import torch.utils.checkpoint

ad = pytest.importorskip("unsloth.utils.attention_dispatch")
G = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
SA = pytest.importorskip("unsloth_zoo.saved_attention")
block = pytest.importorskip("unsloth.models._block_compile")

H, KV, D, S = 4, 2, 16, 32


@pytest.fixture
def auto_mode(monkeypatch):
    # What unsloth_zoo's automatic mode gives on a measured GPU, with a threshold below S.
    monkeypatch.delenv("UNSLOTH_GC_SAVE_ATTENTION", raising = False)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION", True)
    monkeypatch.setattr(ad, "_SAVE_ATTENTION_MIN_SEQ", 16)
    monkeypatch.setattr(ad, "_saved_attention_call", SA.saved_attention_call)
    monkeypatch.setattr(block, "_IN_COMPILED_REGION", [False])
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _smart_checkpoint(monkeypatch):
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", G.UnslothCheckpointFunction)


def _attend(Q, K, V):
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = Q.shape[2],
        kv_seq_len = K.shape[2],
        n_heads = H,
        head_dim = D,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        training = True,
    )
    return ad.run_attention(config = config, context = context, Q = Q, K = K, V = V)


def _block(q, k, v):
    return (_attend(q * 1.0, k, v) * 2.0).sum()


def _inputs():
    g = torch.Generator().manual_seed(0)
    return [torch.randn(1, n, S, D, generator = g).requires_grad_() for n in (H, KV, KV)]


def _graph_breaks(fullgraph = False):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    q, k, v = _inputs()
    out = torch.compile(_block, fullgraph = fullgraph)(q, k, v)
    out.backward()
    torch.testing.assert_close(out, _block(*_inputs()))
    return sum(counters["graph_break"].values())


def _context(q_len):
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = KV, n_groups = H // KV)
    context = ad.AttentionContext(
        bsz = 1,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = H,
        head_dim = D,
        requires_grad = True,
        seq_info = None,
        attention_mask = None,
        causal_mask = None,
        training = True,
    )
    return config, context


def test_auto_mode_routes_only_under_the_unsloth_checkpoint(monkeypatch, auto_mode):
    # use_gradient_checkpointing True / False: torch's own CheckpointFunction, no frame ever.
    assert torch.utils.checkpoint.CheckpointFunction.__name__ != "UnslothCheckpointFunction"
    assert ad._auto_saves_attention(*_context(S)) is False
    _smart_checkpoint(monkeypatch)
    assert ad._auto_saves_attention(*_context(S)) is True
    assert ad._auto_saves_attention(*_context(8)) is False


@pytest.mark.parametrize("smart", [False, True])
def test_auto_mode_adds_no_graph_break_outside_unsloth_block(monkeypatch, auto_mode, smart):
    # A user's torch.compile (not Unsloth's block): the long call stays in the graph.
    if smart:
        _smart_checkpoint(monkeypatch)
    assert _graph_breaks() == 0


def test_auto_mode_compiles_under_fullgraph(monkeypatch, auto_mode):
    _smart_checkpoint(monkeypatch)
    assert _graph_breaks(fullgraph = True) == 0


def test_auto_mode_still_routes_inside_unsloth_block(monkeypatch, auto_mode):
    # Unsloth's own block (fullgraph=False, a break is legal) under the smart checkpoint keeps routing.
    _smart_checkpoint(monkeypatch)
    monkeypatch.setattr(block, "_IN_COMPILED_REGION", [True])
    assert _graph_breaks() >= 1


def test_explicit_setting_is_unchanged(monkeypatch, auto_mode):
    # UNSLOTH_GC_SAVE_ATTENTION=1 (min_seq None) keeps routing every training call, as before.
    monkeypatch.setattr(ad, "_SAVE_ATTENTION_MIN_SEQ", None)
    assert _graph_breaks() >= 1


@pytest.mark.parametrize("setting", ["0", "false"])
def test_kill_switch_set_after_import_stops_routing(monkeypatch, auto_mode, setting):
    # zoo creates no frame once the setting leaves "auto": the routed call would be a break for nothing.
    _smart_checkpoint(monkeypatch)
    monkeypatch.setenv("UNSLOTH_GC_SAVE_ATTENTION", setting)
    assert ad._auto_saves_attention(*_context(S)) is False
    monkeypatch.setattr(block, "_IN_COMPILED_REGION", [True])
    assert _graph_breaks() == 0
