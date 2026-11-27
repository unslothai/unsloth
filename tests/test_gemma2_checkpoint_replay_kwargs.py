# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Reentrant checkpointing replays each layer after the loop: a Gemma2 layer that is not a
GradientCheckpointingLayer (transformers < 4.53) must replay with its own use_sliding_window,
not the last layer's, else the recomputed forward and its gradients differ from the eager ones."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models.llama as llama_module

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

N_LAYERS = 4


class _WindowLayer(torch.nn.Module):
    """Plain nn.Module like transformers 4.52's Gemma2DecoderLayer; output depends on the window flag."""

    def __init__(self, idx, seen):
        super().__init__()
        self.idx = idx
        self.seen = seen
        self.weight = torch.nn.Parameter(torch.full((1,), 1.0 + 0.1 * idx))
        # Skips the shared rotary build: these layers never read position embeddings.
        self.self_attn = types.SimpleNamespace(rotary_emb = None)

    def forward(
        self,
        hidden_states,
        causal_mask = None,
        *args,
        **kwargs,
    ):
        sliding = kwargs.get("use_sliding_window")
        self.seen.append((self.idx, sliding))
        scale = 2.0 if sliding else 3.0
        return (hidden_states + torch.sin(hidden_states * self.weight * scale),)


def _run(gradient_checkpointing, monkeypatch):
    # transformers >= 4.53 Gemma2 layers are GradientCheckpointingLayer and take the other branch.
    monkeypatch.setattr(llama_module, "GradientCheckpointingLayer", type(None))
    # The final norm's Triton kernel needs a GPU; the replay under test is before it.
    monkeypatch.setattr(llama_module, "fast_rms_layernorm", lambda norm, X, **_: norm(X))
    config = transformers.Gemma2Config(
        vocab_size = 32,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = N_LAYERS,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 4,
        sliding_window = 4,
    )
    torch.manual_seed(0)
    model = transformers.Gemma2Model(config)
    model.train()
    del model.rotary_emb
    seen = []
    model.layers = torch.nn.ModuleList([_WindowLayer(i, seen) for i in range(N_LAYERS)])
    model.max_seq_length = 64
    model.gradient_checkpointing = gradient_checkpointing
    input_ids = torch.randint(1, 32, (1, 6))
    out = llama_module.LlamaModel_fast_forward(
        model, input_ids = input_ids, attention_mask = None, use_cache = False
    )
    out.last_hidden_state.float().pow(2).sum().backward()
    grads = [layer.weight.grad.clone() for layer in model.layers]
    return seen, grads


def test_replay_uses_each_layers_sliding_window(monkeypatch):
    seen, grads = _run(True, monkeypatch)
    _, eager = _run(False, monkeypatch)
    torch.testing.assert_close(torch.cat(grads), torch.cat(eager))
    forward, replay = seen[:N_LAYERS], seen[N_LAYERS:]
    assert forward == [(i, i % 2 == 0) for i in range(N_LAYERS)]
    # Backward recomputes every layer, last first; each with its own flag.
    assert sorted(replay) == forward, replay
