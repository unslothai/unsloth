# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Packed rows (packing = True, now the SFT default) through the compiled decoder blocks: the number of
documents in a packed row changes every batch, and packed_seq_lengths ([documents]) reached every block
unmarked, so each block and the attention frame after its graph break recompiled once on the second packed
batch (FastModel Qwen3, Kaggle T4: 2 recompiles after step 1, guard
"tensor 'kwargs['packed_seq_lengths']' size mismatch at index 0"). CPU: no accelerator needed."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "_u_block_compile_packed", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _counted_compile(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = counter, fullgraph = False, dynamic = None),
    )
    return counter


def _layer_class():
    class PackedAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            position_ids = None,
            packed_seq_lengths = None,
            **kwargs,
        ):
            cu = torch.nn.functional.pad(packed_seq_lengths.cumsum(0), (1, 0))
            out = self.proj(hidden_states)
            return out * (cu[-1].to(out.dtype) / hidden_states.shape[1])

    class PackedDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = PackedAttention()
            self.norm = torch.nn.LayerNorm(8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
            position_ids = None,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = self.self_attn(
                self.norm(hidden_states), position_ids = position_ids, **kwargs
            )
            return residual + hidden_states

    return PackedDecoderLayer


def _step(layer, lengths):
    total = sum(lengths)
    x = torch.randn(1, total, 8, requires_grad = True)
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None]
    out = layer(
        x,
        position_ids = position_ids,
        packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
    )
    out.sum().backward()


def test_packed_document_count_change_does_not_recompile(monkeypatch):
    M = _load()
    counter = _counted_compile(monkeypatch)
    torch._dynamo.reset()
    cls = _layer_class()
    M._wrap(cls, {})
    layer = cls().train()
    _step(layer, [5, 4, 3])  # step 0: the only compile
    before = counter.frame_count
    _step(layer, [6, 6, 2, 2])  # another document count (and total)
    _step(layer, [3, 3, 3, 3, 4])
    _step(layer, [7, 9])
    extra = counter.frame_count - before
    torch._dynamo.reset()
    assert extra == 0, f"{extra} recompiled frame(s) after the first packed batch"


def test_mark_packed_dynamic_marks_document_dims_only(monkeypatch):
    M = _load()
    marked = []
    monkeypatch.setattr(
        torch._dynamo, "maybe_mark_dynamic", lambda t, d: marked.append((tuple(t.shape), d))
    )
    M._mark_packed_dynamic(
        {
            "packed_seq_lengths": torch.tensor([3, 4], dtype = torch.int32),
            "cu_seq_lens_q": torch.tensor([0, 3, 7], dtype = torch.int32),
            "position_ids": torch.zeros(1, 7, dtype = torch.long),  # the sequence mark covers these
            "max_length_q": 4,
            "cu_seqlens": None,
        }
    )
    assert sorted(marked) == [((2,), 0), ((3,), 0)]
