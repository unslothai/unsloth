# SPDX-License-Identifier: AGPL-3.0-only
"""Kimi-K3 KDA A_log is stored zero-padded past num_heads; fla's backward needs it narrowed."""

import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "unsloth_remote_code_shims_under_test",
    os.path.join(_HERE, "..", "unsloth", "models", "remote_code_shims.py"),
)
shims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shims)


def _remote_attention_class():
    mod = types.ModuleType("transformers_modules.fake_kimi.modeling_kimi_linear")
    sys.modules[mod.__name__] = mod

    class KimiDeltaAttention(nn.Module):
        def __init__(self, heads, stored):
            super().__init__()
            self.num_heads = heads
            self.A_log = nn.Parameter(stored.clone())

        def forward(self, g):
            return _KdaGate.apply(g, self.A_log)

    KimiDeltaAttention.__module__ = mod.__name__
    return KimiDeltaAttention


class _KdaGate(torch.autograd.Function):
    # Mirrors fla/ops/kda/gate.py: forward reads A_log[:H], backward returns dA.view_as(A_log).
    @staticmethod
    def forward(ctx, g, A_log):
        H = g.shape[-1]
        ctx.save_for_backward(g, A_log)
        return g * A_log[:H].exp()

    @staticmethod
    def backward(ctx, dy):
        g, A_log = ctx.saved_tensors
        H = g.shape[-1]
        dA = (dy * g * A_log[:H].exp()).sum(0)
        return dy * A_log[:H].exp(), dA.view_as(A_log)


def _model(attn):
    m = nn.Module()
    m.attn = attn
    return m


def test_zero_padded_a_log_is_narrowed_and_backward_shape_matches():
    cls = _remote_attention_class()
    stored = torch.cat([torch.randn(3), torch.zeros(1)])
    model = _model(cls(3, stored))
    shims.apply_remote_code_shims(model)
    assert model.attn.A_log.shape == (3,)
    assert torch.equal(model.attn.A_log.detach(), stored[:3])
    model.attn(torch.randn(2, 3)).sum().backward()
    assert model.attn.A_log.grad.shape == (3,)


def test_nonzero_tail_and_exact_size_are_left_alone():
    cls = _remote_attention_class()
    m1 = _model(cls(3, torch.randn(4)))
    m2 = _model(cls(4, torch.randn(4)))
    shims.apply_remote_code_shims(m1)
    shims.apply_remote_code_shims(m2)
    assert m1.attn.A_log.shape == (4,) and m2.attn.A_log.shape == (4,)


def test_non_remote_modules_untouched():
    class Local(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 3
            self.A_log = nn.Parameter(torch.cat([torch.randn(3), torch.zeros(1)]))

    model = _model(Local())
    shims.apply_remote_code_shims(model)
    assert model.attn.A_log.shape == (4,)
