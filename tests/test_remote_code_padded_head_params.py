# SPDX-License-Identifier: AGPL-3.0-only
"""Kimi-K3 ships KDA A_log zero-padded ([128] for 96 heads). Quantized loads keep the padded tensor and the
fla backward (`dA.view_as(A_log)`) fails, so apply_remote_code_shims narrows it to num_heads."""

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

    KimiDeltaAttention.__module__ = mod.__name__
    return KimiDeltaAttention


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
    assert model.attn.A_log.requires_grad
    # The KDA gate backward does grad.view_as(A_log) with a per-head gradient.
    per_head_grad = torch.ones(3)
    assert per_head_grad.view_as(model.attn.A_log).shape == (3,)


def test_nonzero_tail_and_exact_size_are_left_alone():
    cls = _remote_attention_class()
    m1 = _model(cls(3, torch.randn(4)))  # tail not zero: not padding
    m2 = _model(cls(4, torch.randn(4)))  # already the right size
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
