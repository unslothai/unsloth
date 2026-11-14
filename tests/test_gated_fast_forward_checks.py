# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fast-forward checks must see through the pre_patch gate (class forward is a wrapper, so `is` fails)."""

import torch


def _gated_class(fast):
    from unsloth.models import llama as unsloth_llama

    def stock(self, *args, **kwargs):
        return "stock"

    cls = type("GatedAttention", (torch.nn.Module,), {"forward": fast})
    cls.forward = unsloth_llama._gated_method(cls, "forward", fast, stock)
    assert getattr(cls.forward, "_unsloth_gated", False)
    return cls


class _Layer(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class _Stack(torch.nn.Module):
    def __init__(self, attn_cls, marked):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(attn_cls()) for _ in range(2)])
        if marked:
            from unsloth.models.llama import mark_unsloth_fast_path
            mark_unsloth_fast_path(self)


def test_context_parallel_accepts_gated_fast_attention():
    from unsloth import context_parallel as cp
    from unsloth.models.llama import LlamaAttention_fast_forward

    cls = _gated_class(LlamaAttention_fast_forward)
    assert cp._supports_context_parallel(_Stack(cls, marked = True), "ulysses")
    assert cp._supports_context_parallel(_Stack(cls, marked = True), "ring")
    assert not cp._supports_context_parallel(_Stack(cls, marked = False), "ulysses")


def test_context_parallel_still_refuses_other_attention():
    from unsloth import context_parallel as cp

    def other(self, *args, **kwargs):
        return "other"

    cls = _gated_class(other)
    assert not cp._supports_context_parallel(_Stack(cls, marked = True), "ulysses")
    plain = type("PlainAttention", (torch.nn.Module,), {"forward": other})
    assert not cp._supports_context_parallel(_Stack(plain, marked = True), "ulysses")


def test_ungated_fast_attention_still_accepted():
    from unsloth import context_parallel as cp
    from unsloth.models.llama import LlamaAttention_fast_forward

    cls = type("FastAttention", (torch.nn.Module,), {"forward": LlamaAttention_fast_forward})
    assert cp._supports_context_parallel(_Stack(cls, marked = False), "ulysses")


def test_async_grpo_sees_gated_fast_backbone():
    from unsloth.models import rl
    from unsloth.models.llama import LlamaModel_fast_forward, mark_unsloth_fast_path

    cls = _gated_class(LlamaModel_fast_forward)
    backbone = cls()
    assert not rl._is_unsloth_fast_backbone(backbone)
    mark_unsloth_fast_path(backbone)
    assert rl._is_unsloth_fast_backbone(backbone)
    assert not rl._is_unsloth_fast_backbone(None)
    direct = type("FastBackbone", (torch.nn.Module,), {"forward": LlamaModel_fast_forward})()
    assert rl._is_unsloth_fast_backbone(direct)
