# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The LoRA fold gate: fold only while every denoiser sits whole on the card with no offload hook (torch only)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")


def _hooked(module, kind):
    if kind == "accelerate":
        module._hf_hook = object()
    elif kind == "group":
        module._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    elif kind == "block_group":
        child = next(module.children())
        child._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    elif kind == "cache":
        module._diffusers_hook = types.SimpleNamespace(hooks = {"first_block_cache": object()})
    return module


@pytest.mark.parametrize(
    "policy, memory_mode, hook, on_card, expected",
    [
        ("none", "auto", None, False, True),  # the existing rule, unchanged
        ("group", "auto", None, True, True),  # text encoder grouped, denoiser whole on the card
        ("group", "auto", "cache", True, True),  # a non-offload hook does not move weights
        ("group", "auto", "group", True, False),  # streamed (or pinned-group) transformer
        ("group", "auto", "block_group", True, False),  # a hook on one block is enough
        ("group", "auto", "accelerate", True, False),
        ("group", "auto", None, False, False),  # demoted to host or split
        ("group", "low_vram", None, True, False),  # low_vram keeps the denoiser streamed
        ("model", "auto", None, True, False),
        ("streaming", "auto", None, True, False),
        ("sequential", "auto", None, True, False),
        (None, "auto", None, True, False),
    ],
)
def test_fold_gate_needs_a_resident_denoiser(
    monkeypatch, policy, memory_mode, hook, on_card, expected
):
    from core.inference import diffusion_lora_fuse as lora_fuse

    monkeypatch.setattr(lora_fuse, "_on_card", lambda module: on_card)
    denoiser = _hooked(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)), hook)
    assert lora_fuse.denoisers_resident(policy, memory_mode, [denoiser]) is expected


def test_on_card_refuses_host_and_meta():
    from core.inference import diffusion_lora_fuse as lora_fuse
    assert not lora_fuse._on_card(torch.nn.Linear(4, 4))
    assert not lora_fuse._on_card(torch.nn.Linear(4, 4, device = "meta"))
