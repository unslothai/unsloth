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
        ("none", "auto", None, False, True),
        ("group", "auto", None, True, True),
        ("group", "auto", "cache", True, True),  # a non-offload hook does not move weights
        ("group", "auto", "group", True, False),
        ("group", "auto", "block_group", True, False),  # a hook on one block is enough
        ("group", "auto", "accelerate", True, False),
        ("group", "auto", None, False, False),
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

    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")
    monkeypatch.setattr(lora_fuse, "_on_card", lambda module: on_card)
    denoiser = _hooked(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)), hook)
    assert lora_fuse.denoisers_resident(policy, memory_mode, [denoiser]) is expected


@pytest.mark.parametrize(
    "env, group_expected",
    [
        (None, False),
        ("", False),
        ("auto", False),
        ("0", False),
        ("1", True),
        ("on", True),
        ("true", True),
    ],
)
def test_group_fold_is_opt_in(monkeypatch, env, group_expected):
    """Unset keeps group offload unfused (PEFT's branch is closer to fp32); no offload folds as before."""
    from core.inference import diffusion_lora_fuse as lora_fuse

    if env is None:
        monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    else:
        monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, env)
    monkeypatch.setattr(lora_fuse, "_on_card", lambda module: True)
    denoiser = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert lora_fuse.denoisers_resident("group", "auto", [denoiser]) is group_expected
    assert lora_fuse.denoisers_resident("none", "auto", [denoiser]) is True


def test_on_card_refuses_host_and_meta():
    from core.inference import diffusion_lora_fuse as lora_fuse
    assert not lora_fuse._on_card(torch.nn.Linear(4, 4))
    assert not lora_fuse._on_card(torch.nn.Linear(4, 4, device = "meta"))
