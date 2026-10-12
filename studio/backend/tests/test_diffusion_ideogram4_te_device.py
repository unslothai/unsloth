# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 under group offload with streamed text encoders: ``text_encoder.device`` reads CPU while the hooks
return every tapped state on the GPU, so ``encode_prompt``'s mask multiply raised a device mismatch. CPU only."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_ideogram4 as ide  # noqa: E402


class _State:
    """A tensor stand-in on a named device (CPU CI has no second real device)."""

    def __init__(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.moves: list = []

    def to(self, device):
        self.moves.append(device)
        return _State(device)


def _streamed_encoder_states(text_encoder, token_ids, attention_mask, pos_2d):
    # What leaf-level group offload does: every layer, and so every returned state, runs on the onload device.
    return [_State("cuda:0") for _ in range(3)]


class _Pipe:
    _get_text_encoder_hidden_states = staticmethod(_streamed_encoder_states)

    def __init__(self, **components):
        self.components = components


def _load_with_stubs(monkeypatch):
    monkeypatch.setattr(ide, "_patch_create_causal_mask", lambda: None)
    monkeypatch.setattr(ide, "load_ideogram4_text_encoder", lambda *a, **k: "te")
    monkeypatch.setattr(ide, "load_krea2_tokenizer", lambda *a, **k: "tok")
    monkeypatch.setattr(ide, "load_ideogram4_transformer", lambda *a, **k: "dit")
    fake_diffusers = types.SimpleNamespace(
        Ideogram4Pipeline = _Pipe,
        AutoencoderKLFlux2 = types.SimpleNamespace(from_pretrained = lambda *a, **k: "vae"),
        FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
            from_pretrained = lambda *a, **k: "sched"
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    return ide.load_ideogram4_pipeline("unsloth/ideogram-4-fp8", torch.bfloat16)


def test_streamed_text_encoder_states_land_on_the_mask_device(monkeypatch):
    pipe = _load_with_stubs(monkeypatch)
    mask = torch.ones(1, 4, dtype = torch.long)  # on the CPU, where text_encoder.device pointed it
    states = pipe._get_text_encoder_hidden_states("te", mask, mask, mask)
    assert [s.device for s in states] == [mask.device] * 3


def test_matching_devices_are_left_alone():
    same = _State("cpu")
    guarded = ide._hidden_states_on_mask_device(lambda *a: [same])
    assert guarded("te", None, torch.ones(1), None)[0] is same and same.moves == []


def test_guard_is_idempotent(monkeypatch):
    pipe = _load_with_stubs(monkeypatch)
    first = pipe._get_text_encoder_hidden_states
    assert ide.install_text_encoder_device_guard(pipe) is True
    assert pipe._get_text_encoder_hidden_states is first
