# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The speed tier keeps AutoencoderKLQwenImage21 contiguous (channels_last measured slower); other VAEs unchanged."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_speed as ds_mod

torch = pytest.importorskip("torch")


def _vae(name: str):
    calls: list = []

    def to(self, memory_format = None):
        calls.append(memory_format)
        return self

    cls = type(name, (), {"to": to})
    return cls(), calls


def test_qwen_image_21_vae_stays_contiguous():
    vae, calls = _vae("AutoencoderKLQwenImage21")
    assert ds_mod._vae_channels_last(types.SimpleNamespace(vae = vae), None) is False
    assert calls == []


@pytest.mark.parametrize("name", ["AutoencoderKL", "AutoencoderKLQwenImage", "AutoencoderKLWan"])
def test_other_vaes_still_go_channels_last(name):
    vae, calls = _vae(name)
    assert ds_mod._vae_channels_last(types.SimpleNamespace(vae = vae), None) is True
    assert calls == [torch.channels_last]


def test_real_qwen_image_21_vae_class_name_matches_the_deny_list():
    mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage21")
    assert mod.AutoencoderKLQwenImage21.__name__ in ds_mod._VAE_CHANNELS_LAST_DENY
