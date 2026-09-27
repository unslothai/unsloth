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


def test_cudnn_benchmark_is_skipped_for_a_qwen_image_21_pipeline():
    vae, _calls = _vae("AutoencoderKLQwenImage21")
    assert ds_mod._cudnn_benchmark_pointless(types.SimpleNamespace(vae = vae)) is True


@pytest.mark.parametrize("name", ["AutoencoderKL", "AutoencoderKLQwenImage", "AutoencoderKLWan"])
def test_cudnn_benchmark_kept_for_other_vaes(name):
    vae, _calls = _vae(name)
    assert ds_mod._cudnn_benchmark_pointless(types.SimpleNamespace(vae = vae)) is False


def test_cudnn_benchmark_kept_when_a_unet_denoises(monkeypatch):
    vae, _calls = _vae("AutoencoderKLQwenImage21")
    monkeypatch.setattr(ds_mod, "_denoiser_unet", lambda pipe: object())
    assert ds_mod._cudnn_benchmark_pointless(types.SimpleNamespace(vae = vae)) is False


def test_apply_speed_optims_leaves_cudnn_benchmark_alone_for_qwen_image_21(monkeypatch):
    vae, _calls = _vae("AutoencoderKLQwenImage21")
    pipe = types.SimpleNamespace(vae = vae)
    enabled: list = []
    monkeypatch.setattr(ds_mod, "_enable_cudnn_benchmark", lambda logger: enabled.append(1) or True)
    monkeypatch.setattr(ds_mod, "_compile_repeated_blocks", lambda *a, **k: False)
    monkeypatch.setattr(ds_mod, "compile_eligible", lambda *a, **k: False)
    target = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    family = types.SimpleNamespace(name = "qwen-image-2.1", supports_torch_compile = True)
    applied = ds_mod.apply_speed_optims(pipe, target, is_gguf = False, family = family, speed_mode = "default")
    assert applied["cudnn_benchmark"] is False and enabled == []
    assert applied["channels_last"] is False
