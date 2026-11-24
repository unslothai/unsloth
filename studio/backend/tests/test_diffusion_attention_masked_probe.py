# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto attention asks torch's native dispatch the way a DiT's block attention really calls it: a DiT that masks every
block call (Krea 2) is asked with a key-padding mask. Torch's FLASH takes no mask, so on A100 / L4 its native call runs
mem-efficient attention and auto keeps the cuDNN pin; unmasked DiTs keep native flash."""

from __future__ import annotations

import types

import pytest
import torch

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend, select_attention_backend

CUDNN = int(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
FLASH = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
EFFICIENT = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)


class _Dit:
    def __init__(self, head_dim = 128):
        self.config = {"attention_head_dim": head_dim}
        self.set_to = None

    def set_attention_backend(self, name):
        self.set_to = name


class Krea2Transformer2DModel(_Dit):
    pass


class FluxTransformer2DModel(_Dit):
    pass


def _target(dtype = torch.bfloat16):
    return types.SimpleNamespace(device = "cuda", dtype = dtype)


def _card(monkeypatch, cap, masked_pick):
    """A card whose native dispatch picks FLASH (cap < 10) / CUDNN (Blackwell) unmasked and ``masked_pick`` masked."""
    att._NATIVE_DISPATCH_CACHE.clear()
    att._NATIVE_FLASH_CACHE.clear()
    monkeypatch.delenv(att.NATIVE_FLASH_ATTN_ENV, raising = False)
    monkeypatch.delenv(att.BLACKWELL_NATIVE_ATTN_ENV, raising = False)
    monkeypatch.setenv(att.CUDNN_HEAD_DIM_PROBE_ENV, "0")
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: cap)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(att, "guard_rocm_fused_sdpa", lambda target, logger = None: None)
    monkeypatch.setattr(att, "_device_capability", lambda device: cap)
    asked = []

    def choice(
        device,
        dtype,
        dim,
        masked = False,
    ):
        asked.append((dim, masked))
        if masked:
            return masked_pick
        return CUDNN if cap >= (10, 0) else FLASH

    monkeypatch.setattr(att, "_native_sdp_choice", choice)
    return asked


@pytest.fixture(autouse = True)
def _clean_caches():
    yield
    att._NATIVE_DISPATCH_CACHE.clear()
    att._NATIVE_FLASH_CACHE.clear()


def _auto(dit, target = None):
    target = target or _target()
    pick = select_attention_backend(target, "auto", speed_active = True)
    return apply_attention_backend(types.SimpleNamespace(transformer = dit), pick, target = target)


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_a_masked_dit_keeps_the_cudnn_pin_on_a_flash_faster_card(monkeypatch, cap):
    asked = _card(monkeypatch, cap, EFFICIENT)
    dit = Krea2Transformer2DModel(128)
    engaged = _auto(dit)
    assert engaged == "_native_cudnn" and dit.set_to == "_native_cudnn"
    assert asked == [(128, True)]


@pytest.mark.parametrize("cap", [(8, 0), (8, 9)])
def test_an_unmasked_dit_still_runs_native_flash(monkeypatch, cap):
    asked = _card(monkeypatch, cap, EFFICIENT)
    dit = FluxTransformer2DModel(128)
    assert _auto(dit) is None
    assert dit.set_to in (None, "native")
    assert asked == [(128, False)]


def test_a_masked_dit_goes_native_once_torch_flash_takes_the_mask(monkeypatch):
    # the rule follows torch: should native dispatch pick FLASH for the masked call, auto leaves it native
    _card(monkeypatch, (8, 9), FLASH)
    assert _auto(Krea2Transformer2DModel(128)) is None


def test_a_mixed_pipe_asks_masked_when_any_dit_masks(monkeypatch):
    asked = _card(monkeypatch, (8, 0), EFFICIENT)
    pipe = types.SimpleNamespace(
        transformer = FluxTransformer2DModel(128), transformer_2 = Krea2Transformer2DModel(128)
    )
    assert att._native_flash_measured_faster(pipe, _target()) is False
    assert asked == [(128, True)]


def test_masked_and_unmasked_answers_are_cached_apart(monkeypatch):
    asked = _card(monkeypatch, (8, 9), EFFICIENT)
    flux = types.SimpleNamespace(transformer = FluxTransformer2DModel(128))
    krea = types.SimpleNamespace(transformer = Krea2Transformer2DModel(128))
    assert att._native_flash_measured_faster(flux, _target()) is True
    assert att._native_flash_measured_faster(krea, _target()) is False
    assert att._native_flash_measured_faster(flux, _target()) is True
    assert asked == [(128, False), (128, True)]


@pytest.mark.parametrize("masked_pick, native", [(EFFICIENT, False), (CUDNN, True)])
def test_blackwell_asks_masked_too(monkeypatch, masked_pick, native):
    asked = _card(monkeypatch, (10, 0), masked_pick)
    assert (
        att._native_dispatches_cudnn(
            types.SimpleNamespace(transformer = Krea2Transformer2DModel(128)), _target()
        )
        is native
    )
    assert asked == [(128, True)]


def test_the_masked_probe_passes_a_key_padding_mask(monkeypatch):
    seen = {}

    def fake(
        q,
        k,
        v,
        attn_mask = None,
        **kw,
    ):
        seen["mask"] = attn_mask
        return EFFICIENT if attn_mask is not None else FLASH

    monkeypatch.setattr(torch, "_fused_sdp_choice", fake)
    assert att._native_sdp_choice("cpu", torch.bfloat16, 128) == FLASH
    assert seen["mask"] is None
    assert att._native_sdp_choice("cpu", torch.bfloat16, 128, masked = True) == EFFICIENT
    mask = seen["mask"]
    assert mask.dtype == torch.bool and tuple(mask.shape) == (1, 1, 1, 256)
    assert bool(mask[..., :-1].all()) and not bool(mask[..., -1])


def test_the_table_names_dits_whose_every_block_call_is_masked():
    assert "Krea2Transformer2DModel" in att.MASKED_ATTENTION_DITS
    assert "FluxTransformer2DModel" not in att.MASKED_ATTENTION_DITS
