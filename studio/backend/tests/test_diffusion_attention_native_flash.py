# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto attention on cards where native flash measured faster than the cuDNN pin: native SDPA when torch's own
dispatch picks FLASH at every DiT head dim; the pin on every other card, for an explicit request and with the kill
switch."""

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


def _target(dtype = torch.bfloat16):
    return types.SimpleNamespace(device = "cuda", dtype = dtype)


@pytest.fixture
def ampere(monkeypatch):
    """An NVIDIA SM80 card whose native dispatch picks FLASH up to head_dim 256 (torch 2.14 / A100)."""
    att._NATIVE_DISPATCH_CACHE.clear()
    att._NATIVE_FLASH_CACHE.clear()
    monkeypatch.delenv(att.NATIVE_FLASH_ATTN_ENV, raising = False)
    monkeypatch.delenv(att.BLACKWELL_NATIVE_ATTN_ENV, raising = False)
    monkeypatch.setenv(att.CUDNN_HEAD_DIM_PROBE_ENV, "0")
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (8, 0))
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(att, "guard_rocm_fused_sdpa", lambda target, logger = None: None)
    state = {"cap": (8, 0), "asked": []}
    monkeypatch.setattr(att, "_device_capability", lambda device: state["cap"])

    def choice(device, dtype, dim):
        state["asked"].append((dtype, dim))
        return FLASH if dim <= 256 else EFFICIENT

    monkeypatch.setattr(att, "_native_sdp_choice", choice)
    yield state
    att._NATIVE_DISPATCH_CACHE.clear()
    att._NATIVE_FLASH_CACHE.clear()


def test_the_measured_table_holds_only_pre_blackwell_cards():
    assert att.FLASH_FASTER_CAPABILITIES
    assert all(cap < (10, 0) for cap in att.FLASH_FASTER_CAPABILITIES)
    assert (9, 0) not in att.FLASH_FASTER_CAPABILITIES  # Hopper unmeasured: keeps the pin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_auto_leaves_native_flash_on_a_measured_card(ampere, dtype, head_dim):
    dit = _Dit(head_dim)
    target = _target(dtype)
    pick = select_attention_backend(target, "auto", speed_active = True)
    assert pick == "_native_cudnn" and isinstance(pick, att._AutoCudnn)
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = dit), pick, target = target) is None
    )
    assert dit.set_to in (None, "native")
    assert ampere["asked"] == [(dtype, head_dim)]


def test_an_explicit_cudnn_request_is_still_pinned(ampere):
    dit = _Dit(128)
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = dit), "_native_cudnn", target = _target()
    )
    assert engaged == "_native_cudnn" and dit.set_to == "_native_cudnn"
    assert ampere["asked"] == []


@pytest.mark.parametrize(
    "case",
    [
        "unmeasured_card",
        "hopper",
        "not_flash",
        "fp32",
        "kill_switch",
        "unknown_head_dim",
        "no_target",
    ],
)
def test_auto_keeps_the_cudnn_pin_wherever_flash_is_unproven(ampere, monkeypatch, case):
    dit = _Dit(512 if case == "not_flash" else 128)
    target = _target(torch.float32 if case == "fp32" else torch.bfloat16)
    if case == "unmeasured_card":
        ampere["cap"] = (8, 6)
    if case == "hopper":
        ampere["cap"] = (9, 0)
    if case == "kill_switch":
        monkeypatch.setenv(att.NATIVE_FLASH_ATTN_ENV, "0")
    if case == "unknown_head_dim":
        dit.config = {}
    pick = select_attention_backend(target, "auto", speed_active = True)
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = dit), pick, target = None if case == "no_target" else target
    )
    assert engaged == "_native_cudnn" and dit.set_to == "_native_cudnn"
    assert type(engaged) is str


def test_mixed_head_dims_need_flash_at_every_one(ampere):
    pipe = types.SimpleNamespace(transformer = _Dit(128), transformer_2 = _Dit(512))
    assert att._native_flash_measured_faster(pipe, _target()) is False
    ampere["asked"].clear()
    att._NATIVE_FLASH_CACHE.clear()
    one = types.SimpleNamespace(transformer = _Dit(64))
    assert att._native_flash_measured_faster(one, _target()) is True
    assert att._native_flash_measured_faster(one, _target()) is True
    assert ampere["asked"] == [(torch.bfloat16, 64)]  # cached per (device, dtype, head_dim)


def test_an_unaskable_probe_keeps_the_pin(ampere, monkeypatch):
    def boom(device, dtype, dim):
        raise RuntimeError("no device")

    monkeypatch.setattr(att, "_native_sdp_choice", boom)
    assert (
        att._native_flash_measured_faster(types.SimpleNamespace(transformer = _Dit(128)), _target())
        is False
    )


def test_speed_off_never_reaches_the_check(ampere):
    assert select_attention_backend(_target(), "auto", speed_active = False) is None
    assert ampere["asked"] == []
