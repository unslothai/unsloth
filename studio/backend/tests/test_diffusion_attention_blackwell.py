# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto attention on Blackwell: native SDPA when torch's own dispatch already runs cuDNN at every DiT head dim (the
pin changes no kernel and costs a per-call wrapper), the cuDNN pin everywhere else and for an explicit request."""

from __future__ import annotations

import types

import pytest
import torch

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend, select_attention_backend

CUDNN = int(torch.nn.attention.SDPBackend.CUDNN_ATTENTION)
FLASH = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION)


class _Dit:
    def __init__(self, head_dim = 128):
        self.config = {"attention_head_dim": head_dim}
        self.set_to = None

    def set_attention_backend(self, name):
        self.set_to = name


def _target(dtype = torch.bfloat16):
    return types.SimpleNamespace(device = "cuda", dtype = dtype)


@pytest.fixture
def blackwell(monkeypatch):
    """An NVIDIA SM100 card whose native dispatch picks cuDNN up to head_dim 128 and FLASH above (torch 2.13 / B200)."""
    att._NATIVE_DISPATCH_CACHE.clear()
    monkeypatch.delenv(att.BLACKWELL_NATIVE_ATTN_ENV, raising = False)
    monkeypatch.setenv(att.CUDNN_HEAD_DIM_PROBE_ENV, "0")
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (10, 0))
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: "cuda:0")
    monkeypatch.setattr(att, "guard_rocm_fused_sdpa", lambda target, logger = None: None)
    state = {"cap": (10, 0), "asked": []}
    monkeypatch.setattr(att, "_device_capability", lambda device: state["cap"])

    def choice(device, dtype, dim):
        state["asked"].append((dtype, dim))
        return CUDNN if dim <= 128 else FLASH

    monkeypatch.setattr(att, "_native_sdp_choice", choice)
    yield state
    att._NATIVE_DISPATCH_CACHE.clear()


def test_auto_pick_still_reads_as_cudnn_for_every_caller(blackwell):
    pick = select_attention_backend(_target(), "auto", speed_active = True)
    assert pick == "_native_cudnn" and isinstance(pick, att._AutoCudnn)
    explicit = select_attention_backend(_target(), "cudnn", speed_active = True)
    assert explicit == "_native_cudnn" and not isinstance(explicit, att._AutoCudnn)
    assert select_attention_backend(_target(), "auto", speed_active = False) is None


def test_blackwell_auto_leaves_native_when_dispatch_already_picks_cudnn(blackwell):
    dit = _Dit(128)
    pick = select_attention_backend(_target(), "auto", speed_active = True)
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = dit), pick, target = _target())
        is None
    )
    assert dit.set_to in (None, "native")
    assert blackwell["asked"] == [(torch.bfloat16, 128)]


def test_an_explicit_cudnn_request_is_still_pinned_on_blackwell(blackwell):
    dit = _Dit(128)
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = dit), "_native_cudnn", target = _target()
    )
    assert engaged == "_native_cudnn" and dit.set_to == "_native_cudnn"
    assert blackwell["asked"] == []


@pytest.mark.parametrize(
    "case",
    ["hopper", "head_dim_256", "fp32", "kill_switch", "unknown_head_dim", "no_target"],
)
def test_auto_keeps_the_cudnn_pin_wherever_native_could_differ(blackwell, monkeypatch, case):
    dit = _Dit(256 if case == "head_dim_256" else 128)
    target = _target(torch.float32 if case == "fp32" else torch.bfloat16)
    if case == "hopper":
        blackwell["cap"] = (9, 0)
    if case == "kill_switch":
        monkeypatch.setenv(att.BLACKWELL_NATIVE_ATTN_ENV, "0")
    if case == "unknown_head_dim":
        dit.config = {}
    pick = select_attention_backend(target, "auto", speed_active = True)
    engaged = apply_attention_backend(
        types.SimpleNamespace(transformer = dit), pick, target = None if case == "no_target" else target
    )
    assert engaged == "_native_cudnn" and dit.set_to == "_native_cudnn"
    assert type(engaged) is str  # the marker never leaks past apply (status, cache keys)


def test_mixed_head_dims_need_cudnn_at_every_one(blackwell):
    pipe = types.SimpleNamespace(transformer = _Dit(128), transformer_2 = _Dit(256))
    assert att._native_dispatches_cudnn(pipe, _target()) is False
    blackwell["asked"].clear()
    att._NATIVE_DISPATCH_CACHE.clear()
    assert (
        att._native_dispatches_cudnn(types.SimpleNamespace(transformer = _Dit(64)), _target()) is True
    )
    assert (
        att._native_dispatches_cudnn(types.SimpleNamespace(transformer = _Dit(64)), _target()) is True
    )
    assert blackwell["asked"] == [(torch.bfloat16, 64)]  # cached per (device, dtype, head_dim)


def test_an_unaskable_probe_keeps_the_pin(blackwell, monkeypatch):
    def boom(device, dtype, dim):
        raise RuntimeError("no device")

    monkeypatch.setattr(att, "_native_sdp_choice", boom)
    assert (
        att._native_dispatches_cudnn(types.SimpleNamespace(transformer = _Dit(128)), _target())
        is False
    )
