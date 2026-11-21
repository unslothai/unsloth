# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto's frozen fp8 base: picked on sm89+ from 0.9x dense free VRAM up to the bf16 band, and loaded host-first so
the bf16 base is never whole on the device (the conversion streams Linear by Linear)."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_frozen_fp8 as ff8


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 32, bias = False)
        self.ff = nn.Linear(32, 64)
        self.odd = nn.Linear(30, 32)  # not 16-aligned: stays bf16
        self.norm = nn.LayerNorm(32)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.proj_out = nn.Linear(32, 32)
        self.to(torch.bfloat16).requires_grad_(False)


def test_convert_frozen_fp8_streams_host_linears_to_the_device():
    model = _Tiny()
    names = {f"blocks.{i}.{n}" for i in range(2) for n in ("to_q", "ff")}
    done, failed = ff8.convert_frozen_fp8(model, names, device = "meta")
    assert (done, failed) == (4, 0)
    for i in range(2):
        b = model.blocks[i]
        assert isinstance(b.to_q, ff8.FrozenFp8Linear) and b.to_q.weight.device.type == "meta"
        assert b.ff.weight.dtype == torch.float8_e4m3fn and b.ff.bias.device.type == "meta"
        # Untargeted modules are left where they were for the caller's final move.
        assert b.odd.weight.device.type == "cpu" and b.norm.weight.device.type == "cpu"


def test_convert_frozen_fp8_without_device_keeps_the_old_in_place_behaviour():
    model = _Tiny()
    done, _ = ff8.convert_frozen_fp8(model, {"blocks.0.to_q"})
    assert done == 1 and model.blocks[0].to_q.weight.device.type == "cpu"


def test_apply_fp8_training_lands_a_host_base_on_the_device(monkeypatch):
    model = _Tiny()
    events = []
    ok = dit._apply_fp8_training(model, events.append, None, device = "meta")
    assert ok is True
    assert all(p.device.type == "meta" for p in model.parameters())
    assert all(b.device.type == "meta" for b in model.buffers())
    assert isinstance(model.blocks[0].to_q, ff8.FrozenFp8Linear)
    assert type(model.blocks[0].odd) is nn.Linear and type(model.proj_out) is nn.Linear


def test_apply_fp8_training_falls_back_to_bf16_on_the_device(monkeypatch):
    model = _Tiny()

    def boom(*a, **k):
        raise RuntimeError("no fp8 here")

    monkeypatch.setattr(dit, "_convert_fp8", boom)
    events = []
    assert dit._apply_fp8_training(model, events.append, None, device = "meta") is False
    assert all(p.device.type == "meta" for p in model.parameters())
    assert type(model.blocks[0].to_q) is nn.Linear


def test_fp8_without_a_hosted_file_loads_on_the_host(monkeypatch):
    moved = []

    class _T(nn.Module):
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *a, **k):
            moved.append(a)
            return self

    monkeypatch.setattr(dit, "_load_fp8_prequant_transformer", lambda *a, **k: None)
    cfg = types.SimpleNamespace(base_model = "x/y", hf_token = None)
    dit._load_dit_transformer(_T, cfg, "cuda", "fp8")
    assert moved == []
    dit._load_dit_transformer(_T, cfg, "cuda", "bf16")
    assert moved == [("cuda",)]


@pytest.mark.parametrize(
    "free, cap, expected",
    [
        (19.0, (8, 9), "bf16"),  # > 1.5x
        (16.4, (8, 9), "fp8"),  # 1.33x
        (11.5, (8, 9), "fp8"),  # 0.93x: streamed fp8, not nf4
        (10.9, (8, 9), "nf4"),  # 0.89x
        (16.4, (8, 6), "int8"),
        (11.5, (8, 6), "nf4"),
    ],
)
def test_pick_auto_precision_fp8_band(free, cap, expected):
    assert dit._pick_auto_precision(False, "cuda", free, 12.3, cap, True, True) == expected


def test_auto_fp8_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIT_AUTO_FP8", "0")
    assert dit._auto_fp8_enabled() is False
    monkeypatch.setenv("UNSLOTH_DIT_AUTO_FP8", "1")
    assert dit._auto_fp8_enabled() is True


@pytest.mark.parametrize(
    "family, expected", [("qwen-image", "int8"), ("moss-image", "int8"), ("z-image", "fp8")]
)
def test_auto_fp8_respects_the_family_train_deny(monkeypatch, family, expected):
    class _FakeCuda:
        @staticmethod
        def mem_get_info():
            return (int(1.3 * dit._SPECS[family].dense_bf16_gb * 1e9), int(80 * 1e9))

        @staticmethod
        def get_device_capability():
            return (8, 9)

    monkeypatch.delenv("UNSLOTH_DIT_AUTO_FP8", raising = False)
    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    monkeypatch.setattr(dit, "has_functional_torchao", lambda: True)
    monkeypatch.setattr(dit, "repo_is_prequantized", lambda repo: False)
    spec = dit._SPECS[family]
    cfg = types.SimpleNamespace(
        base_precision = "auto", base_model = "some/dense-base", mixed_precision = "bf16"
    )
    assert dit._resolve_base_precision(cfg, spec, "cuda") == expected
