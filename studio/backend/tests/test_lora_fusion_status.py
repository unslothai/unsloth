# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The image status says why an active LoRA set runs unfolded (not opted in, offloaded denoiser, quantized bake), from
the same decision the fold itself takes."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_lora_fuse as lora_fuse
from models.inference import DiffusionStatusResponse


class _Pipe:
    def __init__(
        self,
        loras = (("a", "p", 1.0),),
        baked = False,
    ):
        self._unsloth_loras = loras
        self._unsloth_loras_baked = baked


def _state(pipe, offload = "none"):
    return types.SimpleNamespace(pipe = pipe, offload_policy = offload, memory_mode = None)


@pytest.fixture
def opted_in(monkeypatch):
    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")


def test_not_opted_in_says_how_to_opt_in(monkeypatch):
    monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    reason = lora_fuse.lora_fusion_skip_reason(_Pipe(), "none", None, [])
    assert reason == lora_fuse.SKIP_NOT_OPTED_IN
    assert "UNSLOTH_DIFFUSION_LORA_FUSE=1" in reason


def test_an_offloaded_denoiser_is_named(opted_in, monkeypatch):
    monkeypatch.setattr(lora_fuse, "denoisers_resident", lambda *_a: False)
    assert lora_fuse.lora_fusion_skip_reason(_Pipe(), "model", None, []) == lora_fuse.SKIP_OFFLOADED


def test_a_quantized_bake_is_named_even_when_opted_out(monkeypatch):
    monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    assert (
        lora_fuse.lora_fusion_skip_reason(_Pipe(baked = True), "none", None, [])
        == lora_fuse.SKIP_QUANT_BAKED
    )


def test_a_resident_opted_in_set_is_eligible(opted_in):
    assert lora_fuse.lora_fusion_skip_reason(_Pipe(), "none", None, []) is None


def test_status_entry(opted_in, monkeypatch):
    assert diffusion_mod._lora_fusion_status(_state(_Pipe(loras = ()))) is None
    monkeypatch.setattr(lora_fuse, "fused", lambda pipe: True)
    assert diffusion_mod._lora_fusion_status(_state(_Pipe())) == {
        "fused": True,
        "skipped_reason": None,
    }
    monkeypatch.setattr(lora_fuse, "fused", lambda pipe: False)
    monkeypatch.setattr(lora_fuse, "denoisers_resident", lambda *_a: False)
    assert diffusion_mod._lora_fusion_status(_state(_Pipe(), offload = "model")) == {
        "fused": False,
        "skipped_reason": lora_fuse.SKIP_OFFLOADED,
    }


def test_status_entry_never_raises():
    assert diffusion_mod._lora_fusion_status(object()) is None


def test_the_fold_takes_the_same_decision(opted_in, monkeypatch):
    """_fuse_constant_loras folds only when lora_fusion_skip_reason says nothing, so status cannot drift from it."""
    calls = []
    monkeypatch.setattr(lora_fuse, "fused", lambda pipe: False)
    monkeypatch.setattr(lora_fuse, "fuse", lambda pipe, d, log: calls.append("fuse") or False)
    monkeypatch.setattr(diffusion_mod, "_lora_fuse_denoisers", lambda pipe: [object()])
    backend = object.__new__(diffusion_mod.DiffusionBackend)

    monkeypatch.setattr(lora_fuse, "lora_fusion_skip_reason", lambda *_a: "skip")
    backend._fuse_constant_loras(_state(_Pipe()), _Pipe())
    assert calls == []

    monkeypatch.setattr(lora_fuse, "lora_fusion_skip_reason", lambda *_a: None)
    backend._fuse_constant_loras(_state(_Pipe()), _Pipe())
    assert calls == ["fuse"]


def test_the_response_model_carries_the_field():
    entry = {"fused": False, "skipped_reason": lora_fuse.SKIP_NOT_OPTED_IN}
    assert DiffusionStatusResponse(loaded = True, lora_fusion = entry).lora_fusion == entry
