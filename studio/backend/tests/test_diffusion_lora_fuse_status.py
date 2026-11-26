# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA fusion is skipped when the user chose speed mode Off (bit-exact), and the reason reaches the status."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion
from core.inference import diffusion_lora_fuse as lora_fuse


def _state(
    speed_mode = "off",
    source = "explicit",
    offload = "none",
):
    return types.SimpleNamespace(
        speed_mode = speed_mode,
        resolved = {"speed_mode": {"value": speed_mode, "source": source}},
        offload_policy = offload,
        memory_mode = "auto",
        cuda_graphs = (),
    )


def _pipe():
    return types.SimpleNamespace(unet = torch.nn.Linear(4, 4), _unsloth_loras = (("a", "/x", 1.0),))


@pytest.fixture
def fuse_calls(monkeypatch):
    calls = []
    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")
    monkeypatch.setattr(diffusion, "_lora_fuse_dits", lambda pipe: [])
    monkeypatch.setattr(lora_fuse, "_on_card", lambda module: True)

    def fake_fuse(
        pipe,
        denoisers,
        logger = None,
    ):
        calls.append(denoisers)
        setattr(pipe, "_unsloth_lora_fused", True)
        return True

    monkeypatch.setattr(lora_fuse, "fuse", fake_fuse)
    monkeypatch.setattr(diffusion, "invalidate_cfg_replica", lambda *a, **k: None)
    return calls


def _run(state, pipe):
    diffusion.DiffusionBackend._fuse_constant_loras(None, state, pipe)
    return diffusion.lora_fuse_status(pipe)


def test_speed_off_chosen_by_user_skips_fusion(fuse_calls):
    pipe = _pipe()
    assert _run(_state("off", "explicit"), pipe) == "bit_exact"
    assert fuse_calls == []
    assert not lora_fuse.fused(pipe)


def test_unset_speed_that_resolves_to_off_still_fuses(fuse_calls):
    pipe = _pipe()
    assert _run(_state("off", "auto"), pipe) == "fused"
    assert len(fuse_calls) == 1


def test_compiled_speed_fuses(fuse_calls):
    pipe = _pipe()
    assert _run(_state("default", "explicit"), pipe) == "fused"
    assert len(fuse_calls) == 1


def test_offload_reports_not_fused(fuse_calls):
    pipe = _pipe()
    assert _run(_state("default", "explicit", offload = "model"), pipe) == "offload"
    assert fuse_calls == []


def test_fusion_env_off_reports_disabled(fuse_calls, monkeypatch):
    monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    pipe = _pipe()
    assert _run(_state("default", "explicit"), pipe) == "disabled"
    assert fuse_calls == []


def test_status_is_none_without_loras(fuse_calls):
    pipe = _pipe()
    _run(_state("default", "explicit"), pipe)
    pipe._unsloth_loras = ()
    assert diffusion.lora_fuse_status(pipe) is None


def test_status_response_keeps_the_fuse_field():
    from models.inference import DiffusionStatusResponse
    assert DiffusionStatusResponse(loaded = True, lora_fuse = "offload").lora_fuse == "offload"
