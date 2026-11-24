# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Under a streamed tier the LTX-2 VAE encoder waits on the host; ``vae.encode`` brings it to the card for the call
only, and the VRAM floor no longer counts it."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_ltx2 as L  # noqa: E402


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(L.LTX2_PARK_VAE_ENCODER_ENV, raising = False)


class AutoencoderKLLTX2Video(torch.nn.Module):
    """Stand-in with the LTX-2 VAE's class name: an encoder used only by ``encode``, a decoder by ``decode``."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(256, 256, bias = False)  # 256 KiB fp32
        self.decoder = torch.nn.Linear(256, 512, bias = False)  # 512 KiB fp32
        self.seen = []

    def encode(self, x):
        self.seen.append(str(next(self.encoder.parameters()).device))
        return self.encoder(x)


@pytest.fixture
def card(monkeypatch):
    """CPU stands in for the card: ``Module.to`` records where each module was sent, ``_module_device`` reads it."""
    where: dict[int, str] = {}
    real_to = torch.nn.Module.to

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is not None:
            for m in self.modules():
                where[id(m)] = str(target)
        return real_to(self, "cpu")

    monkeypatch.setattr(torch.nn.Module, "to", to)
    monkeypatch.setattr(
        L,
        "_module_device",
        lambda m: torch.device(where.get(id(m), "cuda:0")) if m is not None else None,
    )
    return where


def _encoder_device(vae, where):
    return where.get(id(vae.encoder), "cuda:0")


def test_streamed_tier_parks_the_encoder_and_encode_brings_it_back_for_the_call(card):
    vae = AutoencoderKLLTX2Video()
    pipe = types.SimpleNamespace(vae = vae)
    moved = L.park_ltx2_vae_encoder(pipe, "streaming")
    assert moved == 256 * 256 * 4
    assert _encoder_device(vae, card) == "cpu"
    x = torch.randn(2, 256)
    expected = x @ vae.encoder.weight.T
    got = vae.encode(x)
    assert torch.equal(got, expected)
    assert vae.seen == [
        "cpu"
    ]  # the stand-in's tensors stay on CPU; the move itself is what is recorded
    assert _encoder_device(vae, card) == "cpu"
    # a later move back onto the card is undone by the next call (the request guard runs it), wrapper not stacked
    vae.encoder.to("cuda:0")
    wrapped = vae.encode
    assert L.park_ltx2_vae_encoder(pipe, "streaming") == 256 * 256 * 4
    assert vae.encode is wrapped and _encoder_device(vae, card) == "cpu"
    assert L.park_ltx2_vae_encoder(pipe, "streaming") == 0


def test_encode_runs_with_the_encoder_on_the_card(card):
    vae = AutoencoderKLLTX2Video()
    trail = []
    inner = vae.encode

    def spy(x):
        trail.append(_encoder_device(vae, card))
        return inner(x)

    vae.encode = spy
    L.park_ltx2_vae_encoder(types.SimpleNamespace(vae = vae), "group")
    vae.encode(torch.randn(1, 256))
    assert trail == ["cuda:0"] and _encoder_device(vae, card) == "cpu"


@pytest.mark.parametrize("policy", ["none", "model"])
def test_resident_and_whole_module_tiers_keep_the_encoder(card, policy):
    vae = AutoencoderKLLTX2Video()
    assert L.park_ltx2_vae_encoder(types.SimpleNamespace(vae = vae), policy) == 0
    assert _encoder_device(vae, card) == "cuda:0" and "encode" not in vae.__dict__


def test_other_vaes_and_the_kill_switch_keep_the_encoder(card, monkeypatch):
    other = AutoencoderKLLTX2Video()
    other.__class__ = type("AutoencoderKLWan", (AutoencoderKLLTX2Video,), {})
    assert L.park_ltx2_vae_encoder(types.SimpleNamespace(vae = other), "streaming") == 0
    monkeypatch.setenv(L.LTX2_PARK_VAE_ENCODER_ENV, "0")
    vae = AutoencoderKLLTX2Video()
    assert L.park_ltx2_vae_encoder(types.SimpleNamespace(vae = vae), "streaming") == 0
    assert _encoder_device(vae, card) == "cuda:0"


def test_vram_floor_leaves_out_a_parked_encoder(monkeypatch):
    from core.inference.video import _video_offload_vram_floor_mib

    def _module(mib):
        module = torch.nn.Module()
        module.register_buffer("w", torch.zeros(mib * 1024 * 1024 // 2, dtype = torch.bfloat16))
        return module

    vae = AutoencoderKLLTX2Video()
    vae.encoder = _module(6)
    vae.decoder = _module(8)
    pipe = types.SimpleNamespace(
        components = {"transformer": _module(10), "text_encoder": _module(12), "vae": vae}
    )

    def plan(policy):
        return types.SimpleNamespace(
            offload_policy = policy, stream_text_encoders = True, stream_transformer = True
        )

    # streaming: decoder only (8) plus the largest streamed unit (12); the decode phase holds the decoder alone
    assert _video_offload_vram_floor_mib(pipe, plan("streaming")) == 20
    assert _video_offload_vram_floor_mib(pipe, plan("streaming"), phase = "decode") == 8
    # whole-module offload is unchanged: the largest component, encoder included
    assert _video_offload_vram_floor_mib(pipe, plan("model")) == 14
    monkeypatch.setenv(L.LTX2_PARK_VAE_ENCODER_ENV, "0")
    assert _video_offload_vram_floor_mib(pipe, plan("streaming")) == 26
    assert _video_offload_vram_floor_mib(pipe, plan("streaming"), phase = "decode") == 14
