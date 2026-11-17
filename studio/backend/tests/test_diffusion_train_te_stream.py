# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for streaming DiT-training text encoders through the card (diffusion_te_stream): the auto policy
streams only encoders that do not fit, keeps as many leading layers resident as the budget allows, honours the kill
switch, and the trainer's whole-module move skips a streamed encoder."""

from __future__ import annotations

import pytest
import torch

import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_te_stream as ts

MIB = 1 << 20


class _Layer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim, bias = False)


class _Encoder(torch.nn.Module):
    """embed + N equal layers + a small ModuleList that must not be mistaken for the layers."""

    def __init__(
        self,
        n_layers = 8,
        dim = 64,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(16, dim)
        self.layers = torch.nn.ModuleList(_Layer(dim) for _ in range(n_layers))
        self.heads = torch.nn.ModuleList(torch.nn.Linear(dim, 2) for _ in range(2))
        self.to(torch.bfloat16)


class _Pipe:
    def __init__(self, enc):
        self.text_encoder = enc


class _FakeStreamer:
    made = []

    def __init__(self, encoder, layers, streamed, device, pin_budget):
        self.streamed = list(streamed)
        self.pinned_bytes = 0
        _FakeStreamer.made.append(self)


@pytest.fixture
def fake_cuda(monkeypatch):
    """A CUDA host with a settable amount of free VRAM; the streamer itself is recorded, not run."""
    state = {"free": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (state["free"], 80 << 30))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(ts, "TextEncoderStreamer", _FakeStreamer)
    monkeypatch.setattr(ts, "_pin_budget", lambda: 0)
    # Layer moves are bookkeeping here: the host has no card.
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
    _FakeStreamer.made = []
    monkeypatch.delenv(ts.TE_STREAM_ENV, raising = False)
    return state


def _sizes(enc):
    layers = ts.find_layers(enc)
    sizes = [ts._module_bytes(layer) for layer in layers]
    return sizes, ts._module_bytes(enc) - sum(sizes)


def test_mode_parsing(monkeypatch):
    for raw, want in (
        ("", "auto"),
        ("auto", "auto"),
        ("0", "off"),
        ("off", "off"),
        ("1", "on"),
        ("on", "on"),
    ):
        monkeypatch.setenv(ts.TE_STREAM_ENV, raw)
        assert ts.te_stream_mode() == want


def test_find_layers_picks_the_heaviest_list():
    enc = _Encoder()
    assert ts.find_layers(enc) is enc.layers


def test_plan_resident_fits_window_then_leading_layers():
    sizes = [10] * 8
    assert ts._plan_resident(sizes, (ts.DEPTH + 1) * 10 - 1) == 0
    assert ts._plan_resident(sizes, (ts.DEPTH + 1) * 10 + 35) == 3
    assert ts._plan_resident(sizes, 10**9) == 8


def test_auto_leaves_an_encoder_that_fits(fake_cuda):
    enc = _Encoder()
    fake_cuda["free"] = ts._module_bytes(enc) + ts._MARGIN_BYTES + MIB
    assert ts.place_text_encoders(_Pipe(enc), "cuda") == []
    assert _FakeStreamer.made == []


def test_auto_streams_only_the_layers_that_do_not_fit(fake_cuda):
    enc = _Encoder()
    sizes, nonlayer = _sizes(enc)
    # Room for the non-layer weights, the streaming window and exactly two resident layers.
    fake_cuda["free"] = ts._MARGIN_BYTES + nonlayer + (ts.DEPTH + 1) * max(sizes) + 2 * sizes[0] + 1
    assert ts.place_text_encoders(_Pipe(enc), "cuda") == ["text_encoder"]
    (made,) = _FakeStreamer.made
    assert made.streamed == list(range(2, len(sizes)))
    assert getattr(enc, ts.STREAMER_ATTR) is made


def test_on_streams_every_layer_even_when_it_fits(fake_cuda, monkeypatch):
    monkeypatch.setenv(ts.TE_STREAM_ENV, "on")
    enc = _Encoder()
    fake_cuda["free"] = 80 << 30
    assert ts.place_text_encoders(_Pipe(enc), "cuda") == ["text_encoder"]
    assert _FakeStreamer.made[0].streamed == list(range(len(enc.layers)))


@pytest.mark.parametrize("device", ["cpu", "xpu", "mps"])
def test_non_cuda_devices_keep_the_whole_module_move(fake_cuda, monkeypatch, device):
    monkeypatch.setenv(ts.TE_STREAM_ENV, "on")
    assert ts.place_text_encoders(_Pipe(_Encoder()), device) == []


def test_kill_switch_and_rocm_auto(fake_cuda, monkeypatch):
    fake_cuda["free"] = 0
    monkeypatch.setenv(ts.TE_STREAM_ENV, "off")
    assert ts.place_text_encoders(_Pipe(_Encoder()), "cuda") == []
    monkeypatch.delenv(ts.TE_STREAM_ENV)
    monkeypatch.setattr(torch.version, "hip", "7.1", raising = False)
    assert ts.place_text_encoders(_Pipe(_Encoder()), "cuda") == []
    assert _FakeStreamer.made == []


def test_quantized_or_placed_encoders_are_left_alone(fake_cuda):
    fake_cuda["free"] = 0
    enc = _Encoder()
    enc.is_quantized = True
    assert ts.place_text_encoders(_Pipe(enc), "cuda") == []
    placed = _Encoder().to("meta")
    placed.embed.weight = torch.nn.Parameter(torch.empty(0, device = "meta"))
    assert ts.place_text_encoders(_Pipe(placed), "cuda") == []


def test_trainer_skips_the_whole_move_for_a_streamed_encoder(monkeypatch):
    """Before: _encoders_to_device always called enc.to(device), so an encoder bigger than the card OOMed (and the
    OOM was swallowed, leaving it split across devices)."""
    moved = []

    class _Enc:
        def to(self, device):
            moved.append(device)
            return self

    streamed, whole = _Enc(), _Enc()
    setattr(streamed, ts.STREAMER_ATTR, object())
    pipe = type(
        "P", (), {"text_encoder": streamed, "text_encoder_2": whole, "text_encoder_3": None}
    )()
    monkeypatch.setattr(ts, "place_text_encoders", lambda p, d: ["text_encoder"])
    dit._encoders_to_device(pipe, "cuda")
    assert moved == ["cuda"]
