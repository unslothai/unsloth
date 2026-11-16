# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The unset switch on strided processors (``install_auto``): a DiT listed in ``AUTO_DITS`` races its incumbent like
the dispatch route (accuracy check, card race, 2048 / long races, ``MIN_TOKENS`` floor); every other DiT, speed off,
ROCm, the kill switches and an explicit attention request leave its processors as they were. CPU: the card is
faked."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_dispatch as D

_TARGET = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


class _Processor:
    _unsloth_int8 = False
    _unsloth_int8_min_tokens = 0
    _parallel_config = None
    _attention_backend = "_native_cudnn"


class _Attention(torch.nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(heads * head_dim, heads * head_dim)])
        self.processor = _Processor()


def _pipe(cls_name: str = "Flux3Transformer2DModel", head_dims = (128, 128, 64, 96)):
    dit = type(cls_name, (torch.nn.Module,), {})()
    dit.blocks = torch.nn.ModuleList([_Attention(2, d) for d in head_dims])
    return types.SimpleNamespace(transformer = dit)


def _flags(pipe):
    return [
        (m.processor._unsloth_int8, m.processor._unsloth_int8_min_tokens)
        for m in pipe.transformer.blocks
    ]


@pytest.fixture
def card(monkeypatch):
    """A faked sm89 card; ``found`` is what the accuracy check + card race return; ``wins`` the extra race points."""
    from core.inference import diffusion_flux3_attn, rocm_bf16

    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(D.DISPATCH_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I, "_secondary_cards", lambda *a, **k: ())
    strided = []
    monkeypatch.setattr(
        diffusion_flux3_attn,
        "ensure_strided_processors",
        lambda pipe, target, logger = None: strided.append(pipe) or 4,
    )
    state = types.SimpleNamespace(
        found = {"ok": True, "min_tokens": 0, "impl": "cuda"}, wins = {}, seen = [], strided = strided
    )

    def verdict(
        device,
        dtype,
        heads,
        head_dim,
        incumbent = None,
        incumbent_name = "native",
        timed = True,
    ):
        state.seen.append((heads, head_dim, incumbent_name, timed))
        return dict(state.found)

    monkeypatch.setattr(I, "verdict", verdict)
    monkeypatch.setattr(D, "_race_wins", lambda *a, **k: state.wins.get(a[-1], False))
    return state


def test_listed_family_races_and_floors_at_min_tokens(card):
    """Won at both race sizes: the floor is MIN_TOKENS (a forced load runs every length), so short text-only calls
    stay on the incumbent. Timed, never forced; the head dim the card has no kernel for is left alone."""
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe) == [(True, D.MIN_TOKENS)] * 3 + [(False, 0)]
    assert card.seen and all(timed for *_, timed in card.seen)
    assert {name for _, _, name, _ in card.seen} == {"_native_cudnn"}
    assert card.strided == [pipe]
    # the forced path keeps its own floor (0): the request by name is unchanged
    forced = _pipe()
    assert (
        I.install_int8_attention(
            forced.transformer, device = "cuda", dtype = torch.bfloat16, force = True
        )
        == 3
    )
    assert _flags(forced)[:3] == [(True, 0)] * 3


def test_floor_follows_the_extra_races(card):
    """Won at 4096 only: the 2048 race places the floor; lost at both: only the long race can route (long calls)."""
    card.found = {"ok": True, "min_tokens": 2560, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe)[0] == (True, (4096 + D.MID_RACE_TOKENS) // 2)
    card.found = {"ok": True, "min_tokens": None, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0
    assert _flags(pipe) == [(False, 0)] * 4
    card.wins[D.LONG_RACE_TOKENS[0]] = True
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 3
    assert _flags(pipe)[0] == (True, sum(D.LONG_RACE_TOKENS) // 2)


def test_accuracy_check_failure_keeps_the_incumbent(card):
    card.found = {"ok": False, "min_tokens": None, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET) == 0
    assert _flags(pipe) == [(False, 0)] * 4


@pytest.mark.parametrize(
    "case",
    [
        "unlisted",
        "speed_off",
        "env_off",
        "env_force",
        "dispatch_off",
        "rocm",
        "cpu",
        "no_transformer",
    ],
)
def test_auto_route_stays_off(card, monkeypatch, case):
    """Never races: an unlisted DiT, speed off (bit-identical), the kill switches, the env switch 1 (the forced
    install runs instead), ROCm, a CPU load, a pipeline with no transformer."""
    from core.inference import rocm_bf16

    pipe = _pipe("ZImageTransformer2DModel" if case == "unlisted" else "Flux3Transformer2DModel")
    target = _TARGET
    speed = True
    if case == "speed_off":
        speed = False
    elif case == "env_off":
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, "0")
    elif case == "env_force":
        monkeypatch.setenv(I.INT8_ATTENTION_ENV, "1")
    elif case == "dispatch_off":
        monkeypatch.setenv(D.DISPATCH_ENV, "0")
    elif case == "rocm":
        monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: True)
    elif case == "cpu":
        target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    elif case == "no_transformer":
        pipe = types.SimpleNamespace(transformer = None)
    assert I.install_auto(pipe, target, speed_active = speed) == 0
    assert not card.seen and not card.strided
    if case != "no_transformer":
        assert _flags(pipe) == [(False, 0)] * 4


def test_apply_attention_backend_races_only_an_automatic_pick(card, monkeypatch):
    """No request / "auto": the strided auto install runs over a native pick and its name is returned. An explicit
    backend request, or a non-native pick, never races."""
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "_native_cudnn")
    monkeypatch.setattr(D, "install", lambda *a, **k: 0)
    calls = []
    monkeypatch.setattr(I, "install_auto", lambda *a, **k: calls.append(k.get("speed_active")) or 2)
    for request in (None, "", "auto", "AUTO"):
        assert (
            A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, requested = request)
            == I.BACKEND
        )
    assert calls == [True] * 4
    for request in ("native", "sdpa", "cudnn", "flash"):
        assert (
            A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, requested = request)
            == "_native_cudnn"
        )
    assert len(calls) == 4
    assert (
        A.apply_attention_backend(object(), "_native_cudnn", target = _TARGET, speed_active = False)
        == I.BACKEND
    )
    assert calls[-1] is False
    monkeypatch.setattr(A, "_apply_attention_backend", lambda *a, **k: "sage_hub")
    assert A.apply_attention_backend(object(), "sage", target = _TARGET) == "sage_hub"
    assert len(calls) == 5


def test_dispatch_route_skips_the_strided_processors_of_a_listed_family(card, monkeypatch):
    """FLUX.3 is listed for the unset switch, but its strided processors are the strided install's: the dispatch
    route takes none of them (it would only take stock processors the strided install could not convert)."""
    pipe = _pipe()
    monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])
    assert D.routable_modules(pipe.transformer) == []
    assert D.install(pipe, _TARGET) == 0


def test_the_race_is_logged_and_counted(card):
    """With Studio's logger (it always passes one) the install reports each verdict and the routed count."""
    import logging

    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("test_int8_auto")
    log.setLevel(logging.INFO)
    log.addHandler(Grab())
    card.found = {"ok": True, "min_tokens": 2560, "impl": "cuda"}
    pipe = _pipe()
    assert I.install_auto(pipe, _TARGET, logger = log) == 3
    assert any("2048 race min_tokens=" in r for r in records), records
    assert any("on 3 attention modules" in r for r in records), records
    card.found = {"ok": True, "min_tokens": 0, "impl": "cuda", "long": {"ok": True}}
    forced = _pipe()
    assert (
        I.install_int8_attention(
            forced.transformer, device = "cuda", dtype = torch.bfloat16, force = True, logger = log
        )
        == 3
    )
    assert any("long_ok=True" in r and "(forced)" in r for r in records), records
