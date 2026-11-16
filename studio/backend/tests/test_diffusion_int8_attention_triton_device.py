# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton launches on the current device and stream, so every INT8 attention Triton launch runs under the card of
its operands (``quantise``, ``attend``, ``int8_attention``, ``half_attention``; fused and per-quantiser grids), and a
call on the current card does not switch at all. CPU only: fake CUDA tensors on cuda:1, a recorded current device,
and launchers that record the device they were launched under."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from torch._subclasses import fake_tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from core.inference import diffusion_int8_attention_triton as T


class _Launcher:
    def __init__(self, log, name, state):
        self.log, self.name, self.state = log, name, state

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.log.append((self.name, self.state["device"]))

        return launch


@pytest.fixture
def card(monkeypatch):
    state = {"device": 0, "switches": 0}
    log = []

    @contextlib.contextmanager
    def device(index):
        index = index.index if isinstance(index, torch.device) else int(index)
        prev, state["device"] = state["device"], index
        state["switches"] += 1
        try:
            yield
        finally:
            state["device"] = prev

    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["device"])
    monkeypatch.setattr(torch.cuda, "device", device)
    # A fake tensor's first use of a device makes a real tensor there (torch.zeros on ROCm builds), which fails for
    # the fake cuda:1 on a one-GPU host; the launchers here are recorded, so no real device is needed.
    monkeypatch.setattr(fake_tensor, "init_gpu_context", lambda device: None, raising = False)
    names = ("stats", "quant_rows", "quant_v", "transpose_v", "prep", "kv", "attn")
    kk = types.SimpleNamespace(**{n: _Launcher(log, n, state) for n in names})
    monkeypatch.setattr(T, "kernels", lambda: kk)
    monkeypatch.setattr(T, "prefetch_fix", lambda: False)
    monkeypatch.setattr(T, "_PLANS", {})
    return state, log


def _qkv(index, D = 128):
    q = torch.empty(1, 200, 2, D, device = f"cuda:{index}", dtype = torch.float16)
    return q, torch.empty_like(q), torch.empty_like(q)


@pytest.mark.parametrize("pv", sorted(T.PV_MODES))
@pytest.mark.parametrize("fused", [True, False])
def test_int8_attention_launches_on_the_operands_card(card, monkeypatch, pv, fused):
    state, log = card
    if pv == "half" and (not torch.backends.cuda.is_built() or 0 < torch.cuda.device_count() < 2):
        # The fake copy into the half V buffer reaches the CUDA runtime: it needs a CUDA build, and with real GPUs
        # present, a real second card behind cuda:1.
        pytest.skip("the fake half V copy needs a CUDA build and, with GPUs present, a second card")
    if not fused:
        monkeypatch.setattr(T, "GRID_LIMIT", 0)
    with FakeTensorMode():
        q, k, v = _qkv(1)
        T.int8_attention(q, k, v, dict(T.DEFAULT_CONFIGS[128]), pv = pv)
    assert log and {d for _, d in log} == {1}, log
    assert {n for n, _ in log} >= (
        {"prep", "kv", "attn"} if fused else {"quant_rows", "stats", "attn"}
    )
    assert state["device"] == 0


def test_quantise_and_attend_launch_on_the_operands_card(card):
    state, log = card
    with FakeTensorMode():
        q, k, v = _qkv(1)
        p = T.quantise(q, k, v, dict(T.DEFAULT_CONFIGS[128]))
        T.attend(p)
    assert [n for n, _ in log] == ["prep", "kv", "attn"]
    assert {d for _, d in log} == {1}


def test_half_attention_launches_on_the_operands_card(card):
    state, log = card
    with FakeTensorMode():
        q, k, v = _qkv(1, D = 64)
        T.half_attention(q, k, v)
    assert log == [("attn", 1)]


def test_same_card_does_not_switch(card):
    state, log = card
    with FakeTensorMode():
        q, k, v = _qkv(0)
        T.int8_attention(q, k, v, dict(T.DEFAULT_CONFIGS[128]))
        T.half_attention(q, k, v)
    assert {d for _, d in log} == {0}
    assert state["switches"] == 0


def test_dispatcher_runs_the_triton_path_on_the_operands_card(card, monkeypatch):
    # Studio's dispatcher holds torch.cuda.device(q.device) and calls the unchecked entry point
    from core.inference import diffusion_int8_attention as I

    state, log = card
    checks = []
    real = T._on_card
    monkeypatch.setattr(T, "_on_card", lambda t: checks.append(t) or real(t))
    with FakeTensorMode():
        q, k, v = _qkv(1)
        I._run(q, k, v, "triton", dict(T.DEFAULT_CONFIGS[128]))
    assert [n for n, _ in log] == ["prep", "kv", "attn"]
    assert {d for _, d in log} == {1}
    assert checks == [] and state["device"] == 0


def test_fast_launch_is_off_without_triton(monkeypatch):
    # The direct launch reads triton.runtime; a host without Triton (the CPU CI legs) must take the regular launch.
    monkeypatch.setattr(T, "triton", None)
    monkeypatch.setitem(T._FAST_LAUNCH, "ok", True)
    assert T._fast_launch_ok() is False
