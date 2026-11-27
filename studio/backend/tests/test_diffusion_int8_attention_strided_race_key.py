# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 dispatch race names a strided cuDNN incumbent by the call each card actually times: the strided call on
the pin's probed cards at its probed dtype, diffusers' contiguous call everywhere else."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.models.attention_dispatch")

from core.inference import diffusion_int8_attention_dispatch as D  # noqa: E402
from core.inference import diffusion_qwenimage21_attn as Q  # noqa: E402

_TARGET = types.SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)


@pytest.fixture(autouse = True)
def _clean():
    D._DECISIONS.clear()
    yield
    D._DECISIONS.clear()


def test_strided_race_name_follows_the_call_the_card_times():
    pin = Q.StridedCudnn((0,), torch.bfloat16)
    assert D._race_name(pin, 0, torch.bfloat16) == "dispatch:_native_cudnn:strided"
    # an unprobed card or another dtype runs diffusers' contiguous call: it shares the plain verdict
    assert D._race_name(pin, 1, torch.bfloat16) == "dispatch:_native_cudnn"
    assert D._race_name(pin, 0, torch.float16) == "dispatch:_native_cudnn"
    assert D._race_name(Q.StridedCudnn((0, 1), torch.bfloat16), 1, torch.bfloat16).endswith(
        ":strided"
    )
    assert D._race_name(Q.StridedCudnn((0,)), 0, torch.float16).endswith(":strided")


@pytest.mark.parametrize(
    "cards,dtype,expected",
    [
        ((0,), torch.bfloat16, {0: ":strided", 1: ""}),
        ((0, 1), torch.bfloat16, {0: ":strided", 1: ":strided"}),
        ((0, 1), torch.float16, {0: "", 1: ""}),
    ],
)
def test_dispatch_race_keys_each_card_by_its_timed_call(monkeypatch, cards, dtype, expected):
    """Each card races the pin; its verdict name says whether its timed incumbent is strided."""
    from core.inference import diffusion_int8_attention as I8
    from core.inference import rocm_bf16

    monkeypatch.delenv(I8.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I8, "int8_attention_op", lambda: None)
    monkeypatch.setattr(I8, "implementation", lambda index, head_dim: "cuda")
    monkeypatch.setattr(I8, "register_route", lambda *a, **k: None)
    monkeypatch.setattr(D, "_secondary_cards", lambda *a, **k: (1,))
    monkeypatch.setattr(D, "install_dispatch_route", lambda: 1)
    monkeypatch.setattr(D, "original_dispatch", lambda: (lambda *a, **k: None))
    names = {}

    def card_floor(device, dtype_, heads, head_dim, run, name, *a):
        names[device.index] = name
        return {"ok": True, "impl": "cuda"}, 1024

    monkeypatch.setattr(D, "_card_floor", card_floor)

    class Proc:
        _parallel_config = None

        def __call__(self, *a, **k):
            return dispatch_attention_fn  # noqa: F821 - only its name is read

    dit = type("WanTransformer3DModel", (torch.nn.Module,), {})()
    m = torch.nn.Module()
    m.heads, m.head_dim, m.processor = 2, 64, Proc()
    m.processor._attention_backend = Q.StridedCudnn(cards, dtype)
    dit.attn = m
    monkeypatch.setattr(D, "_dits", lambda p: [p.transformer])
    assert D.install(types.SimpleNamespace(transformer = dit), _TARGET) == 1
    assert names == {c: "dispatch:_native_cudnn" + suffix for c, suffix in expected.items()}
