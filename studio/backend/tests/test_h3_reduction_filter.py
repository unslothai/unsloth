# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 pins inductor's reduction-config filter on every arch (kill switch ``UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK=1``)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402


def _h3():
    from core.inference import video_families
    return next(f for f in video_families._FAMILIES if f.name == "minimax-h3")


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (9, 0), (10, 0), (12, 0), None])
def test_h3_filters_reductions_on_every_arch(monkeypatch, cap):
    monkeypatch.setattr(cc, "_device_capability", lambda: cap)
    assert _h3().filter_reduction_configs is True
    assert cc.family_filters_reductions(_h3()) is True


def test_the_filter_reaches_h3s_regional_compile(monkeypatch):
    seen = {}

    def fake_compile(pipe, logger, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", fake_compile)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, backend = "cuda")
    ds.apply_speed_optims(
        types.SimpleNamespace(), target, is_gguf = False, family = _h3(), speed_mode = ds.SPEED_DEFAULT
    )
    assert seen["filter_reductions"] is True
