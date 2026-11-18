# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5 (480p and 720p repacks, one DiT) pins inductor's reduction-config filter on every arch.

Without it, fresh G4 (sm120) processes rendered one seed into different clips (LPIPS 0.16 to 0.29): near-tied norm
reduction configs are benchmarked per process and sum in different orders. Kill switch
``UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK=1``.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_NAMES = ("hunyuanvideo-1.5", "hunyuanvideo-1.5-720p")


def _family(name):
    from core.inference import video_families
    return next(f for f in video_families._FAMILIES if f.name == name)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 9), (9, 0), (10, 0), (12, 0), None])
def test_hv15_filters_reductions_on_every_arch(monkeypatch, name, cap):
    monkeypatch.setattr(cc, "_device_capability", lambda: cap)
    assert _family(name).filter_reduction_configs is True
    assert cc.family_filters_reductions(_family(name)) is True


def _filter_reaching_compile(monkeypatch, family):
    seen = {}

    def fake_compile(pipe, logger, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", fake_compile)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, backend = "cuda")
    ds.apply_speed_optims(
        types.SimpleNamespace(), target, is_gguf = False, family = family, speed_mode = ds.SPEED_DEFAULT
    )
    return seen["filter_reductions"]


@pytest.mark.parametrize("name", _NAMES)
def test_the_filter_reaches_hv15s_regional_compile(monkeypatch, name):
    monkeypatch.setattr(cc, "_device_capability", lambda: (12, 0))
    assert _filter_reaching_compile(monkeypatch, _family(name)) is True


@pytest.mark.parametrize("name", _NAMES)
def test_the_kill_switch_restores_the_per_process_pick(monkeypatch, name):
    monkeypatch.setenv(cc.DYNAMIC_SCALE_RBLOCK_ENV, "1")
    kwargs = {"fullgraph": True, "dynamic": True}
    assert ds.pin_reduction_configs(kwargs) is False
    assert "options" not in kwargs
    # the family still asks; the switch alone turns the pin off
    assert _family(name).filter_reduction_configs is True
