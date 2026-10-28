# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FP8 + NVFP4 cache tags compose to one string whatever order they were set in (torch keys its caches on it)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch.compiler.config as cc  # noqa: E402

from core.inference import diffusion_fp8_unit_scale as fp8  # noqa: E402
from core.inference import diffusion_nvfp4_act_quant as nvfp4  # noqa: E402


@pytest.fixture(autouse = True)
def _restore_tag():
    saved = cc.cache_key_tag
    yield
    cc.cache_key_tag = saved


def _apply(steps, start = ""):
    cc.cache_key_tag = start
    for step in steps:
        step()
    return cc.cache_key_tag


FP8 = lambda: fp8._set_compile_cache_tag("256")  # noqa: E731
NVFP4_ON = lambda: nvfp4._set_compile_cache_tag(nvfp4._TAG_ON)  # noqa: E731


def test_setter_order_does_not_change_the_tag():
    assert _apply([FP8, NVFP4_ON]) == _apply([NVFP4_ON, FP8])


def test_setting_again_does_not_change_the_tag():
    once = _apply([FP8, NVFP4_ON])
    assert _apply([FP8, NVFP4_ON, FP8]) == once
    assert _apply([FP8, NVFP4_ON, NVFP4_ON, FP8, NVFP4_ON]) == once


def test_values_and_removal_still_count():
    on = _apply([FP8, NVFP4_ON])
    off = _apply([FP8, lambda: nvfp4._set_compile_cache_tag(nvfp4._TAG_OFF)])
    other_rows = _apply([lambda: fp8._set_compile_cache_tag("512"), NVFP4_ON])
    assert len({on, off, other_rows}) == 3
    removed = _apply([FP8, NVFP4_ON, lambda: fp8._set_compile_cache_tag(None)])
    assert "unsloth_fp8_unit_scale=" not in removed and "unsloth_nvfp4_act_quant=1" in removed


def test_foreign_tokens_are_kept_in_place():
    tag = _apply([NVFP4_ON, FP8], start = "user_tag;other=2")
    assert tag.split(";")[:2] == ["user_tag", "other=2"]
    assert tag == _apply([FP8, NVFP4_ON], start = "user_tag;other=2")
