# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The FP8 / NVFP4 cache tags reach every context that compiles (#15055 follow-up).

torch 2.12+ keeps ``torch.compiler.config`` overrides in a ContextVar: a tag set on the load thread is invisible to the
render thread and the background compile thread unless it is recorded for ``diffusion_compile_config.apply()``.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("torch")
import torch.compiler.config as cc  # noqa: E402

from core.inference import diffusion_compile_config as compile_config  # noqa: E402
from core.inference import diffusion_fp8_unit_scale as fp8  # noqa: E402
from core.inference import diffusion_nvfp4_act_quant as nvfp4  # noqa: E402


@pytest.fixture(autouse = True)
def _clean():
    saved = cc.cache_key_tag
    compile_config._reset_for_tests()
    yield
    compile_config._reset_for_tests()
    cc.cache_key_tag = saved


def _in_thread(fn):
    out = {}

    def run():
        out["value"] = fn()

    t = threading.Thread(target = run)
    t.start()
    t.join()
    return out["value"]


def _render_context_tag():
    """What a render / background-compile thread sees: both call ``apply()`` before compiling."""

    def read():
        compile_config.apply()
        return cc.cache_key_tag

    return _in_thread(read)


def test_kill_switch_tag_set_on_the_load_thread_reaches_the_render_thread(monkeypatch):
    monkeypatch.setenv(nvfp4.FAST_ACT_QUANT_ENV, "0")
    assert _in_thread(lambda: (nvfp4.install_nvfp4_act_quant_patch(), cc.cache_key_tag)[1]) == (
        "unsloth_nvfp4_act_quant=0"
    )
    assert _render_context_tag() == "unsloth_nvfp4_act_quant=0"


def test_fp8_and_nvfp4_tags_set_on_the_load_thread_reach_the_render_thread():
    _in_thread(
        lambda: (fp8._set_compile_cache_tag("256"), nvfp4._set_compile_cache_tag(nvfp4._TAG_ON))
    )
    assert _render_context_tag() == "unsloth_fp8_unit_scale=256;unsloth_nvfp4_act_quant=1"


def test_reset_on_a_later_load_reaches_the_render_thread():
    _in_thread(lambda: nvfp4._set_compile_cache_tag(nvfp4._TAG_OFF))
    _in_thread(nvfp4.reset_compile_cache_tag)
    assert _render_context_tag() == ""
