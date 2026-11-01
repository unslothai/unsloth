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
    fp8.uninstall()
    compile_config._reset_for_tests()
    yield
    fp8.uninstall()
    compile_config._reset_for_tests()
    cc.cache_key_tag = saved


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:
            out["error"] = exc

    t = threading.Thread(target = run)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
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


def test_nvfp4_reset_in_a_fresh_context_keeps_the_recorded_fp8_tag():
    _in_thread(lambda: (fp8._set_compile_cache_tag("256"), nvfp4._set_compile_cache_tag("1")))
    _in_thread(nvfp4.reset_compile_cache_tag)
    assert _render_context_tag() == "unsloth_fp8_unit_scale=256"


def test_real_fp8_installer_keeps_the_tag_across_two_loads(monkeypatch):
    f8t = pytest.importorskip("torchao.quantization.quantize_.workflows.float8.float8_tensor")
    monkeypatch.delenv(fp8.UNIT_SCALE_ENV, raising = False)
    monkeypatch.delenv(fp8.MIN_ROWS_ENV, raising = False)
    monkeypatch.setattr(fp8, "_device_caps", lambda: {0: (8, 9)})

    def load():
        nvfp4.reset_compile_cache_tag()
        return fp8.install()

    assert _in_thread(load) is True
    wrapped = f8t.addmm_float8_unwrapped_inference
    assert _render_context_tag() == "unsloth_fp8_unit_scale=256"
    assert _in_thread(load) is True
    assert fp8.stats()["installed"] is True
    assert f8t.addmm_float8_unwrapped_inference is wrapped
    assert _render_context_tag() == "unsloth_fp8_unit_scale=256"


@pytest.mark.parametrize("mode, rows, tag", [("auto", "256", "256"), ("force", "512", "512:force")])
def test_repeated_install_restores_the_installed_tag_in_a_fresh_context(
    monkeypatch, mode, rows, tag
):
    f8t = pytest.importorskip("torchao.quantization.quantize_.workflows.float8.float8_tensor")
    monkeypatch.setenv(fp8.UNIT_SCALE_ENV, mode)
    monkeypatch.setenv(fp8.MIN_ROWS_ENV, rows)
    monkeypatch.setattr(fp8, "_device_caps", lambda: {0: (8, 9)})
    assert _in_thread(fp8.install) is True
    wrapped = f8t.addmm_float8_unwrapped_inference
    # These settings are captured by the installed closure, so later env changes must not relabel it.
    monkeypatch.setenv(fp8.UNIT_SCALE_ENV, "0")
    monkeypatch.setenv(fp8.MIN_ROWS_ENV, "1024")
    installed, current = _in_thread(lambda: (fp8.install(), cc.cache_key_tag))
    assert installed is True
    assert current == f"unsloth_fp8_unit_scale={tag}"
    assert _render_context_tag() == current
    assert f8t.addmm_float8_unwrapped_inference is wrapped
