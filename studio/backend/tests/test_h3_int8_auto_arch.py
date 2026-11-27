# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video: Auto races INT8 attention on every arch (sm89 again), with an exact start wherever it engages (the
first denoise steps on 16-bit attention, test_h3_int8_exact_start.py); explicit On and the env switch are as before. ``h3_int8_load_gate`` is the load's decision, driven here directly; one AST check pins that the H3 video
load uses it."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.inference import diffusion_int8_attention as I8
from core.inference import video_minimax_h3_attn as A

BACKEND_DIR = Path(__file__).resolve().parents[1]
CONFIG_REASON = (
    "Auto engages it on MiniMax-H3 video only over Studio's hosted INT8 denoiser without adapters"
)
SPEED_REASON = "speed_mode=off keeps the output bit-identical, so the kernel is not installed"


def _gate(**kw):
    """The evidenced Auto config on an A100 unless overridden."""
    args = dict(
        mode = "auto",
        attention_backend = None,
        auto_attention = True,
        speed_on = True,
        hosted_int8 = True,
        comfy_checkpoint = False,
        adapters = False,
        capability = (8, 0),
    )
    args.update(kw)
    return A.h3_int8_load_gate(**args)


def _mode(
    monkeypatch,
    env,
    request = None,
):
    """The load's resolved mode for an env switch value and a per-load request, as video.py resolves it."""
    if env is None:
        monkeypatch.delenv(I8.INT8_ATTENTION_ENV, raising = False)
    else:
        monkeypatch.setenv(I8.INT8_ATTENTION_ENV, env)
    return I8.resolve_policy(request).mode


def test_sm89_auto_races_with_an_exact_start():
    gate = _gate(capability = (8, 9))
    assert (gate.forced, gate.auto, gate.reason) == (False, True, None)
    assert gate.exact_steps == A.H3_INT8_EXACT_START_STEPS > 0
    assert A.h3_int8_auto_off_reason((8, 9)) is None


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0), (10, 0), (12, 0), (7, 5), None])
def test_auto_races_on_every_arch(capability):
    # None is ROCm / an unknown card; the exact start comes with Auto on every one
    gate = _gate(capability = capability)
    assert (gate.forced, gate.auto, gate.reason) == (False, True, None)
    assert gate.exact_steps == A.H3_INT8_EXACT_START_STEPS
    assert A.h3_int8_auto_off_reason(capability) is None


def test_no_arch_keeps_auto_off():
    assert A.H3_INT8_AUTO_OFF_ARCHS == frozenset()


@pytest.mark.parametrize(
    "env, pick",
    [(None, "on"), ("1", None), ("0", "on"), ("auto", "on")],
)
def test_forced_bypasses_the_arch(monkeypatch, env, pick):
    gate = _gate(mode = _mode(monkeypatch, env, pick), capability = (8, 9))
    assert (gate.forced, gate.auto, gate.reason) == (True, False, None)


def test_explicit_int8_backend_pick_forces_unless_off(monkeypatch):
    gate = _gate(attention_backend = I8.BACKEND, auto_attention = False, capability = (8, 9))
    assert (gate.forced, gate.auto, gate.reason) == (True, False, None)
    off = _gate(mode = "off", attention_backend = I8.BACKEND, auto_attention = False, capability = (8, 9))
    assert (off.forced, off.auto, off.reason) == (False, False, None)


@pytest.mark.parametrize("capability", [(8, 9), (8, 0)])
@pytest.mark.parametrize("env, pick", [("0", None), (None, "off"), ("1", "off")])
def test_off_stays_off_without_a_reason(monkeypatch, capability, env, pick):
    gate = _gate(mode = _mode(monkeypatch, env, pick), capability = capability)
    assert gate == A.H3Int8Gate(forced = False, auto = False, reason = None)


@pytest.mark.parametrize("capability", [(8, 9), (8, 0)])
def test_speed_off_wins_over_everything_but_off(capability):
    for mode in ("auto", "force"):
        gate = _gate(mode = mode, speed_on = False, capability = capability)
        assert gate.auto is False and gate.reason == SPEED_REASON


def test_explicit_attention_pick_keeps_it_before_the_arch():
    gate = _gate(attention_backend = "flash", auto_attention = False, capability = (8, 9))
    assert (gate.forced, gate.auto) == (False, False)
    assert gate.reason == "attention_backend='flash' was picked explicitly; Auto keeps it"


@pytest.mark.parametrize(
    "kw", [dict(hosted_int8 = False), dict(comfy_checkpoint = True), dict(adapters = True)]
)
@pytest.mark.parametrize("capability", [(8, 9), (8, 0), None])
def test_off_config_auto_keeps_the_config_reason_on_any_arch(kw, capability):
    gate = _gate(capability = capability, **kw)
    assert (gate.forced, gate.auto, gate.reason) == (False, False, CONFIG_REASON)


def test_h3_video_load_uses_the_gate():
    tree = ast.parse((BACKEND_DIR / "core" / "inference" / "video.py").read_text(encoding = "utf-8"))
    load = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_h3_modular_pipeline"
    )
    calls = [
        n
        for n in ast.walk(load)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "h3_int8_load_gate"
    ]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    # the capability of the card the load runs on
    assert (
        kw["capability"]
        == "cuda_capability(str(getattr(umem_target, 'torch_device', None) or device))"
    )
    used = {
        n.attr
        for n in ast.walk(load)
        if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) == "h3_int8_gate"
    }
    assert used == {"forced", "auto", "reason", "exact_steps"}
