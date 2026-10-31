# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
import types

import pytest

from core.inference import diffusion_qwenimage21 as q
from core.inference import diffusion_qwenimage21_eager as e

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_diffusion_qwenimage21 import _inputs, _model, _render  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(e.EAGER_FAST_ENV, raising = False)
    monkeypatch.delenv(q.FAST_STEP_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    q.uninstall()
    e.uninstall()
    yield
    q.uninstall()
    e.uninstall()


def _valid(inp):
    inp["encoder_hidden_states_mask"] = torch.ones(
        inp["encoder_hidden_states"].shape[:2], dtype = torch.long
    )
    return inp


def _armed(m):
    assert q.install()
    assert e.install_for_pipe(types.SimpleNamespace(transformer = m))
    return m


@pytest.mark.parametrize(
    "case", [{}, {"cond": (4, 4)}, {"batch": 2, "pad": 3}, {"batch": 2, "cond": (4, 8), "pad": 2}]
)
def test_broadcast_modulation_and_the_eager_qkv_are_bit_identical(case):
    m = _model()
    inp = _valid(_inputs(**case)) if not case.get("pad") else _inputs(**case)
    ref = _render(m, inp)
    _armed(m)
    before = dict(e.STATS)
    got = _render(m, inp)
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert e.STATS["mod_broadcast"] > before["mod_broadcast"]


def test_a_demoted_load_runs_stock_outside_a_pending_background_compile():
    m = _model()
    _armed(m)
    inp = _inputs()
    before = e.STATS["mod_broadcast"]
    e.demote(types.SimpleNamespace(transformer = m))
    _render(m, inp)
    assert e.STATS["mod_broadcast"] == before


def test_state_rules(monkeypatch):
    m = _model()
    attn = m.transformer_blocks[0].attn
    with torch.inference_mode():
        assert not e.active(attn)
        _armed(m)
        assert e.active(m) and e.active(attn)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert not e.active(attn)
        monkeypatch.undo()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        e.demote(types.SimpleNamespace(transformer = m))
        assert not e.active(attn)
        from core.inference import diffusion_bg_compile as bg

        token = bg._FORCE_EAGER.set(True)
        try:
            assert e.active(attn) and e.active(m)
        finally:
            bg._FORCE_EAGER.reset(token)
    with torch.enable_grad():
        e.set_state(m, e.EAGER)
        assert not e.active(attn)


def test_demote_leaves_an_unarmed_load_stock():
    m = _model()
    e.demote(types.SimpleNamespace(transformer = m))
    assert e._ATTR not in m.__dict__


def test_kill_switch_and_other_models_stay_stock(monkeypatch):
    m = _model()
    monkeypatch.setenv(e.EAGER_FAST_ENV, "0")
    assert not e.install_for_pipe(types.SimpleNamespace(transformer = m))
    assert e._ATTR not in m.__dict__
    monkeypatch.delenv(e.EAGER_FAST_ENV)
    other = torch.nn.Linear(2, 2)
    assert not e.install_for_pipe(types.SimpleNamespace(transformer = other))


def test_a_drifted_function_keeps_the_stock_step(monkeypatch):
    monkeypatch.setitem(e._FINGERPRINTS, "_qwenimage21_prepare_qkv", frozenset({"0" * 16}))
    stock = qmod._qwenimage21_prepare_qkv
    assert not e.install()
    assert qmod._qwenimage21_prepare_qkv is stock


def test_install_is_idempotent_and_uninstall_restores():
    stock = qmod._qwenimage21_prepare_qkv
    assert e.install() and e.install()
    wrapped = qmod._qwenimage21_prepare_qkv
    assert wrapped is not stock and wrapped.__wrapped__ is stock
    from core.inference.diffusion_autoopt.patches import source_digest

    assert source_digest(wrapped) == source_digest(stock)
    e.uninstall()
    assert qmod._qwenimage21_prepare_qkv is stock


def test_installed_diffusers_matches_the_fingerprints():
    assert e.why_unsupported(qmod) is None


def test_cpu_tensors_never_reach_the_kernel():
    norm = qmod.RMSNorm(16, eps = 1e-6)
    x = torch.randn(1, 4, 2, 16)
    freqs = torch.polar(torch.ones(4, 8), torch.rand(4, 8))
    assert e.install()
    assert e._norm_rope(x, norm, freqs) is None


def test_the_eager_tier_rmsnorm_patch_still_takes_the_kernel(monkeypatch):
    from core.inference import diffusion_eager_patches as ep

    assert e.install()
    norm = qmod.RMSNorm(16, eps = 1e-6)
    assert e._stock_norm(norm)
    monkeypatch.setattr(qmod.RMSNorm, "forward", ep._rmsnorm_forward)
    assert e._stock_norm(norm)
    monkeypatch.setattr(qmod.RMSNorm, "forward", lambda self, x: x)
    assert not e._stock_norm(norm)
