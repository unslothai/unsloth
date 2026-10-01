# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_qwenimage_rope.py``: the Qwen-Image RoPE table entry swap."""

from __future__ import annotations

import pytest

from core.inference import diffusion_qwenimage21_rope as q21
from core.inference import diffusion_qwenimage_rope as qr

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage")

needs_cuda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and not getattr(torch.version, "hip", None)
        and q21.inductor_addcmul_is_fma()
    ),
    reason = "needs CUDA and inductor's fma addcmul lowering",
)


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(qr.QWEN_REAL_ROPE_ENV, raising = False)
    yield
    qr.uninstall()


def test_stock_entry_fingerprint_matches():
    entry = qmod.ROPE_PER_DEVICE["cuda"]
    assert q21._digest(entry.func) in qr._FINGERPRINT
    assert entry.keywords == {"use_real": False}


def test_disabled_and_other_models_are_noops(monkeypatch):
    assert qr.install(torch.nn.Linear(4, 4)) is False
    monkeypatch.setenv(qr.QWEN_REAL_ROPE_ENV, "0")
    assert qr.disabled()


@needs_cuda
def test_compiled_rope_bit_identical_to_complex_product():
    index = torch.cuda.current_device()
    stock = qmod.ROPE_PER_DEVICE["cuda"]
    assert qr._patch_table(index)
    patched = qmod.ROPE_PER_DEVICE["cuda"]
    assert patched is not stock
    g = torch.Generator(device = "cpu").manual_seed(0)
    x = torch.randn(2, 300, 24, 128, generator = g).cuda().to(torch.bfloat16)
    ang = torch.rand(300, 64, generator = g) * 6.28
    freqs = torch.polar(torch.ones_like(ang), ang).to(torch.complex64).cuda()
    ref = stock(x, freqs)
    torch._dynamo.reset()
    out = torch.compile(lambda a, f: qmod.ROPE_PER_DEVICE["cuda"](a, f))(x, freqs)
    assert torch.equal(out, ref)
    # Eager calls keep the stock complex path.
    assert torch.equal(patched(x, freqs), ref)
    qr.uninstall()
    assert qmod.ROPE_PER_DEVICE["cuda"] is stock


def _fake_probe(monkeypatch):
    probed = []
    monkeypatch.setattr(q21, "inductor_addcmul_is_fma", lambda: True)
    monkeypatch.setattr(q21, "_addcmul_lowering", lambda: (True, False))
    monkeypatch.setattr(q21, "_FUSION", {})
    monkeypatch.setattr(q21, "probe_fusion", lambda dev: probed.append(dev.index) or ("x", "x"))
    return probed


def test_every_device_is_probed_once_the_table_is_patched(monkeypatch):
    probed = _fake_probe(monkeypatch)
    assert qr._patch_table(0)
    assert qr._patch_table(1)
    assert probed == [0, 1]
    assert q21._FUSION == {0: ("x", "x"), 1: ("x", "x")}


class QwenImageTransformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


@pytest.mark.parametrize("env", [qr.QWEN_REAL_ROPE_ENV, q21.REAL_ROPE_ENV])
def test_kill_switch_on_a_later_load_restores_the_table(monkeypatch, env):
    _fake_probe(monkeypatch)
    stock = qmod.ROPE_PER_DEVICE["cuda"]
    assert qr._patch_table(0)
    assert qmod.ROPE_PER_DEVICE["cuda"] is not stock
    monkeypatch.setenv(env, "0")
    assert qr.install(QwenImageTransformer2DModel()) is False
    assert qmod.ROPE_PER_DEVICE["cuda"] is stock
