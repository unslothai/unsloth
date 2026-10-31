# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3's remote Triton conv is turned off only for a model on an RDNA1 (gfx101x) card."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.rdna1_conv_fallback import disable_lfm3_triton_conv_on_rdna1

LFM3 = "transformers_modules.LiquidAI.LFM3_hyphen_1_dot_7B.abc.modeling_lfm3"
OMNI_TEXT = "transformers_modules.LiquidAI.LFM3_hyphen_Omni_hyphen_2B.def.modeling_lfm3"
OTHER = "transformers_modules.someone.other.ghi.modeling_other"


def _original(hidden_states, weight):
    return True


def _fake_torch(*, hip, arch):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+rocm7.14.0" if hip else "2.11.0+cu130"
    torch.version = types.SimpleNamespace(hip = "7.14" if hip else None)
    props = types.SimpleNamespace(gcnArchName = arch)
    torch.cuda = types.SimpleNamespace(
        current_device = lambda: 0,
        get_device_properties = lambda index = None: props,
    )
    return torch


def _model(device_type, index = 0):
    param = types.SimpleNamespace(device = types.SimpleNamespace(type = device_type, index = index))
    return types.SimpleNamespace(parameters = lambda: iter([param]))


@pytest.fixture
def remote_modules(monkeypatch):
    mods = {}
    for name in (LFM3, OMNI_TEXT):
        mod = types.ModuleType(name)
        mod.can_use_lfm3_conv = _original
        mods[name] = mod
    other = types.ModuleType(OTHER)
    other.something_else = _original
    mods[OTHER] = other
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return mods


@pytest.mark.parametrize("arch", ["gfx1010", "gfx1010:xnack-", "gfx1012"])
def test_rdna1_gate_patched_in_every_remote_module(remote_modules, arch):
    log = types.SimpleNamespace(lines = [])
    log.info = log.lines.append
    patched = disable_lfm3_triton_conv_on_rdna1(
        _model("cuda"), _fake_torch(hip = True, arch = arch), log
    )
    assert sorted(patched) == sorted([LFM3, OMNI_TEXT])
    for name in (LFM3, OMNI_TEXT):
        assert remote_modules[name].can_use_lfm3_conv(None, None) is False
    assert not hasattr(remote_modules[OTHER], "can_use_lfm3_conv")
    assert len(log.lines) == 1 and "gfx101" in log.lines[0]


@pytest.mark.parametrize(
    "hip, arch, device",
    [
        (True, "gfx1030", "cuda"),
        (True, "gfx1034", "cuda"),
        (True, "gfx1201", "cuda"),
        (True, "gfx1100", "cuda"),
        (True, "gfx1010", "cpu"),  # ROCm torch, model kept on CPU
        (False, "", "cuda"),  # CUDA
        (False, "", "mps"),
        (False, "", "cpu"),
    ],
)
def test_gate_untouched_elsewhere(remote_modules, hip, arch, device):
    patched = disable_lfm3_triton_conv_on_rdna1(_model(device), _fake_torch(hip = hip, arch = arch))
    assert patched == []
    for name in (LFM3, OMNI_TEXT):
        assert remote_modules[name].can_use_lfm3_conv is _original


def test_model_without_gate_is_a_noop(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("transformers_modules.")]:
        monkeypatch.delitem(sys.modules, name)
    plain = types.ModuleType(OTHER)
    monkeypatch.setitem(sys.modules, OTHER, plain)
    assert (
        disable_lfm3_triton_conv_on_rdna1(_model("cuda"), _fake_torch(hip = True, arch = "gfx1010"))
        == []
    )


def test_unreadable_model_or_props_is_a_noop(remote_modules):
    torch = _fake_torch(hip = True, arch = "gfx1010")

    def _boom(index = None):
        raise RuntimeError("no device")

    torch.cuda.get_device_properties = _boom
    assert disable_lfm3_triton_conv_on_rdna1(_model("cuda"), torch) == []
    assert disable_lfm3_triton_conv_on_rdna1(object(), _fake_torch(hip = True, arch = "gfx1010")) == []
    assert remote_modules[LFM3].can_use_lfm3_conv is _original


def _tensor(index):
    return types.SimpleNamespace(device = types.SimpleNamespace(type = "cuda", index = index))


def test_gate_stays_on_for_a_non_rdna1_card_in_the_same_process(remote_modules):
    # Mixed host: an RDNA1 model is loaded first, then the same remote code runs on another card.
    archs = {0: "gfx1010", 1: "gfx1100"}
    torch = _fake_torch(hip = True, arch = "gfx1010")
    torch.cuda.get_device_properties = lambda index = None: types.SimpleNamespace(
        gcnArchName = archs[index]
    )
    disable_lfm3_triton_conv_on_rdna1(_model("cuda", 0), torch)
    assert disable_lfm3_triton_conv_on_rdna1(_model("cuda", 0), torch) == []  # no double wrap
    for name in (LFM3, OMNI_TEXT):
        gate = remote_modules[name].can_use_lfm3_conv
        assert gate(_tensor(0), None) is False
        assert gate(_tensor(1), None) is True
