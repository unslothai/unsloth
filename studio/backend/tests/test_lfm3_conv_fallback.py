# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LFM3's remote Triton conv is turned off only on a ROCm card Triton cannot run on."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import lfm3_conv_fallback as fallback
from core.inference.lfm3_conv_fallback import disable_lfm3_triton_conv_where_unsupported

LFM3 = "transformers_modules.LiquidAI.LFM3_hyphen_1_dot_7B.abc.modeling_lfm3"
OMNI_TEXT = "transformers_modules.LiquidAI.LFM3_hyphen_Omni_hyphen_2B.def.modeling_lfm3"
OTHER = "transformers_modules.someone.other.ghi.modeling_other"

_KERNEL = object()


def _original(hidden_states, weight):
    return _KERNEL


def _fake_torch(*, hip, arch = "gfx1010"):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+rocm7.14.0" if hip else "2.11.0+cu130"
    torch.version = types.SimpleNamespace(hip = "7.14" if hip else None)
    torch.cuda = types.SimpleNamespace(
        current_device = lambda: 0,
        get_device_properties = lambda index = None: types.SimpleNamespace(gcnArchName = arch),
    )
    return torch


def _model(device_type, index = 0):
    param = types.SimpleNamespace(device = types.SimpleNamespace(type = device_type, index = index))
    return types.SimpleNamespace(parameters = lambda: iter([param]))


def _tensor(index, device_type = "cuda"):
    return types.SimpleNamespace(device = types.SimpleNamespace(type = device_type, index = index))


def _probe(results):
    calls = []

    def runs_on(index):
        calls.append(index)
        return results[index]

    runs_on.calls = calls
    return runs_on


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


@pytest.mark.parametrize("arch", ["gfx1010", "gfx1030", "gfx1036"])
def test_gate_patched_where_triton_cannot_run(remote_modules, arch):
    log = types.SimpleNamespace(lines = [])
    log.info = log.lines.append
    patched = disable_lfm3_triton_conv_where_unsupported(
        _model("cuda"), _fake_torch(hip = True, arch = arch), log, _probe({0: False})
    )
    assert sorted(patched) == sorted([LFM3, OMNI_TEXT])
    for name in (LFM3, OMNI_TEXT):
        assert remote_modules[name].can_use_lfm3_conv(_tensor(0), None) is False
    assert not hasattr(remote_modules[OTHER], "can_use_lfm3_conv")
    assert len(log.lines) == 1 and arch in log.lines[0]


@pytest.mark.parametrize(
    "hip, device",
    [
        (
            True,
            "cuda",
        ),  # Triton runs on the card (gfx1010 on ROCm Triton 3.5, gfx1100, gfx1151 ...)
        (True, "cpu"),
        (False, "cuda"),
        (False, "mps"),
        (False, "cpu"),
    ],
)
def test_gate_untouched_elsewhere(remote_modules, hip, device):
    probe = _probe({0: True})
    patched = disable_lfm3_triton_conv_where_unsupported(
        _model(device), _fake_torch(hip = hip), None, probe
    )
    assert patched == []
    if not hip:
        assert probe.calls == []
    for name in (LFM3, OMNI_TEXT):
        assert remote_modules[name].can_use_lfm3_conv is _original


def test_model_without_gate_is_a_noop(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("transformers_modules.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, OTHER, types.ModuleType(OTHER))
    assert (
        disable_lfm3_triton_conv_where_unsupported(
            _model("cuda"), _fake_torch(hip = True), None, _probe({0: False})
        )
        == []
    )


def test_unreadable_model_is_a_noop(remote_modules):
    assert (
        disable_lfm3_triton_conv_where_unsupported(
            object(), _fake_torch(hip = True), None, _probe({0: False})
        )
        == []
    )
    assert remote_modules[LFM3].can_use_lfm3_conv is _original


def test_missing_core_probe_keeps_the_kernel(remote_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", None)
    assert fallback._core_probe()(0) is True
    assert disable_lfm3_triton_conv_where_unsupported(_model("cuda"), _fake_torch(hip = True)) == []


def test_a_supported_card_in_the_same_process_keeps_the_kernel(remote_modules):
    probe = _probe({0: False, 1: True})
    torch = _fake_torch(hip = True)
    disable_lfm3_triton_conv_where_unsupported(_model("cuda", 0), torch, None, probe)
    assert disable_lfm3_triton_conv_where_unsupported(_model("cuda", 0), torch, None, probe) == []
    for name in (LFM3, OMNI_TEXT):
        gate = remote_modules[name].can_use_lfm3_conv
        assert gate(_tensor(0), None) is False
        assert gate(_tensor(1), None) is _KERNEL
        assert gate(_tensor(None, "cpu"), None) is False
