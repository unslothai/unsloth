# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm ``aiter`` without diffusers' local AITER backend (0.41+) runs the auto pick, not native SDPA; all else unchanged."""

from __future__ import annotations

import enum
import sys
import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import select_attention_backend


def _target(device = "cuda"):
    return types.SimpleNamespace(device = device, dtype = "torch.bfloat16")


def _fake_dispatch(monkeypatch, names, can_use):
    Names = enum.Enum("AttentionBackendName", {n.upper(): n for n in names})
    mod = types.ModuleType("diffusers.models.attention_dispatch")
    mod.AttentionBackendName = Names
    if can_use is not None:
        mod._CAN_USE_AITER_ATTN = can_use
    models = types.ModuleType("diffusers.models")
    models.attention_dispatch = mod
    pkg = types.ModuleType("diffusers")
    pkg.models = models
    monkeypatch.setitem(sys.modules, "diffusers", pkg)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_dispatch", mod)


# 0.42.1 / 0.41.0 enum (abridged) vs 0.36.0, which still had the local backend.
_HUB_ONLY = ("native", "flash", "_native_cudnn", "aiter_fa2_hub", "flash_4_hub")
_LOCAL = ("native", "flash", "_native_cudnn", "aiter")


@pytest.mark.parametrize(
    "names, can_use, usable",
    [
        (_HUB_ONLY, None, False),
        (_LOCAL, False, False),  # 0.36 without the aiter package
        (_LOCAL, True, True),
    ],
)
def test_diffusers_aiter_usable(monkeypatch, names, can_use, usable):
    _fake_dispatch(monkeypatch, names, can_use)
    assert att._diffusers_aiter_usable() is usable


def test_diffusers_aiter_usable_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_dispatch", None)
    assert att._diffusers_aiter_usable() is False


def _rocm(monkeypatch, *, flash_verified):
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    monkeypatch.setattr(att, "_rocm_auto_flash", lambda target, family = None: flash_verified)


@pytest.mark.parametrize("flash_verified", [True, False])
@pytest.mark.parametrize(
    "speed_active, speed_unset", [(True, False), (False, True), (False, False)]
)
def test_dead_aiter_on_rocm_is_the_auto_pick(
    monkeypatch, flash_verified, speed_active, speed_unset
):
    _rocm(monkeypatch, flash_verified = flash_verified)
    _fake_dispatch(monkeypatch, _HUB_ONLY, None)
    kw = dict(speed_active = speed_active, speed_unset = speed_unset)
    auto = select_attention_backend(_target(), "auto", **kw)
    assert select_attention_backend(_target(), "aiter", **kw) == auto
    if flash_verified and (speed_active or speed_unset):
        # gfx1151 FLUX.3 klein: the default flash pick, not native (1.40 vs 1.70 s/step, AMD run 45035677532).
        assert auto == att.ROCM_FLASH_CANDIDATE


def test_usable_aiter_on_rocm_still_honored(monkeypatch):
    _rocm(monkeypatch, flash_verified = True)
    _fake_dispatch(monkeypatch, _LOCAL, True)
    assert select_attention_backend(_target(), "aiter", speed_active = True) == "aiter"
    assert att.explicit_attention_decline(_target(), "aiter") is None


@pytest.mark.parametrize("names, can_use", [(_HUB_ONLY, None), (_LOCAL, True)])
def test_aiter_off_rocm_unchanged(monkeypatch, names, can_use):
    _fake_dispatch(monkeypatch, names, can_use)
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: True)
    assert select_attention_backend(_target(), "aiter", speed_active = True) is None
    assert att.explicit_attention_decline(_target(), "aiter")[1] == "unsupported"
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: False)
    assert select_attention_backend(_target("mps"), "aiter", speed_active = True) is None
    assert att.explicit_attention_decline(_target("mps"), "aiter")[1] == "unsupported"


# origin/main answers: (speed on, speed off, decline status).
_C = att.ROCM_FLASH_CANDIDATE
_U = "unsupported"
_FROZEN = {
    True: {
        "auto": ("_native_cudnn", None, None),
        "cudnn": ("_native_cudnn", "_native_cudnn", None),
        "flash": ("flash", "flash", None),
        "flash2": ("flash", "flash", None),
        "flash3": ("_flash_3_hub", "_flash_3_hub", None),
        "flash4": (None, None, _U),
        "native": (None, None, None),
        "sage": ("sage", "sage", None),
        "sdpa": (None, None, None),
        "unsloth_int8": ("_native_cudnn", "_native_cudnn", None),
        "xformers": ("xformers", "xformers", None),
        "flash_sdpa": ("_native_flash", "_native_flash", None),
    },
    False: {
        "auto": (_C, None, None),
        "cudnn": (None, None, _U),
        "flash": ("flash", "flash", None),
        "flash2": ("flash", "flash", None),
        "flash3": (None, None, _U),
        "flash4": (None, None, _U),
        "native": (None, None, None),
        "sage": (None, None, _U),
        "sdpa": (None, None, None),
        "unsloth_int8": (_C, _C, None),
        "xformers": (None, None, _U),
        "flash_sdpa": (None, None, _U),
    },
}


def test_frozen_table_covers_every_other_alias():
    assert set(_FROZEN[True]) == set(_FROZEN[False]) == set(att.ATTN_ALIASES) - {"aiter"}


@pytest.mark.parametrize("names, can_use", [(_HUB_ONLY, None), (_LOCAL, True)])
@pytest.mark.parametrize("nvidia", [True, False])
@pytest.mark.parametrize("alias", sorted(set(att.ATTN_ALIASES) - {"aiter"}))
def test_every_other_pick_unchanged(monkeypatch, names, can_use, nvidia, alias):
    _fake_dispatch(monkeypatch, names, can_use)
    monkeypatch.setattr(att, "_is_cuda_nvidia", lambda target: nvidia)
    monkeypatch.setattr(att, "_rocm_auto_flash", lambda target, family = None: True)
    monkeypatch.setattr(att, "_rocm_flash_attn_runs", lambda target: True)
    monkeypatch.setattr(att, "_cuda_capability", lambda: (9, 0))
    monkeypatch.setattr(att, "_cudnn_attention_supported", lambda: True)
    on = select_attention_backend(_target(), alias, speed_active = True)
    off = select_attention_backend(_target(), alias, speed_active = False)
    declined = att.explicit_attention_decline(_target(), alias)
    got = (on, off, declined[1] if declined else None)
    assert got == _FROZEN[nvidia][alias]


def test_dead_aiter_status_says_fell_back_to_auto(monkeypatch):
    _rocm(monkeypatch, flash_verified = True)
    _fake_dispatch(monkeypatch, _HUB_ONLY, None)
    why, status = att.explicit_attention_decline(_target(), "aiter")
    assert status == "fell_back" and "automatic pick" in why
    entry = att.attention_resolved_entry("aiter", "flash", None, target = _target())
    assert entry[1] == "flash" and entry[3] == "fell_back" and "automatic pick" in entry[2]
