# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fused Wan kernels take the sequence length L unspecialized: L only splits the row id into (sample, token), so
a new resolution or clip length whose L % 16 differs reuses the binary instead of compiling another. CPU only (the
kernels are built, not launched)."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("triton")

from core.inference import video_wan_fused as wf  # noqa: E402


def _jit(name: str):
    fn = wf._kernels()[name]
    for v in inspect.getclosurevars(fn).nonlocals.values():
        if type(v).__name__ == "JITFunction" and v.__name__ == "_" + name:
            return v
    raise AssertionError(name)


@pytest.fixture
def kernels(monkeypatch):
    def build(env = None):
        if env is None:
            monkeypatch.delenv(wf.WAN_FUSED_ANY_LENGTH_ENV, raising = False)
        else:
            monkeypatch.setenv(wf.WAN_FUSED_ANY_LENGTH_ENV, env)
        wf._kernels.cache_clear()
        return {n: _jit(n) for n in ("modnorm", "gate_residual", "affine_norm", "rope")}

    yield build
    wf._kernels.cache_clear()


def _unspecialized(fn) -> list:
    return [p.name for p in fn.params if p.do_not_specialize]


def test_length_is_not_specialized(kernels):
    ks = kernels()
    assert _unspecialized(ks["modnorm"]) == ["L"]
    assert _unspecialized(ks["gate_residual"]) == ["L"]
    assert _unspecialized(ks["rope"]) == ["L"]
    # D bounds the vectorized hidden axis and the strides set load alignment: those stay specialized
    assert _unspecialized(ks["affine_norm"]) == []


def test_kill_switch_specializes_again(kernels):
    ks = kernels("0")
    assert all(_unspecialized(k) == [] for k in ks.values())
