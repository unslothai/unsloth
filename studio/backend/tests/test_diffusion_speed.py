# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the opt-in diffusion speed layer (``diffusion_speed.py``).

Hermetic: torch is stubbed via ``sys.modules`` only where a path needs it, so the
gating logic and the best-effort applier run without a GPU or real diffusers.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    compile_eligible,
    normalize_speed_mode,
)


def _target(
    *,
    device = "cuda",
    dtype = "bfloat16",
    compile_ok = True,
):
    return types.SimpleNamespace(
        device = device,
        dtype = dtype,
        supports_default_torch_compile = compile_ok,
    )


def _family(*, compile_ok = True):
    return types.SimpleNamespace(supports_torch_compile = compile_ok)


def _stub_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # _is_bfloat16 compares by identity then str fallback
    torch.channels_last = "channels_last"
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(matmul = types.SimpleNamespace(allow_tf32 = False)),
        cudnn = types.SimpleNamespace(allow_tf32 = False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_speed_mode():
    assert normalize_speed_mode(None) == SPEED_OFF
    assert normalize_speed_mode("") == SPEED_OFF
    assert normalize_speed_mode("MAX") == SPEED_MAX
    with pytest.raises(ValueError):
        normalize_speed_mode("ludicrous")


# ── compile gating ────────────────────────────────────────────────────────────


def test_compile_eligible_requires_bf16_cuda_friendly(monkeypatch):
    _stub_torch(monkeypatch)
    # The happy path: bf16, CUDA, compile-friendly family (GGUF is eligible too).
    assert compile_eligible(_target(), family = _family()) is True
    # fp16 (non-bf16) is excluded.
    assert compile_eligible(_target(dtype = "float16"), family = _family()) is False
    # A family flagged not compile-friendly (Z-Image) is excluded.
    assert compile_eligible(_target(), family = _family(compile_ok = False)) is False
    # No compile support (e.g. ROCm/XPU/MPS) is excluded.
    assert compile_eligible(_target(compile_ok = False), family = _family()) is False


# ── applier ───────────────────────────────────────────────────────────────────


class _Pipe:
    def __init__(
        self,
        *,
        with_compile = False,
        with_fuse = False,
    ) -> None:
        self.vae = types.SimpleNamespace(mem_format = None, to = self._vae_to)
        self.transformer = types.SimpleNamespace()
        if with_compile:
            self.transformer.compile_repeated_blocks = self._compile
        if with_fuse:
            self.fuse_qkv_projections = self._fuse
        self.compiled = False
        self.fused = False

    def _vae_to(self, *, memory_format):
        self.vae.mem_format = memory_format

    def _compile(self, *, fullgraph, dynamic):
        self.compiled = True

    def _fuse(self):
        self.fused = True


def test_speed_off_applies_nothing(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(pipe, _target(), family = _family(), speed_mode = SPEED_OFF)
    assert applied == {"channels_last": False, "tf32": False, "fused_qkv": False, "compiled": False}
    assert pipe.vae.mem_format is None and pipe.compiled is False


def test_speed_default_channels_last_and_compile_when_eligible(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(pipe, _target(), family = _family(), speed_mode = SPEED_DEFAULT)
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last
    assert applied["compiled"] is True and pipe.compiled is True
    # default does not flip TF32 or fuse QKV.
    assert applied["tf32"] is False and applied["fused_qkv"] is False


def test_speed_max_enables_tf32_and_fused_qkv(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(pipe, _target(), family = _family(), speed_mode = SPEED_MAX)
    assert applied["tf32"] is True and torch.backends.cuda.matmul.allow_tf32 is True
    assert applied["fused_qkv"] is True and pipe.fused is True


def test_speed_max_tf32_only_on_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _Pipe()
    applied = apply_speed_optims(
        pipe,
        _target(device = "mps", compile_ok = False),
        family = _family(),
        speed_mode = SPEED_MAX,
    )
    assert applied["tf32"] is False  # not CUDA -> no TF32


def test_apply_tolerates_missing_optims(monkeypatch):
    _stub_torch(monkeypatch)
    # A bare pipe (no vae.to, no compile, no fuse) must not crash.
    bare = types.SimpleNamespace(vae = None, transformer = types.SimpleNamespace())
    applied = apply_speed_optims(bare, _target(), family = _family(), speed_mode = SPEED_MAX)
    assert applied["channels_last"] is False and applied["fused_qkv"] is False
