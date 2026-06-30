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

from core.inference import diffusion_speed as ds_mod
from core.inference.diffusion_speed import (
    SPEED_DEFAULT,
    SPEED_EAGER,
    SPEED_MAX,
    SPEED_OFF,
    apply_speed_optims,
    compile_eligible,
    normalize_speed_mode,
    resolve_speed_mode,
    restore_backend_flags,
    snapshot_backend_flags,
)


def _stub_gguf_accel(monkeypatch):
    """Replace the real compiled-dequant installer (which touches torch.compile /
    diffusers) with a recorder, so the tier-gating logic in apply_speed_optims is tested
    in isolation. Returns a dict of how many times it was called."""
    called = {"compiled_dequant": 0}

    def _install(logger = None):
        called["compiled_dequant"] += 1
        return True

    monkeypatch.setattr(ds_mod.gguf_compile, "install_compiled_dequant", _install)
    return called


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
        cudnn = types.SimpleNamespace(allow_tf32 = False, benchmark = False),
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


def test_resolve_speed_mode_gguf_auto_default():
    # Unset (None) -> default for GGUF (near-lossless), off for dense.
    assert resolve_speed_mode(None, is_gguf = True) == SPEED_DEFAULT
    assert resolve_speed_mode(None, is_gguf = False) == SPEED_OFF
    # An explicit value is honored verbatim, including an explicit opt-out to off.
    assert resolve_speed_mode("off", is_gguf = True) == SPEED_OFF
    assert resolve_speed_mode("max", is_gguf = True) == SPEED_MAX
    assert resolve_speed_mode("max", is_gguf = False) == SPEED_MAX


# ── compile gating ────────────────────────────────────────────────────────────


def test_compile_eligible_requires_bf16_cuda_friendly(monkeypatch):
    _stub_torch(monkeypatch)
    # The happy path: bf16, CUDA, compile-friendly family.
    assert compile_eligible(_target(), is_gguf = False, family = _family()) is True
    # GGUF is now compile-eligible too (measured ~2.3x, PSNR ~37 dB vs eager).
    assert compile_eligible(_target(), is_gguf = True, family = _family()) is True
    # fp16 (non-bf16) is excluded.
    assert compile_eligible(_target(dtype = "float16"), is_gguf = False, family = _family()) is False
    # A family flagged not compile-friendly is excluded.
    assert compile_eligible(_target(), is_gguf = False, family = _family(compile_ok = False)) is False
    # No compile support (e.g. XPU/MPS) is excluded.
    assert compile_eligible(_target(compile_ok = False), is_gguf = False, family = _family()) is False


# ── backend-flag snapshot / restore (TF32 / cudnn.benchmark leak guard) ────────


def test_snapshot_restore_backend_flags(monkeypatch):
    torch = _stub_torch(monkeypatch)
    snap = snapshot_backend_flags()
    assert snap == {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False}
    # An opt-in max run flips the globals on...
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # ...and restore puts them back, so a later `off` load is bit-identical again.
    restore_backend_flags(snap)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False


def test_restore_backend_flags_tolerates_none():
    restore_backend_flags(None)  # no torch needed, no-op


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

    def _compile(self, **kwargs):
        self.compiled = True
        self.compile_kwargs = kwargs

    def _fuse(self):
        self.fused = True


def test_speed_off_applies_nothing(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_OFF
    )
    assert applied == {
        "channels_last": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fused_qkv": False,
        "compiled": False,
        "compiled_dequant": False,
    }
    assert pipe.vae.mem_format is None and pipe.compiled is False
    # off must not touch any process-wide flag (bit-identical reference path).
    assert torch.backends.cudnn.benchmark is False


def test_speed_default_dense_falls_back_to_regional_compile(monkeypatch):
    # A DENSE model has no GGUF dequant to compile, so `default` falls back to the
    # regional block compile (its only compile lever) -- and no GGUF accelerators.
    torch = _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last
    assert applied["compiled"] is True and pipe.compiled is True
    # default compiles with dynamic=True and no autotune mode (fast cold start,
    # resolution-robust, sidesteps the CUDA-graph crash).
    assert pipe.compile_kwargs == {"fullgraph": True, "dynamic": True}
    # default also autotunes the VAE convs but does NOT flip TF32 or fuse QKV.
    assert applied["cudnn_benchmark"] is True and torch.backends.cudnn.benchmark is True
    assert applied["tf32"] is False and applied["fused_qkv"] is False
    # No GGUF dequant on a dense model.
    assert applied["compiled_dequant"] is False
    assert called == {"compiled_dequant": 0}


def test_speed_default_gguf_compiles_only_dequant(monkeypatch):
    # GGUF `default` is the LIGHT path: compile ONLY the dequant op chain, NOT the
    # regional block compile.
    _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = True, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["channels_last"] is True
    assert applied["compiled_dequant"] is True
    # The transformer block is NOT regionally compiled under GGUF default.
    assert applied["compiled"] is False and pipe.compiled is False
    assert called == {"compiled_dequant": 1}


def test_speed_eager_gguf_installs_no_accelerator(monkeypatch):
    # eager = lossless-but-no-compile: neither the compiled dequant nor the regional
    # block compile run; only the process-wide lossless levers (channels_last, cudnn)
    # and the shared/per-arch eager monkey-patches (installed elsewhere) engage.
    _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = True, family = _family(), speed_mode = SPEED_EAGER
    )
    assert applied["compiled_dequant"] is False and applied["compiled"] is False
    assert pipe.compiled is False
    assert called == {"compiled_dequant": 0}


def test_speed_max_gguf_regional_compile_not_dequant(monkeypatch):
    # GGUF `max` = the FULL regional block compile (which fuses the dequant inline), so
    # the standalone compiled dequant is deliberately OFF.
    _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = True, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled"] is True and pipe.compiled is True
    assert pipe.compile_kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert applied["compiled_dequant"] is False
    assert called == {"compiled_dequant": 0}


def test_speed_default_cudnn_benchmark_only_on_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe,
        _target(device = "mps", compile_ok = False),
        is_gguf = True,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert applied["cudnn_benchmark"] is False  # not CUDA -> no autotune flip


def test_speed_max_enables_tf32_and_fused_qkv(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["tf32"] is True and torch.backends.cuda.matmul.allow_tf32 is True
    assert applied["fused_qkv"] is True and pipe.fused is True
    # max opts into autotuned kernels (static shapes); CUDA-graph modes are avoided.
    assert pipe.compile_kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert pipe.compile_kwargs["dynamic"] is False


def test_speed_max_tf32_only_on_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _Pipe()
    applied = apply_speed_optims(
        pipe,
        _target(device = "mps", compile_ok = False),
        is_gguf = True,
        family = _family(),
        speed_mode = SPEED_MAX,
    )
    assert applied["tf32"] is False  # not CUDA -> no TF32


def test_apply_tolerates_missing_optims(monkeypatch):
    _stub_torch(monkeypatch)
    # A bare pipe (no vae.to, no compile, no fuse) must not crash.
    bare = types.SimpleNamespace(vae = None, transformer = types.SimpleNamespace())
    applied = apply_speed_optims(
        bare, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["channels_last"] is False and applied["fused_qkv"] is False
