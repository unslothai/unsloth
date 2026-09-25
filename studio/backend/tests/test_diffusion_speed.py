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


@pytest.fixture(autouse = True)
def _compile_runtime_independent_of_the_host(monkeypatch):
    """Keep these tests off the HOST's toolchain, which is what "hermetic" above claims.

    ``torch_compile_runtime_available`` asks whether THIS machine can run inductor, and on
    Windows that means asking whether a Triton wheel is installed. Without this, every
    compile-tier assertion in the file fails on a Windows checkout with no ``triton-windows``
    for a reason that has nothing to do with tiering (measured: 15 failures on a
    ``windows-latest`` runner, all green on Linux and macOS). Pin the non-Windows branch; the
    tests that are *about* Windows set ``sys.platform`` themselves and a later setattr wins.
    The lru_cache is dropped either side so one test's answer is never another test's."""
    ds_mod.torch_compile_runtime_available.cache_clear()
    monkeypatch.setattr(ds_mod.sys, "platform", "linux")
    yield
    ds_mod.torch_compile_runtime_available.cache_clear()


def _stub_torch(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # _is_bfloat16 compares by identity then str fallback
    torch.channels_last = "channels_last"
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(matmul = types.SimpleNamespace(allow_tf32 = False)),
        cudnn = types.SimpleNamespace(allow_tf32 = False, benchmark = False),
    )
    # Said explicitly so the CUDA-graph arm refuses deterministically, whatever the host has.
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    # The VAE-decode compile wraps a bound method; identity wrap is enough for tests.
    torch.compile = lambda fn, **kwargs: fn
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
    # The video backend passes a dense default of `default` (clips amortise the compile); it must not affect GGUF or explicit values.
    assert resolve_speed_mode(None, is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_DEFAULT
    assert resolve_speed_mode("off", is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_OFF


# ── compile gating ────────────────────────────────────────────────────────────


def test_compile_eligible_requires_bf16_cuda_friendly(monkeypatch):
    _stub_torch(monkeypatch)
    # The happy path: bf16, CUDA, compile-friendly family.
    assert compile_eligible(_target(), is_gguf = False, family = _family()) is True
    # GGUF is compile-eligible too (measured ~2.3x, PSNR ~37 dB vs eager).
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
    # The plain stub torch has no _inductor and no get_float32_matmul_precision, so none of those keys appear.
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


def test_snapshot_partial_when_some_backends_missing(monkeypatch):
    # A build without cuda.matmul (CPU/MPS) must still snapshot + restore the flags it does have.
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(),  # no .matmul
        cudnn = types.SimpleNamespace(benchmark = True),  # no .allow_tf32
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    snap = snapshot_backend_flags()
    assert snap == {"cudnn_benchmark": True}
    torch.backends.cudnn.benchmark = False
    restore_backend_flags(snap)
    assert torch.backends.cudnn.benchmark is True


def test_restore_is_independent_per_flag(monkeypatch):
    # A read-only / failing attribute must not abort restoring the remaining flags.
    torch = _stub_torch(monkeypatch)

    class _NoMatmulSet:
        @property
        def allow_tf32(self):
            return False

        @allow_tf32.setter
        def allow_tf32(self, value):
            raise RuntimeError("read-only on this build")

    torch.backends.cuda.matmul = _NoMatmulSet()
    snap = {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False}
    torch.backends.cudnn.benchmark = True
    restore_backend_flags(snap)  # matmul setter raises, cudnn still restored
    assert torch.backends.cudnn.benchmark is False


# ── applier ───────────────────────────────────────────────────────────────────


class _Pipe:
    def __init__(
        self,
        *,
        with_compile = False,
        with_fuse = False,
        with_second_dit = False,
    ) -> None:
        self.vae = types.SimpleNamespace(mem_format = None, to = self._vae_to)
        self.transformer = types.SimpleNamespace()
        if with_compile:
            self.transformer.compile_repeated_blocks = self._compile
        if with_fuse:
            self.fuse_qkv_projections = self._fuse
        self.compiled = False
        self.fused = False
        # A dual-DiT family (Ideogram) carries a second denoiser expert that runs every step.
        self.second_compiled = False
        if with_second_dit:
            self.unconditional_transformer = types.SimpleNamespace()
            if with_compile:
                self.unconditional_transformer.compile_repeated_blocks = self._compile2

    def _vae_to(self, *, memory_format):
        self.vae.mem_format = memory_format

    def _compile(self, **kwargs):
        self.compiled = True
        self.compile_kwargs = kwargs

    def _compile2(self, **kwargs):
        self.second_compiled = True

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
        "compiled_vae_decode": False,
        "fp16_accum": False,
        "cuda_graph": False,
    }
    assert pipe.vae.mem_format is None and pipe.compiled is False
    # off must not touch any process-wide flag (the bit-identical reference path).
    assert torch.backends.cudnn.benchmark is False


def test_speed_compiles_both_dits_for_dual_dit_family(monkeypatch):
    # A dual-DiT family runs BOTH DiTs each step, so the regional block compile must engage on both or one runs eager while status claims compiled.
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True, with_second_dit = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.compiled is True and pipe.second_compiled is True


def test_speed_default_dense_falls_back_to_regional_compile(monkeypatch):
    # A DENSE model has no GGUF dequant to compile, so `default` falls back to the regional block compile with no GGUF accelerators.
    torch = _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last
    assert applied["compiled"] is True and pipe.compiled is True
    # default compiles with dynamic=True and no autotune mode: fast cold start, resolution-robust, sidesteps the CUDA-graph crash.
    assert pipe.compile_kwargs == {"fullgraph": True, "dynamic": True}
    # default also autotunes the VAE convs but does NOT flip TF32 or fuse QKV.
    assert applied["cudnn_benchmark"] is True and torch.backends.cudnn.benchmark is True
    assert applied["tf32"] is False and applied["fused_qkv"] is False
    # No GGUF dequant on a dense model.
    assert applied["compiled_dequant"] is False
    assert called == {"compiled_dequant": 0}


def test_offload_active_drops_fullgraph(monkeypatch):
    # Offload installs a torch.compiler.disable'd onload hook, so fullgraph=True crashes at the first denoise step (as an active step cache does): it must drop to False.
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe,
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        offload_active = True,
    )
    assert applied["compiled"] is True
    assert pipe.compile_kwargs["fullgraph"] is False


def test_speed_default_gguf_compiles_only_dequant(monkeypatch):
    # GGUF `default` is the LIGHT path: compile ONLY the dequant op chain, NOT the regional block compile.
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
    # eager = lossless-but-no-compile: only the process-wide lossless levers and the eager monkey-patches engage.
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
    # GGUF `max` is the FULL regional block compile (which fuses the dequant inline), so the standalone compiled dequant is OFF.
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


@pytest.mark.parametrize(
    "hip, version",
    [("7.13.0", "2.11.0+rocm7.13.0"), (None, "2.9.1+rocmsdk20251116")],
    ids = ["version_hip", "rocm_tag_only"],
)
def test_speed_default_skips_cudnn_benchmark_on_rocm(monkeypatch, hip, version):
    torch = _stub_torch(monkeypatch)
    torch.version = types.SimpleNamespace(hip = hip)
    torch.__version__ = version
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = True, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["cudnn_benchmark"] is False
    assert torch.backends.cudnn.benchmark is False


def test_speed_max_enables_tf32_and_fused_qkv(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["tf32"] is True and torch.backends.cuda.matmul.allow_tf32 is True
    assert applied["fused_qkv"] is True and pipe.fused is True
    # max opts into autotuned kernels with automatic dynamic (static until a dimension changes, so a new prompt
    # length does not recompile every time); CUDA-graph modes are avoided.
    assert pipe.compile_kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert pipe.compile_kwargs["dynamic"] is None
    assert ds_mod.auto_dynamic_active(pipe) is True


def test_default_tier_does_not_mark_automatic_dynamic(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    assert pipe.compile_kwargs["dynamic"] is True
    assert ds_mod.auto_dynamic_active(pipe) is False
    assert isinstance(ds_mod.dynamo_graph_count(), int)


def test_max_tier_auto_dynamic_dit_shapes_are_not_tracked_as_static(monkeypatch):
    # A generalised DiT reuses one graph for unseen shapes; only the graph-count delta dirties its bundle.
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert ds_mod.auto_dynamic_active(pipe) is True
    assert ds_mod.compiled_shapes_are_static(pipe, SPEED_MAX) is False
    # A max-tier pipe whose DiT did not compile auto-dynamic keeps per-shape tracking.
    plain = _Pipe(with_compile = True)
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_MAX) is True
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_DEFAULT) is False


# ── U-Net whole-module compile fallback (SDXL) ─────────────────────────────────


class UNet2DConditionModel:
    """Fake with the diffusers class NAME the fallback keys on: no
    compile_repeated_blocks (U-Nets ship no _repeated_blocks), but Module.compile."""

    def __init__(self):
        self.compile_kwargs = None

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs


class _SomeOtherUNet(UNet2DConditionModel):
    pass


class _UNetPipe:
    def __init__(self, unet = None):
        self.mem_format = None
        self.fused = False
        self.vae = types.SimpleNamespace(to = self._vae_to, decode = lambda z: z)
        self.unet = UNet2DConditionModel() if unet is None else unet

    def _vae_to(self, *, memory_format):
        self.mem_format = memory_format

    def fuse_qkv_projections(self):
        self.fused = True


def test_unet_whole_compile_default_tier(monkeypatch):
    # SDXL's UNet has no _repeated_blocks, so `default` falls back to a whole-module STATIC compile (measured 1.61x at
    # LPIPS 0.034): fullgraph on, dynamic OFF. The U-Net recipe also fuses QKV and compiles the VAE decode.
    _stub_torch(monkeypatch)
    pipe = _UNetPipe()
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.unet.compile_kwargs == {"fullgraph": True, "dynamic": False}
    assert applied["fused_qkv"] is True and pipe.fused is True
    assert applied["compiled_vae_decode"] is True


def test_dit_default_tier_keeps_fuse_and_vae_decode_off(monkeypatch):
    # The DiT default tier is unchanged: fused QKV measured exactly neutral so it stays max-only, and the VAE decode stays eager.
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert applied["fused_qkv"] is False and pipe.fused is False
    assert applied["compiled_vae_decode"] is False


def test_unet_whole_compile_offload_drops_fullgraph(monkeypatch):
    # Offload hooks graph-break exactly as on the regional path.
    _stub_torch(monkeypatch)
    pipe = _UNetPipe()
    applied = apply_speed_optims(
        pipe,
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        offload_active = True,
    )
    assert applied["compiled"] is True
    assert pipe.unet.compile_kwargs == {"fullgraph": False, "dynamic": False}


def test_unet_whole_compile_max_tier_mode(monkeypatch):
    _stub_torch(monkeypatch)
    pipe = _UNetPipe()
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled"] is True
    assert pipe.unet.compile_kwargs == {
        "fullgraph": True,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
    }


def test_unet_whole_compile_gated_by_class_name(monkeypatch):
    # An unlisted U-Net class (unmeasured architecture) stays eager rather than paying an unvalidated whole-module compile.
    _stub_torch(monkeypatch)
    pipe = _UNetPipe(unet = _SomeOtherUNet())
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is False
    assert pipe.unet.compile_kwargs is None


def test_unet_whole_compile_failure_degrades_to_eager(monkeypatch):
    _stub_torch(monkeypatch)

    class _Boom(UNet2DConditionModel):
        def compile(self, **kwargs):
            raise RuntimeError("no dynamo on this build")

    _Boom.__name__ = "UNet2DConditionModel"
    pipe = _UNetPipe(unet = _Boom())
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is False  # best-effort: load proceeds eager


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


# ── fp16 accumulation (consumer fp16-GEMM fast path) ──────────────────────────


def _stub_torch_fp16_accum(
    monkeypatch,
    *,
    consumer = True,
    with_flag = True,
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.channels_last = "channels_last"
    matmul_attrs = {"allow_tf32": False}
    if with_flag:
        matmul_attrs["allow_fp16_accumulation"] = False
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(matmul = types.SimpleNamespace(**matmul_attrs)),
        cudnn = types.SimpleNamespace(allow_tf32 = False, benchmark = False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    import core.inference.diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda device = None: consumer)
    return torch


def test_snapshot_captures_fp16_accum_when_present(monkeypatch):
    torch = _stub_torch_fp16_accum(monkeypatch)
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    snap = snapshot_backend_flags()
    assert snap["matmul_fp16_accum"] is True
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    restore_backend_flags(snap)
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is True


def test_snapshot_skips_fp16_accum_on_older_torch(monkeypatch):
    _stub_torch_fp16_accum(monkeypatch, with_flag = False)
    snap = snapshot_backend_flags()
    assert "matmul_fp16_accum" not in snap
    restore_backend_flags(snap)  # nothing to restore, no error


def test_fp16_accum_engages_on_consumer_cuda(monkeypatch):
    torch = _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    applied = apply_speed_optims(
        _Pipe(), _target(), is_gguf = True, family = _family(), speed_mode = "default"
    )
    assert applied["fp16_accum"] is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is True


def test_fp16_accum_skipped_on_datacenter(monkeypatch):
    torch = _stub_torch_fp16_accum(monkeypatch, consumer = False)
    _stub_gguf_accel(monkeypatch)
    applied = apply_speed_optims(
        _Pipe(), _target(), is_gguf = True, family = _family(), speed_mode = "default"
    )
    assert applied["fp16_accum"] is False
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


def test_fp16_accum_respects_kill_switch(monkeypatch):
    _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DISABLE_FP16_ACCUM", "1")
    applied = apply_speed_optims(
        _Pipe(), _target(), is_gguf = True, family = _family(), speed_mode = "default"
    )
    assert applied["fp16_accum"] is False


@pytest.mark.parametrize("value", ["TRUE", "Yes", "On", " true "])
def test_fp16_accum_kill_switch_is_case_insensitive(monkeypatch, value):
    # The escape hatch must honor the common boolean spellings, so UNSLOTH_DISABLE_FP16_ACCUM=TRUE is not ignored.
    _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DISABLE_FP16_ACCUM", value)
    applied = apply_speed_optims(
        _Pipe(), _target(), is_gguf = True, family = _family(), speed_mode = "default"
    )
    assert applied["fp16_accum"] is False


def test_fp16_accum_respects_family_deny_list(monkeypatch):
    _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    monkeypatch.setattr(ds_mod, "_FP16_ACCUM_DENY", frozenset({"fragile-family"}))
    fam = types.SimpleNamespace(supports_torch_compile = True, name = "fragile-family")
    applied = apply_speed_optims(_Pipe(), _target(), is_gguf = True, family = fam, speed_mode = "default")
    assert applied["fp16_accum"] is False


def test_fp16_accum_skipped_when_flag_missing(monkeypatch):
    _stub_torch_fp16_accum(monkeypatch, consumer = True, with_flag = False)
    _stub_gguf_accel(monkeypatch)
    applied = apply_speed_optims(
        _Pipe(), _target(), is_gguf = True, family = _family(), speed_mode = "default"
    )
    assert applied["fp16_accum"] is False


def test_fp16_accum_not_touched_off_cuda(monkeypatch):
    torch = _stub_torch_fp16_accum(monkeypatch, consumer = True)
    applied = apply_speed_optims(
        _Pipe(),
        _target(device = "mps"),
        is_gguf = False,
        family = _family(),
        speed_mode = "eager",
    )
    assert applied["fp16_accum"] is False
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


def test_fp16_accum_denied_on_fp16_dtype_below_max(monkeypatch):
    # fp16 compute is where the accumulator width changes results (measured same-seed drift, mean 2-5%), so the quality-neutral tiers refuse it.
    torch = _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    for mode in ("eager", "default"):
        applied = apply_speed_optims(
            _Pipe(),
            _target(dtype = "float16"),
            is_gguf = True,
            family = _family(),
            speed_mode = mode,
        )
        assert applied["fp16_accum"] is False
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


def test_fp16_accum_allowed_on_fp16_dtype_under_max(monkeypatch):
    # max already trades exactness for speed, so the 2x fp16 accumulate joins that tier for fp16 pipelines.
    torch = _stub_torch_fp16_accum(monkeypatch, consumer = True)
    _stub_gguf_accel(monkeypatch)
    applied = apply_speed_optims(
        _Pipe(with_compile = True, with_fuse = True),
        _target(dtype = "float16"),
        is_gguf = True,
        family = _family(),
        speed_mode = "MAX",
    )
    assert applied["fp16_accum"] is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is True


# ── inductor precision-cast emulation (compile-vs-eager numeric parity) ─────────


def _stub_inductor_config(
    monkeypatch,
    torch,
    *,
    emulate = False,
):
    """Attach a fake ``_inductor.config`` to the stubbed torch module (diffusion_speed
    resolves it as attributes off the imported torch, never via sys.modules -- so the
    real torch._inductor lingering in sys.modules cannot leak into stubbed tests)."""
    cfg = types.SimpleNamespace(emulate_precision_casts = emulate)
    torch._inductor = types.SimpleNamespace(config = cfg)
    return cfg


def test_regional_compile_enables_emulate_precision_casts(monkeypatch):
    # Inductor's fused pointwise kernels keep intermediates in fp32 where eager rounds to bf16 between ops, which compounds
    # over a denoise. emulate_precision_casts restores eager's rounding at zero measured cost, so the regional compile sets it.
    torch = _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    cfg = _stub_inductor_config(monkeypatch, torch, emulate = False)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert cfg.emulate_precision_casts is True


def test_snapshot_restores_emulate_precision_casts(monkeypatch):
    # The flag is process-global, so unload must restore the pre-load value like the TF32 / cudnn.benchmark globals.
    torch = _stub_torch(monkeypatch)
    cfg = _stub_inductor_config(monkeypatch, torch, emulate = False)
    snap = snapshot_backend_flags()
    assert snap["inductor_emulate_precision_casts"] is False
    cfg.emulate_precision_casts = True
    restore_backend_flags(snap)
    assert cfg.emulate_precision_casts is False


def test_missing_inductor_config_is_tolerated(monkeypatch):
    # A build without torch._inductor (or with the flag renamed) must break neither the snapshot nor the compile path.
    _stub_torch(monkeypatch)  # the stub torch has no _inductor attribute
    _stub_gguf_accel(monkeypatch)
    snap = snapshot_backend_flags()
    assert "inductor_emulate_precision_casts" not in snap
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True


def test_regional_compile_arms_cache_hook_inners(monkeypatch):
    # Production engages the step cache BEFORE compile, so the regional compile pass must re-arm the installed cache hooks
    # with compiled inner forwards, else every computed step runs eager under the hook's torch.compiler.disable.
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    from core.inference import diffusion_cache as dc_mod

    armed = []
    monkeypatch.setattr(
        dc_mod,
        "_compile_hooked_block_inners",
        lambda transformer, logger = None: armed.append(transformer) or 1,
    )
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert armed == [pipe.transformer]


# ── the inductor runtime gate ────────────────────────────────────────────────
# The Unsloth workers already refuse torch.compile when Triton is missing on Windows; the diffusion
# and video backends run in the SERVER process, which those gates never reach.


def _clear_runtime_cache():
    from core.inference.diffusion_speed import torch_compile_runtime_available
    torch_compile_runtime_available.cache_clear()


def _set_crt_headers(monkeypatch, reachable: bool):
    from core import _msvc_env
    monkeypatch.setattr(_msvc_env, "crt_headers_reachable", lambda: reachable)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_torchdynamo_disable_is_honored_on_every_platform(monkeypatch, platform):
    from core.inference import diffusion_speed as ds_mod

    # compile_eligible reads torch to test the dtype, and without the stub it returns False for
    # every input -- which would make the assertions below pass whatever the gate did.
    _stub_torch(monkeypatch)
    # Both platforms, or the name is a claim the test never checks. The positive control must
    # clear the Windows toolchain question first, or the negatives hold for the wrong reason.
    monkeypatch.setattr(ds_mod.sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
        _set_crt_headers(monkeypatch, True)
    _clear_runtime_cache()
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    # The positive control. Without it the two `is False` lines below prove nothing.
    assert ds_mod.compile_eligible(_target(), is_gguf = False, family = _family()) is True

    _clear_runtime_cache()
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    assert ds_mod.torch_compile_runtime_available() is False
    assert ds_mod.compile_eligible(_target(), is_gguf = False, family = _family()) is False
    _clear_runtime_cache()
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    assert ds_mod.torch_compile_runtime_available() is True
    assert ds_mod.compile_eligible(_target(), is_gguf = False, family = _family()) is True
    _clear_runtime_cache()


def test_windows_without_triton_falls_back_to_eager(monkeypatch):
    """A compile call on a Windows install with no Triton wheel is not an error at compile time --
    it fails at the first forward, mid-generation. Decide it here instead."""
    from core.inference import diffusion_speed as ds_mod

    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    monkeypatch.setattr(ds_mod.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "triton", None)  # `import triton` -> ImportError
    _clear_runtime_cache()
    assert ds_mod.torch_compile_runtime_available() is False
    assert ds_mod.compile_eligible(_target(), is_gguf = False, family = _family()) is False

    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    _set_crt_headers(monkeypatch, True)
    _clear_runtime_cache()
    assert ds_mod.torch_compile_runtime_available() is True
    _clear_runtime_cache()


def test_windows_with_triton_but_no_msvc_falls_back_to_eager(monkeypatch):
    from core.inference import diffusion_speed as ds_mod

    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    monkeypatch.setattr(ds_mod.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    _set_crt_headers(monkeypatch, False)
    _clear_runtime_cache()
    assert ds_mod.torch_compile_runtime_available() is False

    _set_crt_headers(monkeypatch, True)
    _clear_runtime_cache()
    assert ds_mod.torch_compile_runtime_available() is True
    _clear_runtime_cache()


def test_gguf_dequant_respects_the_runtime_gate(monkeypatch):
    from core.inference import diffusion_speed as ds_mod

    _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    monkeypatch.setattr(ds_mod.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))

    _set_crt_headers(monkeypatch, False)
    _clear_runtime_cache()
    applied = apply_speed_optims(
        object(),
        _target(),
        is_gguf = True,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert called["compiled_dequant"] == 0
    assert not applied.get("compiled_dequant")

    _set_crt_headers(monkeypatch, True)
    _clear_runtime_cache()
    applied = apply_speed_optims(
        object(),
        _target(),
        is_gguf = True,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert called["compiled_dequant"] == 1
    assert applied.get("compiled_dequant") is True
    _clear_runtime_cache()


def test_linux_and_mac_are_not_asked_about_triton(monkeypatch):
    """Only Windows ships without it, and a probe import on a healthy Linux box is pure cost."""
    from core.inference import diffusion_speed as ds_mod

    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    monkeypatch.setitem(sys.modules, "triton", None)
    for platform_name in ("linux", "darwin"):
        monkeypatch.setattr(ds_mod.sys, "platform", platform_name)
        _clear_runtime_cache()
        assert ds_mod.torch_compile_runtime_available() is True
    _clear_runtime_cache()


_TORCHAO_FLAGS = (
    "coordinate_descent_tuning",
    "coordinate_descent_check_all_directions",
    "force_fuse_int_mm_with_mul",
    "fx_graph_cache",
)


def _stub_full_inductor_config(torch):
    cfg = types.SimpleNamespace(
        emulate_precision_casts = False,
        triton = types.SimpleNamespace(unique_kernel_names = False),
        **{name: False for name in _TORCHAO_FLAGS},
    )
    torch._inductor = types.SimpleNamespace(config = cfg)
    return cfg


def _stub_matmul_precision(
    torch,
    calls: list,
    initial = "highest",
):
    cell = {"v": initial}

    def _get():
        return cell["v"]

    def _set(v):
        calls.append(("precision", v))
        cell["v"] = v

    torch.get_float32_matmul_precision = _get
    torch.set_float32_matmul_precision = _set
    return cell


def test_snapshot_restores_torchao_inductor_flags(monkeypatch):
    torch = _stub_torch(monkeypatch)
    cfg = _stub_full_inductor_config(torch)
    snap = snapshot_backend_flags()
    for name in _TORCHAO_FLAGS:
        assert snap[f"inductor_{name}"] is False
    assert snap["inductor_triton_unique_kernel_names"] is False
    for name in _TORCHAO_FLAGS:
        setattr(cfg, name, True)
    cfg.triton.unique_kernel_names = True
    cfg.emulate_precision_casts = True
    restore_backend_flags(snap)
    for name in _TORCHAO_FLAGS:
        assert getattr(cfg, name) is False, name
    assert cfg.triton.unique_kernel_names is False
    assert cfg.emulate_precision_casts is False


def test_snapshot_restores_float32_matmul_precision(monkeypatch):
    torch = _stub_torch(monkeypatch)
    calls: list = []
    cell = _stub_matmul_precision(torch, calls, initial = "highest")
    snap = snapshot_backend_flags()
    assert snap["matmul_precision"] == "highest"
    cell["v"] = "high"
    restore_backend_flags(snap)
    assert cell["v"] == "highest"


def test_matmul_precision_is_restored_before_tf32(monkeypatch):
    torch = _stub_torch(monkeypatch)
    calls: list = []
    _stub_matmul_precision(torch, calls, initial = "highest")

    class _Matmul:
        def __init__(self):
            self._tf32 = False

        @property
        def allow_tf32(self):
            return self._tf32

        @allow_tf32.setter
        def allow_tf32(self, v):
            calls.append(("tf32", v))
            self._tf32 = v

    torch.backends.cuda.matmul = _Matmul()
    snap = snapshot_backend_flags()
    calls.clear()
    restore_backend_flags(snap)
    kinds = [k for k, _ in calls]
    assert kinds.index("precision") < kinds.index("tf32"), calls


def test_snapshot_skips_inductor_flags_a_build_lacks(monkeypatch):
    torch = _stub_torch(monkeypatch)
    cfg = _stub_inductor_config(monkeypatch, torch, emulate = False)
    snap = snapshot_backend_flags()
    assert snap["inductor_emulate_precision_casts"] is False
    for name in _TORCHAO_FLAGS:
        assert f"inductor_{name}" not in snap
    assert "inductor_triton_unique_kernel_names" not in snap
    assert "matmul_precision" not in snap
    cfg.emulate_precision_casts = True
    restore_backend_flags(snap)
    assert cfg.emulate_precision_casts is False


def test_video_snapshot_precedes_transformer_quant():
    """A failed load and an unload must both restore the pre-quant backend flags."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        snaps = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "snapshot_backend_flags"
        ]
        quants = [
            c.lineno
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "quantize_transformer"
        ]
        if snaps and quants:
            assert (
                min(snaps) < min(quants)
            ), f"{node.name}: snapshot_backend_flags at {snaps} must precede quantize_transformer at {quants}"
            return
    raise AssertionError(
        "no video.py function calls both snapshot_backend_flags and quantize_transformer"
    )


def _stub_cuda_graph(
    monkeypatch,
    *,
    eligible = True,
    reason = "ok",
):
    """Replace ``core.inference.diffusion_cuda_graph`` with a recorder that captures nothing.

    Into BOTH sys.modules and the package attribute: ``from . import X`` reads the attribute when
    an earlier import already bound it, and falls back to sys.modules only when it has not."""
    import core.inference as inference_pkg

    calls = {"eligible": [], "installs": 0}

    stub = types.ModuleType("core.inference.diffusion_cuda_graph")

    def _graph_eligible(target, **kwargs):
        calls["eligible"].append(kwargs)
        return eligible, reason

    def _install_cuda_graphs(pipe, *, logger = None):
        calls["installs"] += 1
        return ("h",)

    stub.graph_eligible = _graph_eligible
    stub.install_cuda_graphs = _install_cuda_graphs
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_cuda_graph", stub)
    monkeypatch.setattr(inference_pkg, "diffusion_cuda_graph", stub, raising = False)
    return calls


@pytest.mark.parametrize("mode", [SPEED_DEFAULT, SPEED_MAX])
def test_cuda_graph_engages_on_compile_tiers(monkeypatch, mode):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = mode)
    assert applied["cuda_graph"] is True
    assert calls["installs"] == 1
    assert pipe._unsloth_cuda_graph_reason == "ok"


@pytest.mark.parametrize("mode", [SPEED_OFF, SPEED_EAGER])
def test_cuda_graph_skipped_below_the_compile_tiers(monkeypatch, mode):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = mode)
    assert applied["cuda_graph"] is False
    assert calls["installs"] == 0 and calls["eligible"] == []


def test_cuda_graph_refusal_stashes_the_reason(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch, eligible = False, reason = "cpu offload active")
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["cuda_graph"] is False
    assert calls["installs"] == 0
    assert pipe._unsloth_cuda_graph_reason == "cpu offload active"


def test_cuda_graph_default_is_forwarded_as_family_default(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch)
    apply_speed_optims(
        _Pipe(with_compile = True),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        cuda_graph_default = False,
    )
    assert calls["eligible"][0]["family_default"] is False
    apply_speed_optims(
        _Pipe(with_compile = True),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert calls["eligible"][1]["family_default"] is True


def test_cuda_graph_cache_engaged_overrides_cache_active_for_the_graph_arm(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch)
    common = dict(is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    apply_speed_optims(
        _Pipe(with_compile = True), _target(), cache_active = True, cache_engaged = False, **common
    )
    assert calls["eligible"][0]["cache_active"] is False
    apply_speed_optims(
        _Pipe(with_compile = True), _target(), cache_active = False, cache_engaged = True, **common
    )
    assert calls["eligible"][1]["cache_active"] is True
    apply_speed_optims(_Pipe(with_compile = True), _target(), cache_active = True, **common)
    assert calls["eligible"][2]["cache_active"] is True


def test_cuda_graph_install_failure_leaves_the_load_usable(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    calls = _stub_cuda_graph(monkeypatch)

    def _boom(pipe, *, logger = None):
        calls["installs"] += 1
        raise RuntimeError("CUDA out of memory during capture")

    sys.modules["core.inference.diffusion_cuda_graph"].install_cuda_graphs = _boom
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["cuda_graph"] is False and calls["installs"] == 1
    assert applied["compiled"] is True  # the rest of the tier still engaged


# ── runtime compile failure falls back to eager ──────────────────────────────
# compile_repeated_blocks is lazy: inductor runs on the first forward, inside generate(). A lowering bug there
# (Qwen-Image-2.1 int8 / fp8: inductor CantSplit) used to fail every render; the guard drops that DiT to eager.


class _BackendCompilerFailed(Exception):
    pass


class _OutOfMemory(RuntimeError):
    pass


def _stub_torch_compile_errors(monkeypatch):
    torch = _stub_torch(monkeypatch)
    torch._dynamo = types.SimpleNamespace(
        exc = types.SimpleNamespace(BackendCompilerFailed = _BackendCompilerFailed)
    )
    torch.OutOfMemoryError = _OutOfMemory
    return torch


class _Block:
    """A Module.compile'd block: __call__ routes through _compiled_call_impl when set, like nn.Module."""

    def __init__(self, compiled) -> None:
        self.eager_calls = 0
        self._compiled_call_impl = compiled

    def _call_impl(self, x):
        self.eager_calls += 1
        return x + 1

    def __call__(self, x):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(x)
        return self._call_impl(x)


class _Dit:
    def __init__(self, blocks) -> None:
        self.blocks = blocks

    def modules(self):
        return [self, *self.blocks]


def test_compile_failure_at_first_forward_falls_back_to_eager(monkeypatch):
    _stub_torch_compile_errors(monkeypatch)
    calls = {"compiled": 0}

    def broken(x):
        calls["compiled"] += 1
        raise _BackendCompilerFailed("CantSplit: 4096*s87 - 4096*s89 not divisible by s87 - s89")

    blocks = [_Block(broken), _Block(broken)]
    dit = _Dit(blocks)
    assert ds_mod.guard_compiled_blocks(dit) == 2
    assert blocks[0](1) == 2  # the failed call itself is answered eagerly
    assert blocks[1](1) == 2
    assert blocks[0](5) == 6
    # One failure flips the whole DiT: the broken lowering is attempted once, not per block or per call.
    assert calls["compiled"] == 1
    assert all(b._compiled_call_impl is None for b in blocks)
    pipe = types.SimpleNamespace(transformer = dit)
    assert "CantSplit" in ds_mod.compile_fallback_error(pipe)


def test_compiled_block_that_works_is_untouched(monkeypatch):
    _stub_torch_compile_errors(monkeypatch)
    block = _Block(lambda x: x * 10)
    dit = _Dit([block])
    ds_mod.guard_compiled_blocks(dit)
    ds_mod.guard_compiled_blocks(dit)  # idempotent: no double wrap
    assert block(3) == 30
    assert block.eager_calls == 0
    assert ds_mod.compile_fallback_error(types.SimpleNamespace(transformer = dit)) is None


def test_settle_fallback_keeps_compiled_while_another_dit_still_compiles(monkeypatch):
    # Dual-DiT loads fall back per DiT. While the second expert still runs compiled, status keeps "compiled" (the LoRA
    # gate and the compile-cache shape registry read it) and only gains the fallback marker; once both fell back,
    # "compiled" goes.
    _stub_torch_compile_errors(monkeypatch)

    def broken(x):
        raise _BackendCompilerFailed("CantSplit")

    first = _Dit([_Block(broken)])
    second_block = _Block(broken)
    second_block_ok = _Block(lambda x: x * 2)
    second = _Dit([second_block_ok])
    ds_mod.guard_compiled_blocks(first)
    ds_mod.guard_compiled_blocks(second)
    pipe = types.SimpleNamespace(transformer = first, transformer_2 = second)
    state = types.SimpleNamespace(speed_optims = ("compiled", "cuda_graph"))
    first.blocks[0](1)
    assert "CantSplit" in ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("compiled", "cuda_graph", "compile_fallback_eager")
    ds_mod.settle_compile_fallback(state, pipe)  # idempotent
    assert state.speed_optims == ("compiled", "cuda_graph", "compile_fallback_eager")

    third = _Dit([second_block])
    ds_mod.guard_compiled_blocks(third)
    pipe.transformer_2 = third
    third.blocks[0](1)
    ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("cuda_graph", "compile_fallback_eager")


def test_settle_fallback_sees_a_modular_workflows_named_partition(monkeypatch):
    # MiniMax-H3's reference workflow compiles transformer_ref through a view; settlement runs on the bare pipe.
    _stub_torch_compile_errors(monkeypatch)

    def broken(x):
        raise _BackendCompilerFailed("CantSplit")

    ref = _Dit([_Block(broken)])
    ds_mod.guard_compiled_blocks(ref)
    pipe = types.SimpleNamespace(transformer_ref = ref)
    state = types.SimpleNamespace(speed_optims = ("compiled",))
    ref.blocks[0](1)
    assert "CantSplit" in ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("compile_fallback_eager",)


def test_settle_fallback_is_a_noop_without_a_failure(monkeypatch):
    _stub_torch_compile_errors(monkeypatch)
    dit = _Dit([_Block(lambda x: x)])
    ds_mod.guard_compiled_blocks(dit)
    state = types.SimpleNamespace(speed_optims = ("compiled",))
    assert ds_mod.settle_compile_fallback(state, types.SimpleNamespace(transformer = dit)) is None
    assert state.speed_optims == ("compiled",)


class _BackendOutOfMemory(RuntimeError):
    pass


_BackendOutOfMemory.__name__ = "OutOfMemoryError_"


@pytest.mark.parametrize("inner", ["message", "backend_class"])
def test_noncanonical_oom_under_a_compile_error_is_not_swallowed(monkeypatch, inner):
    # A compile-time allocation failure can surface as a plain RuntimeError("... out of memory ...") or a
    # backend-specific OutOfMemoryError_ under BackendCompilerFailed; the batch backoff must still see it.
    _stub_torch_compile_errors(monkeypatch)

    def fails(x):
        cause = (
            RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")
            if inner == "message"
            else (_BackendOutOfMemory("allocation failed"))
        )
        try:
            raise cause
        except RuntimeError as err:
            raise _BackendCompilerFailed("autotune failed") from err

    block = _Block(fails)
    dit = _Dit([block])
    ds_mod.guard_compiled_blocks(dit)
    with pytest.raises(_BackendCompilerFailed) as raised:
        block(1)
    assert block.eager_calls == 0
    assert ds_mod.compile_fallback_error(types.SimpleNamespace(transformer = dit)) is None
    # What the image / video handlers receive is the outer compiler error; their backoff must classify it as OOM.
    from core.inference.diffusion_batched import is_oom_error

    assert is_oom_error(raised.value)


@pytest.mark.parametrize("kind", ["runtime", "oom"])
def test_non_compile_errors_are_not_swallowed(monkeypatch, kind):
    # Only a failure while BUILDING the graph is safe to retry eagerly. A kernel error, and an OOM even when inductor
    # wrapped it, must reach the caller (the OOM backoff splits the batch on it).
    _stub_torch_compile_errors(monkeypatch)

    def fails(x):
        if kind == "runtime":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        try:
            raise _OutOfMemory("CUDA out of memory")
        except _OutOfMemory as oom:
            raise _BackendCompilerFailed("autotune ran out") from oom

    block = _Block(fails)
    dit = _Dit([block])
    ds_mod.guard_compiled_blocks(dit)
    with pytest.raises((RuntimeError, _BackendCompilerFailed)):
        block(1)
    assert block.eager_calls == 0
    assert ds_mod.compile_fallback_error(types.SimpleNamespace(transformer = dit)) is None


class _TorchaoWeight:
    pass


_TorchaoWeight.__module__ = "torchao.quantization.quantize_.workflows.int8.int8_tensor"


def test_torchao_dit_compiles_with_automatic_dynamic(monkeypatch):
    # dynamic=True made Qwen-Image-2.1's attention-output cat fuse into torchao's per-row activation-quant reduction,
    # which inductor cannot split; automatic dynamic (None) compiles it. A bf16 DiT keeps dynamic=True.
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    quant = _Pipe(with_compile = True)
    quant.transformer.parameters = lambda: iter([_TorchaoWeight()])
    apply_speed_optims(quant, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    assert quant.compile_kwargs["dynamic"] is None
    assert ds_mod.compiled_shapes_are_static(quant, SPEED_DEFAULT) is True

    dense = _Pipe(with_compile = True)
    dense.transformer.parameters = lambda: iter([object()])
    apply_speed_optims(dense, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    assert dense.compile_kwargs["dynamic"] is True
    assert ds_mod.compiled_shapes_are_static(dense, SPEED_DEFAULT) is False


def test_torchao_payload_wrapped_in_a_plain_parameter_still_counts():
    # Some torchao builds keep Linear.weight an nn.Parameter whose .data is the subclass.
    wrapped = types.SimpleNamespace(data = _TorchaoWeight())
    assert ds_mod._carries_torchao_weights(
        types.SimpleNamespace(parameters = lambda: iter([wrapped]))
    )
    other = types.SimpleNamespace(data = object())
    assert not ds_mod._carries_torchao_weights(
        types.SimpleNamespace(parameters = lambda: iter([other]))
    )


def test_compile_dynamic_is_the_value_the_cache_fingerprint_keys_on():
    # diffusion.py fingerprints compile-cache bundles with compile_dynamic, so an automatic-dynamic torchao build
    # never reuses a bundle written by the old explicit-dynamic path.
    quant = types.SimpleNamespace(parameters = lambda: iter([_TorchaoWeight()]))
    dense = types.SimpleNamespace(parameters = lambda: iter([object()]))
    assert ds_mod.compile_dynamic(quant, True) is None
    assert ds_mod.compile_dynamic(quant, False) is False
    assert ds_mod.compile_dynamic(dense, True) is True
    assert ds_mod.compile_dynamic(None, True) is True


def test_max_tier_compiles_torchao_dit_with_automatic_dynamic(monkeypatch):
    # max compiles every DiT with automatic dynamic, so a torchao DiT never gets dynamic=True (CantSplit) there either.
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    pipe.transformer.parameters = lambda: iter([_TorchaoWeight()])
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert pipe.compile_kwargs["dynamic"] is None
    assert ds_mod.auto_dynamic_active(pipe) is True
    assert ds_mod.compiled_shapes_are_static(pipe, SPEED_MAX) is False


def test_auto_dynamic_active_follows_the_torchao_marker():
    dit = types.SimpleNamespace()
    pipe = types.SimpleNamespace(transformer = dit)
    assert ds_mod.auto_dynamic_active(pipe) is False
    dit._unsloth_auto_dynamic = True
    assert ds_mod.auto_dynamic_active(pipe) is True
    assert isinstance(ds_mod.dynamo_graph_count(), int)


def test_real_rope_installed_before_the_qwen_image_21_block_compile_only(monkeypatch):
    from core.inference import diffusion_qwenimage21_rope as rope

    _stub_torch(monkeypatch)
    order = []
    monkeypatch.setattr(rope, "install", lambda logger = None: order.append("rope") or True)
    pipe = _Pipe(with_compile = True)
    real_compile = pipe._compile
    pipe.transformer.compile_repeated_blocks = lambda **kw: order.append("compile") or real_compile(
        **kw
    )
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    assert order == ["compile"]  # any other DiT keeps the stock RoPE

    order.clear()
    QwenImage21Transformer2DModel = type("QwenImage21Transformer2DModel", (), {})
    pipe = _Pipe(with_compile = True)
    pipe.transformer = QwenImage21Transformer2DModel()
    pipe.transformer.compile_repeated_blocks = lambda **kw: order.append("compile")
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert order == ["rope", "compile"] and applied["compiled"] is True

    # A failing install never costs the compile.
    order.clear()
    monkeypatch.setattr(
        rope, "install", lambda logger = None: (_ for _ in ()).throw(RuntimeError("probe"))
    )
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert order == ["compile"] and applied["compiled"] is True
