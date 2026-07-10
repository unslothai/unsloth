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
    # The video backend passes a dense default of `default` (clips amortise the
    # compile within one run); it must not affect GGUF or explicit values.
    assert resolve_speed_mode(None, is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_DEFAULT
    assert resolve_speed_mode("off", is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_OFF


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


def test_snapshot_partial_when_some_backends_missing(monkeypatch):
    # A build/platform without cuda.matmul (e.g. CPU/MPS) must still snapshot + restore the
    # flags it does have, rather than skipping the whole snapshot on one missing attribute.
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
        "fp16_accum": False,
    }
    assert pipe.vae.mem_format is None and pipe.compiled is False
    # off must not touch any process-wide flag (bit-identical reference path).
    assert torch.backends.cudnn.benchmark is False


def test_speed_compiles_both_dits_for_dual_dit_family(monkeypatch):
    # A dual-DiT family (Ideogram: transformer + unconditional_transformer) runs BOTH DiTs each
    # denoise step, so the regional block compile must engage on both, not just the first --
    # otherwise the second DiT runs eager while status reports compile as engaged.
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True, with_second_dit = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.compiled is True and pipe.second_compiled is True


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


def test_offload_active_drops_fullgraph(monkeypatch):
    # Group/model/sequential offload installs a torch.compiler.disable'd onload hook;
    # compiling with fullgraph=True then crashes at the first denoise step. Same reason
    # as an active step cache -> fullgraph must drop to False when offload is planned.
    # (Dense model: on this branch GGUF `default` takes the compiled-dequant path.)
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
    # The documented safety escape hatch must honor the common boolean spellings, not only
    # lowercase "1"/"true"/"yes": an operator setting UNSLOTH_DISABLE_FP16_ACCUM=TRUE to stop
    # fp16-accumulation drift would otherwise be silently ignored.
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
    # fp16 compute is where the accumulator width actually changes results (measured
    # same-seed drift, mean 2-5%): the quality-neutral tiers must refuse it.
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
    # max already trades exactness for speed (conv algos, max-autotune), so the 2x
    # fp16 accumulate joins that tier for fp16 pipelines.
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
    # Inductor's fused pointwise kernels keep intermediates in fp32 where eager rounds
    # to bf16 between ops; over a multi-step denoise that compounds to a visible drift
    # (LPIPS 0.221 vs bit-exact on HunyuanVideo-1.5-720p). emulate_precision_casts
    # restores eager's rounding at zero measured speed cost (LPIPS 0.052), so the
    # regional compile path must switch it on.
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
    # The flag is process-global, so the unload path must restore the pre-load value
    # exactly like the TF32 / cudnn.benchmark globals.
    torch = _stub_torch(monkeypatch)
    cfg = _stub_inductor_config(monkeypatch, torch, emulate = False)
    snap = snapshot_backend_flags()
    assert snap["inductor_emulate_precision_casts"] is False
    cfg.emulate_precision_casts = True
    restore_backend_flags(snap)
    assert cfg.emulate_precision_casts is False


def test_missing_inductor_config_is_tolerated(monkeypatch):
    # A build without torch._inductor (or with the flag renamed) must neither break the
    # snapshot nor the compile path.
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
    # The production load order engages the step cache BEFORE compile, so the regional
    # compile pass must re-arm the already-installed cache hooks with compiled inner
    # forwards (otherwise every computed step runs eager under the hook's
    # torch.compiler.disable; measured 1.69 vs 1.09 s/step on HunyuanVideo-1.5-720p).
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
