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

from core.inference import diffusion_compile_config as compile_config
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
def _fresh_compile_knobs():
    compile_config._reset_for_tests()
    yield
    compile_config._reset_for_tests()


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
    torch.float16 = "float16"
    torch.channels_last = "channels_last"
    torch.contiguous_format = "contiguous_format"
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(matmul = types.SimpleNamespace(allow_tf32 = False)),
        cudnn = types.SimpleNamespace(allow_tf32 = False, benchmark = False),
    )
    # explicit so the CUDA-graph arm refuses deterministically on any host
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.compile_calls = []

    def _compile(fn, **kwargs):
        torch.compile_calls.append(kwargs)
        return fn

    torch.compile = _compile
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_normalize_speed_mode():
    assert normalize_speed_mode(None) == SPEED_OFF
    assert normalize_speed_mode("") == SPEED_OFF
    assert normalize_speed_mode("MAX") == SPEED_MAX
    with pytest.raises(ValueError):
        normalize_speed_mode("ludicrous")


def test_resolve_speed_mode_gguf_auto_default():
    assert resolve_speed_mode(None, is_gguf = True) == SPEED_DEFAULT
    assert resolve_speed_mode(None, is_gguf = False) == SPEED_OFF
    assert resolve_speed_mode("off", is_gguf = True) == SPEED_OFF
    assert resolve_speed_mode("max", is_gguf = True) == SPEED_MAX
    assert resolve_speed_mode("max", is_gguf = False) == SPEED_MAX
    # video passes dense_default (clips amortise compile); it must not affect GGUF or explicit values
    assert resolve_speed_mode(None, is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_DEFAULT
    assert resolve_speed_mode("off", is_gguf = False, dense_default = SPEED_DEFAULT) == SPEED_OFF


def test_compile_eligible_requires_bf16_cuda_friendly(monkeypatch):
    _stub_torch(monkeypatch)
    assert compile_eligible(_target(), is_gguf = False, family = _family()) is True
    # GGUF is compile-eligible too (measured ~2.3x, PSNR ~37 dB vs eager)
    assert compile_eligible(_target(), is_gguf = True, family = _family()) is True
    # fp16 is excluded when the card capability cannot be read (no probe in this stub)
    assert compile_eligible(_target(dtype = "float16"), is_gguf = False, family = _family()) is False
    assert compile_eligible(_target(), is_gguf = False, family = _family(compile_ok = False)) is False
    assert compile_eligible(_target(compile_ok = False), is_gguf = False, family = _family()) is False


def _stub_torch_capability(
    monkeypatch,
    cap,
    seen = None,
):
    torch = _stub_torch(monkeypatch)
    torch.float16 = "float16"

    def get_device_capability(*args):
        if seen is not None:
            seen.append(args)
        return cap

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True, get_device_capability = get_device_capability
    )
    return torch


def _fp16_target(**kw):
    t = _target(dtype = "float16")
    for k, v in kw.items():
        setattr(t, k, v)
    return t


@pytest.mark.parametrize("cap", [(7, 5), (8, 6), (8, 9), (12, 0)])
def test_compile_eligible_fp16_on_turing_or_newer(monkeypatch, cap):
    _stub_torch_capability(monkeypatch, cap)
    assert compile_eligible(_fp16_target(), is_gguf = False, family = _family()) is True
    assert compile_eligible(_fp16_target(backend = "cuda"), is_gguf = True, family = _family()) is True


@pytest.mark.parametrize("cap", [(7, 0), (6, 1), (6, 0), (5, 2)])
def test_compile_eligible_fp16_stays_eager_below_turing(monkeypatch, cap):
    _stub_torch_capability(monkeypatch, cap)
    assert compile_eligible(_fp16_target(), is_gguf = False, family = _family()) is False


def test_compile_eligible_fp16_asks_the_selected_card(monkeypatch):
    seen = []
    _stub_torch_capability(monkeypatch, (7, 5), seen)
    assert compile_eligible(_fp16_target(ordinal = 3), is_gguf = False, family = _family()) is True
    assert seen == [(3,)]


def test_compile_eligible_fp16_refusals(monkeypatch):
    _stub_torch_capability(monkeypatch, (7, 5))
    fam = types.SimpleNamespace(supports_torch_compile = True, fp16_incompatible = True)
    assert compile_eligible(_fp16_target(), is_gguf = False, family = fam) is False
    assert compile_eligible(_fp16_target(backend = "rocm"), is_gguf = False, family = _family()) is False
    assert (
        compile_eligible(_fp16_target(), is_gguf = False, family = _family(compile_ok = False)) is False
    )
    assert (
        compile_eligible(
            _fp16_target(supports_default_torch_compile = False), is_gguf = False, family = _family()
        )
        is False
    )
    assert compile_eligible(_target(dtype = "float32"), is_gguf = False, family = _family()) is False


def test_compile_eligible_fp16_probe_failure_stays_eager(monkeypatch):
    torch = _stub_torch_capability(monkeypatch, (7, 5))

    def boom(*args):
        raise RuntimeError("no device")

    torch.cuda.get_device_capability = boom
    assert compile_eligible(_fp16_target(), is_gguf = False, family = _family()) is False


def test_fp16_compile_is_explicit_tier_only(monkeypatch):
    _stub_torch_capability(monkeypatch, (7, 5))
    assert ds_mod.fp16_compile_explicit_only(_fp16_target()) is True
    assert compile_eligible(_fp16_target(), is_gguf = False, family = _family()) is True
    assert ds_mod.fp16_compile_explicit_only(_target()) is False
    assert ds_mod.fp16_compile_explicit_only(_target(dtype = "float32")) is False


def test_compile_eligible_bf16_never_probes_capability(monkeypatch):
    seen = []
    _stub_torch_capability(monkeypatch, (7, 0), seen)
    fam = types.SimpleNamespace(supports_torch_compile = True, fp16_incompatible = True)
    assert compile_eligible(_target(), is_gguf = False, family = fam) is True
    assert seen == []


def test_apply_speed_optims_compiles_fp16_dit(monkeypatch):
    _stub_torch_capability(monkeypatch, (7, 5))
    monkeypatch.setattr(ds_mod, "_compile_repeated_blocks", lambda *a, **k: True)
    applied = apply_speed_optims(
        types.SimpleNamespace(),
        _fp16_target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert applied["compiled"] is True


def _unet_pipe():
    UNet2DConditionModel = type("UNet2DConditionModel", (), {})
    return types.SimpleNamespace(unet = UNet2DConditionModel())


@pytest.mark.parametrize("mode", [SPEED_DEFAULT, SPEED_MAX])
def test_fp16_unet_stays_eager_under_offload(monkeypatch, mode):
    _stub_torch_capability(monkeypatch, (7, 5))
    calls = []
    monkeypatch.setattr(ds_mod, "_compile_repeated_blocks", lambda *a, **k: calls.append(1) or True)
    monkeypatch.setattr(ds_mod, "_fuse_qkv", lambda *a, **k: True)
    off = apply_speed_optims(
        _unet_pipe(),
        _fp16_target(),
        is_gguf = False,
        family = _family(),
        speed_mode = mode,
        offload_active = True,
    )
    assert off["compiled"] is False and calls == []
    if mode == SPEED_DEFAULT:
        assert off["fused_qkv"] is False
    resident = apply_speed_optims(
        _unet_pipe(),
        _fp16_target(),
        is_gguf = False,
        family = _family(),
        speed_mode = mode,
        offload_active = False,
    )
    assert resident["compiled"] is True


def test_fp16_offloaded_dit_and_bf16_unet_still_compile(monkeypatch):
    _stub_torch_capability(monkeypatch, (7, 5))
    monkeypatch.setattr(ds_mod, "_compile_repeated_blocks", lambda *a, **k: True)
    monkeypatch.setattr(ds_mod, "_fuse_qkv", lambda *a, **k: True)
    dit = apply_speed_optims(
        types.SimpleNamespace(),
        _fp16_target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        offload_active = True,
    )
    assert dit["compiled"] is True
    bf16 = apply_speed_optims(
        _unet_pipe(),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        offload_active = True,
    )
    assert bf16["compiled"] is True
    assert ds_mod.fp16_unet_offloaded(_target(), _unet_pipe(), offload_active = True) is False
    assert ds_mod.fp16_unet_offloaded(_fp16_target(), _unet_pipe(), offload_active = True) is True
    assert ds_mod.fp16_unet_offloaded(_fp16_target(), _unet_pipe(), offload_active = False) is False


def test_snapshot_restore_backend_flags(monkeypatch):
    torch = _stub_torch(monkeypatch)
    snap = snapshot_backend_flags()
    # the stub torch has no _inductor or get_float32_matmul_precision
    assert snap == {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    restore_backend_flags(snap)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False


def test_restore_backend_flags_tolerates_none():
    restore_backend_flags(None)


def test_snapshot_partial_when_some_backends_missing(monkeypatch):
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        cuda = types.SimpleNamespace(),
        cudnn = types.SimpleNamespace(benchmark = True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    snap = snapshot_backend_flags()
    assert snap == {"cudnn_benchmark": True}
    torch.backends.cudnn.benchmark = False
    restore_backend_flags(snap)
    assert torch.backends.cudnn.benchmark is True


def test_restore_is_independent_per_flag(monkeypatch):
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
    restore_backend_flags(snap)
    assert torch.backends.cudnn.benchmark is False


class AutoencoderKL(types.SimpleNamespace):
    """The class NAME matters: ``auto`` keys the allow list off it."""


class AutoencoderKLWan(types.SimpleNamespace):
    """A video VAE."""


class _Pipe:
    def __init__(
        self,
        *,
        with_compile = False,
        with_fuse = False,
        with_second_dit = False,
        vae_cls = AutoencoderKL,
    ) -> None:
        self.vae = vae_cls(mem_format = None, to = self._vae_to, decode = lambda z: z)
        self.transformer = types.SimpleNamespace()
        if with_compile:
            self.transformer.compile_repeated_blocks = self._compile
        if with_fuse:
            self.fuse_qkv_projections = self._fuse
        self.compiled = False
        self.fused = False
        # dual-DiT families (Ideogram) run a second denoiser expert every step
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
        "vae_fp16_decode": False,
        "vae_single_frame": False,
        "vae_fused": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fused_qkv": False,
        "compiled": False,
        "compiled_dequant": False,
        "rocm_query_chunks": False,
        "compiled_vae_decode": False,
        "fp16_accum": False,
        "cuda_graph": False,
        "int8_gemm": False,
    }
    assert pipe.vae.mem_format is None and pipe.compiled is False
    # off is the bit-identical reference path: it must not touch process-wide flags
    assert torch.backends.cudnn.benchmark is False


def test_speed_compiles_both_dits_for_dual_dit_family(monkeypatch):
    # dual-DiT runs both DiTs each step, so both must compile or status lies
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True, with_second_dit = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.compiled is True and pipe.second_compiled is True


def test_speed_default_dense_falls_back_to_regional_compile(monkeypatch):
    torch = _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["channels_last"] is False and pipe.vae.mem_format == torch.contiguous_format
    assert applied["compiled"] is True and pipe.compiled is True
    # dynamic=True, no autotune: fast cold start, resolution-robust, avoids the CUDA-graph crash
    assert pipe.compile_kwargs == {"fullgraph": True, "dynamic": True}
    assert applied["cudnn_benchmark"] is True and torch.backends.cudnn.benchmark is True
    assert applied["tf32"] is False and applied["fused_qkv"] is False
    assert applied["compiled_dequant"] is False
    assert called == {"compiled_dequant": 0}


def test_offload_active_drops_fullgraph(monkeypatch):
    # the offload onload hook is compiler-disabled, so fullgraph=True would crash at step 1
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
    # GGUF default compiles only the dequant op chain, not the regional block compile
    _stub_torch(monkeypatch)
    called = _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = True, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["channels_last"] is False
    assert applied["compiled_dequant"] is True
    assert applied["compiled"] is False and pipe.compiled is False
    assert called == {"compiled_dequant": 1}


def test_speed_eager_gguf_installs_no_accelerator(monkeypatch):
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
    # GGUF max fuses the dequant inline via the block compile, so standalone dequant compile is off
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
    assert applied["cudnn_benchmark"] is False


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


def test_speed_default_respects_family_cudnn_benchmark_opt_out(monkeypatch):
    torch = _stub_torch(monkeypatch)
    family = types.SimpleNamespace(supports_torch_compile = True, cudnn_benchmark = False)
    applied = apply_speed_optims(
        _Pipe(with_compile = True), _target(), is_gguf = False, family = family, speed_mode = SPEED_DEFAULT
    )
    assert applied["cudnn_benchmark"] is False
    assert torch.backends.cudnn.benchmark is False
    family = types.SimpleNamespace(supports_torch_compile = True, cudnn_benchmark = True)
    applied = apply_speed_optims(
        _Pipe(with_compile = True), _target(), is_gguf = False, family = family, speed_mode = SPEED_DEFAULT
    )
    assert applied["cudnn_benchmark"] is True and torch.backends.cudnn.benchmark is True


def test_cudnn_benchmark_opt_out_image_families():
    from core.inference.diffusion_families import _FAMILIES
    off = {fam.name for fam in _FAMILIES if not fam.cudnn_benchmark}
    assert off == {"qwen-image", "flux.1", "z-image", "sdxl"}


def test_speed_max_enables_tf32_and_fused_qkv(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["tf32"] is True and torch.backends.cuda.matmul.allow_tf32 is True
    assert applied["fused_qkv"] is True and pipe.fused is True
    # automatic dynamic avoids recompiling per prompt length; CUDA-graph modes are avoided
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
    # a generalised DiT reuses one graph; only the graph-count delta dirties its bundle
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert ds_mod.auto_dynamic_active(pipe) is True
    assert ds_mod.compiled_shapes_are_static(pipe, SPEED_MAX) is False
    plain = _Pipe(with_compile = True)
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_MAX) is True
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_DEFAULT) is False


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
        self.vae = AutoencoderKL(to = self._vae_to, decode = lambda z: z)
        self.unet = UNet2DConditionModel() if unet is None else unet

    def _vae_to(self, *, memory_format):
        self.mem_format = memory_format

    def fuse_qkv_projections(self):
        self.fused = True


def test_unet_whole_compile_default_tier(monkeypatch):
    # SDXL UNet has no _repeated_blocks: whole-module static compile (measured 1.61x, LPIPS 0.034)
    _stub_torch(monkeypatch)
    pipe = _UNetPipe()
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.unet.compile_kwargs == {"fullgraph": True, "dynamic": False}
    assert applied["fused_qkv"] is True and pipe.fused is True
    assert applied["compiled_vae_decode"] is True


@pytest.mark.parametrize("tier", [SPEED_DEFAULT, SPEED_MAX])
def test_a_unet_keeps_its_decode_recipe_and_bundle_key_on_every_tier(monkeypatch, tier):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _UNetPipe()
    applied = apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = tier)
    assert applied["compiled_vae_decode"] is True
    assert {"fullgraph": False, "dynamic": True} in torch.compile_calls
    assert not any(call.get("mode") for call in torch.compile_calls)
    assert ds_mod.vae_decode_compile_allowed(pipe, tier) is False


def test_dit_default_tier_keeps_fuse_off_and_leaves_the_vae_decode_eager(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    assert type(pipe.vae).__name__ in ds_mod._VAE_COMPILE_ALLOW
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert applied["fused_qkv"] is False and pipe.fused is False
    assert applied["compiled_vae_decode"] is False
    assert torch.compile_calls == []
    assert ds_mod.vae_decode_compile_allowed(pipe, SPEED_DEFAULT) is False
    assert ds_mod.vae_decode_compile_allowed(pipe, SPEED_MAX) is True


@pytest.mark.parametrize("tier", [SPEED_EAGER, SPEED_DEFAULT])
def test_eager_vae_decode_keeps_contiguous_weights(monkeypatch, tier):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = tier)
    assert applied["compiled_vae_decode"] is False
    assert applied["channels_last"] is False and pipe.vae.mem_format == torch.contiguous_format


@pytest.mark.parametrize("tier", [SPEED_EAGER, SPEED_DEFAULT])
def test_fused_vae_keeps_its_channels_last_weights(monkeypatch, tier):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True)

    def fused(p, logger):
        p.vae._unsloth_vae_fused_cl_weights = True
        return True

    monkeypatch.setattr(ds_mod, "_install_fused_vae", fused)
    applied = apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = tier)
    assert applied["vae_fused"] is True
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last


@pytest.mark.parametrize(
    "backend, device", [("rocm", "cuda"), ("mps", "mps"), ("xpu", "xpu"), ("cpu", "cpu")]
)
def test_eager_vae_decode_layout_unchanged_off_nvidia(monkeypatch, backend, device):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    target = _target(device = device)
    target.backend = backend
    applied = apply_speed_optims(
        pipe, target, is_gguf = False, family = _family(), speed_mode = SPEED_EAGER
    )
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last


@pytest.mark.parametrize(
    "dtype, expect_channels_last",
    [("torch.bfloat16", True), ("float16", True), ("torch.float32", False)],
)
def test_offloaded_eager_decode_keeps_channels_last_unless_fp32(
    monkeypatch, dtype, expect_channels_last
):
    # eager 16-bit contiguous decode peaks ~0.63 GiB/MP higher; fp32 is smaller contiguous
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    pipe.vae.dtype = dtype
    applied = apply_speed_optims(
        pipe,
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_EAGER,
        offload_active = True,
    )
    assert applied["channels_last"] is expect_channels_last
    assert pipe.vae.mem_format == (
        torch.channels_last if expect_channels_last else torch.contiguous_format
    )


def test_unmeasured_vae_class_keeps_channels_last(monkeypatch):
    torch = _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, vae_cls = type("HYVAE2D", (types.SimpleNamespace,), {}))
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_EAGER
    )
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last


def test_compiled_vae_decode_keeps_channels_last(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True)
    pipe.vae.dtype = "torch.bfloat16"
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled_vae_decode"] is True
    assert applied["channels_last"] is True and pipe.vae.mem_format == torch.channels_last


@pytest.mark.parametrize(
    "dtype, force_upcast, expect_channels_last",
    [
        ("torch.float32", False, False),
        ("float16", True, False),
        ("float16", False, True),
        ("bfloat16", True, True),
    ],
)
def test_unet_compiled_decode_in_fp32_keeps_contiguous_weights(
    monkeypatch, dtype, force_upcast, expect_channels_last
):
    # SDXL decodes force_upcast fp16 VAEs in fp32, where channels_last measured 0.89x (T4)
    torch = _stub_torch(monkeypatch)
    pipe = _UNetPipe()
    pipe.vae.dtype = dtype
    pipe.vae.config = types.SimpleNamespace(force_upcast = force_upcast)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled_vae_decode"] is True
    assert applied["channels_last"] is expect_channels_last
    assert pipe.mem_format == (
        torch.channels_last if expect_channels_last else torch.contiguous_format
    )


def test_dit_default_tier_vae_decode_compile_forced_on_by_env(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True and applied["compiled_vae_decode"] is True
    assert torch.compile_calls == [{"fullgraph": False, "dynamic": True}]
    assert ds_mod.vae_decode_compile_allowed(pipe, SPEED_DEFAULT) is True


def test_dit_vae_decode_compile_opts_out_by_env(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "0")
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled"] is True and applied["compiled_vae_decode"] is False
    assert torch.compile_calls == []
    assert ds_mod.vae_decode_compile_allowed(pipe, SPEED_MAX) is False


def test_dit_vae_decode_compile_deny_set_and_force(monkeypatch):
    assert "AutoencoderKLQwenImage" in ds_mod._VAE_COMPILE_DENY
    _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True)
    monkeypatch.setattr(
        ds_mod,
        "_VAE_COMPILE_DENY",
        frozenset({type(pipe.vae).__name__}),
    )
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled_vae_decode"] is False
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    applied = apply_speed_optims(
        _Pipe(with_compile = True),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
    )
    assert applied["compiled_vae_decode"] is True


def test_video_vae_stays_eager_under_auto_and_compiles_once_per_pipe(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True, vae_cls = AutoencoderKLWan)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True and applied["compiled_vae_decode"] is False
    assert torch.compile_calls == []
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    for _ in range(2):
        applied = apply_speed_optims(
            pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
        )
        assert applied["compiled_vae_decode"] is True
    assert torch.compile_calls == [{"fullgraph": False, "dynamic": True}]


def test_dit_vae_decode_compile_max_tier_autotunes(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled_vae_decode"] is True
    assert torch.compile_calls == [
        {"fullgraph": False, "dynamic": True, "mode": "max-autotune-no-cudagraphs"}
    ]


@pytest.mark.parametrize("vae_name", ["AutoencoderKL", "AutoencoderKLFlux2"])
def test_max_tier_vae_decode_compiles_dynamic_once_for_every_resolution(monkeypatch, vae_name):
    """Automatic dynamic paid a generalising VAE recompile on the second resolution (minutes on max); dynamic=True
    compiles once, and a VAE compile never marks the load automatic-dynamic."""
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    pipe = _Pipe(with_compile = True, vae_cls = type(vae_name, (AutoencoderKL,), {}))
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["compiled_vae_decode"] is True
    assert [c["dynamic"] for c in torch.compile_calls] == [True]
    assert not getattr(pipe.vae, "_unsloth_auto_dynamic", False)


def test_qwen_image_vae_decode_stays_eager_with_the_single_frame_path(monkeypatch):
    """Not worth a compile on max (see _VAE_COMPILE_DENY), and never keyed on the marker installed after the
    compile-cache fingerprint: the loader and apply_speed_optims must give the same answer."""
    _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    AutoencoderKLQwenImage = type("AutoencoderKLQwenImage", (), {})
    pipe = types.SimpleNamespace(vae = AutoencoderKLQwenImage())
    assert ds_mod._vae_decode_compile_allowed(pipe, SPEED_MAX) is False
    pipe.vae._unsloth_single_frame = True
    assert ds_mod._vae_decode_compile_allowed(pipe, SPEED_MAX) is False
    assert ds_mod._vae_decode_compile_allowed(pipe, SPEED_DEFAULT) is False
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    assert ds_mod._vae_decode_compile_allowed(pipe, SPEED_DEFAULT) is True


def test_unet_vae_decode_compile_ignores_the_env(monkeypatch):
    _stub_torch(monkeypatch)
    monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "0")
    pipe = _UNetPipe()
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled_vae_decode"] is True


def test_video_wan_vae_decode_is_denied_on_measurement(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    assert "AutoencoderKLWan" in ds_mod._VAE_COMPILE_DENY
    monkeypatch.setattr(
        ds_mod, "_VAE_COMPILE_ALLOW", ds_mod._VAE_COMPILE_ALLOW | {"AutoencoderKLWan"}
    )
    applied = apply_speed_optims(
        _Pipe(with_compile = True, vae_cls = AutoencoderKLWan),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_DEFAULT,
        cuda_graph_default = False,
    )
    assert applied["compiled"] is True and applied["compiled_vae_decode"] is False
    assert torch.compile_calls == []


def test_video_wan_vae_decode_stays_denied_on_max(monkeypatch):
    torch = _stub_torch(monkeypatch)
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    applied = apply_speed_optims(
        _Pipe(with_compile = True, vae_cls = AutoencoderKLWan),
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_MAX,
        cuda_graph_default = False,
    )
    assert applied["compiled_vae_decode"] is False
    assert torch.compile_calls == []


def test_unet_whole_compile_offload_drops_fullgraph(monkeypatch):
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
    # unmeasured U-Net classes stay eager rather than pay an unvalidated compile
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
    assert applied["compiled"] is False


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
    assert applied["tf32"] is False


def test_apply_tolerates_missing_optims(monkeypatch):
    _stub_torch(monkeypatch)
    bare = types.SimpleNamespace(vae = None, transformer = types.SimpleNamespace())
    applied = apply_speed_optims(
        bare, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX
    )
    assert applied["channels_last"] is False and applied["fused_qkv"] is False


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
    restore_backend_flags(snap)


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
    # fp16 accumulation drifts same-seed output 2-5%, so quality-neutral tiers refuse it
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
    # inductor keeps fp32 intermediates where eager rounds to bf16; emulate_precision_casts fixes it
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
    # the flag is process-global, so unload must restore it
    torch = _stub_torch(monkeypatch)
    cfg = _stub_inductor_config(monkeypatch, torch, emulate = False)
    snap = snapshot_backend_flags()
    assert snap["inductor_emulate_precision_casts"] is False
    cfg.emulate_precision_casts = True
    restore_backend_flags(snap)
    assert cfg.emulate_precision_casts is False


def test_missing_inductor_config_is_tolerated(monkeypatch):
    # builds without torch._inductor (or a renamed flag) must break neither snapshot nor compile
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    snap = snapshot_backend_flags()
    assert "inductor_emulate_precision_casts" not in snap
    pipe = _Pipe(with_compile = True)
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True


def test_regional_compile_arms_cache_hook_inners(monkeypatch):
    # the step cache engages before compile, so compile must re-arm its hooks with compiled forwards
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


# workers refuse compile without Triton on Windows, but diffusion runs in the server process


def _clear_runtime_cache():
    from core.inference.diffusion_speed import torch_compile_runtime_available
    torch_compile_runtime_available.cache_clear()


def _set_crt_headers(monkeypatch, reachable: bool):
    from core import _msvc_env
    monkeypatch.setattr(_msvc_env, "crt_headers_reachable", lambda: reachable)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_torchdynamo_disable_is_honored_on_every_platform(monkeypatch, platform):
    from core.inference import diffusion_speed as ds_mod

    # without the stub compile_eligible is always False, so the asserts would pass vacuously
    _stub_torch(monkeypatch)
    # the positive control must clear the Windows toolchain check or negatives hold for the wrong reason
    monkeypatch.setattr(ds_mod.sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
        _set_crt_headers(monkeypatch, True)
    _clear_runtime_cache()
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising = False)
    # positive control: without it the `is False` asserts prove nothing
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
    monkeypatch.setitem(sys.modules, "triton", None)
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
    assert applied["compiled"] is True


class _StreamBlock:
    """A repeated block whose ``forward`` source is what the detector reads."""

    def forward(self, hidden_states, encoder_hidden_states, temb):  # pragma: no cover - never run
        return hidden_states, encoder_hidden_states


class FluxSingleTransformerBlock(_StreamBlock):
    """Named on ``_STREAM_MERGING_BLOCKS``, so it is recognised without reading source."""


class _MergingByArgOrderA(_StreamBlock):
    def forward(self, hidden_states, encoder_hidden_states, temb):  # pragma: no cover
        import torch  # source fixture: only the text is read
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim = 1)
        return hidden_states, encoder_hidden_states


class _MergingByArgOrderB(_StreamBlock):
    def forward(self, hidden_states, encoder_hidden_states, temb):  # pragma: no cover
        import torch
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim = 1)
        return hidden_states, encoder_hidden_states


class _DualStreamBlock(_StreamBlock):
    """Takes BOTH streams but keeps them separate (Qwen-Image / SD3 shape), so it stays dynamic."""

    def forward(self, hidden_states, encoder_hidden_states, temb):  # pragma: no cover
        return hidden_states + 1, encoder_hidden_states + 1


@pytest.fixture
def no_divisibility_proof(monkeypatch):
    """The static fallback only exists for a torch whose inductor cannot prove the stream-merge split."""
    monkeypatch.setattr(ds_mod, "_divisibility_proof_available", lambda: False)


def _dit(*block_classes):
    blocks = [cls() for cls in block_classes]
    dit = types.SimpleNamespace(
        _repeated_blocks = [cls.__name__ for cls in block_classes],
        named_modules = lambda: [("", None)] + [(f"blocks.{i}", b) for i, b in enumerate(blocks)],
    )
    return dit


def test_class_merges_streams_is_crash_confirmed_names_only_by_default():
    assert ds_mod._class_merges_streams(FluxSingleTransformerBlock) is True
    assert ds_mod._class_merges_streams(_MergingByArgOrderA) is False
    assert ds_mod._class_merges_streams(_MergingByArgOrderB) is False
    assert ds_mod._class_merges_streams(_DualStreamBlock) is False


def test_class_merges_streams_broad_sweep_is_opt_in():
    assert ds_mod._class_merges_streams(_MergingByArgOrderA, True) is True
    assert ds_mod._class_merges_streams(_MergingByArgOrderB, True) is True
    assert ds_mod._class_merges_streams(_DualStreamBlock, True) is False


def test_class_merges_streams_without_source_falls_back_to_the_name_list(monkeypatch):
    import inspect

    monkeypatch.setattr(
        inspect, "getsource", lambda _obj: (_ for _ in ()).throw(OSError("no source"))
    )
    ds_mod._class_merges_streams.cache_clear()
    assert ds_mod._class_merges_streams(_MergingByArgOrderA, True) is False
    assert ds_mod._class_merges_streams(FluxSingleTransformerBlock, True) is True
    ds_mod._class_merges_streams.cache_clear()


@pytest.mark.usefixtures("no_divisibility_proof")
def test_dits_merge_streams_honours_the_opt_in_env(monkeypatch):
    monkeypatch.delenv(ds_mod._STREAM_MERGE_DETECT_ENV, raising = False)
    assert ds_mod._dits_merge_streams([_dit(_MergingByArgOrderA)]) is False
    monkeypatch.setenv(ds_mod._STREAM_MERGE_DETECT_ENV, "1")
    assert ds_mod._dits_merge_streams([_dit(_MergingByArgOrderA)]) is True
    monkeypatch.delenv(ds_mod._STREAM_MERGE_DETECT_ENV, raising = False)
    assert ds_mod._dits_merge_streams([_dit(FluxSingleTransformerBlock)]) is True


@pytest.mark.usefixtures("no_divisibility_proof")
def test_dits_merge_streams_scans_every_denoiser():
    assert ds_mod._dits_merge_streams([]) is False
    assert ds_mod._dits_merge_streams([types.SimpleNamespace()]) is False
    assert ds_mod._dits_merge_streams([_dit(_DualStreamBlock)]) is False
    assert ds_mod._dits_merge_streams([_dit(_MergingByArgOrderA)]) is False
    assert ds_mod._dits_merge_streams([_dit(_DualStreamBlock, FluxSingleTransformerBlock)]) is True
    assert (
        ds_mod._dits_merge_streams([_dit(_DualStreamBlock), _dit(FluxSingleTransformerBlock)])
        is True
    )


@pytest.mark.usefixtures("no_divisibility_proof")
def test_speed_default_compiles_stream_merging_dit_with_static_shapes(monkeypatch):
    """FLUX.1 regression: dynamic=True cannot be codegen'd for a stream-merging block."""
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    pipe.transformer._repeated_blocks = ["FluxSingleTransformerBlock"]
    block = FluxSingleTransformerBlock()
    pipe.transformer.named_modules = lambda: [("blocks.0", block)]
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.compile_kwargs == {"fullgraph": True, "dynamic": False}
    assert pipe.compile_kwargs["dynamic"] is not None


@pytest.mark.usefixtures("no_divisibility_proof")
def test_compiled_shapes_are_static_reports_the_stream_merging_downgrade(monkeypatch):
    _stub_torch(monkeypatch)
    merging = types.SimpleNamespace(transformer = _dit(FluxSingleTransformerBlock))
    plain = types.SimpleNamespace(transformer = _dit(_DualStreamBlock))
    assert ds_mod.compiled_shapes_are_static(merging, SPEED_DEFAULT) is True
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_DEFAULT) is False
    assert ds_mod.compiled_shapes_are_static(plain, SPEED_MAX) is True
    assert ds_mod.compiled_shapes_are_static(merging, SPEED_OFF) is False
    assert ds_mod.compiled_shapes_are_static(merging, SPEED_EAGER) is False


def test_the_loader_keys_the_compile_bundle_on_the_vae_decode_decision():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert src.count('"vae_decode": vae_decode_compile_allowed(') == 2
    assert '"vae_decode": vae_decode_compile_allowed(pipe, effective_speed)' in src
    assert '"vae_decode": vae_decode_compile_allowed(state.pipe, SPEED_DEFAULT)' in src
    assert ds_mod.vae_decode_compile_allowed is not None


def test_stream_merging_dit_compiles_dynamic_once_inductor_proves_the_split(monkeypatch):
    """With the divisibility proof (torch 2.14+ or the backport) FLUX.1 compiles dynamic like every other DiT."""
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    monkeypatch.setattr(ds_mod, "_divisibility_proof_available", lambda: True)
    pipe = _Pipe(with_compile = True)
    pipe.transformer._repeated_blocks = ["FluxSingleTransformerBlock"]
    block = FluxSingleTransformerBlock()
    pipe.transformer.named_modules = lambda: [("blocks.0", block)]
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert applied["compiled"] is True
    assert pipe.compile_kwargs == {"fullgraph": True, "dynamic": True}
    assert ds_mod._dits_merge_streams([_dit(FluxSingleTransformerBlock)]) is False
    assert ds_mod.compiled_shapes_are_static(pipe, SPEED_DEFAULT) is False


def test_divisibility_proof_probe_never_raises(monkeypatch):
    from core.inference import diffusion_inductor_backports as bp
    monkeypatch.setattr(bp, "proof_available", lambda: (_ for _ in ()).throw(RuntimeError("probe")))
    assert ds_mod._divisibility_proof_available() is False


def test_speed_max_keeps_automatic_dynamic_for_a_stream_merging_dit(monkeypatch):
    """The static exception is the default tier's alone: max compiles every DiT with automatic dynamic."""
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    pipe = _Pipe(with_compile = True)
    pipe.transformer._repeated_blocks = ["FluxSingleTransformerBlock"]
    block = FluxSingleTransformerBlock()
    pipe.transformer.named_modules = lambda: [("blocks.0", block)]
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert pipe.compile_kwargs["dynamic"] is None
    assert ds_mod.auto_dynamic_active(pipe) is True


# compile_repeated_blocks is lazy: inductor lowering bugs surface inside generate()


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
    assert blocks[0](1) == 2
    assert blocks[1](1) == 2
    assert blocks[0](5) == 6
    # one failure flips the whole DiT: the broken lowering is tried once
    assert calls["compiled"] == 1
    assert all(b._compiled_call_impl is None for b in blocks)
    pipe = types.SimpleNamespace(transformer = dit)
    assert "CantSplit" in ds_mod.compile_fallback_error(pipe)


def test_compiled_block_that_works_is_untouched(monkeypatch):
    _stub_torch_compile_errors(monkeypatch)
    block = _Block(lambda x: x * 10)
    dit = _Dit([block])
    ds_mod.guard_compiled_blocks(dit)
    ds_mod.guard_compiled_blocks(dit)
    assert block(3) == 30
    assert block.eager_calls == 0
    assert ds_mod.compile_fallback_error(types.SimpleNamespace(transformer = dit)) is None


def test_settle_fallback_keeps_compiled_while_another_dit_still_compiles(monkeypatch):
    # status keeps compiled while either expert still runs compiled (LoRA gate and cache read it)
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
    ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("compiled", "cuda_graph", "compile_fallback_eager")

    third = _Dit([second_block])
    ds_mod.guard_compiled_blocks(third)
    pipe.transformer_2 = third
    third.blocks[0](1)
    ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("cuda_graph", "compile_fallback_eager")


def test_settle_fallback_sees_a_modular_workflows_named_partition(monkeypatch):
    # MiniMax-H3 compiles transformer_ref through a view; settlement runs on the bare pipe
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
    # compile OOM may be a plain RuntimeError or wrapped in BackendCompilerFailed
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
    from core.inference.diffusion_batched import is_oom_error

    assert is_oom_error(raised.value)


@pytest.mark.parametrize("kind", ["runtime", "oom"])
def test_non_compile_errors_are_not_swallowed(monkeypatch, kind):
    # only graph-build failures retry eagerly; kernel errors and OOM reach the caller for backoff
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


class _Vae:
    """A VAE whose ``decode`` is a class method, as on a diffusers AutoencoderKL."""

    def __init__(self) -> None:
        self.eager_calls = 0

    def decode(self, z):
        self.eager_calls += 1
        return z + 1


def _stub_lazy_compile(monkeypatch, failure):
    """torch.compile that returns fine and fails only on the first call, like dynamo / inductor lowering."""
    torch = _stub_torch_compile_errors(monkeypatch)
    calls = {"compiled": 0}

    def _compile(fn, **kwargs):
        torch.compile_calls.append(kwargs)

        def compiled(*args, **kw):
            calls["compiled"] += 1
            raise failure()

        return compiled

    torch.compile = _compile
    return calls


def test_vae_decode_compile_failure_at_first_decode_falls_back_to_eager(monkeypatch):
    calls = _stub_lazy_compile(
        monkeypatch, lambda: _BackendCompilerFailed("LoweringException: no lowering for aten.foo")
    )
    vae = _Vae()
    pipe = types.SimpleNamespace(vae = vae)
    assert ds_mod._compile_vae_decode(pipe, None, max_autotune = True) is True
    assert "decode" in vae.__dict__
    assert vae.decode(1) == 2
    assert vae.eager_calls == 1
    assert "decode" not in vae.__dict__
    assert vae.decode(5) == 6
    assert calls["compiled"] == 1
    assert "LoweringException" in vae._unsloth_compile_decode_error


def test_vae_decode_fallback_is_settled_into_status_and_not_recompiled(monkeypatch):
    # dual-DiT runs apply_speed_optims twice: a fallen-back decode must not read as compiled
    calls = _stub_lazy_compile(monkeypatch, lambda: _BackendCompilerFailed("LoweringException"))
    vae = _Vae()
    pipe = types.SimpleNamespace(vae = vae)
    state = types.SimpleNamespace(speed_optims = ("compiled", "compiled_vae_decode"))
    assert ds_mod.settle_compile_fallback(state, pipe) is None
    assert ds_mod._compile_vae_decode(pipe, None) is True
    assert vae.decode(1) == 2
    assert vae._unsloth_compiled_decode is False
    assert "LoweringException" in ds_mod.settle_compile_fallback(state, pipe)
    assert state.speed_optims == ("compiled", "compile_fallback_eager")
    assert ds_mod._compile_vae_decode(pipe, None) is False
    assert "decode" not in vae.__dict__
    assert vae.decode(5) == 6
    assert calls["compiled"] == 1


def test_vae_decode_compile_fallback_restores_an_instance_decode(monkeypatch):
    _stub_torch_compile_errors(monkeypatch)
    original = lambda z: z * 3  # noqa: E731 - an instance attribute, as on the SimpleNamespace fakes
    vae = types.SimpleNamespace(decode = original)

    def broken(z):
        raise _BackendCompilerFailed("CantSplit")

    vae.decode = ds_mod._guard_compiled_decode(vae, broken, original, None)
    assert vae.decode(2) == 6
    assert vae.decode is original


@pytest.mark.parametrize("forced", [False, True])
def test_a_tiled_dit_decode_stays_eager_unless_the_compile_is_forced(monkeypatch, forced):
    # tiled decode unrolls its tile loop: minutes of first-render compile
    if forced:
        monkeypatch.setenv(ds_mod.COMPILE_VAE_ENV, "1")
    else:
        monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    calls = {"compiled": 0, "eager": 0}

    def compiled(z):
        calls["compiled"] += 1
        return z

    def eager(z):
        calls["eager"] += 1
        return z

    vae = types.SimpleNamespace(decode = eager, use_tiling = True)
    vae.decode = ds_mod._guard_compiled_decode(
        vae, compiled, eager, None, eager_when_tiled = ds_mod._vae_eager_when_tiled(None)
    )
    vae.decode(1)
    assert calls == ({"compiled": 1, "eager": 0} if forced else {"compiled": 0, "eager": 1})
    vae.use_tiling = False
    vae.decode(1)
    assert calls["compiled"] == (2 if forced else 1)


@pytest.mark.parametrize("offload_active", [False, True])
def test_vae_decode_compile_fallback_relayouts_an_eager_16bit_decode(monkeypatch, offload_active):
    monkeypatch.delenv(ds_mod.COMPILE_VAE_ENV, raising = False)
    _stub_lazy_compile(monkeypatch, lambda: _BackendCompilerFailed("LoweringException"))
    torch = sys.modules["torch"]
    pipe = _Pipe(with_compile = True)
    pipe.vae.dtype = "torch.bfloat16"
    applied = apply_speed_optims(
        pipe,
        _target(),
        is_gguf = False,
        family = _family(),
        speed_mode = SPEED_MAX,
        offload_active = offload_active,
    )
    assert applied["compiled_vae_decode"] is True
    assert pipe.vae.mem_format == torch.channels_last
    assert pipe.vae.decode(1) == 1
    assert pipe.vae._unsloth_compiled_decode is False
    assert pipe.vae.mem_format == (
        torch.channels_last if offload_active else torch.contiguous_format
    )


@pytest.mark.parametrize("kind", ["runtime", "oom"])
def test_vae_decode_non_compile_errors_are_not_swallowed(monkeypatch, kind):
    def failure():
        if kind == "runtime":
            return RuntimeError("CUDA error: an illegal memory access was encountered")
        try:
            raise _OutOfMemory("CUDA out of memory")
        except _OutOfMemory as oom:
            try:
                raise _BackendCompilerFailed("autotune ran out") from oom
            except _BackendCompilerFailed as outer:
                return outer

    _stub_lazy_compile(monkeypatch, failure)
    vae = _Vae()
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None) is True
    with pytest.raises((RuntimeError, _BackendCompilerFailed)):
        vae.decode(1)
    assert vae.eager_calls == 0
    assert "decode" in vae.__dict__


class _TorchaoWeight:
    pass


_TorchaoWeight.__module__ = "torchao.quantization.quantize_.workflows.int8.int8_tensor"


def test_torchao_dit_compiles_with_automatic_dynamic(monkeypatch):
    # dynamic=True fuses Qwen-Image-2.1's attn cat into torchao's act-quant reduction (CantSplit)
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
    # some torchao builds keep Linear.weight a Parameter whose .data is the subclass
    wrapped = types.SimpleNamespace(data = _TorchaoWeight())
    assert ds_mod._carries_torchao_weights(
        types.SimpleNamespace(parameters = lambda: iter([wrapped]))
    )
    other = types.SimpleNamespace(data = object())
    assert not ds_mod._carries_torchao_weights(
        types.SimpleNamespace(parameters = lambda: iter([other]))
    )


def test_compile_dynamic_is_the_value_the_cache_fingerprint_keys_on():
    # bundles are fingerprinted with compile_dynamic, so old explicit-dynamic bundles are not reused
    quant = types.SimpleNamespace(parameters = lambda: iter([_TorchaoWeight()]))
    dense = types.SimpleNamespace(parameters = lambda: iter([object()]))
    assert ds_mod.compile_dynamic(quant, True) is None
    assert ds_mod.compile_dynamic(quant, False) is False
    assert ds_mod.compile_dynamic(dense, True) is True
    assert ds_mod.compile_dynamic(None, True) is True


def test_max_tier_compiles_torchao_dit_with_automatic_dynamic(monkeypatch):
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


def test_automatic_dynamic_compile_arms_the_prompt_length_allowlist(monkeypatch):
    from core.inference import diffusion_dynamic_text

    armed = []
    monkeypatch.setattr(
        diffusion_dynamic_text,
        "install",
        lambda t, logger = None, *, dynamic = None: armed.append((t, dynamic)) or True,
    )
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True, with_fuse = True)
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert armed == [(pipe.transformer, None)]

    # install() arms only unbacked sources (dense MiniMax-H3's temb)
    armed.clear()
    _stub_torch(monkeypatch)
    pipe = _Pipe(with_compile = True)
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT)
    assert armed == [(pipe.transformer, True)]


@pytest.mark.usefixtures("no_divisibility_proof")
def test_speed_max_compiles_a_quantised_stream_merging_dit_static(monkeypatch):
    """A torchao FLUX block under automatic dynamic hits CantSplit on the first new resolution and drops to eager;
    max compiles it static instead, and reports its artifacts as per-shape."""
    _stub_torch(monkeypatch)
    _stub_gguf_accel(monkeypatch)
    monkeypatch.setattr(ds_mod, "_carries_torchao_weights", lambda module: True)
    pipe = _Pipe(with_compile = True)
    pipe.transformer._repeated_blocks = ["FluxSingleTransformerBlock"]
    block = FluxSingleTransformerBlock()
    pipe.transformer.named_modules = lambda: [("blocks.0", block)]
    apply_speed_optims(pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_MAX)
    assert pipe.compile_kwargs["dynamic"] is False
    assert ds_mod.auto_dynamic_active(pipe) is False
    assert ds_mod.compiled_shapes_are_static(pipe, SPEED_MAX) is True


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
    assert order == ["compile"]

    order.clear()
    QwenImage21Transformer2DModel = type("QwenImage21Transformer2DModel", (), {})
    pipe = _Pipe(with_compile = True)
    pipe.transformer = QwenImage21Transformer2DModel()
    pipe.transformer.compile_repeated_blocks = lambda **kw: order.append("compile")
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert order == ["rope", "compile"] and applied["compiled"] is True

    order.clear()
    monkeypatch.setattr(
        rope, "install", lambda logger = None: (_ for _ in ()).throw(RuntimeError("probe"))
    )
    applied = apply_speed_optims(
        pipe, _target(), is_gguf = False, family = _family(), speed_mode = SPEED_DEFAULT
    )
    assert order == ["compile"] and applied["compiled"] is True


def test_family_compiles_regionally_reads_the_repeated_blocks_declaration(monkeypatch):
    from core.inference.diffusion_speed import family_compiles_regionally

    diffusers = types.ModuleType("diffusers")
    diffusers.Blocked = type("Blocked", (), {"_repeated_blocks": ["Block"]})
    diffusers.Unblocked = type("Unblocked", (), {"_repeated_blocks": []})
    diffusers.Undeclared = type("Undeclared", (), {})
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    def fam(cls, denoiser_attr = "transformer"):
        return types.SimpleNamespace(transformer_class = cls, denoiser_attr = denoiser_attr)

    assert family_compiles_regionally(fam("Blocked")) is True
    assert family_compiles_regionally(fam("Unblocked")) is False
    assert family_compiles_regionally(fam("Undeclared")) is True
    assert family_compiles_regionally(fam("NotInThisDiffusers")) is True
    assert family_compiles_regionally(fam("Unblocked", denoiser_attr = "unet")) is True
    assert family_compiles_regionally(fam(None)) is True
    assert family_compiles_regionally(None) is True


def test_family_compiles_regionally_closes_the_dynamo_import_window_first(monkeypatch):
    import utils.torch_warmup as warmup

    from core.inference.diffusion_speed import family_compiles_regionally

    events: list = []
    monkeypatch.setattr(warmup, "close_dynamo_import_window", lambda _log: events.append("guard"))

    class _Diffusers(types.ModuleType):
        def __getattr__(self, name):
            events.append("probe")
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "diffusers", _Diffusers("diffusers"))
    fam = types.SimpleNamespace(transformer_class = "Lumina2Transformer2DModel")
    assert family_compiles_regionally(fam) is True
    assert events[:2] == ["guard", "probe"]


def test_pinned_denoiser_engages_the_int8_gemm_after_placement(monkeypatch):
    """Pinned after placement: the GEMM installs with offload_active False, only on a compiled DiT."""
    calls = []
    fake = types.ModuleType("core.inference.diffusion_int8_gemm")

    def _install(
        transformer,
        logger = None,
        offload_active = False,
    ):
        calls.append(offload_active)
        return 0 if offload_active else 60

    fake.install = _install
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_int8_gemm", fake)
    dit = types.SimpleNamespace()
    pipe = types.SimpleNamespace(_unsloth_cuda_graph_reason = "offload active")
    monkeypatch.setattr(ds_mod, "_denoiser_dits", lambda p: [dit])

    applied = {"compiled": True, "int8_gemm": False, "cuda_graph": False}
    ds_mod.engage_pinned_denoisers(pipe, applied)
    assert calls == [False] and applied["int8_gemm"] and dit._unsloth_int8_gemm == 60
    assert not applied["cuda_graph"]
    assert pipe._unsloth_cuda_graph_reason == "denoiser pinned resident under offload hooks"

    calls.clear()
    eager = {"compiled": False, "int8_gemm": False}
    ds_mod.engage_pinned_denoisers(pipe, eager)
    assert calls == [] and not eager["int8_gemm"]


@pytest.mark.parametrize(
    "offload_active, denoiser_offloaded, expected",
    [(True, False, False), (True, True, True), (True, None, True), (False, None, False)],
)
def test_int8_gemm_install_follows_the_denoiser_placement(
    monkeypatch, offload_active, denoiser_offloaded, expected
):
    """Only a moving denoiser keeps the stock GEMM; one pinned under the others' offload rotation takes the fused one."""
    from core.inference import diffusion_int8_gemm, diffusion_speed as ds_mod

    seen = []
    monkeypatch.setattr(
        diffusion_int8_gemm,
        "install",
        lambda t, logger = None, offload_active = False: seen.append(offload_active) or 0,
    )

    class _DiT:
        def compile_repeated_blocks(self, **kwargs):
            return None

    monkeypatch.setattr(ds_mod, "_denoiser_dits", lambda pipe: [_DiT()])
    ds_mod._compile_repeated_blocks(
        types.SimpleNamespace(),
        None,
        offload_active = offload_active,
        denoiser_offloaded = denoiser_offloaded,
    )
    assert seen == [expected]
