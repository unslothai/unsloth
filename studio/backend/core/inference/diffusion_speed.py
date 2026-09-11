# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in speed optimisations for the local diffusion backend.

Off by default, so the default render path stays bit-identical (the regression harness checks
this). On opt-in it applies the near-lossless speedups in the diffusers-recommended order
(channels_last + cudnn.benchmark -> compile, with TF32 / fused-QKV under "max"):

  off     - nothing (default; bit-identical reference).
  eager   - everything lossless EXCEPT torch.compile: channels_last VAE + cudnn.benchmark +
            attention backend + the shared eager monkey-patches (fused RMSNorm / AdaLayerNorm
            + per-arch addcmul, see diffusion_eager_patches.py / diffusion_arch_patches.py). The
            fast first-image / casual path, no compile tax.
  default - LIGHT compile. GGUF: compile ONLY the dequant op chain (~70-80% of eager GGUF time)
            for ~1.24-1.64x at a small one-time compile (~7.5-10.4s), zero extra VRAM,
            resolution-invariant; the block stays eager. Dense: no dequant, so falls back to
            regional compile of the repeated block; a U-Net (SDXL, no repeated-block list) gets a
            whole-module STATIC compile (1.61x at LPIPS 0.034, see ``_UNET_WHOLE_COMPILE``).
  max     - FULL compile: regional max-autotune compile of the repeated block (fuses GGUF dequant
            + matmul/norm/elementwise in one graph -- ~3.2x on GGUF Z-Image, PSNR ~36 dB, above
            the Q4 noise floor) plus TF32 matmul and fused QKV.

``default`` is the cheap always-amortising compile; ``max`` pays the larger regional tax for the
bigger warm speedup. The compiled dequant is skipped under ``max`` (the regional compile subsumes
it; a separate compiled dequant would break that graph). ``supports_torch_compile`` + bf16/CUDA
checks gate regional compile.

Kernel-level switches for the NVFP4 flashinfer backend live with their own modules and are listed
here because this is where an operator looks for a speed knob. All are safe to leave unset.

  ``UNSLOTH_NVFP4_FAST_BIAS=auto|0|1``  the Triton bias pass over the FP4 GEMM's output, eager path
                                        only, bit-identical to ``add_`` and 1.6x to 3.7x faster
                                        (``diffusion_nvfp4_bias.py``). ``0`` restores ``add_``.
  ``UNSLOTH_NVFP4_FAST_DISPATCH=auto|0|1``  cache FlashInfer's per-call dispatch state instead of
                                        rebuilding it every GEMM: 61.8 -> 18.1 us of host time per
                                        call, bit-identical (``diffusion_nvfp4_dispatch.py``). Off
                                        unless the installed flashinfer is on the exact allowlist
                                        AND a runtime bit-identity check passes on the device; ``1``
                                        skips the version check only, never the check.
  ``UNSLOTH_NVFP4_ZERO_BUFFER=1``       restores the full M x N memset in place of the 1-element PDL
                                        ordering barrier. Strictly slower, +3.9 to +97 us per GEMM
                                        (``diffusion_nvfp4_ops.py``).
  ``UNSLOTH_NVFP4_BACKEND=auto|torchao|flashinfer``  which NVFP4 kernels to run at all.

The flags this flips (TF32, cudnn.benchmark) are PROCESS-WIDE, so ``snapshot_backend_flags`` /
``restore_backend_flags`` let the caller restore prior values at unload, keeping a later ``off``
load bit-identical. torch imported lazily.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import Any, Optional

from . import diffusion_gguf_compile as gguf_compile

SPEED_OFF = "off"
SPEED_EAGER = "eager"
SPEED_DEFAULT = "default"
SPEED_MAX = "max"
SPEED_MODES = (SPEED_OFF, SPEED_EAGER, SPEED_DEFAULT, SPEED_MAX)


# (attribute, snapshot key). All but the first are what torchao's recommended_inductor_config_setter() flips.
_INDUCTOR_FLAGS = (
    ("emulate_precision_casts", "inductor_emulate_precision_casts"),
    ("coordinate_descent_tuning", "inductor_coordinate_descent_tuning"),
    ("coordinate_descent_check_all_directions", "inductor_coordinate_descent_check_all_directions"),
    ("force_fuse_int_mm_with_mul", "inductor_force_fuse_int_mm_with_mul"),
    ("fx_graph_cache", "inductor_fx_graph_cache"),
)
_INDUCTOR_TRITON_FLAGS = (("unique_kernel_names", "inductor_triton_unique_kernel_names"),)


def snapshot_backend_flags() -> Optional[dict]:
    """Capture the process-wide torch backend flags this layer may mutate, for restore on unload. None
    without torch. Each flag is read defensively: a build missing one still captures the rest."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch -> nothing to snapshot/restore
        return None
    state: dict[str, Any] = {}
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is not None and hasattr(matmul, "allow_tf32"):
        state["matmul_tf32"] = bool(matmul.allow_tf32)
    if matmul is not None and hasattr(matmul, "allow_fp16_accumulation"):
        state["matmul_fp16_accum"] = bool(matmul.allow_fp16_accumulation)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        if hasattr(cudnn, "allow_tf32"):
            state["cudnn_tf32"] = bool(cudnn.allow_tf32)
        if hasattr(cudnn, "benchmark"):
            state["cudnn_benchmark"] = bool(cudnn.benchmark)
    inductor_cfg = _inductor_config()
    if inductor_cfg is not None:
        for attr, key in _INDUCTOR_FLAGS:
            if hasattr(inductor_cfg, attr):
                state[key] = bool(getattr(inductor_cfg, attr))
        triton_cfg = getattr(inductor_cfg, "triton", None)
        if triton_cfg is not None:
            for attr, key in _INDUCTOR_TRITON_FLAGS:
                if hasattr(triton_cfg, attr):
                    state[key] = bool(getattr(triton_cfg, attr))
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if callable(getter):
        try:
            state["matmul_precision"] = str(getter())
        except Exception:  # noqa: BLE001 - unreadable on this build: restore the rest
            pass
    return state


def restore_backend_flags(state: Optional[dict]) -> None:
    """Restore the flags captured by ``snapshot_backend_flags``. No-op on None. Each flag is
    restored independently so one failure can't leak the others."""
    if not state:
        return
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch -> nothing to restore
        return

    def _set(obj: Any, attr: str, key: str) -> None:
        if obj is not None and key in state and hasattr(obj, attr):
            try:
                setattr(obj, attr, state[key])
            except Exception:  # noqa: BLE001 - best-effort per-flag restore
                pass

    # FIRST: on some builds set_float32_matmul_precision also writes matmul.allow_tf32.
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if state.get("matmul_precision") and callable(setter):
        try:
            setter(state["matmul_precision"])
        except Exception:  # noqa: BLE001 - best-effort per-flag restore
            pass
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    _set(matmul, "allow_tf32", "matmul_tf32")
    _set(matmul, "allow_fp16_accumulation", "matmul_fp16_accum")
    cudnn = getattr(torch.backends, "cudnn", None)
    _set(cudnn, "allow_tf32", "cudnn_tf32")
    _set(cudnn, "benchmark", "cudnn_benchmark")
    inductor_cfg = _inductor_config()
    for attr, key in _INDUCTOR_FLAGS:
        _set(inductor_cfg, attr, key)
    triton_cfg = getattr(inductor_cfg, "triton", None) if inductor_cfg is not None else None
    for attr, key in _INDUCTOR_TRITON_FLAGS:
        _set(triton_cfg, attr, key)


def _inductor_config() -> Any:
    """``torch._inductor.config`` or None. Read as attributes off the imported torch (not a
    submodule import) so a stubbed/partial torch reports None instead of a stale sys.modules hit."""
    try:
        import torch
        return getattr(getattr(torch, "_inductor", None), "config", None)
    except Exception:  # noqa: BLE001 - no inductor -> nothing to snapshot/set
        return None


def normalize_speed_mode(value: Optional[str]) -> str:
    """Lower/strip a requested speed mode (dashes ok); None / "" -> off."""
    if value is None:
        return SPEED_OFF
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized:
        return SPEED_OFF
    if normalized not in SPEED_MODES:
        raise ValueError(
            f"Unsupported diffusion speed_mode '{value}'. Use one of: {', '.join(SPEED_MODES)}."
        )
    return normalized


def resolve_speed_mode(
    value: Optional[str],
    *,
    is_gguf: bool,
    dense_default: str = SPEED_OFF,
) -> str:
    """The effective speed mode when the caller leaves it UNSET (``None``).

    GGUF defaults to ``default``: compiles only the hot dequant op chain (~70-80% of eager GGUF
    time) for ~1.24-1.64x at a small compile, zero extra VRAM, perturbation below the quant noise
    floor. Dense resolves to ``dense_default``: the image backend keeps ``off`` (bit-identical
    first generations, deferred engagement), the video backend passes ``default`` (a clip denoise
    amortises the compile within one generation). An explicit value (incl. ``"off"``) is honored."""
    if value is None:
        return SPEED_DEFAULT if is_gguf else dense_default
    return normalize_speed_mode(value)


@lru_cache(maxsize = 1)
def torch_compile_runtime_available() -> bool:
    """Whether THIS process can actually run an inductor compile.

    Inductor needs Triton, and Windows is the one supported platform whose normal install has no
    Triton wheel. The three Unsloth workers (inference / training / export) already gate on this
    import and set ``TORCHDYNAMO_DISABLE=1`` when it fails, but the diffusion and video backends
    run in the SERVER process, which those gates never reach, so ask it once here.

    ``TORCHDYNAMO_DISABLE`` is honored on every platform: a compile under it is a silent no-op that
    would otherwise be recorded as an engaged optimisation. Cached, since neither answer can change
    inside a process and this runs on every load."""
    if os.environ.get("TORCHDYNAMO_DISABLE", "").strip() not in ("", "0"):
        return False
    if sys.platform != "win32":
        return True
    try:
        import triton  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 -- absent or broken Triton means eager, never a failed load
        return False
    try:
        from .._msvc_env import crt_headers_reachable  # noqa: PLC0415
        return crt_headers_reachable()
    except Exception:  # noqa: BLE001 -- this runs during load; never fail it over a probe
        return True


def compile_eligible(target: Any, *, is_gguf: bool, family: Any) -> bool:
    """Whether the denoiser's repeated block should be regionally compiled.

    Only on CUDA (incl. ROCm), for a bf16 transformer, on a compile-friendly family, in a process
    that can run inductor. ``is_gguf`` no longer disqualifies (GGUF compiles fine and ~2.3x
    faster); the param is kept for compat."""
    del is_gguf
    if not torch_compile_runtime_available():
        return False
    if not bool(getattr(target, "supports_default_torch_compile", False)):
        return False
    if not bool(getattr(family, "supports_torch_compile", True)):
        return False
    return _is_bfloat16(getattr(target, "dtype", None))


def _is_bfloat16(dtype: Any) -> bool:
    try:
        import torch
        return dtype is torch.bfloat16
    except Exception:
        return str(dtype).endswith("bfloat16")


def apply_speed_optims(
    pipe: Any,
    target: Any,
    *,
    is_gguf: bool,
    family: Any,
    speed_mode: str = SPEED_OFF,
    cache_active: bool = False,
    offload_active: bool = False,
    cuda_graph_default: bool = True,
    cache_engaged: Optional[bool] = None,
    logger: Any = None,
) -> dict[str, bool]:
    """Apply the opt-in speed optims for ``speed_mode`` to a built pipeline, BEFORE placement /
    offload. Returns which engaged; every step is best-effort (unsupported ones are skipped).

    ``offload_active`` (offload policy != none) installs ``@torch.compiler.disable``d onload hooks,
    so the compile must drop ``fullgraph`` (like an active step cache) or it crashes at step 1.

    ``cuda_graph_default`` is what the CUDA-graph arm assumes for a family that declares nothing:
    True on the image backend, False on video, where ``supports_cuda_graph`` opts in.

    ``cache_active`` also covers a step cache that may still toggle on at generation time. The
    CUDA-graph arm refuses only on ``cache_engaged``: the caller bypasses per chunk if it toggles."""
    applied = {
        "channels_last": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fp16_accum": False,
        "fused_qkv": False,
        "compiled": False,
        "compiled_dequant": False,
        "compiled_vae_decode": False,
        "cuda_graph": False,
    }
    mode = normalize_speed_mode(speed_mode)
    # TF32 (max) and cudnn.benchmark (any non-off CUDA load) are process-global; the caller restores them so a later
    # `off` load never inherits them.
    if mode == SPEED_OFF:
        return applied

    on_cuda = getattr(target, "device", None) == "cuda"
    family_allows_compile = bool(getattr(family, "supports_torch_compile", True))

    # Lossless: a channels-last VAE speeds up its convs with no numeric change.
    applied["channels_last"] = _vae_channels_last(pipe, logger)

    if on_cuda:
        applied["cudnn_benchmark"] = _enable_cudnn_benchmark(logger)

    if on_cuda:
        applied["fp16_accum"] = _enable_fp16_accumulation(
            family, logger, dtype = getattr(target, "dtype", None), speed_mode = mode
        )

    # The compile lever, per tier. default = LIGHT: GGUF compiles ONLY the dequant op chain (cheap, VRAM-free,
    # resolution-invariant); dense falls back to the regional block compile. max = FULL: regional max-autotune compile
    # of the repeated block. eager = no compile.
    if mode == SPEED_DEFAULT:
        # Asked directly: this arm never reaches compile_eligible(), unlike the dense arm below.
        if is_gguf and on_cuda and family_allows_compile and torch_compile_runtime_available():
            applied["compiled_dequant"] = gguf_compile.install_compiled_dequant(logger)
        elif compile_eligible(target, is_gguf = is_gguf, family = family):
            # A U-Net (SDXL) fuses QKV BEFORE its whole-module compile: 36.3 vs 39.3 ms/step (LPIPS 0.033). DiTs were
            # neutral, so they keep the fuse on max only.
            if _denoiser_unet(pipe) is not None:
                applied["fused_qkv"] = _fuse_qkv(pipe, logger)
            applied["compiled"] = _compile_repeated_blocks(
                pipe,
                logger,
                max_autotune = False,
                cache_active = cache_active,
                offload_active = offload_active,
            )
    elif mode == SPEED_MAX and compile_eligible(target, is_gguf = is_gguf, family = family):
        applied["compiled"] = _compile_repeated_blocks(
            pipe,
            logger,
            max_autotune = True,
            cache_active = cache_active,
            offload_active = offload_active,
        )

    # A compiled U-Net family also compiles the VAE decode (4.98 to 4.25 s over 4 images, LPIPS unchanged). DiTs skip
    # it. dynamic=True keeps it resolution-robust.
    if applied["compiled"] and _denoiser_unet(pipe) is not None:
        applied["compiled_vae_decode"] = _compile_vae_decode(pipe, logger)

    if mode == SPEED_MAX:
        if on_cuda:
            applied["tf32"] = _enable_tf32(logger)
        applied["fused_qkv"] = _fuse_qkv(pipe, logger)

    # Deliberately NOT gated on applied["compiled"]: a launch-bound step survives the compile.
    if mode in (SPEED_DEFAULT, SPEED_MAX):
        cuda_graph = None
        ok, reason = False, "cuda graph layer unavailable"
        try:
            from . import diffusion_cuda_graph as cuda_graph  # noqa: PLC0415 - import cycle
            ok, reason = cuda_graph.graph_eligible(
                target,
                family = family,
                pipe = pipe,
                offload_active = offload_active,
                cache_active = cache_active if cache_engaged is None else bool(cache_engaged),
                speed_mode = mode,
                family_default = cuda_graph_default,
                logger = logger,
            )
        except Exception as exc:  # noqa: BLE001 - an unimportable graph layer means eager, never a failed load
            _warn(logger, "cuda graph eligibility", exc)
        # Stashed either way: status reports WHY graphs are off, not just that they are.
        try:
            pipe._unsloth_cuda_graph_reason = reason
        except Exception:  # noqa: BLE001
            pass
        if ok and cuda_graph is not None:
            try:
                applied["cuda_graph"] = bool(cuda_graph.install_cuda_graphs(pipe, logger = logger))
            except Exception as exc:  # noqa: BLE001 - the load proceeds eager
                _warn(logger, "cuda graph capture", exc)

    return applied


def _vae_channels_last(pipe: Any, logger: Any) -> bool:
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "to"):
        return False
    try:
        import torch
        vae.to(memory_format = torch.channels_last)
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "channels_last", exc)
        return False


# U-Net denoisers ship no ``_repeated_blocks``, so the regional compile cannot reach them; these classes get a
# WHOLE-module STATIC compile. SDXL (B200, 30 steps / 1024px): 26.9 vs 45.9 ms/step, 1.61x end-to-end at LPIPS 0.034.
# Static means a recompile per (height, width, batch).
_UNET_WHOLE_COMPILE: frozenset[str] = frozenset({"UNet2DConditionModel"})


def _denoiser_unet(pipe: Any) -> Any:
    """The pipe's U-Net denoiser when its class is on the whole-compile list, else None."""
    unet = getattr(pipe, "unet", None)
    if unet is not None and type(unet).__name__ in _UNET_WHOLE_COMPILE:
        return unet
    return None


def compiled_shapes_are_static(pipe: Any, speed_mode: Optional[str]) -> bool:
    """Whether this load's compiled artifacts are per-(width, height, batch).

    ``max`` compiles regional blocks dynamic=False and U-Net whole-module is always static;
    ``default`` DiT compiles dynamic=True (one artifact across shapes) EXCEPT for a stream-merging
    DiT, which is static there too (see ``_STREAM_MERGING_BLOCKS``). The compile-cache layer keys on
    this to re-save its bundle when a session hits an uncovered shape."""
    mode = normalize_speed_mode(speed_mode)
    if mode == SPEED_MAX:
        return True
    if mode != SPEED_DEFAULT:
        return False
    if _denoiser_unet(pipe) is not None:
        return True
    return _dits_merge_streams(_denoiser_dits(pipe))


def _denoiser_dits(pipe: Any) -> list:
    """Every DiT the denoise loop runs: the primary ``transformer`` plus a second expert some
    families carry (Ideogram's ``unconditional_transformer``, an MoE ``transformer_2``). Speed /
    attention optims must reach ALL of them (mirroring the offload path), else the second DiT runs
    eager / native while status over-reports the optim as engaged."""
    dits: list = []
    for attr in ("transformer", "transformer_2", "unconditional_transformer"):
        m = getattr(pipe, attr, None)
        if m is not None and m not in dits:
            dits.append(m)
    return dits


# Repeated blocks measured to make inductor raise CantSplit under dynamic = True: a block that
# concatenates the text and image streams gets two separate dynamic symbols, and inductor's Mod
# never cancels an Add over an Add, so the split it needs is unprovable. ``dynamic = False`` is the
# only escape (mark_static is overridden, and ``dynamic = None`` crashes on the second shape).
# MEASURED, not inferred: the merge is necessary but NOT sufficient, so nothing joins this set on
# code reading alone.
_STREAM_MERGING_BLOCKS: frozenset[str] = frozenset({"FluxSingleTransformerBlock"})

# The same cat matched on source. OFF by default because it over-flags: the escape hatch for a new
# family that crashes before its class can be named above.
_STREAM_MERGE_DETECT_ENV = "UNSLOTH_STATIC_STREAM_MERGE_DETECT"
_STREAM_MERGE_SOURCE = re.compile(
    r"torch\.cat\(\s*\[\s*(?:encoder_hidden_states\s*,\s*hidden_states"
    r"|hidden_states\s*,\s*encoder_hidden_states)\s*\]"
)


@lru_cache(maxsize = None)
def _class_merges_streams(cls: type, broad: bool = False) -> bool:
    """Whether one repeated-block CLASS is known to need a static compile. ``broad`` is an argument
    rather than an env read inside the body, so the memo cannot outlive it."""
    if cls.__name__ in _STREAM_MERGING_BLOCKS:
        return True
    if not broad:
        return False
    try:
        import inspect  # noqa: PLC0415 - only reached under the opt-in broad sweep
        source = inspect.getsource(cls.forward)
    except Exception:  # noqa: BLE001 - no source (frozen / C ext) means fall back to the name list
        return False
    return bool(_STREAM_MERGE_SOURCE.search(source))


def _dits_merge_streams(dits: list) -> bool:
    """Whether ANY denoiser DiT's repeated blocks merge the streams, so that load's regional
    compile must be static. One check per distinct block class, not per instance."""
    broad = os.environ.get(_STREAM_MERGE_DETECT_ENV) == "1"
    seen: set[type] = set()
    for transformer in dits:
        names = set(getattr(transformer, "_repeated_blocks", ()) or ())
        if not names:
            continue
        try:
            modules = list(transformer.named_modules())
        except Exception:  # noqa: BLE001 - a probe, never a failed load
            continue
        for _name, sub in modules:
            cls = type(sub)
            if cls.__name__ not in names or cls in seen:
                continue
            seen.add(cls)
            if _class_merges_streams(cls, broad):
                return True
    return False


def _compile_repeated_blocks(
    pipe: Any,
    logger: Any,
    *,
    max_autotune: bool = False,
    cache_active: bool = False,
    offload_active: bool = False,
) -> bool:
    dits = [
        t for t in _denoiser_dits(pipe) if callable(getattr(t, "compile_repeated_blocks", None))
    ]
    unet = _denoiser_unet(pipe) if not dits else None
    if not dits and unet is None:
        return False
    # default: dynamic=True, fast cold start, no recompile on resolution change. max: max-autotune-no-cudagraphs +
    # dynamic=False, a few % more for a longer compile and a recompile per resolution. Inductor's own cudagraph modes
    # fail on the regional block -- "accessing tensor output of CUDAGraphs that has been overwritten" -- so the tier
    # stays on -no-cudagraphs and the capture is taken one level up, at the denoiser module.
    # The one exception to "default is dynamic": see _STREAM_MERGING_BLOCKS.
    static_shapes = max_autotune or _dits_merge_streams(dits)
    if static_shapes and not max_autotune and logger is not None:
        logger.info(
            "diffusion.speed: regional compile is static; this DiT merges the text and image streams "
            "inside its repeated block and cannot be codegen'd with dynamic sequence lengths",
        )
    kwargs: dict[str, Any] = {
        "fullgraph": not (cache_active or offload_active),
        "dynamic": not static_shapes,
    }
    if max_autotune:
        kwargs["mode"] = "max-autotune-no-cudagraphs"
    try:
        import torch

        # Heterogeneous-block DiTs (Z-Image needs ~11 graphs) exceed dynamo's default recompile_limit of 8, where a
        # resident load hard-errors under fullgraph, so raise it to 64. NOT force_parameter_static_shapes=False: no win
        # and ~6x slower.
        dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_cfg is not None:
            for _limit_attr in ("recompile_limit", "cache_size_limit"):  # name varies by torch ver
                if hasattr(dynamo_cfg, _limit_attr):
                    setattr(dynamo_cfg, _limit_attr, max(getattr(dynamo_cfg, _limit_attr) or 0, 64))
        # Match eager intermediate rounding in inductor's fused pointwise kernels: they keep chains in fp32 where eager
        # materialises bf16 between ops, a per-forward delta a multi-step denoise amplifies. Measured LPIPS vs eager:
        # Qwen-Image 0.019 to 0.006, HunyuanVideo-1.5-720p 0.221 to 0.052, at ~zero cost. Process-global, so
        # snapshot_backend_flags restores it on unload.
        inductor_cfg = _inductor_config()
        if inductor_cfg is not None and hasattr(inductor_cfg, "emulate_precision_casts"):
            inductor_cfg.emulate_precision_casts = True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "compile_repeated_blocks", exc)
        return False
    if unet is not None:
        # Whole-module static compile for the U-Net classes above. fullgraph mirrors the regional decision; dynamic is
        # ALWAYS False, so each new (height, width, batch) pays its own compile. ``Module.compile`` keeps the module
        # identity.
        unet_kwargs: dict[str, Any] = {"fullgraph": kwargs["fullgraph"], "dynamic": False}
        if max_autotune:
            unet_kwargs["mode"] = "max-autotune-no-cudagraphs"
        try:
            unet.compile(**unet_kwargs)
            return True
        except Exception as exc:  # noqa: BLE001 - optimisation only
            _warn(logger, "unet whole-module compile", exc)
            return False
    # Compile every denoiser DiT (dual-DiT families run both); a per-DiT failure degrades only that one to eager.
    engaged = False
    for transformer in dits:
        try:
            transformer.compile_repeated_blocks(**kwargs)
            engaged = True
        except Exception as exc:  # noqa: BLE001 - optimisation only
            _warn(logger, "compile_repeated_blocks", exc)
            continue
        # A step cache engaged BEFORE this compile already wrapped each block forward in a disabled hook, so the compute
        # branch would run eager and forfeit the regional compile. Re-point the hooks' inner forward at compiled
        # wrappers (no-op without them).
        try:
            from .diffusion_cache import _compile_hooked_block_inners
            _compile_hooked_block_inners(transformer, logger)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            _warn(logger, "cache-hook inner compile", exc)
    return engaged


def _compile_vae_decode(pipe: Any, logger: Any) -> bool:
    """torch.compile the VAE ``decode`` bound method in place (U-Net families; caller gates).
    Instance-level assignment: the pipe owns it and the module object is untouched."""
    vae = getattr(pipe, "vae", None)
    decode = getattr(vae, "decode", None) if vae is not None else None
    if not callable(decode):
        return False
    try:
        import torch
        vae.decode = torch.compile(decode, fullgraph = False, dynamic = True)
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "vae decode compile", exc)
        return False


def _enable_cudnn_benchmark(logger: Any) -> bool:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "cudnn_benchmark", exc)
        return False


def _enable_tf32(logger: Any) -> bool:
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "tf32", exc)
        return False


# Families the overflow harness found to produce non-finite activations under fp16 accumulation. Empty by measurement:
# no overflow across all six families.
_FP16_ACCUM_DENY: frozenset[str] = frozenset()


def _enable_fp16_accumulation(
    family: Any,
    logger: Any,
    *,
    dtype: Any = None,
    speed_mode: Optional[str] = None,
) -> bool:
    """Turn on fp16-accumulated fp16 GEMMs for consumer GPUs (~2x the fp32-accumulate rate;
    datacenter parts keep the safer default). Gated on: torch 2.10+ exposing the flag, a consumer
    device, the family not in _FP16_ACCUM_DENY, UNSLOTH_DISABLE_FP16_ACCUM unset, and -- when
    compute dtype IS fp16 (the only case results change) -- the ``max`` tier (bf16 loads are
    bit-identical, so any tier). The caller's snapshot/restore returns the flag on unload."""
    import os

    if os.environ.get("UNSLOTH_DISABLE_FP16_ACCUM", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    name = str(getattr(family, "name", family or "")).lower()
    if name in _FP16_ACCUM_DENY:
        return False
    if str(dtype).replace("torch.", "") == "float16" and speed_mode != SPEED_MAX:
        return False
    try:
        import torch

        matmul = torch.backends.cuda.matmul
        if not hasattr(matmul, "allow_fp16_accumulation"):
            return False
        from .diffusion_transformer_quant import _is_consumer_gpu

        if not _is_consumer_gpu():
            return False
        matmul.allow_fp16_accumulation = True
        return True
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "fp16_accum", exc)
        return False


def _fuse_qkv(pipe: Any, logger: Any) -> bool:
    # Prefer the pipe-level fuse (covers every component); else fuse each denoiser DiT so a dual-DiT family fuses BOTH
    # experts.
    fn = getattr(pipe, "fuse_qkv_projections", None)
    if callable(fn):
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 - optimisation only
            _warn(logger, "fuse_qkv_projections", exc)
            return False
    engaged = False
    for transformer in _denoiser_dits(pipe):
        tfn = getattr(transformer, "fuse_qkv_projections", None)
        if callable(tfn):
            try:
                tfn()
                engaged = True
            except Exception as exc:  # noqa: BLE001 - optimisation only
                _warn(logger, "fuse_qkv_projections", exc)
    return engaged


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.speed: %s failed: %s", what, exc)
