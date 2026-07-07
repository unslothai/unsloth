# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in speed optimisations for the local diffusion backend.

Off by default, so the default render path stays bit-identical to a plain run (the
property the regression harness checks). When the operator opts in, this applies the
near-lossless speedups in the order the diffusers guides recommend
(channels_last + cudnn.benchmark -> compile, with TF32 / fused-QKV under "max"):

  off     - nothing (default; bit-identical reference).
  eager   - everything lossless EXCEPT torch.compile: channels_last VAE +
            cudnn.benchmark + the attention backend + the shared eager monkey-patches
            (fused RMSNorm / AdaLayerNorm + per-arch addcmul fusions, see
            diffusion_eager_patches.py / diffusion_arch_patches.py). The fast first-image
            / casual-use path -- no compile tax to amortise.
  default - LIGHT compile. For a GGUF model: channels_last + cudnn.benchmark +
            torch.compile of ONLY the dequant op chain
            (``torch.compile(dequantize_gguf_tensor, dynamic=True)``) -- the dequant is
            ~70-80% of eager GGUF time, so fusing it gives ~1.24-1.64x for a small
            one-time compile (~7.5-10.4s) and ZERO extra VRAM, resolution-invariant
            (the dequant inputs are fixed-shape weights). For a dense (non-GGUF) model
            there is no dequant, so ``default`` falls back to regional torch.compile of
            the denoiser's repeated block (the only compile lever a dense model has).
  max     - the FULL torch.compile: regional max-autotune compile of the denoiser's
            repeated block (which fuses the GGUF dequant AND the matmul/norm/elementwise
            in one graph -- ~3.2x on the GGUF Z-Image transformer, PSNR ~36 dB vs eager,
            well above the Q4 noise floor) plus TF32 matmul and fused QKV projections.

Tier rationale: ``default`` is the cheap, always-amortising compile (compile just the
hot GGUF dequant; the block stays eager) so the first image is fast and VRAM is
untouched; ``max`` pays the larger regional-compile tax for the bigger warm speedup.
The compiled dequant is deliberately skipped under ``max`` -- the regional block compile
subsumes the dequant fusion (a separately-compiled dequant would be traced into that
graph and break it), so ``max`` runs the stock dequant and lets the block compile it. The
per-family ``supports_torch_compile`` flag and the bf16/CUDA checks gate regional compile.

The backend flags this layer flips (TF32, cudnn.benchmark) are PROCESS-WIDE, so
``snapshot_backend_flags`` / ``restore_backend_flags`` let the caller capture the
prior values at load and restore them at unload, keeping a later ``off`` load
bit-identical instead of inheriting a previous ``max`` run's globals. torch is
imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

from . import diffusion_gguf_compile as gguf_compile

SPEED_OFF = "off"
SPEED_EAGER = "eager"
SPEED_DEFAULT = "default"
SPEED_MAX = "max"
SPEED_MODES = (SPEED_OFF, SPEED_EAGER, SPEED_DEFAULT, SPEED_MAX)


def snapshot_backend_flags() -> Optional[dict]:
    """Capture the process-wide torch backend flags this layer may mutate, so the
    caller can restore them on unload. None if torch is unavailable. Each flag is read
    defensively so a build/platform missing one (e.g. no cuda.matmul on CPU/MPS) still
    captures the rest -- otherwise a single missing attribute would skip the whole
    snapshot and a real mutated flag would leak."""
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> nothing to snapshot/restore
        return None
    state: dict[str, bool] = {}
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
    return state


def restore_backend_flags(state: Optional[dict]) -> None:
    """Restore the flags captured by ``snapshot_backend_flags``. No-op on None. Each
    flag is restored independently so one failure can't leave the others leaked."""
    if not state:
        return
    try:
        import torch
    except Exception:  # noqa: BLE001 — no torch -> nothing to restore
        return

    def _set(obj: Any, attr: str, key: str) -> None:
        if obj is not None and key in state and hasattr(obj, attr):
            try:
                setattr(obj, attr, state[key])
            except Exception:  # noqa: BLE001 — best-effort per-flag restore
                pass

    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    _set(matmul, "allow_tf32", "matmul_tf32")
    _set(matmul, "allow_fp16_accumulation", "matmul_fp16_accum")
    cudnn = getattr(torch.backends, "cudnn", None)
    _set(cudnn, "allow_tf32", "cudnn_tf32")
    _set(cudnn, "benchmark", "cudnn_benchmark")


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


def resolve_speed_mode(value: Optional[str], *, is_gguf: bool) -> str:
    """The effective speed mode when the caller leaves it UNSET (``None``).

    A GGUF model defaults to ``default``: it compiles only the hot dequant op chain
    (~70-80% of eager GGUF time) for ~1.24-1.64x at a small one-time compile and zero
    extra VRAM -- a cheap, always-amortising win whose numeric perturbation sits well
    below the quantisation noise floor (the dequant graph is unchanged, just
    Inductor-fused). A dense (non-GGUF) model stays ``off`` / bit-identical, since there
    compile would be the only source of drift. An explicit value -- including ``"off"``
    -- is always honored verbatim."""
    if value is None:
        return SPEED_DEFAULT if is_gguf else SPEED_OFF
    return normalize_speed_mode(value)


def compile_eligible(target: Any, *, is_gguf: bool, family: Any) -> bool:
    """Whether the denoiser's repeated block should be regionally compiled.

    Only on CUDA (incl. ROCm via supports_default_torch_compile), for a bf16
    transformer, on a compile-friendly family. ``is_gguf`` no longer disqualifies:
    ``compile_repeated_blocks`` runs fine on the GGUF transformer (the per-op
    dequant stays eager, the rest of the block compiles) and is ~2.3x faster, so it
    is kept only for signature/logging compatibility."""
    del is_gguf  # GGUF is compile-eligible now; param kept for call-site compat.
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
    logger: Any = None,
) -> dict[str, bool]:
    """Apply the opt-in speed optimisations for ``speed_mode`` to a built pipeline,
    BEFORE placement / offload. Returns which optimisations actually engaged. Every
    step is best-effort: a pipeline that doesn't support one is simply skipped.

    ``offload_active`` is the planned offload policy != none: group/model/sequential
    offloading installs ``@torch.compiler.disable``d onload hooks, so the compile must
    drop ``fullgraph`` (same reason as an active step cache) or it crashes at the first
    denoise step."""
    applied = {
        "channels_last": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fp16_accum": False,
        "fused_qkv": False,
        "compiled": False,
        "compiled_dequant": False,
    }
    mode = normalize_speed_mode(speed_mode)
    # TF32 and cudnn.benchmark are the process-global flags this may flip (TF32 on max,
    # cudnn.benchmark on any non-off CUDA load). The caller snapshots them before this
    # call and restores on unload / failed load via snapshot_backend_flags /
    # restore_backend_flags, so a later `off` load -- or chat inference in the same
    # process -- never inherits them. We keep no separate bookkeeping here.
    if mode == SPEED_OFF:
        return applied

    on_cuda = getattr(target, "device", None) == "cuda"
    family_allows_compile = bool(getattr(family, "supports_torch_compile", True))

    # Lossless: a channels-last VAE speeds up its convolutions with no numeric change.
    applied["channels_last"] = _vae_channels_last(pipe, logger)

    # Near-lossless: let cuDNN autotune the fixed-shape VAE convs (CUDA only). It may
    # pick a different conv algorithm, so it is a "default"-tier (not bit-identical) win.
    if on_cuda:
        applied["cudnn_benchmark"] = _enable_cudnn_benchmark(logger)

    # Consumer-only: fp16 GEMMs accumulate in fp16 (~2x on GeForce-class parts, whose
    # fp32-accumulate rate is halved; datacenter HBM parts gain nothing and keep the
    # safer fp32 accumulate). Only affects fp16 matmuls -- bf16 loads were measured
    # bit-identical with the flag on across every family (36/36 same-seed A/B cases),
    # so on the quality-neutral tiers the flag engages only when the compute dtype is
    # NOT fp16. On an fp16 pipeline (the pre-Ampere fallback dtype) the same harness
    # measured real same-seed drift (mean 2-5% on SDXL / FLUX), so fp16 compute gets
    # the 2x accumulate only under ``max``, the tier that already trades exactness for
    # speed. Guarded by a per-family deny-list fed by the overflow validation harness
    # and the UNSLOTH_DISABLE_FP16_ACCUM kill switch.
    if on_cuda:
        applied["fp16_accum"] = _enable_fp16_accumulation(
            family, logger, dtype = getattr(target, "dtype", None), speed_mode = mode
        )

    # --- the compile lever, remapped per tier ----------------------------------------
    # default = LIGHT compile: for a GGUF model, compile ONLY the dequant op chain
    #   (~70-80% of eager GGUF time) -- cheap, VRAM-free, resolution-invariant; the
    #   transformer block stays eager. A dense model has no dequant, so default falls
    #   back to the regional block compile (its only compile lever).
    # max = FULL compile: regional max-autotune compile of the repeated denoiser block
    #   (fuses dequant + matmul + norm + elementwise in one graph). It subsumes the
    #   dequant fusion, so we do NOT also install the standalone compiled dequant here.
    # eager = no compile at all.
    if mode == SPEED_DEFAULT:
        if is_gguf and on_cuda and family_allows_compile:
            applied["compiled_dequant"] = gguf_compile.install_compiled_dequant(logger)
        elif compile_eligible(target, is_gguf = is_gguf, family = family):
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

    if mode == SPEED_MAX:
        # Near-lossless: TF32 matmul (CUDA only) trades a few mantissa bits for speed.
        if on_cuda:
            applied["tf32"] = _enable_tf32(logger)
        applied["fused_qkv"] = _fuse_qkv(pipe, logger)

    return applied


def _vae_channels_last(pipe: Any, logger: Any) -> bool:
    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "to"):
        return False
    try:
        import torch
        vae.to(memory_format = torch.channels_last)
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "channels_last", exc)
        return False


def _compile_repeated_blocks(
    pipe: Any,
    logger: Any,
    *,
    max_autotune: bool = False,
    cache_active: bool = False,
    offload_active: bool = False,
) -> bool:
    transformer = getattr(pipe, "transformer", None)
    fn = getattr(transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        return False
    # default: mode="default" + dynamic=True -- fast cold start, robust to resolution
    # changes (no recompile). max: mode="max-autotune-no-cudagraphs" + dynamic=False --
    # Triton autotuning for a few % more on GEMM/conv-heavy models, at a much longer
    # compile and a recompile per new resolution. The CUDA-graph modes (reduce-overhead
    # / max-autotune) are deliberately NOT used: they crash on the regionally-compiled
    # block because its static output buffer is overwritten across denoise steps.
    #
    # fullgraph drops to False when a step cache OR CPU offloading is engaged: both insert
    # an ``@torch.compiler.disable``d function into the forward -- FBCache's per-step
    # decision, and group/model/sequential offload's ``ModuleGroup.onload_`` streaming hook
    # -- i.e. a graph break, which fullgraph=True rejects ("Skip inlining
    # torch.compiler.disable()d function"). The break is cheap and the rest of the block
    # still compiles.
    kwargs: dict[str, Any] = {
        "fullgraph": not (cache_active or offload_active),
        "dynamic": not max_autotune,
    }
    if max_autotune:
        kwargs["mode"] = "max-autotune-no-cudagraphs"
    try:
        import torch

        # Heterogeneous-block DiTs (e.g. Z-Image) compile ~one graph per distinct block
        # shape through compile_repeated_blocks; Z-Image needs ~11, above dynamo's default
        # recompile_limit of 8. Once the limit is hit a resident load hard-errors under
        # fullgraph (and an offload/cache load silently drops the overflow blocks to eager),
        # so raise it well past that (64) for headroom on larger heterogeneous DiTs. This is
        # diffusers' own documented fix for regional-compile recompilation (their guide bumps
        # cache_size_limit). Deliberately NOT force_parameter_static_shapes=False: it doesn't
        # cut the variant count here and makes each compile ~6x slower (24s -> 143s cold).
        dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_cfg is not None:
            for _limit_attr in ("recompile_limit", "cache_size_limit"):  # name varies by torch ver
                if hasattr(dynamo_cfg, _limit_attr):
                    setattr(dynamo_cfg, _limit_attr, max(getattr(dynamo_cfg, _limit_attr) or 0, 64))
        fn(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "compile_repeated_blocks", exc)
        return False


def _enable_cudnn_benchmark(logger: Any) -> bool:
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "cudnn_benchmark", exc)
        return False


def _enable_tf32(logger: Any) -> bool:
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "tf32", exc)
        return False


# Families the overflow validation harness (scripts/fp16_accum_validate.py) found to
# produce non-finite activations or NEW black frames under fp16 accumulation. Empty by
# measurement: across all six families the harness found no overflow anywhere -- bf16
# loads are bit-identical with the flag on, and fp16 loads stay finite (their same-seed
# drift is why fp16 compute is additionally gated to the ``max`` tier below).
_FP16_ACCUM_DENY: frozenset[str] = frozenset()


def _enable_fp16_accumulation(
    family: Any,
    logger: Any,
    *,
    dtype: Any = None,
    speed_mode: Optional[str] = None,
) -> bool:
    """Turn on fp16-accumulated fp16 GEMMs for consumer GPUs, where they run ~2x the
    fp32-accumulate rate (datacenter HBM parts are not throughput-nerfed, so they keep
    the safer default). Gated on: the torch build exposing the flag (2.10+), a
    consumer-class device, the family not being deny-listed by the overflow harness,
    the UNSLOTH_DISABLE_FP16_ACCUM kill switch being unset, and -- when the pipeline
    compute dtype IS fp16, the only case where the accumulator width changes results --
    the ``max`` tier (measured same-seed drift: mean 2-5%; bf16 loads are bit-identical
    so they engage on any tier). The caller's snapshot/restore pair returns the
    process-wide flag to its prior value on unload."""
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
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "fp16_accum", exc)
        return False


def _fuse_qkv(pipe: Any, logger: Any) -> bool:
    for owner in (pipe, getattr(pipe, "transformer", None)):
        fn = getattr(owner, "fuse_qkv_projections", None)
        if callable(fn):
            try:
                fn()
                return True
            except Exception as exc:  # noqa: BLE001 — optimisation only
                _warn(logger, "fuse_qkv_projections", exc)
                return False
    return False


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.speed: %s failed: %s", what, exc)
