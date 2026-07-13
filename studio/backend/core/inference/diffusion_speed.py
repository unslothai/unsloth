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

The flags this flips (TF32, cudnn.benchmark) are PROCESS-WIDE, so ``snapshot_backend_flags`` /
``restore_backend_flags`` let the caller restore prior values at unload, keeping a later ``off``
load bit-identical. torch imported lazily.
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
    """Capture the process-wide torch backend flags this layer may mutate, for restore on unload.
    None if torch is unavailable. Each flag is read defensively so a build missing one (e.g. no
    cuda.matmul on CPU/MPS) still captures the rest, instead of leaking a real mutated flag."""
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
    inductor_cfg = _inductor_config()
    if inductor_cfg is not None and hasattr(inductor_cfg, "emulate_precision_casts"):
        state["inductor_emulate_precision_casts"] = bool(inductor_cfg.emulate_precision_casts)
    return state


def restore_backend_flags(state: Optional[dict]) -> None:
    """Restore the flags captured by ``snapshot_backend_flags``. No-op on None. Each flag is
    restored independently so one failure can't leak the others."""
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
    _set(_inductor_config(), "emulate_precision_casts", "inductor_emulate_precision_casts")


def _inductor_config() -> Any:
    """``torch._inductor.config`` or None. Read as attributes off the imported torch (not a
    submodule import) so a stubbed/partial torch reports None instead of a stale sys.modules hit."""
    try:
        import torch
        return getattr(getattr(torch, "_inductor", None), "config", None)
    except Exception:  # noqa: BLE001 — no inductor -> nothing to snapshot/set
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


def compile_eligible(target: Any, *, is_gguf: bool, family: Any) -> bool:
    """Whether the denoiser's repeated block should be regionally compiled.

    Only on CUDA (incl. ROCm), for a bf16 transformer, on a compile-friendly family. ``is_gguf``
    no longer disqualifies (GGUF compiles fine and ~2.3x faster); the param is kept for compat."""
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
    """Apply the opt-in speed optims for ``speed_mode`` to a built pipeline, BEFORE placement /
    offload. Returns which engaged; every step is best-effort (unsupported ones are skipped).

    ``offload_active`` (offload policy != none) installs ``@torch.compiler.disable``d onload hooks,
    so the compile must drop ``fullgraph`` (like an active step cache) or it crashes at step 1."""
    applied = {
        "channels_last": False,
        "cudnn_benchmark": False,
        "tf32": False,
        "fp16_accum": False,
        "fused_qkv": False,
        "compiled": False,
        "compiled_dequant": False,
        "compiled_vae_decode": False,
    }
    mode = normalize_speed_mode(speed_mode)
    # TF32 (max) and cudnn.benchmark (any non-off CUDA load) are process-global; the caller
    # snapshots/restores them via snapshot_backend_flags so a later `off` load never inherits them.
    if mode == SPEED_OFF:
        return applied

    on_cuda = getattr(target, "device", None) == "cuda"
    family_allows_compile = bool(getattr(family, "supports_torch_compile", True))

    # Lossless: a channels-last VAE speeds up its convs with no numeric change.
    applied["channels_last"] = _vae_channels_last(pipe, logger)

    # Near-lossless: cuDNN autotunes the fixed-shape VAE convs (CUDA only). May pick a different
    # conv algorithm, so it is a "default"-tier (not bit-identical) win.
    if on_cuda:
        applied["cudnn_benchmark"] = _enable_cudnn_benchmark(logger)

    # Consumer-only: fp16 GEMMs accumulate in fp16 (~2x on GeForce-class parts; datacenter HBM
    # parts gain nothing and keep fp32 accumulate). bf16 loads measured bit-identical with the flag
    # on (36/36 same-seed cases), so on the neutral tiers it engages only when compute dtype is NOT
    # fp16. fp16 pipelines showed same-seed drift (mean 2-5% on SDXL/FLUX), so fp16 compute gets it
    # only under ``max``. Guarded by _FP16_ACCUM_DENY and the UNSLOTH_DISABLE_FP16_ACCUM kill switch.
    if on_cuda:
        applied["fp16_accum"] = _enable_fp16_accumulation(
            family, logger, dtype = getattr(target, "dtype", None), speed_mode = mode
        )

    # --- the compile lever, per tier ---
    # default = LIGHT: GGUF compiles ONLY the dequant op chain (cheap, VRAM-free,
    #   resolution-invariant; block stays eager); dense has no dequant, so falls back to the
    #   regional block compile.
    # max = FULL: regional max-autotune compile of the repeated block (subsumes the dequant fusion,
    #   so no standalone compiled dequant here).
    # eager = no compile.
    if mode == SPEED_DEFAULT:
        if is_gguf and on_cuda and family_allows_compile:
            applied["compiled_dequant"] = gguf_compile.install_compiled_dequant(logger)
        elif compile_eligible(target, is_gguf = is_gguf, family = family):
            # A U-Net (SDXL) fuses QKV BEFORE its whole-module compile: 36.3 vs 39.3 ms/step
            # (LPIPS 0.033). DiTs were neutral under the regional compile (Qwen-Image 6.53 vs
            # 6.52 s), so they keep the fuse on the max tier only.
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

    # A compiled U-Net family also compiles the VAE decode (a real share at SDXL's step rate:
    # 4.98 -> 4.25 s over 4 images, LPIPS unchanged). DiT families skip it (a few % of their
    # generation). dynamic=True keeps it resolution-robust; fullgraph=False tolerates offload hooks.
    if applied["compiled"] and _denoiser_unet(pipe) is not None:
        applied["compiled_vae_decode"] = _compile_vae_decode(pipe, logger)

    if mode == SPEED_MAX:
        # Near-lossless: TF32 matmul (CUDA only) trades mantissa bits for speed.
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


# U-Net denoisers ship no ``_repeated_blocks`` (heterogeneous block mix), so the regional compile
# can't reach them; these classes get a WHOLE-module STATIC ``torch.compile`` instead. On SDXL
# (B200, 30 steps / 1024px): static whole-UNet runs 26.9 ms/step vs the 45.9 ms/step bit-exact
# reference -- 1.61x end-to-end (6.16 -> 3.83 s) at LPIPS 0.034 -- while dynamic=True compiles 5x
# slower for less win, and a regional block compile only reaches 45.0 ms/step. Static shapes mean
# a recompile per new (height, width, batch); the Mega-cache bundle carries each across restarts.
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
    ``default`` DiT compiles dynamic=True (one artifact across shapes). The compile-cache layer
    keys on this to re-save its bundle when a session hits an uncovered shape."""
    mode = normalize_speed_mode(speed_mode)
    if mode == SPEED_MAX:
        return True
    return mode == SPEED_DEFAULT and _denoiser_unet(pipe) is not None


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
    # default: dynamic=True -- fast cold start, no recompile on resolution change. max:
    # mode="max-autotune-no-cudagraphs" + dynamic=False -- Triton autotuning for a few % more, at a
    # longer compile and a recompile per resolution. CUDA-graph modes are NOT used: they crash on
    # the regional block (its static output buffer is overwritten across steps).
    #
    # fullgraph drops to False under a step cache OR offloading: both insert an
    # ``@torch.compiler.disable``d function (FBCache's per-step decision, offload's onload hook),
    # i.e. a graph break fullgraph=True rejects. The break is cheap; the rest still compiles.
    kwargs: dict[str, Any] = {
        "fullgraph": not (cache_active or offload_active),
        "dynamic": not max_autotune,
    }
    if max_autotune:
        kwargs["mode"] = "max-autotune-no-cudagraphs"
    try:
        import torch

        # Heterogeneous-block DiTs (e.g. Z-Image) compile ~one graph per distinct block shape;
        # Z-Image needs ~11, above dynamo's default recompile_limit of 8. Past the limit a resident
        # load hard-errors under fullgraph (offload/cache silently drops blocks to eager), so raise
        # it to 64 (diffusers' documented regional-compile fix). NOT
        # force_parameter_static_shapes=False: no variant-count win here and ~6x slower (24 -> 143s).
        dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
        if dynamo_cfg is not None:
            for _limit_attr in ("recompile_limit", "cache_size_limit"):  # name varies by torch ver
                if hasattr(dynamo_cfg, _limit_attr):
                    setattr(dynamo_cfg, _limit_attr, max(getattr(dynamo_cfg, _limit_attr) or 0, 64))
        # Match eager's intermediate rounding in inductor's fused pointwise kernels: they keep
        # chains in fp32 where eager materialises bf16 between ops, a per-forward delta a multi-step
        # denoise amplifies. Measured LPIPS vs eager: Qwen-Image 0.019 -> 0.006, FLUX.1-dev
        # 0.046 -> 0.029 (+2% step), FLUX.2-klein 0.018 -> 0.017, HunyuanVideo-1.5-720p 0.221 ->
        # 0.052, all at ~zero cost. Process-global, so snapshot_backend_flags restores it on unload.
        inductor_cfg = _inductor_config()
        if inductor_cfg is not None and hasattr(inductor_cfg, "emulate_precision_casts"):
            inductor_cfg.emulate_precision_casts = True
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "compile_repeated_blocks", exc)
        return False
    if unet is not None:
        # Whole-module static compile for the U-Net classes above. fullgraph mirrors the regional
        # decision (in practice only offload lowers it, U-Nets have no CacheMixin); dynamic is
        # ALWAYS False, so each new (height, width, batch) pays its own compile (Mega-cache carries
        # it across restarts). ``Module.compile`` keeps the module identity, so unload/status/LoRA
        # see the same object.
        unet_kwargs: dict[str, Any] = {"fullgraph": kwargs["fullgraph"], "dynamic": False}
        if max_autotune:
            unet_kwargs["mode"] = "max-autotune-no-cudagraphs"
        try:
            unet.compile(**unet_kwargs)
            return True
        except Exception as exc:  # noqa: BLE001 — optimisation only
            _warn(logger, "unet whole-module compile", exc)
            return False
    # Compile every denoiser DiT (dual-DiT families run both); a per-DiT failure degrades only that
    # one to eager.
    engaged = False
    for transformer in dits:
        try:
            transformer.compile_repeated_blocks(**kwargs)
            engaged = True
        except Exception as exc:  # noqa: BLE001 — optimisation only
            _warn(logger, "compile_repeated_blocks", exc)
            continue
        # A step cache engaged BEFORE this compile (the production load order) already wrapped each
        # block's forward in a @torch.compiler.disable'd hook, so the compute branch would run eager
        # on every non-skipped step and forfeit the regional compile (measured 1.69 vs 1.09 s/step
        # on HunyuanVideo-1.5). Re-point the hooks' inner forward at compiled wrappers; no-op when
        # no cache hooks. The toggle path (cache after load) is armed by apply_step_cache. Lazy
        # import to keep the dependency one-directional.
        try:
            from .diffusion_cache import _compile_hooked_block_inners
            _compile_hooked_block_inners(transformer, logger)
        except Exception as exc:  # noqa: BLE001 — optimisation only
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
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "vae decode compile", exc)
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


# Families the overflow harness found to produce non-finite activations / NEW black frames under
# fp16 accumulation. Empty by measurement: no overflow across all six families (bf16 bit-identical,
# fp16 finite; the fp16 same-seed drift is why fp16 compute is gated to ``max`` below).
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
    except Exception as exc:  # noqa: BLE001 — optimisation only
        _warn(logger, "fp16_accum", exc)
        return False


def _fuse_qkv(pipe: Any, logger: Any) -> bool:
    # Prefer the pipe-level fuse (covers every component); else fuse each denoiser DiT so a
    # dual-DiT family fuses BOTH experts.
    fn = getattr(pipe, "fuse_qkv_projections", None)
    if callable(fn):
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001 — optimisation only
            _warn(logger, "fuse_qkv_projections", exc)
            return False
    engaged = False
    for transformer in _denoiser_dits(pipe):
        tfn = getattr(transformer, "fuse_qkv_projections", None)
        if callable(tfn):
            try:
                tfn()
                engaged = True
            except Exception as exc:  # noqa: BLE001 — optimisation only
                _warn(logger, "fuse_qkv_projections", exc)
    return engaged


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.speed: %s failed: %s", what, exc)
