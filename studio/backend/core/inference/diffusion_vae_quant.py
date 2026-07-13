# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto low-precision casting of the diffusion pipeline's VAE (image / video decoder).

The transformer and the companion text encoder both quantise, but the VAE loads dense
and is a real resident chunk (the FLUX.2 / Qwen conv VAEs run to a few GB, the video
Conv3d VAEs more). Unlike the encoders it is CONVOLUTIONAL, so the Linear-only torchao
int8 path does not apply -- there is no int8 Conv3d kernel. Exactly two schemes fit:

  fp8_dynamic - torchao dynamic fp8 COMPUTE, PER-TENSOR granularity. Quantises Conv2d/Conv3d
                weights (not just Linear); torchao auto-skips any conv whose C_out/C_in is not
                a multiple of 16 (so the 3-channel RGB ``conv_out`` head stays dense) and
                torchao 0.17's fp8 conv kernel also rejects POINTWISE (1x1) convs, kept dense
                too. Needs fp8-GEMM silicon (cc >= 8.9) and a resident VAE (the fp8 subclasses
                reject the Module.to() an offload hook uses).
  fp8         - diffusers layerwise casting: 8-bit (e4m3) STORAGE, upcast per layer to the
                compute dtype. Storage-only, so it runs on ANY conv (2D / 3D) on any
                fp8-capable card (cc >= 8.9) and survives group offload.

No int8 (no Conv3d int8 kernel) and no nvfp4 for the VAE. A decoded-image LPIPS / SSIM sweep
vs the dense bf16 VAE (B200) showed layerwise ``fp8`` holds across families (SSIM >= 0.977 on
all but SDXL) while ``fp8_dynamic`` (PerTensor conv compute) is in-bar only on a couple (FLUX.2,
Hunyuan) and catastrophic elsewhere (Qwen-Image SSIM 0.46). So ``auto`` engages layerwise
``fp8`` ONLY (the memory win with no measured quality loss); ``fp8_dynamic`` is an explicit
opt-in, re-gated by a per-family deny list. ``auto`` also SIZE-GATES above a ~1 GB floor: the
small image AutoencoderKLs (~0.2 GB) save ~nothing and slow their tiny decode, while the video
Conv3d VAEs (~2.5 GB) halve to ~1.2 GB at ~2% decode cost. An explicit scheme skips the size
gate. ``none``/``off`` stays dense bf16. A quantised VAE MUST NOT be ``.to(dtype=...)``'d
afterwards (the fp8 subclasses mishandle it), so the loader skips the img2img/inpaint VAE
re-align when a scheme engaged. torch / diffusers / torchao imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

VAE_QUANT_FP8 = "fp8"
VAE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
VAE_QUANT_AUTO = "auto"
# Concrete schemes (excludes "auto") the casters dispatch on.
VAE_QUANT_MODES = (VAE_QUANT_FP8, VAE_QUANT_FP8_DYNAMIC)

# The RGB decoder head + output norms stay dense. torchao's fp8_dynamic path already skips
# ``conv_out`` via its %16 rule, but the layerwise fp8 path has none, so name them for both.
# Substring-matched against the fqn: the pixel head + its group-norm, whose low-magnitude
# outputs the coarse fp8 grid would band.
_VAE_KEEP_DENSE_TOKENS = ("conv_out", "proj_out", "conv_norm_out", "norm_out")

# ``auto`` engages layerwise ``fp8`` ONLY: fp8_dynamic is in-bar on just FLUX.2 / Hunyuan, and
# for a VAE decode (a few % of e2e) its fp8-matmul speedup over storage-only fp8 is negligible,
# so auto never risks it. fp8_dynamic stays an EXPLICIT opt-in (re-gated by the deny list below).
# The select_vae_quant_scheme loop keeps its per-scheme gates generic so re-adding fp8_dynamic
# here later needs no extra plumbing.
_VAE_AUTO_LADDER = (VAE_QUANT_FP8,)

# VAEs whose activation ranges break a scheme at the MODEL level -- from the decoded-image
# LPIPS / SSIM sweep vs the dense bf16 VAE (B200; bar LPIPS <= 0.05, SSIM >= 0.95). A denied
# scheme is skipped by ``auto`` and refused when requested explicitly.
#   sdxl - layerwise fp8 marginal (SSIM 0.935) AND fp8_dynamic worse (0.894): deny BOTH (its
#          small VAE just stays dense, negligible cost).
#   fp8_dynamic-only denials (layerwise fp8 is fine; auto still quantises these): flux.1 /
#     flux.1-kontext 0.935, qwen-image / qwen-image-edit 0.46 (catastrophic), ltx-2 0.942.
#     FLUX.2 and Hunyuan-1.5 pass fp8_dynamic, so they are not denied.
# The vae_force_fp32 families (Wan) are gated separately at the loader.
_VAE_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "sdxl": frozenset({VAE_QUANT_FP8, VAE_QUANT_FP8_DYNAMIC}),
    "flux.1": frozenset({VAE_QUANT_FP8_DYNAMIC}),
    "flux.1-kontext": frozenset({VAE_QUANT_FP8_DYNAMIC}),
    "qwen-image": frozenset({VAE_QUANT_FP8_DYNAMIC}),
    "qwen-image-edit": frozenset({VAE_QUANT_FP8_DYNAMIC}),
    "ltx-2": frozenset({VAE_QUANT_FP8_DYNAMIC}),
}

# Cache of (device, conv ndim) -> bool for the fp8_dynamic conv smoke probe (the 2D and 3D
# torchao conv kernels are separate code paths).
_VAE_DYNAMIC_PROBE_CACHE: dict[tuple[str, int], bool] = {}

# ``auto`` only quantises a VAE big enough for the saving to beat the fp8-decode overhead. The
# split is by kind: image AutoencoderKLs are ~0.15-0.26 GB (halving saves ~0.1 GB at +6-16% on
# their tiny decode -- net negative), while video Conv3d VAEs are ~2.5 GB (halving saves ~1.2 GB
# at only +2%). ~1 GB cleanly separates them; an explicit request skips the floor.
_VAE_AUTO_MIN_BYTES = 1_000_000_000


def normalize_vae_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested VAE quant; None / "" / "none" / "off" -> None, "auto" -> "auto"
    (resolved later by select_vae_quant_scheme). Raises ValueError for an unsupported value."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized == VAE_QUANT_AUTO:
        return VAE_QUANT_AUTO
    if normalized not in VAE_QUANT_MODES:
        raise ValueError(
            f"Unsupported vae_quant '{value}'. Use one of: "
            f"{', '.join((VAE_QUANT_AUTO,) + VAE_QUANT_MODES)}, none/off."
        )
    return normalized


def vae_quant_supported(target: Any, mode: str) -> bool:
    """Whether ``mode`` is usable for ``target``: a CUDA + bf16 device with the fp8 dtype, plus
    (for fp8_dynamic) fp8-GEMM silicon (sm_89+). Both modes need fp8 (no int8 / nvfp4 VAE)."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch

        if getattr(target, "dtype", None) is not torch.bfloat16:
            return False
        if mode == VAE_QUANT_FP8:
            return hasattr(torch, "float8_e4m3fn")
        if mode == VAE_QUANT_FP8_DYNAMIC:
            # Compute fp8 (torch._scaled_mm on the conv weights) needs fp8-GEMM silicon.
            return hasattr(torch, "float8_e4m3fn") and torch.cuda.get_device_capability() >= (8, 9)
    except Exception:
        return False
    return False


def _vae_family_denied(family: Optional[str], scheme: str) -> bool:
    return scheme in _VAE_FAMILY_SCHEME_DENY.get((family or "").strip().lower(), frozenset())


def _vae_param_bytes(vae: Any) -> int:
    """Dense weight footprint of the VAE in bytes, to size-gate ``auto``."""
    try:
        return sum(p.numel() * p.element_size() for p in vae.parameters())
    except Exception:
        # Unknown size -> treat as above the floor rather than wrongly skip.
        return _VAE_AUTO_MIN_BYTES


def _vae_fp8_dynamic_probe(device: str, ndim: int = 2) -> bool:
    """True iff torchao's fp8_dynamic CONV path runs on this build for ``ndim``-D convs: quantise
    a tiny Conv2d/Conv3d(16, 16, 3) (channels %16 so torchao doesn't skip it; a SPATIAL 3x3 kernel,
    NOT 1x1 which torchao 0.17 rejects) with the PerTensor config and run one forward. Cached per
    (device, ndim): the 2D and 3D kernels are separate paths, so a working Conv2d doesn't prove
    Conv3d. Makes an explicit fp8_dynamic request robust to a torchao build lacking the conv path
    -- it fails here and stays dense rather than crashing at the first decode."""
    key = (device, ndim)
    if key in _VAE_DYNAMIC_PROBE_CACHE:
        return _VAE_DYNAMIC_PROBE_CACHE[key]
    ok = False
    try:
        import torch
        from torch import nn
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        conv_cls = nn.Conv3d if ndim == 3 else nn.Conv2d
        conv = conv_cls(16, 16, 3, padding = 1).to(device = device, dtype = torch.bfloat16)
        quantize_(
            conv,
            Float8DynamicActivationFloat8WeightConfig(granularity = PerTensor()),
            filter_fn = lambda m, fqn = "": isinstance(m, conv_cls),
        )
        x = torch.randn(1, 16, *([4] * ndim), device = device, dtype = torch.bfloat16)
        with torch.no_grad():
            conv(x)
        torch.cuda.synchronize()
        ok = True
    except Exception:
        ok = False
    _VAE_DYNAMIC_PROBE_CACHE[key] = ok
    return ok


def _vae_conv_ndims(vae: Any) -> tuple[int, ...]:
    """The conv dimensionalities ``_cast_vae_fp8_dynamic`` would quantise in this VAE: (2,) for
    image AutoencoderKLs, (3,) or (2, 3) for the video Conv3d decoders. Falls back to (2,) when
    the VAE cannot be inspected so the gate never silently widens."""
    ndims: set[int] = set()
    try:
        from torch import nn
        for module in vae.modules():
            if isinstance(module, nn.Conv3d):
                ndims.add(3)
            elif isinstance(module, nn.Conv2d):
                ndims.add(2)
    except Exception:
        ndims = set()
    return tuple(sorted(ndims)) or (2,)


def select_vae_quant_scheme(
    target: Any,
    requested: Optional[str],
    *,
    family: Optional[str] = None,
    offload_active: bool = False,
    force_fp32: bool = False,
    vae: Any = None,
) -> Optional[str]:
    """Resolve the concrete VAE scheme to apply, or None to stay dense bf16.

    An explicit scheme is returned as-is unless family-denied. ``force_fp32`` always stays dense.
    ``auto`` walks ``_VAE_AUTO_LADDER`` (layerwise fp8 ONLY) and returns the first scheme that
    survives the offload policy (fp8_dynamic tensors reject Module.to() -> only fp8 under offload),
    is not family-denied, is hardware-supported, and (fp8_dynamic only) passes a conv smoke probe
    for every conv dim the ``vae`` contains. Returns None when nothing qualifies."""
    requested = normalize_vae_quant(requested)
    if requested is None or force_fp32:
        return None
    if requested != VAE_QUANT_AUTO:
        return None if _vae_family_denied(family, requested) else requested
    from .diffusion_transformer_quant import _capability

    if _capability() is None:
        return None
    device = str(getattr(target, "device", "cuda"))
    for scheme in _VAE_AUTO_LADDER:
        # fp8_dynamic tensors reject Module.to(), so an offload hook crashes -> only fp8 engages.
        if offload_active and scheme == VAE_QUANT_FP8_DYNAMIC:
            continue
        if _vae_family_denied(family, scheme):
            continue
        if not vae_quant_supported(target, scheme):
            continue
        # fp8_dynamic also needs the torchao CONV fp8 kernel to run for every conv dim the VAE
        # contains (Conv3d is a separate path; Conv2d assumed when no VAE was provided).
        if scheme == VAE_QUANT_FP8_DYNAMIC and not all(
            _vae_fp8_dynamic_probe(device, ndim)
            for ndim in (_vae_conv_ndims(vae) if vae is not None else (2,))
        ):
            continue
        return scheme
    return None


def quantize_vae(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    offload_active: bool = False,
    force_fp32: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """Quantise ``pipe.vae`` in place with ``mode`` (auto / fp8 / fp8_dynamic). Returns the engaged
    scheme, or None when disabled, unsupported, force-fp32, family-denied, or nothing was cast.
    ``auto`` resolves via ``select_vae_quant_scheme``; an explicit scheme is re-gated the same way.
    Under offload fp8_dynamic is skipped (its tensors reject Module.to()); fp8 still engages.
    Best-effort: any failure leaves the VAE dense."""
    mode = normalize_vae_quant(mode)
    if mode is None:
        return None
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return None
    if mode == VAE_QUANT_AUTO:
        # Size gate: a small image VAE saves ~nothing and slows its decode, so auto leaves it
        # dense; the big video Conv3d VAEs clear the floor. Explicit requests skip this gate.
        if _vae_param_bytes(vae) < _VAE_AUTO_MIN_BYTES:
            _note(
                logger,
                "vae auto: VAE under the ~1GB size floor; staying dense (quant saves ~nothing)",
            )
            return None
        mode = select_vae_quant_scheme(
            target,
            VAE_QUANT_AUTO,
            family = family,
            offload_active = offload_active,
            force_fp32 = force_fp32,
            vae = vae,
        )
        if mode is None:
            return None
    else:
        # An explicit scheme re-runs the same gates ``auto`` applies.
        if force_fp32:
            _note(logger, f"vae '{mode}' skipped: family runs its VAE in fp32")
            return None
        if _vae_family_denied(family, mode):
            _note(logger, f"vae '{mode}' denied for family '{family}' (out-of-bar; staying dense)")
            return None
        # fp8 streams fine under offload; fp8_dynamic's tensors reject Module.to(), so skip
        # only that one when offload is active.
        if offload_active and mode == VAE_QUANT_FP8_DYNAMIC:
            _note(
                logger,
                f"vae '{mode}' skipped under offload (torchao tensors reject Module.to()); "
                "pin a resident memory mode or use fp8",
            )
            return None
        if not vae_quant_supported(target, mode):
            return None
        # fp8_dynamic also needs torchao's fp8 CONV kernel for EVERY conv dim this VAE contains
        # (Conv3d is a separate path); else the cast succeeds but the first decode crashes.
        if mode == VAE_QUANT_FP8_DYNAMIC:
            device = str(getattr(target, "device", "cuda"))
            if not all(_vae_fp8_dynamic_probe(device, ndim) for ndim in _vae_conv_ndims(vae)):
                _note(
                    logger, "vae 'fp8_dynamic' skipped: torchao build lacks a working fp8 conv path"
                )
                return None
    try:
        if mode == VAE_QUANT_FP8_DYNAMIC:
            _cast_vae_fp8_dynamic(vae, target)
        else:
            _cast_vae_fp8(vae, target)
        return mode
    except Exception as exc:  # noqa: BLE001 — leave the VAE dense
        # fp8_dynamic's quantize_ swaps weights module-by-module, so a mid-pass failure may
        # leave the VAE PARTIALLY quantized -- fail the load for that instead of a dense fallback.
        # raise_if_partially_quantized only recognises torchao parameter subclasses, so it CANNOT
        # see a partial layerwise fp8 mutation (diffusers apply_layerwise_casting installs upcast
        # hooks + fp8 storage in place, leaving no torchao params). Detect a leftover layerwise
        # hook directly and fail closed there too; a clean failure (no hook installed) still falls
        # back to dense (best-effort), matching the fp8 storage-only contract.
        from .diffusion_transformer_quant import raise_if_partially_quantized

        if mode == VAE_QUANT_FP8 and _has_layerwise_casting(vae):
            raise RuntimeError(
                "vae_quant fp8 failed after partially installing layerwise casting (leftover "
                f"fp8 hooks); reload the model instead of a dense fallback (original error: {exc})"
            ) from exc
        raise_if_partially_quantized(vae, what = f"vae_quant {mode}", exc = exc)
        _warn(logger, mode, exc)
        return None


def _cast_vae_fp8_dynamic(vae: Any, target: Any) -> None:
    # torchao dynamic fp8 COMPUTE, PER-TENSOR (NOT the DiT's per-row config): quantises Conv2d /
    # Conv3d weights, not just Linear. Both channel dims must be %16 (torchao skips the conv
    # otherwise, already leaving the RGB head dense); the head / output norms are excluded by name
    # too. A weight.dim() < 2 (bias-only leaf) never matches.
    from torch import nn
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerTensor,
        quantize_,
    )

    def filter_fn(module: Any, fqn: str = "") -> bool:
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() < 2:
            return False
        if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
            return False
        # torchao 0.17's fp8 conv kernel rejects POINTWISE (1x1) convs: they cast fine but crash
        # at the first decode. Leave them dense (cheap 1x1 projections anyway).
        kernel_size = getattr(module, "kernel_size", None)
        if isinstance(kernel_size, tuple) and kernel_size and all(k == 1 for k in kernel_size):
            return False
        name = fqn.lower() if fqn else ""
        return not any(tok in name for tok in _VAE_KEEP_DENSE_TOKENS)

    quantize_(
        vae, Float8DynamicActivationFloat8WeightConfig(granularity = PerTensor()), filter_fn = filter_fn
    )


def _cast_vae_fp8(vae: Any, target: Any) -> None:
    import re

    import torch
    from diffusers.hooks import apply_layerwise_casting
    from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN

    # diffusers layerwise casting: fp8 STORAGE, upcast per forward. Works on any conv (2D / 3D) and
    # survives offload. Keep the RGB head / output norms dense (low-magnitude pixels band on the
    # coarse grid; the diffusers default pattern only covers pos/patch embeds). Literal substrings
    # (re.escape) matched against the fqn.
    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN) + tuple(re.escape(t) for t in _VAE_KEEP_DENSE_TOKENS)
    apply_layerwise_casting(
        vae,
        storage_dtype = torch.float8_e4m3fn,
        compute_dtype = target.dtype,
        skip_modules_pattern = skip,
    )


def _has_layerwise_casting(module: Any) -> bool:
    """True when any submodule still carries a diffusers layerwise-casting hook -- i.e. an
    ``apply_layerwise_casting`` pass mutated the module (installed an fp8-storage upcast hook)
    before failing. torchao's partial-quant detector cannot see these, so a mid-pass layerwise
    failure would otherwise report a dense fallback over a half-cast module. Best-effort: a module
    without ``.modules()`` or a moved diffusers internal returns False (defer to the torchao check).
    """
    try:
        hook_name = "layerwise_casting"
        try:
            from diffusers.hooks.layerwise_casting import _LAYERWISE_CASTING_HOOK
            hook_name = _LAYERWISE_CASTING_HOOK
        except Exception:  # noqa: BLE001 -- const moved: fall back to the stable literal
            pass
        for sub in module.modules():
            registry = getattr(sub, "_diffusers_hook", None)
            get_hook = getattr(registry, "get_hook", None)
            if callable(get_hook) and get_hook(hook_name) is not None:
                return True
    except Exception:  # noqa: BLE001 -- unqueryable module: defer to the torchao check
        return False
    return False


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.vae_quant: (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.vae_quant: %s", msg)
