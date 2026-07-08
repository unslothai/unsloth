# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto low-precision casting of the diffusion pipeline's VAE (image / video decoder).

The transformer and the companion text encoder both quantise, but the VAE loads dense
and is a real resident chunk (the FLUX.2 / Qwen conv VAEs run to a few GB, the video
Conv3d VAEs more). Unlike the encoders it is CONVOLUTIONAL, so the Linear-only torchao
int8 path does not apply -- there is no int8 Conv3d kernel. Exactly two schemes fit:

  fp8_dynamic - torchao dynamic fp8 COMPUTE with PER-TENSOR granularity. Its
                Float8DynamicActivationFloat8WeightConfig quantises Conv2d (4D) and
                Conv3d (5D) weights, not just Linear, and torchao auto-skips any conv
                whose C_out / C_in is not a multiple of 16 (so the 3-channel RGB
                ``conv_out`` head stays dense). Needs fp8-GEMM silicon (cc >= 8.9) and a
                resident (non-offloaded) VAE -- the fp8 tensor subclasses reject the
                Module.to() an offload hook uses.
  fp8         - diffusers layerwise casting: 8-bit (e4m3) STORAGE, upcast per layer to the
                compute dtype. Storage-only, so it runs on ANY conv (2D / 3D) on any
                fp8-capable card (cc >= 8.9) and survives group offload.

There is no int8 (no Conv3d int8 kernel) and no nvfp4 for the VAE. ``auto`` (the loader
default) walks (fp8_dynamic, fp8): fp8_dynamic on resident data-center / Ada+ silicon that
passes a live conv smoke probe, else layerwise fp8, else dense. ``none``/``off`` keeps the
VAE dense bf16; an explicit scheme forces it (re-gated). A quantised VAE MUST NOT be
``.to(dtype=...)``'d afterwards (the fp8 tensor subclasses mishandle it), so the loader
skips the img2img/inpaint VAE re-align when a scheme engaged. torch / diffusers / torchao
are imported lazily so the module stays importable in a no-torch runtime.
"""

from __future__ import annotations

from typing import Any, Optional

VAE_QUANT_FP8 = "fp8"
VAE_QUANT_FP8_DYNAMIC = "fp8_dynamic"
VAE_QUANT_AUTO = "auto"
# Concrete schemes (excludes "auto"): "auto" resolves to one of these via
# select_vae_quant_scheme, and these are the values the casters dispatch on.
VAE_QUANT_MODES = (VAE_QUANT_FP8, VAE_QUANT_FP8_DYNAMIC)

# The RGB decoder head + output normalisations stay dense. torchao's fp8_dynamic path
# already skips the 3-channel ``conv_out`` via its C_out/C_in %16 rule, but the layerwise
# fp8 path has no such rule, so name them explicitly for both. Substring-matched against the
# module fqn: "conv_out"/"proj_out" (the pixel head), "conv_norm_out"/"norm_out" (the head's
# group-norm), whose low-magnitude outputs the coarse fp8 grid would band.
_VAE_KEEP_DENSE_TOKENS = ("conv_out", "proj_out", "conv_norm_out", "norm_out")

# Best-first ``auto`` order: fp8_dynamic (compute fp8 on the conv tensor cores) leads; layerwise
# ``fp8`` (storage-only) is the universal fallback and the sole scheme that survives group offload.
_VAE_AUTO_LADDER = (VAE_QUANT_FP8_DYNAMIC, VAE_QUANT_FP8)

# VAEs whose activation ranges break a scheme at the MODEL level (measured decoded-image
# LPIPS / SSIM vs the dense bf16 VAE). Populated from the accuracy sweep; a denied scheme is
# skipped by ``auto`` and refused when requested explicitly. Empty by default -- the
# vae_force_fp32 video families are gated separately at the loader (they never quantise).
_VAE_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {}

# Cache of device -> bool for the fp8_dynamic conv smoke probe (run once per device).
_VAE_DYNAMIC_PROBE_CACHE: dict[str, bool] = {}


def normalize_vae_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested VAE quant; None / "" / "none" / "off" -> None,
    "auto" -> "auto" (resolved later by select_vae_quant_scheme).

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
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
    """Whether ``mode`` is usable for ``target``: a CUDA device with a bf16 compute dtype, plus
    the fp8 dtype (fp8 layerwise storage) and, for fp8_dynamic, fp8-GEMM silicon (Ada sm_89+ /
    Hopper / Blackwell). There is no int8 / nvfp4 VAE scheme, so both modes need fp8."""
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


def _vae_fp8_dynamic_probe(device: str) -> bool:
    """True iff torchao's fp8_dynamic CONV path runs on this build: quantise a tiny
    Conv2d(16, 16, 1) (channels a multiple of 16 so torchao does not skip it) with the
    PerTensor fp8 config and run one forward. Cached per device. This makes ``auto`` robust
    to a torchao build whose Float8 config lacks the conv path (the Linear-only fp8 the
    transformer probes does not prove the conv path works) -- it fails here and the ladder
    falls to layerwise fp8 rather than crashing at the first decode."""
    if device in _VAE_DYNAMIC_PROBE_CACHE:
        return _VAE_DYNAMIC_PROBE_CACHE[device]
    ok = False
    try:
        import torch
        from torch import nn
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        conv = nn.Conv2d(16, 16, 1).to(device = device, dtype = torch.bfloat16)
        quantize_(
            conv,
            Float8DynamicActivationFloat8WeightConfig(granularity = PerTensor()),
            filter_fn = lambda m, fqn = "": isinstance(m, nn.Conv2d),
        )
        x = torch.randn(1, 16, 4, 4, device = device, dtype = torch.bfloat16)
        with torch.no_grad():
            conv(x)
        torch.cuda.synchronize()
        ok = True
    except Exception:
        ok = False
    _VAE_DYNAMIC_PROBE_CACHE[device] = ok
    return ok


def select_vae_quant_scheme(
    target: Any,
    requested: Optional[str],
    *,
    family: Optional[str] = None,
    offload_active: bool = False,
    force_fp32: bool = False,
) -> Optional[str]:
    """Resolve the concrete VAE scheme to apply, or None to stay dense bf16.

    An explicit scheme is returned as-is unless family-denied (``quantize_vae`` re-gates it).
    ``force_fp32`` (a family that runs its VAE in fp32) always stays dense. ``auto`` walks
    (fp8_dynamic, fp8) and returns the first that: survives the active offload policy (torchao
    fp8_dynamic tensors reject the Module.to() an offload hook uses -> only layerwise fp8 under
    offload), is not family-denied, is hardware-supported, and (fp8_dynamic only) passes a real
    conv smoke probe. Returns None when nothing qualifies (e.g. no CUDA / pre-Ada)."""
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
        # torchao fp8_dynamic produces tensor subclasses that reject Module.to(); an offload hook
        # moves the VAE that way and hard-crashes, so under offload only layerwise fp8 engages.
        if offload_active and scheme == VAE_QUANT_FP8_DYNAMIC:
            continue
        if _vae_family_denied(family, scheme):
            continue
        if not vae_quant_supported(target, scheme):
            continue
        # fp8_dynamic additionally needs the torchao CONV fp8 kernel to actually run on this build.
        if scheme == VAE_QUANT_FP8_DYNAMIC and not _vae_fp8_dynamic_probe(device):
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
    """Quantise ``pipe.vae`` in place with ``mode`` (auto / fp8 / fp8_dynamic). Returns the scheme
    actually engaged, or None when disabled, unsupported, force-fp32, family-denied, or no VAE was
    cast. ``auto`` resolves to the best scheme for the GPU + family via ``select_vae_quant_scheme``;
    an explicit scheme is re-gated the same way. Under offload the torchao fp8_dynamic mode is
    skipped (its tensor subclasses reject the ``Module.to()`` an offload hook uses); layerwise fp8
    still engages. Best-effort: any failure leaves the VAE dense."""
    mode = normalize_vae_quant(mode)
    if mode is None:
        return None
    if mode == VAE_QUANT_AUTO:
        mode = select_vae_quant_scheme(
            target,
            VAE_QUANT_AUTO,
            family = family,
            offload_active = offload_active,
            force_fp32 = force_fp32,
        )
        if mode is None:
            return None
    else:
        # An explicit scheme re-runs the same gates ``auto`` applies in select_vae_quant_scheme.
        if force_fp32:
            _note(logger, f"vae '{mode}' skipped: family runs its VAE in fp32")
            return None
        if _vae_family_denied(family, mode):
            _note(logger, f"vae '{mode}' denied for family '{family}' (out-of-bar; staying dense)")
            return None
        # Layerwise fp8 is storage-only and streams fine under offload; the torchao fp8_dynamic
        # tensors reject Module.to(), so skip only that one when an offload policy is active.
        if offload_active and mode == VAE_QUANT_FP8_DYNAMIC:
            _note(
                logger,
                f"vae '{mode}' skipped under offload (torchao tensors reject Module.to()); "
                "pin a resident memory mode or use fp8",
            )
            return None
        if not vae_quant_supported(target, mode):
            return None
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return None
    try:
        if mode == VAE_QUANT_FP8_DYNAMIC:
            _cast_vae_fp8_dynamic(vae, target)
        else:
            _cast_vae_fp8(vae, target)
        return mode
    except Exception as exc:  # noqa: BLE001 — leave the VAE dense
        _warn(logger, mode, exc)
        return None


def _cast_vae_fp8_dynamic(vae: Any, target: Any) -> None:
    # torchao dynamic fp8 COMPUTE with PER-TENSOR granularity (NOT the DiT's per-row config):
    # Float8DynamicActivationFloat8WeightConfig quantises Conv2d (4D) / Conv3d (5D) weights, not
    # just Linear. Both channel dims must be a multiple of 16 (torchao skips the conv otherwise,
    # which already leaves the 3-channel RGB head dense), and the decoder head / output norms are
    # excluded by name for good measure. A weight.dim() < 2 (e.g. a bias-only leaf) never matches.
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

    # diffusers' layerwise casting stores each supported leaf module's weights in fp8 and upcasts
    # them per forward. Storage-only, so it works on any conv (2D / 3D) and survives offload. Keep
    # the RGB decoder head / output norms dense (their low-magnitude pixels band on the coarse fp8
    # grid); the diffusers default pattern only covers pos/patch embeds. Names are literal
    # substrings (re.escape) matched against the module fqn.
    skip = tuple(DEFAULT_SKIP_MODULES_PATTERN) + tuple(re.escape(t) for t in _VAE_KEEP_DENSE_TOKENS)
    apply_layerwise_casting(
        vae,
        storage_dtype = torch.float8_e4m3fn,
        compute_dtype = target.dtype,
        skip_modules_pattern = skip,
    )


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.vae_quant: (%s) failed: %s", what, exc)


def _note(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.vae_quant: %s", msg)
