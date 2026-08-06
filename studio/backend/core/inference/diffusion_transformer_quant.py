# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision quantisation of the diffusion DiT transformer.

The default GGUF path stores weights 4-bit but DEQUANTISES to bf16 on every matmul, so it
runs at bf16 tensor-core rate: a memory win that costs speed. This module is the opt-in
alternative: load the DENSE bf16 transformer and torchao-quantise it with a
DYNAMIC-ACTIVATION scheme so the matmul runs on low-precision tensor cores. Measured on B200
(Z-Image-Turbo, 1024px/8 steps) vs GGUF+compile (0.802s, LPIPS 0.083): fp8 0.585s (1.37x),
int8 0.603s (1.33x), both at LOWER LPIPS than GGUF -- faster and slightly more accurate, at
a higher-memory dense load. Strictly opt-in; GGUF stays the low-memory default and fallback.

Scheme by architecture (``auto`` picks the best supported, best first):
  nvfp4 / mxfp8 - Blackwell sm_100+ FP4 / MX tensor cores (biggest win; prototype).
  fp8           - Ada / Hopper / Blackwell (sm_89+) fp8 tensor cores.
  int8          - Ampere+ (sm_80+) int8 tensor cores -- the broadest-hardware lever.

Every scheme needs ``torch.compile`` for the speedup (dynamic quant is ~30x slower eager);
the loader compiles the repeated block after this. torch / torchao imported lazily; every
probe is best-effort: an unsupported scheme yields None and the caller loads GGUF.
"""

from __future__ import annotations

from typing import Any, Optional

TQ_INT8 = "int8"
TQ_FP8 = "fp8"
TQ_NVFP4 = "nvfp4"
TQ_MXFP8 = "mxfp8"
TQ_AUTO = "auto"
TQ_SCHEMES = (TQ_INT8, TQ_FP8, TQ_NVFP4, TQ_MXFP8)
TQ_MODES = (TQ_AUTO,) + TQ_SCHEMES

# Schemes whose torchao path asserts a bf16 weight, so their filter skips non-bf16 Linears rather than aborting the pass. On torchao 0.17 / B200: fp8 per-row and mxfp8 assert bf16; nvfp4 and int8 handle fp32/fp16.
_REQUIRE_BF16_SCHEMES = (TQ_FP8, TQ_MXFP8)

# fp8 granularity the runtime uses: per-ROW is REQUIRED for correctness on outlier-heavy DiTs. Stamped into prequant metadata, so a stale per-TENSOR checkpoint is rejected and rebuilt.
FP8_GRANULARITY = "per_row"

# Skip linears below this feature size: a small FLOP share, so leaving them bf16 costs ~nothing.
DEFAULT_MIN_LINEAR_FEATURES = 512

# int8-ONLY name exclusions: torch._int_mm needs activation rows M above 16, and a DiT's AdaLN modulation and timestep / guidance / pooled-text
# embedders run once from a [batch, dim] vector (M = 1), so they crash despite large feature dims. Negligible FLOPs; scaled_mm has no M limit.
_INT8_EXCLUDE_NAME_TOKENS = (
    "norm",  # AdaLN modulation .linear
    "_mod",  # Qwen img_mod / txt_mod
    "modulation",  # Flux.2 double/single_stream_modulation
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",  # Flux/Qwen (pooled-text + timestep); NOT context_embedder
    "pooled",
    # Krea 2 time_embed.linear_2 (M = batch); time_mod_proj is caught by "_mod", and the rest fall under min_features.
    "time_embed",
)


# int8 PER-FAMILY exclusions, on top of _INT8_EXCLUDE_NAME_TOKENS. Qwen-Image MMDiT runs every TEXT-stream Linear at M = actual prompt tokens
# (unpadded, unlike FLUX's 512-token T5), so a short prompt falls under _int_mm's M floor of 16 and the denoise crashes. bf16 there costs ~nothing.
_QWENIMAGE_INT8_EXCLUDES = (
    "txt_in",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "txt_mlp",
)
# HunyuanVideo-1.5's attention trim (this PR) shrinks the text / image streams from their padded
# lengths to the VALID token counts, so every text-stream Linear runs at a tiny M the int8 dynamic
# path cannot handle. Both failures measured on B200:
#   - M = 0 (t2v byt5 / image streams trim to zero tokens): torchao returns the input UNPROJECTED
#     (a quantized 1472 -> 2048 Linear maps [1, 0, 1472] to [1, 0, 1472]), so the 2048-wide
#     cond-type add crashes -> context_embedder_2 / image_embedder;
#   - M <= 16 (a short prompt, or the empty negative prompt's ~6 tokens): torch._int_mm requires
#     M > 16 and raises -> the TokenRefiner and every block's context-stream projections.
# These run at M = text tokens (tens) against the video stream's M ~ 32k+, so bf16 here costs
# nothing measurable and the video-stream linears keep full int8 coverage. "context_embedder"
# also matches "context_embedder_2" (substring check).
_HUNYUAN15_INT8_EXCLUDES = (
    "context_embedder",
    "image_embedder",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff_context",
)
_INT8_FAMILY_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "qwen-image": _QWENIMAGE_INT8_EXCLUDES,
    "qwen-image-edit": _QWENIMAGE_INT8_EXCLUDES,  # same DiT class + unpadded text stream
    "hunyuanvideo-1.5": _HUNYUAN15_INT8_EXCLUDES,
    "hunyuanvideo-1.5-720p": _HUNYUAN15_INT8_EXCLUDES,
}


def exclude_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    """Name tokens to exclude from quantisation for ``scheme`` (optionally family-specific).
    int8 (M>16) skips the M=1 modulation / conditioning-embedder projections
    (_INT8_EXCLUDE_NAME_TOKENS) plus per-family small-M text streams
    (_INT8_FAMILY_EXCLUDE_NAME_TOKENS); other schemes (scaled_mm) exclude nothing.
    ``family=None`` preserves the historical behaviour. Shared by the runtime path and the
    offline prequant builder so they never drift (else an int8 checkpoint bakes the small-M
    projections and crashes at the first denoise on Flux / Qwen)."""
    if scheme == TQ_INT8:
        return _INT8_EXCLUDE_NAME_TOKENS + _INT8_FAMILY_EXCLUDE_NAME_TOKENS.get(
            str(family or "").strip().lower(), ()
        )
    return ()


# Per-arch preference for ``auto``, best first. On Blackwell fp8 leads: on B200 plain fp8 dynamic is faster AND more accurate at DiT shapes,
# while mxfp8 block scaling only adds overhead. nvfp4's FP4 GEMM is real with torch>=2.11 but wins only on very large GEMMs (0.81x on Z-Image
# 1024px, LPIPS 0.166 vs fp8 0.044), so it is kept OUT of the ladder below and stays an explicit opt-in (transformer_quant="nvfp4"): auto must
# never silently drop to a scheme that is both slower and less accurate. Restore the commented Blackwell tier to re-enable it once the FP4
# tensor-core GEMM wins at the DiT's real shapes (hidden ~3072, MLP ~12288, M ~4096) and its accuracy is validated by the prequant gate.
# Consumer / workstation GPUs move int8 first: they halve fp8/fp16 FP32-accumulate.
_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    ((10, 0), (TQ_FP8, TQ_MXFP8, TQ_INT8)),  # Blackwell sm_100+ (nvfp4 is explicit opt-in only)
    # ((10, 0), (TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8)),  # restore to re-enable nvfp4 under auto
    ((8, 9), (TQ_FP8, TQ_INT8)),  # Ada sm_89 / Hopper sm_90
    ((8, 0), (TQ_INT8,)),  # Ampere sm_80 / sm_86
)

# Families whose activation ranges break specific schemes at the MODEL level (the smoke probe only proves the GEMM runs). Measured with the
# 28-pair prequant accuracy gate on B200: qwen-image + mxfp8 does semantic damage at 1024px, + nvfp4 is unusable (LPIPS 0.51). Neither has a
# known fix, so both stay denied and auto falls through to int8 (excellent on Qwen). The deny also applies to an EXPLICIT request.
#
# fp8 was denied here too, for black frames. That cause is gone: it was torchao's fp8 scale chooser having no eps clamp, so qwen's all-zero
# text rows gave scale 0 and NaN, and ``_make_quant_config`` now floors it with ``activation_value_lb``. Re-measured on B200 with the floor,
# and BOTH paths the deny governs clear the 0.10 LPIPS bar, which is why it is safe to drop rather than just the checkpoint half:
#   prequant checkpoint, full 28-pair gate  -> 28/28 PASS (SSIM 0.87-0.99, LPIPS 0.027-0.228, CLIP delta <= 0.019)
#   on-the-fly quantize_, gate's simple_object prompt at its own seeds -> LPIPS 0.048 / 0.033 / 0.027 / 0.063
# against pre-floor plain fp8 at LPIPS 0.712 with SSIM 0.016 and mean luma 0. Keeping the deny cost real speed: it pinned qwen to int8 at
# 1.10x bf16 when compiled fp8 measures 1.21x.
_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "qwen-image": frozenset({TQ_MXFP8, TQ_NVFP4}),
    "qwen-image-edit": frozenset({TQ_MXFP8, TQ_NVFP4}),  # same DiT
}


# Schemes denied for TRAINING on top of the inference table. Training holds a stricter bar because
# the evidence above is rendering evidence: it says a frozen fp8 forward reconstructs the bf16 image,
# not that a LoRA converges when its frozen linears are fp8. Nobody has run that, so qwen fp8 stays
# out of the Train UI until someone does. Delete the entry once a training run is measured.
_FAMILY_TRAIN_SCHEME_DENY: dict[str, frozenset[str]] = {
    "qwen-image": frozenset({TQ_FP8}),
    "qwen-image-edit": frozenset({TQ_FP8}),
}


def _family_denied(family, scheme: str) -> bool:
    return scheme in _FAMILY_SCHEME_DENY.get(str(family or "").strip().lower(), ())


def _family_train_denied(family, scheme: str) -> bool:
    """``_family_denied`` plus the training-only additions. Every inference deny also applies to
    training (a scheme that cannot render cannot train), so this is a superset, never a bypass."""
    key = str(family or "").strip().lower()
    return _family_denied(family, scheme) or scheme in _FAMILY_TRAIN_SCHEME_DENY.get(key, ())


# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}

# Data-center GPU tokens (un-nerfed FP32 accumulate). Matched as whole tokens of get_device_name() so "A4000" is not mistaken for "A40"; anything else is consumer-class.
_DATACENTER_GPU_TOKENS = frozenset(
    {
        "B200",
        "B100",
        "B300",  # Blackwell Ultra data center
        "GB200",
        "GB300",
        "GB10",  # Blackwell data center
        "H200",
        "H100",
        "H800",
        "H20",
        "GH200",  # Grace-Hopper superchip (data center)
        "A100",
        "A800",
        "A30",
        "A40",
        "A16",
        "A10",
        "A2",  # Ampere data center
        "L40",
        "L40S",
        "L4",
        "L20",
        "L2",  # Ada data center
        "V100",
        "P100",
        "P40",
        "T4",  # legacy data center
    }
)


# Professional parts the backend treats as datacenter-class. Matched as phrases since the marker spans tokens.
_PROFESSIONAL_GPU_MARKERS = ("RTX PRO 6000", "RTX 6000 ADA")


def _is_consumer_gpu(device: Any = None) -> bool:
    """Whether the active GPU is consumer-class (GDDR), where fp8 FP32 accumulate is halved so
    fast (FP16) accumulate is a ~2x win. Data-center HBM and professional parts are not nerfed
    (return False -> precise accumulate, fp8 first). Heuristic on the device name: GeForce /
    TITAN -> consumer; a data-center token or professional marker -> not; anything else defaults
    to consumer (fast accumulate is free on data-center, a win on consumer). True on any failure."""
    try:
        import re

        import torch
        name = torch.cuda.get_device_name(device).upper()
    except Exception:  # noqa: BLE001 — no torch / no device -> assume consumer
        return True
    if "GEFORCE" in name or "TITAN" in name:
        return True
    if any(marker in name for marker in _PROFESSIONAL_GPU_MARKERS):
        return False
    tokens = set(re.split(r"[^A-Z0-9]+", name))
    return not (tokens & _DATACENTER_GPU_TOKENS)


def normalize_transformer_quant(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested transformer quant; None / "" / "none" / "off" -> None.

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized not in TQ_MODES:
        raise ValueError(
            f"Unsupported transformer_quant '{value}'. Use one of: {', '.join(TQ_MODES)}."
        )
    return normalized


def dense_transformer_supported(target: Any) -> bool:
    """Whether the dense-source quant path is usable for ``target``: a CUDA device with bf16
    dtype (the only config any torchao dynamic scheme accelerates). Cheap loader pre-check."""
    if getattr(target, "device", None) != "cuda":
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def select_transformer_quant_scheme(
    target: Any,
    requested: Optional[str],
    family: Optional[str] = None,
) -> Optional[str]:
    """The concrete scheme to apply, or None to fall back to GGUF.

    ``auto`` walks the per-arch ladder and returns the first scheme passing a real
    quantise+matmul smoke test, so an unavailable Blackwell fp4/mx kernel lands on fp8/int8
    with no error. An explicit scheme is honored only if supported (else None), never swapped.
    ``family`` applies the measured deny list (``_FAMILY_SCHEME_DENY``): schemes that produce
    black frames / out-of-bar drift are skipped by ``auto`` and refused when explicit."""
    requested = normalize_transformer_quant(requested)
    if requested is None or not dense_transformer_supported(target):
        return None
    device = str(getattr(target, "device", "cuda"))
    if requested != TQ_AUTO:
        if _family_denied(family, requested):
            return None
        return requested if _scheme_supported(requested, device) else None
    cap = _capability()
    if cap is None:
        return None
    for floor, schemes in _AUTO_LADDER:
        if cap >= floor:
            for scheme in _prefer_consumer_scheme(schemes, device):
                if _family_denied(family, scheme):
                    continue
                if _scheme_supported(scheme, device):
                    return scheme
            return None
    return None


def auto_scheme_candidates(target: Any, family: Optional[str] = None) -> tuple[str, ...]:
    """Every scheme ``auto`` would accept on this device, best first.

    ``select_transformer_quant_scheme`` returns only the winner, which is all the load needs
    until the winner turns out to have no hosted prequant AND not to fit dense. The caller then
    needs to know what auto would have picked NEXT, so it can reach a scheme that does have a
    checkpoint instead of dropping to GGUF. Same ladder, same deny list, same smoke probe as the
    auto branch of the selector, so the two can never disagree about what is allowed."""
    if not dense_transformer_supported(target):
        return ()
    device = str(getattr(target, "device", "cuda"))
    cap = _capability()
    if cap is None:
        return ()
    for floor, schemes in _AUTO_LADDER:
        if cap >= floor:
            return tuple(
                scheme
                for scheme in _prefer_consumer_scheme(schemes, device)
                if not _family_denied(family, scheme) and _scheme_supported(scheme, device)
            )
    return ()


def _prefer_consumer_scheme(schemes: tuple[str, ...], device: Any) -> tuple[str, ...]:
    """Reorder an arch tier's schemes for the GPU class. On consumer / workstation cards move
    int8 first: they halve fp8/fp16 FP32-accumulate while int8 runs full-rate, so int8 is as
    fast or faster (and the only path on pre-Ada consumer). Data-center parts keep fp8 first."""
    if TQ_INT8 in schemes and schemes[0] != TQ_INT8 and _is_consumer_gpu(device):
        return (TQ_INT8,) + tuple(s for s in schemes if s != TQ_INT8)
    return schemes


def _capability() -> Optional[tuple[int, int]]:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return (int(major), int(minor))
    except Exception:
        return None


def _scheme_supported(scheme: str, device: str) -> bool:
    """CUDA + (for fp8) the fp8 dtype + a cached quantise+matmul smoke test for ``scheme``."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if scheme == TQ_FP8 and not hasattr(torch, "float8_e4m3fn"):
            return False
    except Exception:
        return False
    return _smoke_probe(scheme, device)


def _smoke_probe(scheme: str, device: str) -> bool:
    """True iff a tiny Linear quantised with ``scheme`` runs one M=32 forward and returns
    finite values. Cached per (scheme, device). Makes ``auto`` robust to a build lacking a
    prototype kernel: it fails here and the ladder moves on, rather than crashing at the first
    real denoise step.

    Half the input is all-zero rows and the output is checked for finiteness, because a kernel
    that runs is not the same as one that is usable. torchao's ``_choose_scale_float8`` has no
    eps clamp, so a zero row yields scale 0 and NaN qdata; that is the root cause behind
    qwen-image's fp8 black frames (its txt stream emits all-zero token rows through a set of
    ``txt_mlp.net.2`` linears that changes run to run, which is why keeping a fixed subset bf16
    does not fix it). ``_make_quant_config`` floors it with ``activation_value_lb``, but only
    when the installed torchao exposes that kwarg, and without a zero row the probe passed on a
    build lacking it. Zero rows are not exotic: every Wan pipeline pads prompt embeddings with
    ``new_zeros`` out to 226 or 512, so the tail rows reaching the DiT are exactly zero.
    Measured on B200, 4096-wide Linear: 412 of 512 rows non-finite without the floor, 0 of 512
    with it. Failing here costs one ladder step down to int8 instead."""
    key = (scheme, device)
    if key in _SMOKE_CACHE:
        return _SMOKE_CACHE[key]
    ok = False
    try:
        import torch
        from torchao.quantization import quantize_

        lin = torch.nn.Linear(512, 512, bias = False).to(device = device, dtype = torch.bfloat16)
        quantize_(lin, _make_quant_config(scheme), filter_fn = make_filter_fn(0))
        # M stays 32 (scaled_mm wants 16-aligned dims); the zero rows go inside it, not after it.
        x = torch.randn(32, 512, device = device, dtype = torch.bfloat16)
        x[16:] = 0
        with torch.no_grad():
            out = lin(x)
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out).all().item())
    except Exception:
        ok = False
    _SMOKE_CACHE[key] = ok
    return ok


def _resolve_fast_accum(fast_accum: Optional[bool]) -> bool:
    """The fp8 ``use_fast_accum`` to apply. ``None`` (auto) is fast accumulate; an explicit bool
    forces it.

    This used to derive from the GPU class, on the theory that only consumer parts pay for FP32
    accumulate. The hardware side of that is right -- NVIDIA's professional whitepapers publish
    equal FP8 rates for both accumulate modes on RTX 6000 Ada (728.5 TFLOPS either way) and RTX PRO
    6000 Blackwell, against an exact halving on RTX 4090/5090 -- but the resulting default was
    still wrong on the cards it was written for, because the cost is in the cuBLAS path, not the
    published rate. Measured:

      * RTX 6000 Ada (sm_89), Z-Image-Turbo 1024x1024 x9: 6.387 s precise vs ~3.1 s fast --
        precise costs 2.05x and is slower than running the GGUF unquantised. (reported in review)
      * B200 (sm_100): the flag is a no-op -- 4096^3 ``_scaled_mm`` at 3023.8 vs 3041.8 TFLOP/s
        (0.994x) with BITWISE-identical output, and 1.213 s vs 1.230 s end to end.
      * consumer parts already resolved to fast here.

    So fast accumulate is a large win where the flag bites and free where it does not. Precise
    accumulate stays one explicit ``transformer_quant_fast_accum: false`` away."""
    return True if fast_accum is None else bool(fast_accum)


def _make_quant_config(scheme: str, fast_accum: Optional[bool] = None) -> Any:
    """The torchao dynamic-activation config for ``scheme`` (lazy imports per branch).

    ``fast_accum`` applies to fp8 only: None auto-detects by GPU class, True/False force it."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    if scheme == TQ_INT8:
        return Int8DynamicActivationInt8WeightConfig()
    if scheme == TQ_FP8:
        # Per-ROW granularity (per-token activation + per-channel weight scale) is REQUIRED: torchao defaults to per-TENSOR, where one Z-Image outlier
        # near 6.6e4 forces a tensor-wide scale that pushes normal values below fp8 resolution and the denoise collapses to noise. _smoke_probe checks
        # per-row scaled_mm, so a build without it falls to int8. fast accumulate (fp8 only) follows GPU class unless forced: consumer cards run fp8
        # ~2x faster with FP16 accumulate. activation_value_lb floors the per-row scale: an ALL-ZERO row otherwise yields scale 0, NaN qdata, black frames.
        import inspect
        from torchao.quantization import PerRow

        fp8_kwargs: dict = {"granularity": PerRow()}
        config_params = inspect.signature(Float8DynamicActivationFloat8WeightConfig).parameters
        if "activation_value_lb" in config_params:
            fp8_kwargs["activation_value_lb"] = 1e-12
        # Pin the plain-torch quantize kernel: the default AUTO switches to the MSLK kernel whenever an mslk package is importable, changing fp8 scale rounding BITWISE and breaking the prequant bit-identity invariant.
        # It is also slower compiled on B200 (an opaque extern call blocks inductor's quantize fusion), so the pin costs nothing.
        if "kernel_preference" in config_params:
            try:
                from torchao.quantization.quantize_.common.kernel_preference import (
                    KernelPreference,
                )
                fp8_kwargs["kernel_preference"] = KernelPreference.TORCH
            except Exception:  # noqa: BLE001 — enum moved: keep the library default
                pass
        try:
            from torchao.float8 import Float8MMConfig
            return Float8DynamicActivationFloat8WeightConfig(
                mm_config = Float8MMConfig(use_fast_accum = _resolve_fast_accum(fast_accum)),
                **fp8_kwargs,
            )
        except Exception:  # noqa: BLE001 — older torchao without the explicit mm knob
            return Float8DynamicActivationFloat8WeightConfig(**fp8_kwargs)
    if scheme == TQ_NVFP4:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        # Select the CUTLASS FP4 path, not the default Triton kernel (which needs MSLK): on a Blackwell box with CUTLASS FP4 but no MSLK the default fails the smoke probe and falls back to GGUF.
        try:
            return NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False)
        except TypeError:  # older torchao without the knob
            return NVFP4DynamicActivationNVFP4WeightConfig()
    if scheme == TQ_MXFP8:
        import torch
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        try:
            return MXDynamicActivationMXWeightConfig(
                activation_dtype = torch.float8_e4m3fn, weight_dtype = torch.float8_e4m3fn
            )
        except (TypeError, AttributeError):
            # TypeError: older torchao without the explicit dtype knobs. AttributeError: a torch build without torch.float8_e4m3fn.
            return MXDynamicActivationMXWeightConfig()
    raise ValueError(f"unknown transformer quant scheme '{scheme}'")


def make_filter_fn(
    min_features: int,
    exclude_name_tokens: tuple[str, ...] = (),
    *,
    require_bf16: bool = False,
    require_divisible: int = 0,
):
    """A torchao ``quantize_`` filter keeping only FLOP-heavy linears: nn.Linear with in/out
    features >= ``min_features`` AND whose fqn contains no ``exclude_name_tokens`` (int8 uses
    these to skip the M=1 projections that crash ``torch._int_mm``). Hides the callback arity.

    ``require_bf16`` also skips any non-bf16 Linear: fp8/mxfp8/nvfp4 assert a bf16 input weight,
    so one non-bf16 Linear (e.g. the fp32 layers Wan/Hunyuan DiTs keep) otherwise raises and
    aborts the ENTIRE pass, leaving the module dense. int8 tolerates non-bf16, so leaves this off.

    ``require_divisible`` (0 disables) skips any Linear whose in/out features are not multiples
    of it: the fp8/fp4 scaled_mm hardware GEMM requires 16-aligned dims and the 32-wide MX block
    scaling cannot tile a ragged dim, so one non-conforming Linear would otherwise crash the
    first real matmul AFTER the quantise pass succeeded (the smoke probe only proves an aligned
    GEMM runs). Leaving such a layer bf16 costs ~nothing."""

    def filter_fn(module: Any, fqn: str = "") -> bool:
        try:
            import torch
            if not isinstance(module, torch.nn.Linear):
                return False
        except Exception:
            return False
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            return False
        if in_features < min_features or out_features < min_features:
            return False
        if require_divisible and (
            in_features % require_divisible or out_features % require_divisible
        ):
            return False
        if exclude_name_tokens:
            name = fqn.lower() if fqn else ""
            if any(tok in name for tok in exclude_name_tokens):
                return False
        if require_bf16:
            import torch
            weight = getattr(module, "weight", None)
            if weight is None or weight.dtype != torch.bfloat16:
                return False
        return True

    return filter_fn


def quantize_transformer(
    pipe: Any,
    target: Any,
    *,
    mode: Optional[str],
    family: Optional[str] = None,
    min_features: int = DEFAULT_MIN_LINEAR_FEATURES,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
) -> Optional[str]:
    """Quantise ``pipe.transformer``'s FLOP-heavy linears in place with the arch-chosen scheme.
    Returns the scheme engaged, or None when disabled / unsupported / failed (caller loads
    GGUF). Best-effort: never raises for an unsupported environment (failure leaves it dense).

    ``fast_accum`` (fp8 only) overrides the per-GPU-class accumulate choice: None auto-detects,
    True/False force it."""
    scheme = select_transformer_quant_scheme(target, mode, family = family)
    if scheme is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    try:
        from torchao.quantization import quantize_

        # int8 skips the M=1 projections; fp8/mxfp8 assert a bf16 weight, so on a mixed-precision DiT they must skip non-bf16 ones. "lora_" keeps a
        # baked adapter's side path high precision. Runtime only: NOT part of exclude_tokens_for_scheme, whose list is baked into prequant metadata.
        exclude = exclude_tokens_for_scheme(scheme, family) + ("lora_",)
        # GEMM tiling floors per scheme: scaled_mm needs 16-aligned dims, MX block scaling 32. int8's _int_mm has no such floor and keeps the historical filter.
        divisible = {TQ_FP8: 16, TQ_NVFP4: 16, TQ_MXFP8: 32}.get(scheme, 0)
        quantize_(
            transformer,
            _make_quant_config(scheme, fast_accum = fast_accum),
            filter_fn = make_filter_fn(
                min_features,
                exclude_name_tokens = exclude,
                require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
                require_divisible = divisible,
            ),
        )
        # Runtime-only diagnostic marker.
        try:
            transformer._unsloth_runtime_quant = scheme
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        return scheme
    except Exception as exc:  # noqa: BLE001 — leave the transformer dense -> GGUF fallback
        _warn(logger, scheme, exc)
        return None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.transformer_quant: %s failed: %s", what, exc)
