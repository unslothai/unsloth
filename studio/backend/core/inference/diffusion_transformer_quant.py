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

# Schemes whose torchao path asserts a bf16 weight, so their filter must skip non-bf16
# Linears (make_filter_fn's require_bf16) rather than aborting the whole pass on a stray fp32
# Linear (e.g. T5's `wo`). Verified on torchao 0.17 / B200: fp8 per-row and mxfp8 assert bf16;
# nvfp4 and int8 quantise fp32/fp16 fine, so they are not gated (keeps those projections quant).
_REQUIRE_BF16_SCHEMES = (TQ_FP8, TQ_MXFP8)

# fp8 granularity the runtime uses: per-ROW is REQUIRED for correctness on outlier-heavy DiTs.
# Stamped into a pre-quant checkpoint's metadata and required by the loader, so a stale
# per-TENSOR checkpoint is rejected and rebuilt rather than reproducing noise.
FP8_GRANULARITY = "per_row"

# Skip linears below this feature size: a small FLOP share, so leaving them bf16 costs ~nothing.
DEFAULT_MIN_LINEAR_FEATURES = 512

# int8-ONLY name exclusions. int8 uses torch._int_mm, which needs activation rows M > 16. A
# DiT's AdaLN modulation projections and timestep / guidance / pooled-text conditioning
# embedders run once from the [batch, dim] vector (M = batch = 1), not per token, so they crash
# _int_mm despite large feature dims (Flux norm1.linear 3072->18432, Qwen img_mod.1, Flux.2
# *_modulation.linear); min_features misses them (they are big), so int8 also skips any Linear
# whose fqn matches a token here. Negligible FLOPs (M=1, once per block), so int8 keeps the full
# speedup on attention/FFN (M = seq). fp8/nvfp4/mxfp8 use scaled_mm (no M limit) and quantise
# these fine, so the exclusion is int8-only. Sequence embedders (M = seq) are NOT excluded.
_INT8_EXCLUDE_NAME_TOKENS = (
    "norm",  # AdaLN modulation .linear
    "_mod",  # Qwen img_mod / txt_mod
    "modulation",  # Flux.2 double/single_stream_modulation
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",  # Flux/Qwen (pooled-text + timestep); NOT context_embedder
    "pooled",
    # Krea 2's time_embed.linear_2 (6144->6144, M = batch); its time_mod_proj is caught by
    # "_mod", and img_in / final_layer.linear / text_fusion.projector fall under min_features.
    "time_embed",
)


# int8 PER-FAMILY name exclusions, on top of _INT8_EXCLUDE_NAME_TOKENS. Qwen-Image's MMDiT
# runs every TEXT-stream Linear at M = actual prompt tokens (the Qwen2.5-VL embeds are not
# padded to a fixed length like FLUX's 512-token T5), so a short prompt ("Cute sloth writing
# on a paper" = 13 tokens, or the near-empty negative prompt) drives torch._int_mm below its
# M > 16 floor and the denoise crashes (measured on B200: "self.size(0) needs to be greater
# than 16, but got 13"). Keep the text stream bf16: it runs at M = tens vs the image stream's
# M ~ 4k+, so the exclusion costs ~nothing and the image stream keeps full int8 coverage.
# txt_mod is already covered by "_mod" in the base list; txt_in is the context embedder.
_QWENIMAGE_INT8_EXCLUDES = (
    "txt_in",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "txt_mlp",
)
_INT8_FAMILY_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "qwen-image": _QWENIMAGE_INT8_EXCLUDES,
    "qwen-image-edit": _QWENIMAGE_INT8_EXCLUDES,  # same DiT class + unpadded text stream
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


# Per-arch preference for ``auto`` -- best first, lower-precision schemes as fallbacks. On
# Blackwell fp8 leads: measured on B200, plain fp8 dynamic is faster AND more accurate than the
# alternatives at the DiT's shapes. mxfp8's block scaling adds overhead with no speed win.
# nvfp4 is also below fp8: the FP4 GEMM is real with torch>=2.11 + torchao's CUTLASS kernel
# (16384^3 GEMM ~3826 TFLOPS, 1.37x fp8) but only beats fp8 on very large GEMMs; at the DiT's
# shapes (hidden ~3072, MLP ~12288, M~4096) it is slower (0.81x on Z-Image 1024px) and less
# accurate (LPIPS 0.166 vs fp8 0.044), so it stays an explicit opt-in, never the auto pick.
#
# DATA-CENTER order. On a consumer / workstation GPU int8 moves first (_prefer_consumer_scheme):
# consumer cards halve fp8/fp16 FP32-accumulate, while int8 (int32 accumulate) runs full-rate.
_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    ((10, 0), (TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8)),  # Blackwell sm_100+
    ((8, 9), (TQ_FP8, TQ_INT8)),  # Ada sm_89 / Hopper sm_90
    ((8, 0), (TQ_INT8,)),  # Ampere sm_80 / sm_86
)

# Families whose activation ranges break specific schemes at the MODEL level (the smoke probe
# only proves the GEMM runs). Measured with the 28-pair prequant accuracy gate on B200:
#   qwen-image + fp8   -> every frame black (luma 0.0000, SSIM 0.016): Qwen's outliers exceed
#                         even per-row fp8's range (the same fp8 matches bf16 on Z-Image/FLUX).
#   qwen-image + mxfp8 -> semantic damage at 1024px (CLIP delta mean 0.0146, worst 0.064/0.102).
#   qwen-image + nvfp4 -> LPIPS mean 0.51 vs bf16: unusable.
# int8 dynamic is excellent on Qwen (LPIPS 0.069 / SSIM 0.958), so auto falls through to it. The
# deny also applies to an EXPLICIT request (returning None gives the same GGUF fallback).
_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "qwen-image": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),
    "qwen-image-edit": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),  # same DiT
}


def _family_denied(family, scheme: str) -> bool:
    return scheme in _FAMILY_SCHEME_DENY.get(str(family or "").strip().lower(), ())


# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}

# Data-center GPU tokens (un-nerfed FP32 accumulate). Matched as whole tokens of
# get_device_name() so workstation "A4000" isn't mistaken for data-center "A40". Anything else
# is treated as consumer-class (FP32-accumulate halved). See developer.nvidia.com/cuda/gpus.
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


# Professional parts the backend treats as datacenter-class (llama_cpp.py _DATACENTER_GPU_RE).
# Matched as phrases since the marker spans tokens, so they aren't misread as consumer.
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
    """True iff a tiny Linear quantised with ``scheme`` runs one M=32 forward. Cached per
    (scheme, device). Makes ``auto`` robust to a build lacking a prototype kernel: it fails
    here and the ladder moves on, rather than crashing at the first real denoise step."""
    key = (scheme, device)
    if key in _SMOKE_CACHE:
        return _SMOKE_CACHE[key]
    ok = False
    try:
        import torch
        from torchao.quantization import quantize_

        lin = torch.nn.Linear(512, 512, bias = False).to(device = device, dtype = torch.bfloat16)
        quantize_(lin, _make_quant_config(scheme), filter_fn = make_filter_fn(0))
        x = torch.randn(32, 512, device = device, dtype = torch.bfloat16)
        with torch.no_grad():
            lin(x)
        torch.cuda.synchronize()
        ok = True
    except Exception:
        ok = False
    _SMOKE_CACHE[key] = ok
    return ok


def _resolve_fast_accum(fast_accum: Optional[bool]) -> bool:
    """The fp8 ``use_fast_accum`` to apply. ``None`` auto-detects by GPU class
    (consumer / workstation -> fast; data-center -> precise); an explicit bool forces it."""
    return _is_consumer_gpu() if fast_accum is None else bool(fast_accum)


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
        # Per-ROW granularity (per-token activation + per-channel weight scale) is REQUIRED for
        # correctness. torchao defaults to per-TENSOR (one scale): a DiT with extreme outliers
        # breaks -- Z-Image MLP activations peak near 6.6e4, so one outlier forces a tensor-wide
        # scale that pushes normal values (~1-30) below fp8 resolution to ~0 and the denoise
        # collapses to noise (B200: per-tensor = noise, per-row = matches bf16). Per-row confines
        # each outlier to its token/channel (also why int8, per-token by default, was always
        # fine). _smoke_probe checks per-row scaled_mm, so a build without it falls to int8.
        #
        # fast accumulate (fp8 only) is chosen by GPU class unless forced: consumer cards run fp8
        # ~2x faster with FP16 accumulate than FP32 (~838 vs ~419 TFLOPS on RTX 50xx); data-center
        # keeps precise accumulate.
        #
        # activation_value_lb floors the dynamic per-row activation scale: an ALL-ZERO token row
        # otherwise yields scale 0 -> NaN qdata -> black frames on torchao's plain-torch kernel
        # path (fused fbgemm/mslk quantize kernels clamp internally, which masks the bug on boxes
        # that have them). Zero rows are real: Wan 2.2 zero-pads its text conditioning and
        # Hunyuan-1.5 / Qwen-Image regenerate zero rows inside their blocks. Weight scales are
        # untouched (weights are never all-zero rows in practice and the floor is 1e-12), so
        # pre-quantized fp8 checkpoints stay valid. The knob exists since the Float8Tensor rework
        # (torchao >= 0.13); older versions keep today's behaviour via the signature check.
        import inspect
        from torchao.quantization import PerRow
        fp8_kwargs: dict = {"granularity": PerRow()}
        config_params = inspect.signature(Float8DynamicActivationFloat8WeightConfig).parameters
        if "activation_value_lb" in config_params:
            fp8_kwargs["activation_value_lb"] = 1e-12
        # Pin the plain-torch quantize kernel. The default AUTO silently switches to the MSLK
        # kernel whenever an mslk package is importable (sm90+), which changes fp8 scale
        # rounding BITWISE (measured: 8/8 FLUX matrices differ, scales ~55% of bytes) -- so a
        # box that merely gains mslk would break the hosted-prequant bit-identity invariant.
        # Measured on B200 the mslk path is also SLOWER compiled (opaque extern call blocks
        # inductor's quantize fusion: FLUX.1 fp8 e2e 1.149 -> 1.624 s), so the pin costs nothing.
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

        # Select the CUTLASS FP4 path, not the default Triton kernel (use_triton_kernel=True
        # needs MSLK): on a Blackwell box with CUTLASS FP4 but no MSLK, the default fails the
        # smoke probe and falls back to GGUF instead of using the FP4 tensor cores.
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
            # TypeError: older torchao without the explicit dtype knobs.
            # AttributeError: a torch build without torch.float8_e4m3fn.
            return MXDynamicActivationMXWeightConfig()
    raise ValueError(f"unknown transformer quant scheme '{scheme}'")


def make_filter_fn(
    min_features: int,
    exclude_name_tokens: tuple[str, ...] = (),
    *,
    require_bf16: bool = False,
):
    """A torchao ``quantize_`` filter keeping only FLOP-heavy linears: nn.Linear with in/out
    features >= ``min_features`` AND whose fqn contains no ``exclude_name_tokens`` (int8 uses
    these to skip the M=1 projections that crash ``torch._int_mm``). Hides the callback arity.

    ``require_bf16`` also skips any non-bf16 Linear: fp8/mxfp8/nvfp4 assert a bf16 input weight,
    so one non-bf16 Linear (e.g. the fp32 layers Wan/Hunyuan DiTs keep) otherwise raises and
    aborts the ENTIRE pass, leaving the module dense. int8 tolerates non-bf16, so leaves this off."""

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

        # int8 skips the M=1 projections; scaled_mm schemes have no M limit but fp8/mxfp8 assert
        # a bf16 weight, so on a mixed-precision DiT (Wan/Hunyuan) they must skip non-bf16 ones or
        # the pass raises. nvfp4 quantises fp32 fine, so it is not gated (see _REQUIRE_BF16_SCHEMES).
        # "lora_" keeps a baked adapter's side path (lora_A/lora_B/lora_embedding) high
        # precision when adapters were attached before this pass; the tiny ranks usually fall
        # under min_features anyway, but an explicit token does not depend on the rank. Runtime
        # only: NOT part of exclude_tokens_for_scheme, whose list is baked into prequant
        # checkpoint metadata (adding it there would reject every existing checkpoint).
        exclude = exclude_tokens_for_scheme(scheme, family) + ("lora_",)
        quantize_(
            transformer,
            _make_quant_config(scheme, fast_accum = fast_accum),
            filter_fn = make_filter_fn(
                min_features,
                exclude_name_tokens = exclude,
                require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
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
