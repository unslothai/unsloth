# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in low-precision quantisation of the diffusion DiT transformer.

The default path loads the transformer as a single-file GGUF, which stores weights
4-bit but DEQUANTISES to bf16 on every matmul -- so it runs at bf16 tensor-core rate
and never touches the int8 / fp8 / fp4 tensor cores. It is a memory win that costs
speed. This module is the opt-in alternative: load the DENSE bf16 transformer from the
base repo and torchao-quantise it with a DYNAMIC-ACTIVATION scheme so the matmul runs
on the low-precision tensor cores. Measured on a B200 (Z-Image-Turbo, 1024px / 8 steps)
vs the GGUF+compile default (0.802s, LPIPS 0.083 vs dense bf16): fp8 dynamic 0.585s
(1.37x), int8 dynamic 0.603s (1.33x), both at LOWER LPIPS than GGUF -- faster AND a hair
more accurate, at the cost of a higher-memory dense load. So it is strictly opt-in; the
loader keeps GGUF as the low-memory default and the fallback.

Scheme by architecture (``auto`` picks the best supported, best first):
  nvfp4 / mxfp8 - Blackwell sm_100+ FP4 / MX tensor cores (biggest win; prototype).
  fp8           - Ada / Hopper / Blackwell (sm_89+) fp8 tensor cores.
  int8          - Ampere+ (sm_80+) int8 tensor cores -- the broadest-hardware lever.

Every scheme needs ``torch.compile`` to realise the speedup (dynamic quant is ~30x
slower eager); the loader already compiles the repeated block AFTER this runs. torch /
torchao are imported lazily so the module stays importable in a no-torch runtime, and
every probe is best-effort: an unsupported scheme yields None and the caller loads GGUF.
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

# Schemes whose torchao path asserts a bf16 weight, so their quantise filter must skip
# non-bf16 Linears (see make_filter_fn's require_bf16) rather than aborting the whole pass on a
# stray fp32 Linear (e.g. T5's fp32 `wo`). Verified on torchao 0.17 / B200: fp8 per-row asserts
# "PerRow quantization only works for bfloat16 precision input weight" and mxfp8 asserts
# "Only supporting bf16 out dtype", but NVFP4's high-precision conversion quantises an fp32
# weight fine (forward included) -- so nvfp4 is deliberately NOT gated here, keeping those fp32
# projections quantised instead of leaving them dense. int8 (torch._int_mm) also quantises
# fp32/fp16 weights fine, so it is not gated either.
_REQUIRE_BF16_SCHEMES = (TQ_FP8, TQ_MXFP8)

# The fp8 weight/activation granularity the runtime config uses (see _make_quant_config):
# per-ROW is REQUIRED for correctness on outlier-heavy DiTs. Stamped into a pre-quantized
# fp8 checkpoint's metadata at build time and required by the loader, so a stale checkpoint
# baked with the old per-TENSOR layout is rejected and rebuilt rather than reproducing noise.
FP8_GRANULARITY = "per_row"

# Skip linears whose in/out features are below this. A small share of the FLOPs, so leaving
# them bf16 costs ~nothing and keeps quality a touch higher.
DEFAULT_MIN_LINEAR_FEATURES = 512

# int8-ONLY name exclusions. The int8 dynamic path uses torch._int_mm, which requires the
# activation row count M > 16. A DiT's AdaLN *modulation* projections and its timestep /
# guidance / pooled-text *conditioning embedders* are computed once from the [batch, dim]
# conditioning vector (M = batch = 1), not per token -- so they crash _int_mm even though
# their feature dims are large (e.g. Flux's norm1.linear 3072->18432, Qwen's img_mod.1, Flux.2's
# *_modulation.linear). min_features does NOT catch them (they are big), so int8 also skips any
# Linear whose fqn matches one of these tokens. They are a negligible share of the FLOPs (run at
# M=1, once per block), so int8 keeps the full speedup on the attention/FFN layers (M = seq).
# fp8 / nvfp4 / mxfp8 use scaled_mm (no M>16 limit) and quantise these layers fine, so the
# exclusion is int8-only. Sequence embedders (context_embedder / x_embedder / txt_in, M = seq)
# are deliberately NOT excluded.
_INT8_EXCLUDE_NAME_TOKENS = (
    "norm",  # AdaLN modulation .linear (norm1 / norm1_context / norm / norm_out)
    "_mod",  # Qwen img_mod / txt_mod
    "modulation",  # Flux.2 double/single_stream_modulation
    "timestep_embed",
    "guidance_embed",
    "time_text_embed",  # Flux/Qwen time_text_embed.* (pooled-text + timestep); NOT context_embedder
    "pooled",
    # Krea 2's Krea2TimestepEmbedding ("time_embed.linear_2", 6144->6144 at M = batch);
    # its other M=1 projection ("time_mod_proj") is already caught by "_mod", and
    # img_in / final_layer.linear / text_fusion.projector fall under min_features.
    "time_embed",
)


# fp8 (and the other per-row scaled_mm schemes) PER-FAMILY name exclusions. Per-row fp8 scales
# each activation ROW by row_amax / 448; a row whose amax is 0 -- a PADDING token in a
# zero-padded conditioning sequence -- yields scale 0 and x / 0 = inf, collapsing the DiT to
# black frames (measured on B200 with the production torch._scaled_mm path via
# scripts/fp8_layer_ablation.py). The fix is to keep the specific Linear that first consumes the
# raw zero-padded sequence in bf16; its bias makes every downstream row non-zero, so the rest of
# the DiT is fp8-clean. This is family-specific because the offending layer's name is:
#   wan2.2 (WanTransformer3DModel): condition_embedder.text_embedder reads the raw 512-token text
#     (~all padding for a short prompt). Excluding the whole condition_embedder (a handful of
#     one-off M=1/M=seq embedders, negligible FLOPs + memory) leaves the entire 30-block
#     attn1/attn2/ffn stack on fp8 -- measured fp8-except-condition_embedder cosine 0.99975 vs bf16,
#     0 non-finite, vs fp8-everywhere = 100% non-finite. int8 (per-token; a zero row rounds to 0,
#     no divide) needs no exclusion, which is why it was always clean and is the fallback.
# HunyuanVideo-1.5 is deliberately absent: its MMDiT masks the padding text tokens to zero INSIDE
# every block, so the per-block context stream (add_*_proj / to_add_out / ff_context) regenerates
# zero rows in every layer -- no small exclude set exists, so it stays fp8-denied -> int8 (see
# _FAMILY_SCHEME_DENY). Applies to every per-row scaled_mm scheme (fp8 today; mxfp8 / nvfp4 inherit
# it if ever un-denied for these families), never to int8.
_FP8_FAMILY_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "wan2.2-ti2v-5b": ("condition_embedder",),
    "wan2.2-t2v-a14b": ("condition_embedder",),  # same DiT class + padded-text conditioning
}


def exclude_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    """Name tokens to exclude from quantisation for ``scheme`` (optionally family-specific).

    int8 (torch._int_mm, M>16) skips the M=1 modulation / conditioning-embedder projections (see
    _INT8_EXCLUDE_NAME_TOKENS) on every family. The per-row scaled_mm schemes (fp8 / mxfp8 / nvfp4)
    exclude nothing by default, but on families whose zero-padded conditioning sequence would divide
    by a zero row scale they skip the offending input embedder (see _FP8_FAMILY_EXCLUDE_NAME_TOKENS).
    ``family=None`` preserves the historical behaviour (int8 tokens, or () otherwise). Shared by the
    runtime quantise path and the offline prequant-checkpoint builder so the two never drift -- a
    checkpoint built offline must skip exactly the layers the runtime path skips, or it bakes a layer
    that then crashes (int8 M=1 -> _int_mm) or infs (fp8 padding row -> scaled_mm) at the first
    denoise step."""
    if scheme == TQ_INT8:
        return _INT8_EXCLUDE_NAME_TOKENS
    return _FP8_FAMILY_EXCLUDE_NAME_TOKENS.get(str(family or "").strip().lower(), ())


# Per-architecture preference order for ``auto`` -- best (fastest, in-bar) first, with
# the lower-precision schemes listed as fallbacks for that arch tier. On Blackwell, fp8
# leads: measured on a B200, plain fp8 dynamic is both faster AND more accurate than the
# alternatives for the DiT's shapes. mxfp8's block scaling adds overhead without a speed
# win, so it sits below fp8. nvfp4 is intentionally DISABLED in the auto ladder (opt-in only;
# see the TODO on the Blackwell tier below): the FP4 tensor-core
# GEMM is real once torch>=2.11 + torchao's CUTLASS FP4 kernel is present (verified: a
# 16384^3 GEMM hits ~3826 TFLOPS, 1.37x fp8), but it only beats fp8 on very large GEMMs.
# At the DiT's actual shapes (hidden ~3072, MLP ~12288, M~4096) it is *slower* than fp8
# (0.81x end-to-end on Z-Image 1024px) AND notably less accurate (LPIPS 0.166 vs fp8's
# 0.044), because FP4's per-forward quant overhead is not amortised and the format is
# coarser. So nvfp4 is kept as an explicit opt-in, never the auto pick for diffusion.
#
# This order is the DATA-CENTER preference. On a consumer / workstation GPU it is reordered
# to put int8 first (see ``_prefer_consumer_scheme``): consumer cards halve fp8/fp16 FP32-
# accumulate throughput, while int8 runs full-rate (int32 accumulate is not nerfed), so int8
# is as fast or faster than fp8 on every consumer NVIDIA / AMD / Intel part.
_AUTO_LADDER: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = (
    # TODO(nvfp4): NVFP4 is disabled in the auto ladder for now. Re-enable it by restoring the
    # commented line below (nvfp4 between fp8 and mxfp8) once the FP4 tensor-core GEMM wins at
    # the DiT's real shapes (hidden ~3072, MLP ~12288, M~4096); today it is slower (0.81x
    # end-to-end on Z-Image 1024px) AND less accurate (LPIPS 0.166 vs fp8's 0.044). nvfp4 stays
    # an explicit opt-in (transformer_quant="nvfp4"); a future MSLK-equipped box may flip the
    # tradeoff. See commit 3a21f12500.
    ((10, 0), (TQ_FP8, TQ_MXFP8, TQ_INT8)),  # Blackwell sm_100+ (nvfp4 disabled; see TODO)
    # ((10, 0), (TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8)),  # restore to re-enable nvfp4 in auto
    ((8, 9), (TQ_FP8, TQ_INT8)),  # Ada sm_89 / Hopper sm_90
    ((8, 0), (TQ_INT8,)),  # Ampere sm_80 / sm_86
)

# Families whose activation ranges break specific dense-quant schemes at the MODEL
# level. The kernel smoke probe below cannot see this (it only proves the GEMM runs);
# these were measured with the 28-pair prequant accuracy gate on a B200
# (scripts/prequant_accuracy_gate.py) and reproduced with on-the-fly quantisation:
#   qwen-image + fp8   -> every frame black (mean luma 0.0000, SSIM 0.016 vs bf16). The
#                         same per-row fp8 that matches bf16 on Z-Image / FLUX: Qwen's
#                         activation outliers exceed even per-row fp8's dynamic range.
#   qwen-image + mxfp8 -> real semantic damage at 1024px (CLIP delta mean 0.0146, worst
#                         cases 0.064 / 0.102 -- 2x the per-case bound).
#   qwen-image + nvfp4 -> LPIPS mean 0.51 vs bf16: unusable.
#   wan2.2 / hunyuanvideo-1.5 + fp8 -> every frame black (mean luma 0.0000, LPIPS ~0.80-0.82
#                         vs bf16), reproduced on B200 with the production torch._scaled_mm
#                         per-row fp8 path (no MSLK): the DiT's activation outliers exceed
#                         per-row fp8's range, the same failure mode as qwen-image. int8
#                         dynamic (per-token) is clean on both (non-black, correct contrast;
#                         First-Block-Cache engages normally instead of over-caching the
#                         degenerate black activations).
# NOTE this is NOT universal across video: ltx-2 + fp8 renders clean (mean 153.7, non-black) on
# the same stack, so it is deliberately NOT denied -- the deny is measured per family, not a
# blanket video rule. int8 dynamic (per-token) is excellent on Qwen (LPIPS mean 0.069 / SSIM
# 0.958) and clean on Wan / Hunyuan, so the auto ladder falls through to it there. mxfp8 / nvfp4
# are denied alongside fp8 on the black-frame families conservatively (the same per-block
# scaled_mm family as the confirmed-black fp8, and mxfp8 is a prototype) so auto lands on the
# battle-tested int8; they can be re-enabled per family once separately validated in-bar, like
# the nvfp4 auto-ladder TODO. The deny also applies to an EXPLICIT request: a scheme that renders
# black frames has no legitimate use, and returning None gives the caller the same fallback
# contract as an unsupported scheme (GGUF build).
_FAMILY_SCHEME_DENY: dict[str, frozenset[str]] = {
    "qwen-image": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),
    "qwen-image-edit": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),  # same DiT + activations
    # Wan2.2 video DiTs (WanTransformer3DModel): plain fp8 rendered black frames, but the failure
    # is a SINGLE input embedder (condition_embedder) dividing by a zero padding-token row -- the
    # 30-block compute stack is fp8-clean. fp8 is therefore allowed here and made safe by the
    # per-family exclude (_FP8_FAMILY_EXCLUDE_NAME_TOKENS keeps condition_embedder bf16). mxfp8 /
    # nvfp4 remain denied (same per-row scaled_mm family, not separately validated) so auto still
    # skips them; both the 5B TI2V and the A14B MoE share the DiT class + activation profile.
    "wan2.2-ti2v-5b": frozenset({TQ_MXFP8, TQ_NVFP4}),
    "wan2.2-t2v-a14b": frozenset({TQ_MXFP8, TQ_NVFP4}),
    # HunyuanVideo-1.5 DiT (HunyuanVideo15Transformer3DModel): fp8 renders black frames (measured,
    # LPIPS 0.82) and -- unlike Wan -- the failure is NOT confinable to an input embedder: its MMDiT
    # masks padding text tokens to zero inside every block, so the per-block context stream
    # (add_*_proj / to_add_out / ff_context) regenerates zero rows layer after layer (measured: fp8
    # on only the main blocks is still 100% non-finite). No small exclude set exists, so fp8 stays
    # denied and auto lands on int8 (clean, per-token). The 480p and 720p repacks share the DiT +
    # activations, so both deny. (ltx-2 fp8 measures clean on the same stack, so it is absent.)
    "hunyuanvideo-1.5": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),
    "hunyuanvideo-1.5-720p": frozenset({TQ_FP8, TQ_MXFP8, TQ_NVFP4}),
}


def _family_denied(family, scheme: str) -> bool:
    return scheme in _FAMILY_SCHEME_DENY.get(str(family or "").strip().lower(), ())


# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}

# Data-center GPU model tokens (un-nerfed FP32 accumulate). Matched as whole tokens of
# torch.cuda.get_device_name(), so the workstation "A4000" is not mistaken for the
# data-center "A40". Anything not here -- GeForce, workstation RTX, or an unknown name --
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


# Professional parts the rest of the backend treats as datacenter-class (see llama_cpp.py
# _DATACENTER_GPU_RE, which applies the same FP32-accum tuning to them). Matched as phrases
# because the marker spans tokens ("RTX PRO 6000", "RTX 6000 ADA"), so they must not be
# misread as consumer (which would put int8 ahead of fp8 and pick fast accumulate).
_PROFESSIONAL_GPU_MARKERS = ("RTX PRO 6000", "RTX 6000 ADA")


def _is_consumer_gpu(device: Any = None) -> bool:
    """Whether the active GPU is consumer-class (GDDR), where fp8 FP32 accumulate is
    throughput-halved so fast (FP16) accumulate is a ~2x win. Data-center HBM parts and
    professional parts (recognised by name) are not nerfed and return False, so they keep
    the higher-precision default accumulate and fp8 first. Heuristic on the device name: a
    GeForce / TITAN name is always consumer; a recognised data-center token or professional
    marker is not; anything else (unknown) defaults to consumer -- the safe choice, since
    fast accumulate is free on data-center and a win on consumer. Best-effort: True on any
    probe failure."""
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
    """Whether the dense-source quant path is usable for ``target``: a CUDA device with
    a bf16 compute dtype (the only configuration any torchao dynamic scheme accelerates).
    A cheap pre-check the loader runs before loading the (large) dense transformer."""
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

    ``auto`` walks the per-arch ladder and returns the first scheme that passes a real
    quantise+matmul smoke test, so on a box where the Blackwell fp4 / mx kernels are
    unavailable it lands on fp8 / int8 with no error. An explicit scheme is honored only
    if supported (else None -> GGUF), never silently swapped for a different one.
    ``family`` additionally applies the measured model-level deny list
    (``_FAMILY_SCHEME_DENY``): schemes that produce black frames or out-of-bar drift on
    that family are skipped by ``auto`` and refused when explicit."""
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


def is_int8_memory_fallback(target: Any, family: Optional[str] = None) -> bool:
    """Whether AUTO DiT quant would land on int8 only as a black-frame / denied FALLBACK on a
    data-center, fp8-capable GPU -- i.e. fp8 would be the arch's pick but is denied/unavailable for
    this family (e.g. HunyuanVideo-1.5). In that case int8 is a MEMORY lever, not a speed win:
    measured on a B200 it is ~7% slower AND less accurate than the dense bf16 + regional-compile
    path, so the loader prefers dense when the DiT fits resident (quantising only when memory-
    constrained). Returns False where int8 is a legitimate accelerator and must be kept:
      - consumer / workstation GPUs (fp8 FP32-accumulate is halved, so int8 is as fast or faster);
      - pre-Ada data-center (sm < 8.9) with no fp8 tensor cores at all (int8 is the only quant);
      - families whose auto scheme is fp8 (Wan / LTX) -- not int8, so nothing to prefer away from.
    Best-effort: any probe failure returns False (keep today's behaviour)."""
    try:
        if _is_consumer_gpu(getattr(target, "device", None)):
            return False
        cap = _capability()
        if cap is None or cap < (8, 9):  # no fp8 tensor cores -> int8 is the genuine accelerator
            return False
        return select_transformer_quant_scheme(target, TQ_AUTO, family = family) == TQ_INT8
    except Exception:  # noqa: BLE001 — optimisation gate only; never break the load
        return False


def _prefer_consumer_scheme(schemes: tuple[str, ...], device: Any) -> tuple[str, ...]:
    """Reorder an arch tier's schemes for the GPU class. On a consumer / workstation card
    move int8 to the front: consumer parts halve fp8/fp16 FP32-accumulate throughput, while
    int8 runs at full rate (int32 accumulate is not nerfed), so int8 is as fast or faster
    than fp8 on every consumer NVIDIA / AMD / Intel GPU (and the only path on pre-Ada
    consumer without fp8 tensor cores). Data-center HBM parts keep fp8 first."""
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
    """True iff a tiny Linear quantised with ``scheme`` runs one M=32 forward without
    error. Cached per (scheme, device). This is what makes ``auto`` robust to a torch /
    torchao build where a prototype (nvfp4 / mxfp8) kernel is unavailable: it fails here
    and the ladder moves on, rather than crashing at the first real denoise step."""
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
    """The torchao dynamic-activation config for ``scheme`` (lazy import; prototype
    import for the Blackwell fp4 / mx schemes is inside the branch that needs it).

    ``fast_accum`` applies to fp8 only: None auto-detects by GPU class, True/False force it."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    if scheme == TQ_INT8:
        return Int8DynamicActivationInt8WeightConfig()
    if scheme == TQ_FP8:
        # Per-ROW granularity (per-token activation scale + per-output-channel weight scale)
        # is REQUIRED for correctness, not just quality. torchao's default fp8 granularity is
        # per-TENSOR: one scale for the whole activation tensor. A DiT with extreme activation
        # outliers breaks under that -- z-image's MLP activations peak near 6.6e4, and a single
        # such outlier forces a tensor-wide scale that pushes every normal value (~1-30) below
        # the fp8 resolution, so they quantise to ~0 and the denoise collapses to pure noise
        # (measured on B200: per-tensor fp8 = noise, per-row fp8 = matches bf16). Per-row
        # confines each outlier to its own token/channel. This is also why int8 dynamic was
        # always fine here: it is per-token by default. The per-row scaled_mm is probed by
        # _smoke_probe, so an arch/build without it falls through the ladder to int8.
        #
        # fast accumulate (fp8 only) is chosen by GPU class unless forced: consumer / workstation
        # cards (GDDR) run the fp8 tensor cores ~2x faster with FP16 (fast) accumulate than FP32
        # (e.g. ~838 vs ~419 TFLOPS on RTX 50xx); data-center HBM parts keep precise accumulate.
        from torchao.quantization import PerRow
        try:
            from torchao.float8 import Float8MMConfig
            return Float8DynamicActivationFloat8WeightConfig(
                granularity = PerRow(),
                mm_config = Float8MMConfig(use_fast_accum = _resolve_fast_accum(fast_accum)),
            )
        except Exception:  # noqa: BLE001 — older torchao without the explicit mm knob
            return Float8DynamicActivationFloat8WeightConfig(granularity = PerRow())
    if scheme == TQ_NVFP4:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        # Select the CUTLASS FP4 path, not the default Triton kernel: torchao defaults
        # use_triton_kernel=True, which needs MSLK installed. On a Blackwell box with the
        # CUTLASS FP4 extension but no MSLK, the default would make the smoke probe fail
        # and silently fall back to GGUF instead of using the FP4 tensor cores.
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
    """A torchao ``quantize_`` filter keeping only the FLOP-heavy linears: nn.Linear with both
    in/out features >= ``min_features`` AND whose fully-qualified name contains none of
    ``exclude_name_tokens`` (used by int8 to skip the M=1 modulation / conditioning-embedder
    projections that crash ``torch._int_mm``). Hides the (module, fqn) callback arity.

    ``require_bf16`` additionally skips any Linear whose weight is not bfloat16. The scaled_mm
    schemes (fp8 / mxfp8 / nvfp4) assert a bf16 input weight, so a single non-bf16 Linear -- e.g.
    the fp32 layers Wan / Hunyuan video DiTs keep for numerical stability -- otherwise raises inside
    ``quantize_`` and aborts the ENTIRE pass, leaving the module silently dense. Gating those layers
    out lets the scheme engage on the bf16 linears and skip the fp32 ones. int8 (torch._int_mm)
    tolerates non-bf16 weights, so it leaves this off and keeps quantising them."""

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
    """Quantise ``pipe.transformer``'s FLOP-heavy linears in place with the arch-chosen
    dynamic scheme. Returns the scheme actually engaged, or None when disabled /
    unsupported / failed -- the caller then loads GGUF instead. Best-effort: it never
    raises for an ordinary unsupported environment (a failure leaves the module dense).

    ``fast_accum`` (fp8 only) overrides the per-GPU-class accumulate choice: None
    auto-detects (fast on consumer, precise on data-center), True/False force it."""
    scheme = select_transformer_quant_scheme(target, mode, family = family)
    if scheme is None:
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    try:
        from torchao.quantization import quantize_

        # int8 (torch._int_mm, M>16) skips the M=1 modulation / conditioning-embedder projections;
        # fp8 / fp4 / mx (scaled_mm) have no M limit but, on a family whose zero-padded conditioning
        # sequence would divide a per-row activation scale by a zero row, skip that one input
        # embedder (Wan's condition_embedder -> _FP8_FAMILY_EXCLUDE_NAME_TOKENS). fp8 and mxfp8 also
        # assert a bf16 weight, so on a mixed-precision DiT (Wan / Hunyuan keep some fp32 linears)
        # they must skip the non-bf16 ones or the whole pass raises and no-ops. nvfp4 quantises fp32
        # weights fine, so it is not gated (see _REQUIRE_BF16_SCHEMES).
        exclude = exclude_tokens_for_scheme(scheme, family)
        quantize_(
            transformer,
            _make_quant_config(scheme, fast_accum = fast_accum),
            filter_fn = make_filter_fn(
                min_features,
                exclude_name_tokens = exclude,
                require_bf16 = scheme in _REQUIRE_BF16_SCHEMES,
            ),
        )
        # Runtime-only marker (torchao tensors are not safetensors-serializable; this
        # backend is inference-only, so this is purely diagnostic).
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
