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

import re as _re
import sys as _sys
import threading as _threading
from typing import Any, Optional

# stdlib-only module (no torch), so this stays inside the "imported lazily" promise above.
from core._torchao_stub import is_stubbed, torch_is_rocm

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
# MiniMax-H3 is the same small-M story with one extra trap. Its adaLN projection is named
# ``adaln_proj``, which no token in _INT8_EXCLUDE_NAME_TOKENS matches: "norm" is the closest and it
# does not appear in the name. On the DENSE checkpoint that projection is Linear(2688 -> 96768), so
# it clears min_features = 512 and gets quantized, then runs at M = 1 and raises
# ``self.size(0) needs to be greater than 16, but got 1`` at the first denoise. Measured: the
# offline builder happily bakes it, and torch.compile then dies on that module.
#
# The pruned-modulation form hides the bug rather than fixing it: there adaln_proj is
# Linear(8 -> 96768), which falls under min_features and is skipped for free (verified on the
# fl2va_pruned build: the filter rejects all 51 of them for min_features, in_features = 8). So the
# exclusion is what makes the DENSE path correct, and is a no-op on the pruned one.
#
# H3's OTHER small-M linears -- context_embedder and the two token_refiner blocks, 13 in total --
# used to be excluded here for the same reason. They are not any more: they are padded instead,
# see _INT8_FAMILY_PAD_NAME_TOKENS below.
_MINIMAX_H3_INT8_EXCLUDES = ("adaln_proj",)
# LTX-2 is audiovisual: every block carries an audio-side stream (audio_attn1 / audio_attn2 /
# audio_ff, fed by audio_proj_in) plus the a2v / v2a cross attentions. A video-only run feeds a
# MINIMAL audio stream -- one token for a still -- so those linears run at M = 1, under the same
# _int_mm floor of 16 as the cases above. The audio stream is deliberately inert on a video-only
# run (isolated modalities, out of the loss, out of the LoRA targets), so leaving it in bf16 costs
# nothing while the video-stream linears keep full int8 coverage. One token covers every audio-side
# name, including video_to_audio_attn, whose keys/values are that same one-token stream.
#
# The audio token alone is not enough. LTX-2 also carries LTX2AdaLayerNormSingle projections whose
# names say nothing about audio -- av_cross_attn_video_scale_shift / av_cross_attn_video_a2v_gate
# (the cross-modality modulation, computed unconditionally in forward, isolate_modalities or not)
# and prompt_adaln / audio_prompt_adaln. Each is a Linear over a BATCH-sized input, so M = batch =
# 1 for a default training run: the same _int_mm floor, reached whatever the audio stream does.
_LTX2_INT8_EXCLUDES = ("audio", "av_cross_attn", "adaln")
_INT8_FAMILY_EXCLUDE_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "qwen-image": _QWENIMAGE_INT8_EXCLUDES,
    "qwen-image-edit": _QWENIMAGE_INT8_EXCLUDES,  # same DiT class + unpadded text stream
    "hunyuanvideo-1.5": _HUNYUAN15_INT8_EXCLUDES,
    "hunyuanvideo-1.5-720p": _HUNYUAN15_INT8_EXCLUDES,
    "minimax-h3": _MINIMAX_H3_INT8_EXCLUDES,
    "ltx-2": _LTX2_INT8_EXCLUDES,
    "ltx-2.3": _LTX2_INT8_EXCLUDES,  # same audiovisual DiT
}


# int8 PER-FAMILY row PADDING, the alternative to excluding a small-M Linear. ``_int_mm``'s floor is
# a GEMM shape constraint, not a numerical one: padding the activation up to 32 rows, running the
# matmul and slicing the result back returns the caller's rows BITWISE unchanged (torchao's int8
# dynamic path scales activations per row, so the pad rows cannot reach them -- see
# diffusion_quant_pad for why that is asserted rather than assumed). So a Linear that would have
# been left dense can be quantized after all, and its weights halve.
#
# On MiniMax-H3 that is context_embedder plus the two token_refiner blocks: 13 Linears, 798 M
# parameters, 0.80 GB of bf16 weights, 3.9% of the whole int8 checkpoint. Measured on B200,
# 640x384 x 124 frames (see h3_quant/README_int8.md for the paired numbers).
#
# Scoped to minimax-h3 deliberately. qwen-image, qwen-image-edit and hunyuanvideo-1.5 have the same
# small-M shape and could adopt this, but each has a PUBLISHED int8 prequant checkpoint whose
# metadata bakes the current exclusion set, and _validate_checkpoint compares that set against
# exclude_tokens_for_scheme: changing it invalidates those artifacts, so they move only together
# with a rebuild and republish of each.
_MINIMAX_H3_INT8_PAD_TOKENS = ("context_embedder", "token_refiner")
_INT8_FAMILY_PAD_NAME_TOKENS: dict[str, tuple[str, ...]] = {
    "minimax-h3": _MINIMAX_H3_INT8_PAD_TOKENS,
}


def pad_tokens_for_scheme(scheme: str, family: Optional[str] = None) -> tuple[str, ...]:
    """Name tokens whose quantized Linears get their activation rows padded to ``_int_mm``'s
    floor rather than being left dense. int8 only (no other scheme has a row floor), and
    per-family. Disjoint from ``exclude_tokens_for_scheme`` by construction: an excluded Linear
    is never quantized, so there is nothing to pad."""
    if scheme != TQ_INT8:
        return ()
    return _INT8_FAMILY_PAD_NAME_TOKENS.get(str(family or "").strip().lower(), ())


def apply_small_m_padding(
    transformer: Any,
    scheme: str,
    family: Optional[str] = None,
    *,
    logger: Any = None,
) -> tuple[str, ...]:
    """Wrap this family's small-M quantized Linears so their GEMM clears ``_int_mm``'s row floor.

    Call AFTER the weights are quantized and in place -- ``quantize_`` on the runtime path, the
    prequant ``load_state_dict`` on the hosted one -- because it reparents the Linears. Returns
    the fqns wrapped, empty for a family with no pad list.

    RAISES if a Linear that should be padded cannot be proven safe to pad. A half-padded
    transformer compiles on the modules that were wrapped and crashes on the ones that were not,
    which is worse than either end state, so the caller must treat this as a failed quantise."""
    tokens = pad_tokens_for_scheme(scheme, family)
    if not tokens:
        return ()
    from .diffusion_quant_pad import matching_linear_fqns, wrap_small_m_linears

    wrapped = wrap_small_m_linears(transformer, matching_linear_fqns(transformer, tokens))
    if wrapped and logger is not None:
        logger.info(
            "diffusion.transformer_quant: padded %d small-M %s linears on %s",
            len(wrapped),
            scheme,
            family,
        )
    return wrapped


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
    # MiniMax-H3's packed sequence runs three modalities through one set of linears, so its
    # activation range is not the per-family range the fp8 filter was measured against. The
    # trainer refuses both outright; declaring it here is what keeps /diffusion/info from
    # advertising a start that is guaranteed to 400.
    "minimax-h3": frozenset({TQ_FP8, TQ_MXFP8}),
}


def _family_denied(family, scheme: str) -> bool:
    return scheme in _FAMILY_SCHEME_DENY.get(str(family or "").strip().lower(), ())


def _family_train_denied(family, scheme: str) -> bool:
    """``_family_denied`` plus the training-only additions. Every inference deny also applies to
    training (a scheme that cannot render cannot train), so this is a superset, never a bypass."""
    key = str(family or "").strip().lower()
    return _family_denied(family, scheme) or scheme in _FAMILY_TRAIN_SCHEME_DENY.get(key, ())


def family_denies_scheme(family: Optional[str], scheme: str) -> bool:
    """Whether the measured deny list rules ``scheme`` out for ``family``, regardless of hardware.

    Public so the refusal message can tell the two "no" answers apart. ``select_transformer_quant_scheme``
    folds a family deny and a missing kernel into the same ``None``, and an EXPLICIT scheme now fails
    closed, so that single answer is the whole explanation the user gets."""
    return _family_denied(family, scheme)


def explain_unusable_scheme(family: Optional[str], scheme: str) -> str:
    """Why ``select_transformer_quant_scheme`` answered None for an EXPLICIT ``scheme``.

    One ``None`` covers three different faults, and the caller turns it into a 409, so the message
    has to separate them: a family the scheme is measured to break, a torchao that cannot run at
    all in this install, and a GPU without the kernels. Only the last is a hardware limit, and only
    the last is what the message used to say."""
    if family_denies_scheme(family, scheme):
        return (
            f"'{scheme}' is ruled out for family '{family}' by the measured accuracy gate (it "
            "renders black frames or fails the quality bar on this DiT), whatever the GPU"
        )
    unavailable = torchao_unavailable_reason()
    if unavailable is not None:
        return (
            f"the torchao in this install cannot run any dense quant scheme ({unavailable}); this "
            "is a package/version problem, not a limit of the GPU"
        )
    return f"'{scheme}' is not usable for family '{family}' on this GPU"


# The reason torchao cannot be used AT ALL, resolved once. None means it imports.
_TORCHAO_UNAVAILABLE: Optional[tuple[Optional[str]]] = None


def torchao_unavailable_reason() -> Optional[str]:
    """Why no dense quant scheme can run in this install, or ``None`` when torchao is usable.

    ``_smoke_probe`` swallows every failure, so a torchao that does not IMPORT is indistinguishable
    from a GPU that lacks the kernels. That did not matter while a declined explicit scheme fell
    back silently; now it is refused, and the message is all the user has. A torch/torchao version
    skew is the common cause and it is not a hardware fault: telling a Blackwell owner "'fp8' is not
    usable on this GPU" when the real error is ``cannot import name 'ScalingType' from
    torch.nn.functional`` points them at the wrong fix entirely."""
    global _TORCHAO_UNAVAILABLE
    if _TORCHAO_UNAVAILABLE is not None:
        return _TORCHAO_UNAVAILABLE[0]
    reason: Optional[str] = None
    if is_stubbed("torchao"):
        # The Windows-ROCm stub imports fine and quantises nothing, so the import test below passes.
        reason = "this platform ships a torchao stub whose quantize_ is a no-op"
    else:
        try:
            from torchao.quantization import quantize_  # noqa: F401
        except Exception as exc:  # noqa: BLE001 — any import failure means the same thing here
            import logging

            # The full text, paths included, goes to the server log; the client gets it stripped.
            # This string is interpolated into the precision-refusal RuntimeError, which both load
            # routes return verbatim as the 409 detail, and an ImportError routinely names the
            # absolute file of the module that raised it.
            logging.getLogger(__name__).warning(
                "torchao is unusable: %s: %s", type(exc).__name__, exc
            )
            reason = _strip_paths(f"{type(exc).__name__}: {exc}")
    _TORCHAO_UNAVAILABLE = (reason,)
    return reason


# An absolute POSIX or Windows path, up to the first whitespace / quote / bracket. Deliberately
# requires a directory separator so dotted module names ("torch.nn.functional") survive intact:
# those are the actionable half of the message.
_ABS_PATH_RE = _re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s'\"()<>|]*[\\/])+[^\s'\"()<>|]*")


def _strip_paths(text: str) -> str:
    """``text`` with server filesystem paths replaced by ``<path>``.

    Refusal reasons reach an authenticated caller as a 409 body. The reason a load was refused is
    the caller's business; where this machine keeps its site-packages is not."""
    return _ABS_PATH_RE.sub("<path>", text)


# Cache of (scheme, device) -> bool so the quantise+matmul smoke test runs once.
_SMOKE_CACHE: dict[tuple[str, str], bool] = {}


def _smoke_cache_device_key(device: str) -> str:
    """``device`` qualified with the current CUDA index, so each card is validated on its own."""
    if device != "cuda":
        return device
    try:
        import torch
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:  # noqa: BLE001 -- an unreadable index falls back to the un-indexed key
        return device


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
    # The Windows-ROCm torchao stub's quantize_ is a no-op, so the smoke probe passes on a
    # still-dense Linear and the transformer gets MARKED quantised without being quantised,
    # giving the wrong VRAM budget and compile policy.
    if is_stubbed("torchao"):
        return False
    # ROCm reports gfx versions through CUDA capability APIs, so _AUTO_LADDER's NVIDIA SM floors
    # misclassify AMD GPUs. Keep ROCm on the GGUF path.
    if torch_is_rocm():
        return False
    try:
        import torch
        return getattr(target, "dtype", None) is torch.bfloat16
    except Exception:
        return False


def dense_transformer_unsupported_reason(target: Any) -> str:
    """Why ``dense_transformer_supported`` rejected the target."""
    if getattr(target, "device", None) == "cuda" and torch_is_rocm():
        return (
            "the dense torchao quant schemes are NVIDIA tensor-core paths (int8 sm_80+, fp8 "
            "sm_89+, fp4/mx sm_100+) and this is a ROCm/AMD GPU, so the checkpoint runs at the "
            "precision it ships with"
        )
    if is_stubbed("torchao"):
        return "this platform ships a torchao stub whose quantize_ is a no-op"
    return "this device cannot run a dense torchao quant (it needs a CUDA GPU in bf16)"


def select_transformer_quant_scheme(
    target: Any,
    requested: Optional[str],
    family: Optional[str] = None,
    *,
    unproven_ok: bool = False,
) -> Optional[str]:
    """The concrete scheme to apply, or None to fall back to GGUF.

    ``auto`` walks the per-arch ladder and returns the first scheme passing a real
    quantise+matmul smoke test, so an unavailable Blackwell fp4/mx kernel lands on fp8/int8
    with no error. An explicit scheme is honored only if supported (else None), never swapped.
    ``family`` applies the measured deny list (``_FAMILY_SCHEME_DENY``): schemes that produce
    black frames / out-of-bar drift are skipped by ``auto`` and refused when explicit.

    ``unproven_ok`` is for the PRE-EVICTION route gate. The smoke test allocates, so a resident
    chat/image/video model can make it fail for want of VRAM rather than for want of a kernel; the
    gate runs before the arbiter has evicted that model, so treating "could not tell" as "not
    supported" refuses a load the very next moment would have run. It answers "is this scheme
    provably unusable here", and the real selection still happens after the handoff."""
    requested = normalize_transformer_quant(requested)
    if requested is None or not dense_transformer_supported(target):
        return None
    device = str(getattr(target, "device", "cuda"))
    if requested != TQ_AUTO:
        if _family_denied(family, requested):
            return None
        return requested if _scheme_supported(requested, device, unproven_ok = unproven_ok) else None
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


def _scheme_supported(
    scheme: str,
    device: str,
    *,
    unproven_ok: bool = False,
) -> bool:
    """CUDA + (for fp8) the fp8 dtype + a cached quantise+matmul smoke test for ``scheme``."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if scheme == TQ_FP8 and not hasattr(torch, "float8_e4m3fn"):
            return False
    except Exception:
        return False
    # Resolve the whole table in one throwaway child, falling back in-process. Nothing above
    # allocates: the context arrives on the first ALLOCATION, not on is_available() or
    # get_device_capability() (0 MiB after both, 614 MiB after the first tensor). Worth a child
    # because /images/download-plan calls assert_precision_available while the user is only
    # STAGING, so a plan alone cost the backend ~700 MiB it can never give back.
    # _smoke_probe's card-qualified key, and the child is asked about that same card: a spawn
    # starts on ordinal 0 whatever this thread is pinned to, so a bare "cuda" would file card 0's
    # verdict under the card the load runs on (and a bare key hid child verdicts entirely).
    card = _smoke_cache_device_key(device)
    if (scheme, card) not in _SMOKE_CACHE:
        with _CHILD_PROBE_LOCK:
            # Re-checked under the lock: the route answers plans concurrently, and a burst of
            # them must not each spawn a child that imports torch.
            if (scheme, card) not in _SMOKE_CACHE:
                table = _child_probe_table(card)
                if table is not None:
                    for name, child_verdict in table.items():
                        # None is the child's out-of-memory: not a verdict, so not cached.
                        if child_verdict is not None:
                            _SMOKE_CACHE[(name, card)] = child_verdict
                    if scheme in table:
                        if table[scheme] is None:
                            # Repeating an OOM probe in the same memory state proves nothing.
                            return unproven_ok
                        # The child ran the same probe. Missing entries still fall through below.
                        return table[scheme]
    return _smoke_probe(scheme, device, unproven_ok = unproven_ok)


# Long enough for a cold interpreter to import torch and torchao and run every probe (measured
# 3.9 s on a B200 host, so this is ~45x headroom for a cold page cache); short enough that a
# child wedged in the driver does not hold the route thread forever. Overrunning it is not a
# verdict either -- the caller falls back in-process.
_CHILD_PROBE_TIMEOUT = 180.0

# Set once the spawn machinery has been shown not to work here (frozen build without
# multiprocessing support, a sandbox that refuses to fork/exec), so the fallback is paid once
# rather than on every miss.
_CHILD_PROBE_UNAVAILABLE = False

# Consecutive OSErrors out of the spawn before that latch is set anyway. An OSError is resource
# pressure and clears, so it is retried; a host that raises one every time still stops paying
# for the attempt after a few misses. Reset by the first spawn that works.
_CHILD_PROBE_SPAWN_ERROR_LIMIT = 3
_CHILD_PROBE_SPAWN_ERRORS = 0

_CHILD_PROBE_LOCK = _threading.Lock()


def _child_probe_table(device: str) -> Optional[dict[str, Optional[bool]]]:
    """Every scheme's verdict from a SHORT-LIVED child process, or None when it could not run.

    The child dies with its CUDA context, which is the whole point: the context is process-wide
    and the driver never returns it, so probing in the backend permanently costs what the probe
    touched. Spawning is not free (3.9 s measured on a B200 host, nearly all of it importing
    torch), so one child answers for every scheme in ``TQ_SCHEMES`` at once rather than one child
    per ladder step: an auto ladder walking three schemes would otherwise pay it three times.

    Verdicts are the child's ``_run_smoke_probe``, the same function the in-process path calls,
    so the two cannot drift. ``None`` for a scheme is an allocator failure, which the caller
    must not cache. ``None`` for the whole table means no child ran -- a frozen build without
    multiprocessing, a sandbox that refuses to spawn, a timeout -- and the caller then probes
    in-process exactly as before, since refusing a load over a missing subprocess would be worse
    than the VRAM it saves."""
    global _CHILD_PROBE_UNAVAILABLE, _CHILD_PROBE_SPAWN_ERRORS
    if _CHILD_PROBE_UNAVAILABLE:
        return None
    import time

    proc = None
    queue = None
    try:
        import multiprocessing as mp

        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )

        # spawn, never fork: the backend is threaded (uvicorn, the arbiter, download workers) and
        # a forked child inherits those locks, and fork does not exist on Windows at all. A build
        # with no spawn support at all raises here, which is what the fallback is for.
        ctx = mp.get_context("spawn")
        with native_path_secret_removed_for_child_start():
            # Inside the scrub, as every other orchestrator here builds theirs. The first
            # spawn-context queue starts multiprocessing's resource tracker, which is exec'd
            # with this environment and outlives every child, so building it above the scrub
            # would strand the lease secret somewhere the child-side scrub cannot reach.
            queue = ctx.Queue()
            # The same entrypoint the inference worker is spawned through: it scrubs the lease
            # secret from the child and binds the child to this process's lifetime, so a wedged
            # probe cannot outlive the backend.
            proc = ctx.Process(
                target = run_without_native_path_secret,
                args = (__name__, "_child_probe_entry", {}, device, TQ_SCHEMES, queue),
                daemon = True,
            )
            proc.start()
        # Adopted like every other spawn site: the bind above is the CHILD arming PDEATHSIG,
        # which is Linux only, and the Windows job object can fail to take when Unsloth already
        # runs inside an incompatible host job. This record is what is left in that case.
        _adopt_probe_pid(proc.pid)
        _CHILD_PROBE_SPAWN_ERRORS = 0
    except Exception as exc:  # noqa: BLE001 — no child here: probe in-process instead
        import logging

        # An OSError is momentary pressure on descriptors, process slots or /dev/shm, not a
        # host that cannot spawn; latching it would hold the backend on the in-process probe,
        # and its ~800 MiB, until restart. Only a deterministic failure latches on sight, plus
        # a run of OSErrors so a host that always refuses is paid for only briefly.
        transient = isinstance(exc, OSError)
        if transient:
            _CHILD_PROBE_SPAWN_ERRORS += 1
        if not transient or _CHILD_PROBE_SPAWN_ERRORS >= _CHILD_PROBE_SPAWN_ERROR_LIMIT:
            _CHILD_PROBE_UNAVAILABLE = True
        logging.getLogger(__name__).info(
            "diffusion.transformer_quant: out-of-process probe unavailable (%s: %s); "
            "probing in-process%s",
            type(exc).__name__,
            exc,
            "" if _CHILD_PROBE_UNAVAILABLE else " and retrying the child on the next miss",
        )
        _close_probe_child(proc, queue)
        return None

    table: Optional[dict[str, Optional[bool]]] = None
    deadline = time.monotonic() + _CHILD_PROBE_TIMEOUT
    try:
        while True:
            try:
                table = queue.get(timeout = 0.5)
                break
            except Exception:  # noqa: BLE001 — Empty, or a queue torn down under us
                pass
            if not proc.is_alive():
                # A put lands through a feeder thread, so it can still be in flight when the
                # child has already exited; one blocking look before calling it a loss.
                try:
                    table = queue.get(timeout = 1.0)
                except Exception:  # noqa: BLE001 — the child really did die empty
                    table = None
                break
            if time.monotonic() >= deadline:
                break
    finally:
        _close_probe_child(proc, queue)
    return table if isinstance(table, dict) else _crashed_child_verdict(proc, device)


# SIGKILL may be OOM and SIGTERM is timeout cleanup, so neither is a scheme verdict.
_PROBE_CRASH_SIGNALS = frozenset({4, 6, 7, 8, 11})  # ILL, ABRT, BUS, FPE, SEGV

# Windows has no signal here: multiprocessing returns GetExitCodeProcess verbatim, so a fault is
# the raw NTSTATUS (access violation 0xC0000005, UCRT abort 0xC0000409), never -SIGSEGV. Only
# TERMINATE is negative, hence the sign test first; the severity bits then separate a fault from
# a deliberate sys.exit.
_NTSTATUS_ERROR_FLOOR = 0xC0000000


def _crashed_child_verdict(proc: Any, device: str) -> Optional[dict[str, Optional[bool]]]:
    """Return an all-false table after a fatal probe crash, otherwise None.

    The child may die before identifying the scheme, and every scheme shares ``quantize_``. Disable
    the table rather than repeating a proven-fatal probe in the backend.
    """
    try:
        exitcode = proc.exitcode if proc is not None else None
    except Exception:  # noqa: BLE001 -- no handle left: nothing proven, fall back as before
        return None
    if exitcode is None:
        return None
    if exitcode < 0:
        if -exitcode not in _PROBE_CRASH_SIGNALS:
            return None
        cause = f"signal {-exitcode}"
    elif _sys.platform == "win32" and exitcode >= _NTSTATUS_ERROR_FLOOR:
        cause = f"status 0x{exitcode:08X}"
    else:
        return None
    import logging

    logging.getLogger(__name__).warning(
        "diffusion.transformer_quant: probe child died on %s quantising on %s; treating "
        "every dense quant scheme as unusable here rather than re-running the same probe in this "
        "process, which would take the server down the same way",
        cause,
        device,
    )
    return {scheme: False for scheme in TQ_SCHEMES}


def _adopt_probe_pid(pid: Optional[int]) -> None:
    """Record the probe child against this process's lifetime, so the shutdown sweep and the
    next startup can both reach it. Best-effort: a probe must not fail over its bookkeeping."""
    try:
        from utils.process_lifetime import adopt_pid
        adopt_pid(pid)
    except Exception:  # noqa: BLE001
        pass


def _forget_probe_pid(pid: Optional[int]) -> None:
    try:
        from utils.process_lifetime import forget_pid
        forget_pid(pid)
    except Exception:  # noqa: BLE001
        pass


def _probe_child_alive(proc: Any) -> bool:
    """Whether the child is KNOWN to be running: a missing or never-started handle raises here,
    and that is not a survivor."""
    try:
        return bool(proc is not None and proc.is_alive())
    except Exception:  # noqa: BLE001
        return False


def _close_probe_child(proc: Any, queue: Any) -> bool:
    """Tear the probe child and its queue down; True once the child is confirmed dead.

    Terminate then kill, as the chat worker's ``_shutdown_subprocess`` does: this child exists
    to hand a CUDA context back, so a wedged one left alive defeats the whole probe. The
    lifetime record is dropped only on confirmed death -- a child that outlives SIGKILL in an
    uninterruptible driver call keeps its breadcrumb, since that record is the only remaining
    handle on the VRAM it is holding. Best-effort at every step: this runs on the failure path
    too, and a probe that could not answer must not also raise."""
    try:
        pid = proc.pid if proc is not None else None
    except Exception:  # noqa: BLE001
        pid = None

    for stop, wait in ((lambda: proc.terminate(), 5.0), (lambda: proc.kill(), 3.0)):
        if not _probe_child_alive(proc):
            break
        for step in (stop, lambda: proc.join(timeout = wait)):
            try:
                step()
            except Exception:  # noqa: BLE001
                pass
    for step in (
        lambda: queue.close() if queue is not None else None,
        lambda: queue.join_thread() if queue is not None else None,
    ):
        try:
            step()
        except Exception:  # noqa: BLE001
            pass

    if _probe_child_alive(proc):
        import logging
        logging.getLogger(__name__).error(
            "diffusion.transformer_quant: probe child pid=%s survived terminate and kill; "
            "keeping its lifetime record so the shutdown sweep can still reach it",
            pid,
        )
        return False
    _forget_probe_pid(pid)
    return True


def _select_probe_card(device: str) -> None:
    """Make ``device``'s ordinal current in this child, so the probe's argument-less CUDA calls
    -- ``synchronize()``, torchao's own capability lookups -- read the card the tensors are on.

    Best-effort: an index this process cannot select still probes, and lands the same failure
    the in-process fallback would."""
    ordinal = device.partition(":")[2]
    if not ordinal.isdigit():
        return
    try:
        import torch
        torch.cuda.set_device(int(ordinal))
    except Exception:  # noqa: BLE001 -- an unselectable index still probes, on the default card
        pass


def _child_probe_entry(device: str, schemes: tuple[str, ...], out: Any) -> None:
    """Child side of ``_child_probe_table``: probe every scheme and post one table back.

    Module-level and stdlib-signature so ``spawn`` can reach it by name. Every scheme is
    attempted even after one fails, because the verdicts are independent and the whole point is
    to pay the CUDA context once."""
    _select_probe_card(device)
    table: dict[str, Optional[bool]] = {}
    for scheme in schemes:
        try:
            table[scheme] = _run_smoke_probe(scheme, device)
        except Exception:  # noqa: BLE001 — _run_smoke_probe already swallows; keep the table whole
            table[scheme] = False
    try:
        out.put(table)
    except Exception:  # noqa: BLE001 — parent gave up; nothing left to say
        pass


def _smoke_probe(
    scheme: str,
    device: str,
    *,
    unproven_ok: bool = False,
) -> bool:
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
    # Keyed by the CARD, not just "cuda": the probe runs on whichever device is current, so on a
    # heterogeneous host a pass on one selected GPU would otherwise stand in for an untested one,
    # and a failure on an older card would reject the scheme on a capable one.
    key = (scheme, _smoke_cache_device_key(device))
    if key in _SMOKE_CACHE:
        return _SMOKE_CACHE[key]
    ok = _run_smoke_probe(scheme, device)
    if ok is None:
        # NOT cached, and NOT a verdict. An out-of-memory says nothing about the scheme, and
        # the probe now runs on the ROUTE thread -- BEFORE the arbiter evicts the resident chat
        # model, which is the whole point of raising the refusal early -- so it meets a full
        # GPU by design. Caching it would refuse every later EXPLICIT request for this scheme
        # for the life of the process, on a host that runs it fine once the eviction has
        # happened; and answering "unsupported" to the pre-eviction gate (``unproven_ok``)
        # refuses THIS load for the same reason the eviction was about to remove.
        return unproven_ok
    _SMOKE_CACHE[key] = ok
    return ok


def _run_smoke_probe(scheme: str, device: str) -> Optional[bool]:
    """The probe body, uncached: True, False, or None for an allocator failure, which is not a
    verdict on the scheme (see ``_smoke_probe``). Split out so the out-of-process child runs the
    exact same code the in-process fallback does and the two verdicts cannot drift."""
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
        return bool(torch.isfinite(out).all().item())
    except Exception as exc:  # noqa: BLE001
        return None if _is_out_of_memory(exc) else False


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is an allocator failure rather than a verdict on the scheme.

    ``torch.OutOfMemoryError`` subclasses ``RuntimeError``, not ``MemoryError``, and moved between
    ``torch.cuda`` and ``torch`` across releases, so both names are tried and the message is the
    backstop (a CUDA OOM surfaced through another wrapper still says "out of memory")."""
    try:
        import torch
        named = tuple(
            t
            for t in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        if named and isinstance(exc, named):
            return True
    except Exception:  # noqa: BLE001 — no torch: the message test below still applies
        pass
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


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
        # Pad this family's small-M linears now that the weights are quantized and in place. Not
        # best-effort: a raise here means the transformer is quantized but not safely compilable,
        # so it falls into the except below and the caller loads GGUF.
        apply_small_m_padding(transformer, scheme, family, logger = logger)
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
