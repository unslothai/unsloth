# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which placement lets a GGUF pick run the hosted pre-quantised denoiser when the memory plan offloads.

A GGUF pick swaps to the hosted int8 / fp8 checkpoint of the same base only while that checkpoint plans RESIDENT;
when it offloads, the GGUF ran instead, and the GGUF route dequantises every block to bf16 per forward then runs
bf16 GEMMs (Qwen-Image-2.1 Q4 on L4 at 12 GB: 0.86 s/step, ~0.16 s of it dequant). The pipeline route already keeps
the hosted checkpoint under any placement ``torchao_offload_plan`` accepts; this gives a GGUF pick the same rule,
with three extra conditions of its own:

- the swap only ever loads a hosted / local pre-quantised checkpoint, never the dense bf16 build (that would page a
  second, larger denoiser through the host and quantise it there);
- an ``auto`` request only takes a checkpoint that is already cached (a GGUF pick never downloads a second
  denoiser), and only when the GGUF is less accurate than the checkpoint (5-bit and below, see ``gguf_outranks_scheme``);
- ``UNSLOTH_DIFFUSION_GGUF_OFFLOAD_PREQUANT=0`` restores the resident-only rule.

Device eligibility is the caller's: the whole branch sits behind ``dense_transformer_supported`` (CUDA bf16 only, so
MPS, CPU, ROCm and fp16 cards such as the T4 keep the GGUF) and ``_planned_quant_scheme`` (no scheme off its SM floor).
"""

from __future__ import annotations

import os
from typing import Any, Optional

# Kill switch: "0" restores the resident-only rule (a GGUF pick whose quantised swap offloads loads the GGUF).
GGUF_OFFLOAD_PREQUANT_ENV = "UNSLOTH_DIFFUSION_GGUF_OFFLOAD_PREQUANT"

# GGUF quants at least as accurate as the hosted W8A8 int8 checkpoint, which an auto request keeps under offload rather
# than trading accuracy for speed. Measured on Qwen-Image-2.1 (1024 px, 25 steps, 8 prompts, LPIPS alex vs Studio's own
# bf16, repeat floor 0.005): Q4_K_M 0.117, Q5_K_M 0.077, int8 0.059, Q6_K 0.042, Q8_0 0.019. So 6-bit and wider keep the
# GGUF; 5-bit and below (K, IQ, MXFP4, legacy Q4_0 / Q5_0) are the less accurate side of the swap.
_GGUF_PREFIXES_KEPT_UNDER_OFFLOAD = ("Q6", "Q8", "F16", "BF16", "F32")
# Every other scheme (fp8: 0.10-0.11 on the same family at 40 steps) only replaces 4-bit and narrower picks.
_GGUF_PREFIXES_KEPT_OTHER_SCHEMES = ("Q5", *_GGUF_PREFIXES_KEPT_UNDER_OFFLOAD)


def gguf_offload_prequant_enabled() -> bool:
    return str(os.environ.get(GGUF_OFFLOAD_PREQUANT_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def gguf_quant_token(gguf_filename: Optional[str]) -> Optional[str]:
    if not gguf_filename:
        return None
    try:
        from hub.utils.gguf import extract_quant_token
        token = extract_quant_token(str(gguf_filename))
    except Exception:  # noqa: BLE001 - an unparseable name is treated as unknown precision
        token = None
    return str(token).upper() if token else None


def gguf_outranks_scheme(gguf_filename: Optional[str], scheme: Optional[str]) -> bool:
    """Whether the picked GGUF is at least as accurate as ``scheme``'s checkpoint, so an auto swap would trade
    accuracy for speed. An unknown quant counts as outranking (keep what the user picked)."""
    token = gguf_quant_token(gguf_filename)
    if token is None:
        return True
    if token.startswith("UD-"):
        token = token[3:]
    kept = (
        _GGUF_PREFIXES_KEPT_UNDER_OFFLOAD
        if str(scheme) == "int8"
        else _GGUF_PREFIXES_KEPT_OTHER_SCHEMES
    )
    return token.startswith(kept)


def gguf_offload_prequant_placement(
    replanned: Any,
    candidate: Any,
    *,
    scheme: Optional[str],
    auto: bool,
    gguf_filename: Optional[str],
    prequant_cached: bool,
    torchao_offload_plan: Any,
) -> tuple[Optional[Any], Optional[str]]:
    """``(placement, None)`` when a GGUF pick should load the pre-quantised checkpoint under ``replanned``'s offload,
    ``(None, reason)`` when it keeps the GGUF for a reason of this rule, ``(None, None)`` when the rule does not apply
    (the caller's own decline stands). ``torchao_offload_plan`` is injected so tests can stub the planner."""
    if not gguf_offload_prequant_enabled():
        return None, None
    if candidate is None or not bool(getattr(candidate, "prequant", False)) or not scheme:
        return None, None
    if auto and not prequant_cached:
        return None, (
            f"the plan offloads the denoiser and the hosted {scheme} checkpoint is not cached; an auto quant never "
            "downloads a second transformer for a GGUF pick"
        )
    if auto and gguf_outranks_scheme(gguf_filename, scheme):
        return None, (
            f"the plan offloads the denoiser, and the picked GGUF ({gguf_quant_token(gguf_filename) or 'unknown quant'}) "
            f"is at least as accurate as the hosted {scheme} checkpoint, so auto keeps the GGUF"
        )
    try:
        placement = torchao_offload_plan(replanned, scheme)
    except Exception:  # noqa: BLE001 - an unanswerable placement keeps the GGUF
        placement = None
    return placement, None


def gguf_offload_swap_reason(
    scheme: str, artifact: Optional[str], placement: Any, gguf_filename: Optional[str]
) -> str:
    """Status tooltip for an engaged swap: names what loaded instead of the GGUF, and why."""
    what = (
        artifact.split(":", 1)[1]
        if artifact and ":" in artifact
        else (artifact or f"the hosted {scheme} checkpoint")
    )
    token = gguf_quant_token(gguf_filename) or "GGUF"
    return (
        f"the {token} GGUF pick was replaced by {what}: the plan offloads the denoiser "
        f"('{getattr(placement, 'offload_policy', 'offload')}'), where the {scheme} checkpoint runs faster than "
        "dequantising the GGUF every step. Set Precision to Off to run the GGUF itself"
    )
