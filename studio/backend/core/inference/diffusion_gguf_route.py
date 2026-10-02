# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Let a GGUF pick whose memory plan offloads load the cached hosted int8 / fp8 checkpoint under that offload.

Only a pre-quantised checkpoint, never a dense bf16 build; ``auto`` takes it only when cached and only for GGUFs less
accurate than it. Device eligibility (CUDA bf16) is the caller's.
"""

from __future__ import annotations

import os
from typing import Any, Optional

# "0" restores the resident-only swap.
GGUF_OFFLOAD_PREQUANT_ENV = "UNSLOTH_DIFFUSION_GGUF_OFFLOAD_PREQUANT"

# GGUFs at least as accurate as int8 (Qwen-Image-2.1 LPIPS vs bf16: Q5_K_M 0.077, int8 0.059, Q6_K 0.042): auto keeps them.
_GGUF_PREFIXES_KEPT_UNDER_OFFLOAD = ("Q6", "Q8", "F16", "BF16", "F32")
# fp8 (LPIPS 0.10-0.11) only replaces 4-bit and narrower.
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
    """Whether the GGUF is at least as accurate as ``scheme``'s checkpoint; an unknown quant counts as outranking."""
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
    """``(placement, None)`` to swap, ``(None, reason)`` to keep the GGUF by this rule, ``(None, None)`` when it
    does not apply."""
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
    what = (
        artifact.split(":", 1)[1]
        if artifact and ":" in artifact
        else (artifact or f"the hosted {scheme} checkpoint")
    )
    token = gguf_quant_token(gguf_filename) or "GGUF"
    return (
        f"the {token} GGUF pick was replaced by {what}: the memory plan offloads "
        f"('{getattr(placement, 'offload_policy', 'offload')}'), where the {scheme} checkpoint runs faster than "
        "dequantising the GGUF every step. Set Precision to Off to run the GGUF itself"
    )
