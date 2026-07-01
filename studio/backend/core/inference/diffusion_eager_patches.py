# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reversible, failure-safe monkey-patches that speed up the diffusion denoiser.

These patch a couple of SHARED diffusers building-block classes (so a tiny patch surface
covers all four Studio families) to make inference faster WITHOUT reducing accuracy and
WITHOUT regressing ``torch.compile``. Both patches are *compile-safe*: they lower to plain
torch ops, are bit-identical or FMA-1-ULP (i.e. equal-or-MORE accurate) vs stock diffusers,
and ``torch.compile`` fuses either form to the same kernel -- so they help the EAGER path
and are neutral under compile.

What it patches (measured on a B200, bf16, DiT shapes):

* ``normalization.RMSNorm.forward`` -> fused ``F.rms_norm`` on the common path (non-NPU,
  no bias, weight None/fp16/bf16; else the exact original, incl. its fp32 quirk). This is
  the standout win (~6-12x per call) because QK-norm runs every attention block on
  Qwen-Image + Z-Image. Bit-identical in bf16.

* ``AdaLayerNormContinuous`` / ``AdaLayerNormZero`` / ``AdaLayerNormZeroSingle`` ->
  the ``norm(x)*(1+scale)+shift`` modulation fused via ``torch.addcmul`` (~1.2x). fp32
  exact; bf16 within 1 ULP (FMA, a single rounding -> more accurate than mul+add). Covers
  flux.1 / flux.2-klein / qwen-image.

Deliberately NOT patched (evidence-based):
* ``FeedForward`` (skipping its eval no-op Dropout) -- measured a small REGRESSION because
  the per-module ``isinstance`` check costs more than the skipped identity dispatch.
* GEGLU / SwiGLU / GELU -- already mul-bound; a real win needs a custom Triton/CUDA kernel
  (out of scope, correctness risk).
* Per-family custom MLP/norm classes and attention -- not shared (no leverage), and
  attention already routes through ``F.scaled_dot_product_attention`` via the existing
  ``set_attention_backend`` dispatcher.

Lifecycle: ``install_compile_safe_patches()`` is idempotent and patches at the class level.
Install it for any active speed tier; the bit-identical ``off`` reference path must run with
the patches UNINSTALLED, so the caller uninstalls on an ``off`` load and on unload. The
Studio CHAT<->DIFFUSION arbiter guarantees a single active diffusion pipe, so class-level
state is safe.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .diffusion_patch_backend import apply_patch, revert_patch

logger = logging.getLogger(__name__)

# Kill-switch: set UNSLOTH_DIFFUSION_EAGER_PATCHES=0 to disable the patches entirely (for
# A/B benchmarking or to rule them out while debugging). Enabled by default.
_ENV_ENABLE = "UNSLOTH_DIFFUSION_EAGER_PATCHES"


def _patches_enabled() -> bool:
    return (os.environ.get(_ENV_ENABLE) or "").strip().lower() not in ("0", "off", "false", "no")


# --------------------------------------------------------------------------- #
# Resolve the diffusers classes we patch. Any import failure -> that patch is
# simply unavailable (None) and is skipped at install time.
# --------------------------------------------------------------------------- #
try:
    from diffusers.models.normalization import (
        AdaLayerNormContinuous as _AdaLayerNormContinuous,
        AdaLayerNormZero as _AdaLayerNormZero,
        AdaLayerNormZeroSingle as _AdaLayerNormZeroSingle,
        RMSNorm as _RMSNorm,
    )
except Exception:  # noqa: BLE001
    _AdaLayerNormContinuous = _AdaLayerNormZero = _AdaLayerNormZeroSingle = _RMSNorm = None

try:
    from diffusers.utils.import_utils import is_torch_npu_available as _is_npu
    _NPU = bool(_is_npu())
except Exception:  # noqa: BLE001
    _NPU = False


# --------------------------------------------------------------------------- #
# Patched forwards. Each mirrors diffusers 0.38 semantics, with the documented
# fused fast path. ``addcmul(input, t1, t2) == input + t1 * t2`` in one fused kernel.
# --------------------------------------------------------------------------- #
def _adaln_continuous_forward(self, x, conditioning_embedding):
    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = torch.chunk(emb, 2, dim = 1)
    # original: self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return torch.addcmul(shift[:, None, :], self.norm(x), 1 + scale[:, None, :])


def _adaln_zero_forward(
    self,
    x,
    timestep = None,
    class_labels = None,
    hidden_dtype = None,
    emb = None,
):
    if self.emb is not None:
        emb = self.emb(timestep, class_labels, hidden_dtype = hidden_dtype)
    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim = 1)
    # original: self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    x = torch.addcmul(shift_msa[:, None], self.norm(x), 1 + scale_msa[:, None])
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _adaln_zero_single_forward(
    self,
    x,
    emb = None,
):
    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa = emb.chunk(3, dim = 1)
    x = torch.addcmul(shift_msa[:, None], self.norm(x), 1 + scale_msa[:, None])
    return x, gate_msa


# Filled in at install time with the ORIGINAL RMSNorm.forward so the guarded fast path
# can fall back for the uncommon (NPU / bias / fp32-weight) cases.
_orig_rmsnorm_forward: Optional[Callable] = None


def _rmsnorm_forward(self, hidden_states):
    # Fall back to the exact original for cases where F.rms_norm is NOT equivalent to
    # diffusers' implementation:
    #   * NPU / bias / fp32-weight -> the original has special handling / an fp32 quirk;
    #   * tuple `dim` -> diffusers always reduces the LAST dim (`mean(-1)`) while
    #     F.rms_norm reduces every dim in `self.dim` (differs for a multi-dim shape);
    #   * dtype mismatch (e.g. fp32 activations into an fp16/bf16-weight norm) -> diffusers
    #     computes the variance in fp32 from the ORIGINAL tensor and only casts before the
    #     weight multiply, so casting first would change the variance.
    if _NPU or self.bias is not None or _orig_rmsnorm_forward is None or len(tuple(self.dim)) != 1:
        return _orig_rmsnorm_forward(self, hidden_states)  # type: ignore[misc]
    weight = self.weight
    if weight is None:
        return F.rms_norm(hidden_states, self.dim, None, self.eps)
    if weight.dtype in (torch.float16, torch.bfloat16) and hidden_states.dtype == weight.dtype:
        # Common DiT path (bf16 activations + bf16 weight): F.rms_norm matches diffusers
        # bit-for-bit (both reduce the variance in fp32 internally), just fused.
        return F.rms_norm(hidden_states, self.dim, weight, self.eps)
    # Mixed dtype / fp32 weight -> keep the exact original behaviour.
    return _orig_rmsnorm_forward(self, hidden_states)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Install / uninstall. All swaps go through the shared patch backend
# (unsloth_zoo patch_function / restore_original): the live original is fingerprinted
# (can_safely_patch, relaxed) so a diffusers that renamed/reordered a forward's params is
# left UNPATCHED instead of miscompiled, and the original is stashed for an exact restore.
# --------------------------------------------------------------------------- #
def _specs():
    # (class, patched_fn)
    return [
        (_AdaLayerNormContinuous, _adaln_continuous_forward),
        (_AdaLayerNormZero, _adaln_zero_forward),
        (_AdaLayerNormZeroSingle, _adaln_zero_single_forward),
        (_RMSNorm, _rmsnorm_forward),
    ]


# Classes whose `forward` we successfully patched, so uninstall reverts exactly those.
_patched: list[type] = []


def install_compile_safe_patches() -> int:
    """Install the shared compile-safe speedup patches (idempotent).

    Returns the number of patches applied. A second call while installed is a no-op.
    """
    global _orig_rmsnorm_forward
    if not _patches_enabled():
        uninstall_patches()  # ensure OFF even if a prior call installed them
        return 0
    if _patched:
        return len(_patched)
    for cls, new_fn in _specs():
        if cls is None:
            continue
        # Capture the live original BEFORE patching so the RMSNorm fast path can fall back
        # to it for the uncommon (NPU / bias / fp32-weight / tuple-dim) cases.
        if cls is _RMSNorm:
            _orig_rmsnorm_forward = cls.forward
        if apply_patch(cls, "forward", new_fn, match_level = "relaxed"):
            _patched.append(cls)
        else:
            logger.warning(
                "eager-patch: skipping %s (signature mismatch / unavailable)",
                getattr(cls, "__name__", cls),
            )
            if cls is _RMSNorm:
                _orig_rmsnorm_forward = None
    logger.info(
        "eager-patch: installed %d/%d shared diffusion patches", len(_patched), len(_specs())
    )
    return len(_patched)


def uninstall_patches() -> None:
    """Restore every patched class to its exact original ``forward`` (idempotent)."""
    global _orig_rmsnorm_forward
    for cls in list(_patched):
        revert_patch(cls, "forward")
    _patched.clear()
    _orig_rmsnorm_forward = None


def is_installed() -> bool:
    """True if any compile-safe patch is currently installed."""
    return bool(_patched)
