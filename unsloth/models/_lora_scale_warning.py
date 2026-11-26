# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One-time warning for low-rank LoRA with ``lora_alpha == r`` (update scale ``alpha / r`` of 1)."""

import numbers
import warnings

__all__ = ["LOW_RANK_ALPHA_MAX_R", "warn_low_rank_lora_alpha"]

# GRPO: r=1 and r=8 with alpha=r under-trained at an LR tuned for alpha=32; r=64 did not.
LOW_RANK_ALPHA_MAX_R = 8

_warned = False


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def warn_low_rank_lora_alpha(
    r,
    lora_alpha,
    use_rslora = False,
):
    """Warn once per process when ``r <= 8``, ``lora_alpha == r`` and rsLoRA is off. Returns True if it warned."""
    global _warned
    if _warned or use_rslora:
        return False
    if not _is_number(r) or not _is_number(lora_alpha):
        return False
    if r <= 0 or r > LOW_RANK_ALPHA_MAX_R or lora_alpha != r:
        return False
    _warned = True
    warnings.warn(
        f"Unsloth: LoRA rank r = {r} with lora_alpha = r gives an update scale of lora_alpha / r = 1, "
        "which is small at low rank. Consider lora_alpha = 16 to 32, or use_rslora = True. "
        "If you keep lora_alpha = r, use a larger learning rate.",
        UserWarning,
        # Attribute to unsloth, not the caller: trl / transformers modules are under ignore filters.
        stacklevel = 2,
    )
    return True
