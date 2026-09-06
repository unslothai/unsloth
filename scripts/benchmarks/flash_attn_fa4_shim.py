"""Make transformers' continuous batching dispatch to Flash Attention 4 on B200.

Two integration gaps between transformers' continuous batching (CB) and the
FA2 varlen path make `attn_implementation="flash_attention_2"` fail out of the
box in transformers 4.57 even with a working FA varlen kernel:

1. CB's `ContinuousBatchProcessor` creates a 4D paged attention mask of shape
   `[1, 1, q_len, k_len]` for every attention implementation except
   `"paged_attention"`. `_flash_attention_forward` then enters the
   `if attention_mask is not None:` branch and calls `_upad_input`, which
   expects a 2D mask and fails (the FA4 kernel ultimately asserts on
   `cu_seqlens_k.shape`).

2. CB passes `max_seqlen_q`/`max_seqlen_k` as model kwargs while
   `_flash_attention_forward` names the parameters `max_length_q`/
   `max_length_k`. The former therefore never bind and the varlen branch
   inside `_flash_attention_forward` invokes the FA kernel with
   `max_seqlen_q=None`.

This module patches around both, and (via the sibling
`site-packages/flash_attn/__init__.py` shim) points the FA2 varlen dispatch
at FA4's Blackwell-capable kernel. Call `apply()` once, before
`model.generate_batch` is invoked.
"""

from __future__ import annotations

import functools


_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import transformers  # noqa: F401  - force load
    from transformers.generation.continuous_batching import continuous_api as _cb
    from transformers import modeling_flash_attention_utils as _fa_utils

    _patch_return_attention_mask(_cb)
    _patch_flash_attention_forward(_fa_utils)

    _APPLIED = True


def _patch_return_attention_mask(cb_module) -> None:
    """Don't materialise a 4D attention mask when the kernel is FA varlen.

    The existing `return_attention_mask` only skips the mask for
    `"paged_attention"`. We extend the skip set to `"flash_attention_2"`
    (and `"flash_attention_3"` for future-proofing) because the varlen path
    relies on `cu_seq_lens_*` and reads no mask.
    """
    _SKIP_MASK_IMPLS = {
        "paged_attention",
        "flash_attention_2",
        "flash_attention_3",
    }

    def return_attention_mask(self) -> bool:
        return self.config._attn_implementation not in _SKIP_MASK_IMPLS

    cb_module.ContinuousBatchProcessor.return_attention_mask = return_attention_mask


def _patch_flash_attention_forward(fa_utils_module) -> None:
    """Accept CB's `max_seqlen_q`/`max_seqlen_k` kwargs as aliases.

    transformers names the parameters `max_length_q`/`max_length_k`, but CB
    (and most downstream call sites) name them `max_seqlen_q`/
    `max_seqlen_k`. We rename at the boundary so callers on either side work
    unchanged.
    """
    original = fa_utils_module._flash_attention_forward

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if kwargs.get("max_length_q") is None and "max_seqlen_q" in kwargs:
            kwargs["max_length_q"] = kwargs.pop("max_seqlen_q")
        if kwargs.get("max_length_k") is None and "max_seqlen_k" in kwargs:
            kwargs["max_length_k"] = kwargs.pop("max_seqlen_k")
        return original(*args, **kwargs)

    fa_utils_module._flash_attention_forward = wrapper

    # Some integration modules imported the function by name before we
    # patched. Re-bind the most common consumers so they pick up the wrapper.
    try:
        from transformers.integrations import flash_attention as _flash_integration

        _flash_integration._flash_attention_forward = wrapper
    except Exception:
        pass
