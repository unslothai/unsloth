# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep bf16-native DiTs in float16 on fp16-only cards (T4 / sm75) instead of promoting them to float32.

``rescale_post_norm``: a branch that overflows float16 but feeds an RMSNorm is divided by a power of two s before the
overflow point and that norm's eps by s**2, since ``rms_norm(y / s, eps / s**2) == rms_norm(y, eps)`` (no clamping).
Attention scales its input (q / k RMSNorm eps scaled too; v and the bias-free out projection are linear); the FFN scales
the ``w3`` half of the gate. Hooks sit on the parents and call ``to_q`` / ``w1`` / ... at run time, so a later LoRA or
quantized Linear sees the same rescale. Kill switch: ``UNSLOTH_STUDIO_FP16_GUARD=0`` restores the float32 promotion.
"""

from __future__ import annotations

import os
from typing import Any, Optional

FP16_GUARD_ENV = "UNSLOTH_STUDIO_FP16_GUARD"
_MARK = "_unsloth_fp16_guard"
_FP16_MAX = 65504.0

# recipe -> (attention, ffn) scale, powers of two (exact in float16); None = finite in plain float16, no patch.
RECIPES: dict[str, Optional[tuple[float, float]]] = {
    "native": None,
    "rescale_post_norm": (16.0, 128.0),
}


def fp16_guard_disabled() -> bool:
    return (os.environ.get(FP16_GUARD_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def family_fp16_guard(fam: Any) -> Optional[str]:
    if fam is None or fp16_guard_disabled():
        return None
    recipe = getattr(fam, "fp16_guard", None)
    return recipe if isinstance(recipe, str) and recipe in RECIPES else None


# Checked before the dtype is chosen: a diffusers release that restructures the block keeps the float32 promotion.
_RECIPE_TOKENS: dict[str, tuple[str, ...]] = {
    "rescale_post_norm": (
        "self.attention_norm2(",
        "self.ffn_norm2(",
        "self.feed_forward(",
        "self.w2(",
        "self.w3(",
        "norm_q",
    ),
}
_SUPPORTED: dict[tuple[str, str], bool] = {}


def _recipe_supported(fam: Any, recipe: str) -> bool:
    tokens = _RECIPE_TOKENS.get(recipe)
    if not tokens:
        return True
    cls_name = getattr(fam, "transformer_class", None)
    if not isinstance(cls_name, str) or not cls_name:
        return False
    key = (cls_name, recipe)
    if key not in _SUPPORTED:
        # Runs from dtype resolution, ahead of load_pipeline's own guard: `import diffusers` imports torch._dynamo.
        try:
            from loggers import get_logger
            from utils.torch_warmup import close_dynamo_import_window
            close_dynamo_import_window(get_logger(__name__))
        except Exception:  # noqa: BLE001, S110 - optimisation only
            pass
        try:
            import importlib
            import inspect
            import sys

            cls = getattr(importlib.import_module("diffusers"), cls_name)
            src = inspect.getsource(sys.modules[cls.__module__])
            _SUPPORTED[key] = all(tok in src for tok in tokens)
        except Exception:  # noqa: BLE001 - unreadable source: keep the safe float32 promotion
            _SUPPORTED[key] = False
    return _SUPPORTED[key]


def fp16_promotes_to_fp32(fam: Any) -> bool:
    """True when a resolved float16 must be promoted to float32 for this family: fp16-incompatible with no usable
    guard (none declared, the kill switch is set, or the installed diffusers block no longer matches the recipe)."""
    if fam is None or not getattr(fam, "fp16_incompatible", False):
        return False
    recipe = family_fp16_guard(fam)
    return recipe is None or not _recipe_supported(fam, recipe)


def _is_fp16(dtype: Any) -> bool:
    try:
        import torch
        return dtype is torch.float16 or dtype == torch.float16
    except Exception:  # noqa: BLE001 - no torch, nothing to guard
        return False


def _scale_eps(norm: Any, factor: float) -> bool:
    eps = getattr(norm, "eps", None)
    if not isinstance(eps, float) or isinstance(eps, bool):
        return False
    norm.eps = eps * factor
    return True


def _finite(x: Any) -> Any:
    import torch
    if x.dtype is torch.float16:
        return torch.nan_to_num(x, nan = 0.0, posinf = _FP16_MAX, neginf = -_FP16_MAX)
    return x


def _attention_pre_hook(inv: float):
    def hook(module, args, kwargs):
        if args:
            return (args[0] * inv, *args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = kwargs["hidden_states"] * inv
            return args, kwargs
        return None

    return hook


def _attention_post_hook(module, args, output):
    import torch
    if isinstance(output, torch.Tensor):
        return _finite(output)
    return None


def _make_ffn_forward(inv: float):
    def forward(self, x):
        import torch.nn.functional as F
        return self.w2(_finite(F.silu(self.w1(x)) * (self.w3(x) * inv)))

    return forward


def _block_sites(block: Any) -> Optional[tuple[Any, ...]]:
    attn = getattr(block, "attention", None)
    ffn = getattr(block, "feed_forward", None)
    norms = tuple(getattr(block, n, None) for n in ("attention_norm2", "ffn_norm2"))
    if attn is None or ffn is None or any(n is None for n in norms):
        return None
    if any(
        getattr(attn, n, None) is None
        for n in ("norm_q", "norm_k", "to_q", "to_k", "to_v", "to_out")
    ):
        return None
    if any(getattr(ffn, n, None) is None for n in ("w1", "w2", "w3")):
        return None
    # The rescale is exact only through bias-free projections.
    linears = [attn.to_q, attn.to_k, attn.to_v, attn.to_out[0], ffn.w2]
    if any(getattr(lin, "bias", None) is not None for lin in linears):
        return None
    eps = [getattr(m, "eps", None) for m in (attn.norm_q, attn.norm_k, *norms)]
    if not all(isinstance(e, float) for e in eps):
        return None
    return attn, ffn, norms[0], norms[1]


def _install_rescale_post_norm(root: Any, attn_scale: float, ffn_scale: float) -> int:
    import types

    sites = 0
    for _name, block in root.named_modules():
        if block.__dict__.get(_MARK) is not None:
            sites += 1
            continue
        found = _block_sites(block)
        if found is None:
            continue
        attn, ffn, attn_norm2, ffn_norm2 = found
        a_inv, f_inv = 1.0 / attn_scale, 1.0 / ffn_scale
        _scale_eps(attn.norm_q, a_inv * a_inv)
        _scale_eps(attn.norm_k, a_inv * a_inv)
        _scale_eps(attn_norm2, a_inv * a_inv)
        _scale_eps(ffn_norm2, f_inv * f_inv)
        handles = (
            attn.register_forward_pre_hook(_attention_pre_hook(a_inv), with_kwargs = True),
            attn.register_forward_hook(_attention_post_hook),
        )
        ffn.forward = types.MethodType(_make_ffn_forward(f_inv), ffn)
        block.__dict__[_MARK] = (attn_scale, ffn_scale, handles)
        sites += 1
    return sites


def install_fp16_guard(
    root: Any,
    recipe: Optional[str],
    dtype: Any,
    logger: Any = None,
) -> int:
    """Install ``recipe`` on every matching block under ``root`` when ``dtype`` is float16. Returns the number of
    guarded blocks (0 = not engaged). Idempotent."""
    if root is None or recipe not in RECIPES or not _is_fp16(dtype) or fp16_guard_disabled():
        return 0
    scales = RECIPES[recipe]
    if scales is None:
        if logger is not None:
            logger.info("diffusion.fp16_guard: recipe=%s (float16 kept, no patch needed)", recipe)
        return 0
    if not callable(getattr(root, "named_modules", None)):
        return 0
    attn_scale, ffn_scale = scales
    sites = _install_rescale_post_norm(root, attn_scale, ffn_scale)
    if logger is not None:
        if sites:
            logger.info(
                "diffusion.fp16_guard: recipe=%s blocks=%d (float16 kept, no float32 promotion)",
                recipe,
                sites,
            )
        else:
            logger.warning(
                "diffusion.fp16_guard: recipe=%s matched no block; float16 output may overflow "
                "(set %s=0 to promote to float32)",
                recipe,
                FP16_GUARD_ENV,
            )
    return sites


def remove_fp16_guard(root: Any) -> int:
    """Undo ``install_fp16_guard`` on ``root`` (eps restored, hooks and forward overrides dropped)."""
    removed = 0
    if root is None:
        return 0
    for _name, block in root.named_modules():
        mark = block.__dict__.pop(_MARK, None)
        if mark is None:
            continue
        attn_scale, ffn_scale, handles = mark
        for handle in handles:
            handle.remove()
        attn, ffn = block.attention, block.feed_forward
        for norm, scale in (
            (attn.norm_q, attn_scale),
            (attn.norm_k, attn_scale),
            (block.attention_norm2, attn_scale),
            (block.ffn_norm2, ffn_scale),
        ):
            _scale_eps(norm, scale * scale)
        ffn.__dict__.pop("forward", None)
        removed += 1
    return removed
