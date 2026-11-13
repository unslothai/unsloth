# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps bf16 DiTs in float16 on fp16-only cards by pre-scaling; hooks run at call time so a later
LoRA or quantized Linear sees the same rescale; UNSLOTH_STUDIO_FP16_GUARD=0 disables."""

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
    # FLUX.3 residual reaches ~1e7: whole stream at 1/1024, LayerNorm eps by 1/1024^2.
    "rescale_residual": (1024.0, 1024.0),
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
    "rescale_residual": (
        "self.context_embedder(encoder_hidden_states)",
        'self.x_embedders["image"](hidden_states)',
        "return hidden_states + mod_gate * (attn_output + ff_output)",
        "encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output",
        "hidden_states = hidden_states + mod_gate * content_output",
        "x = self.gate_fn(x[..., :half]) * x[..., half:]",
        "return self.norm(hidden_states) * (1 + scale) + shift",
        "attn.norm_q(query.float())",
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


def _scale_output_hook(inv: float):
    def hook(module, args, output):
        return output * inv

    return hook


def _scale_input_hook(inv: float):
    def hook(module, args):
        return (args[0] * inv, *args[1:]) if args else None

    return hook


def _compiling() -> bool:
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - no torch.compiler: eager
        return False


def _bias_free(*linears: Any) -> bool:
    return all(lin is not None and getattr(lin, "bias", None) is None for lin in linears)


def _residual_block_sites(block: Any) -> Optional[tuple[Any, ...]]:
    """(norm, attn, act_fn) of a FLUX.3 mode / joint block, whose residual update is gate * (attn + ff)."""
    norm, attn, ff = (getattr(block, n, None) for n in ("norm", "attn", "ff"))
    if norm is None or attn is None or ff is None:
        return None
    to_out = getattr(attn, "to_out", None)
    act = getattr(ff, "act_fn", None)
    if to_out is None or act is None or not hasattr(ff, "linear_in"):
        return None
    try:
        out_proj = to_out[0]
    except (TypeError, IndexError):
        return None
    if not _bias_free(attn.to_q, attn.to_k, attn.to_v, out_proj, getattr(ff, "linear_out", None)):
        return None
    eps = [
        getattr(m, "eps", None)
        for m in (norm, getattr(attn, "norm_q", None), getattr(attn, "norm_k", None))
    ]
    if not all(isinstance(e, float) for e in eps):
        return None
    return norm, attn, act


def _residual_entries(root: Any) -> Optional[list]:
    """Modules that write the residual stream from the inputs (text and per-mode image projections)."""
    ctx = getattr(root, "context_embedder", None)
    xs = getattr(root, "x_embedders", None)
    if ctx is None or xs is None or not callable(getattr(xs, "values", None)):
        return None
    return [ctx, *xs.values()]


def _install_rescale_residual(root: Any, scale: float) -> int:
    """Residual stream at 1/scale: exact via bias-free projections, scale-invariant norms, eps / scale^2."""
    if root.__dict__.get(_MARK) is not None:
        return root.__dict__[_MARK][1]
    entries = _residual_entries(root)
    final = getattr(getattr(root, "norm_out", None), "norm", None)
    if (
        entries is None
        or not _bias_free(*entries)
        or not isinstance(getattr(final, "eps", None), float)
    ):
        return 0
    found = [(b, sites) for b in root.modules() if (sites := _residual_block_sites(b)) is not None]
    if not found:
        return 0
    inv = 1.0 / scale
    eps_scaled = [final]
    handles = []
    for _block, (norm, attn, act) in found:
        eps_scaled += [norm, attn.norm_q, attn.norm_k]
        handles.append(attn.register_forward_pre_hook(_attention_pre_hook(inv), with_kwargs = True))
        handles.append(act.register_forward_hook(_scale_output_hook(inv)))
    for norm in eps_scaled:
        _scale_eps(norm, inv * inv)
    # Re-hook whatever sits in the entry slot per call (e.g. a later LoRA wrapper) so the stream is scaled once.
    entry_hooks: dict[int, Any] = {}

    def sync_entries(
        module,
        args,
        kwargs = None,
    ):
        if _compiling():
            return None
        current = _residual_entries(root) or []
        ids = {id(m) for m in current}
        for key in [k for k in entry_hooks if k not in ids]:
            entry_hooks.pop(key).remove()
        for m in current:
            if id(m) not in entry_hooks:
                entry_hooks[id(m)] = m.register_forward_pre_hook(_scale_input_hook(inv))
        return None

    sync_entries(root, ())
    handles.append(root.register_forward_pre_hook(sync_entries))

    def undo():
        for handle in (*handles, *entry_hooks.values()):
            handle.remove()
        entry_hooks.clear()
        for norm in eps_scaled:
            _scale_eps(norm, scale * scale)

    root.__dict__[_MARK] = (undo, len(found))
    return len(found)


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
    if recipe == "rescale_residual":
        sites = _install_rescale_residual(root, attn_scale)
    else:
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
        if callable(mark[0]):
            mark[0]()
            removed += mark[1]
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
