# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run FLUX.3 Klein Base's conditioning-only work (text and reference-image mode blocks at timestep 0) once per render.

Nothing there reads the latent or the step's timestep, so ComfyUI also runs it once (``comfy/ldm/flux3/model.py``
``preprocess_conds``). A flow-map checkpoint (distilled Klein) feeds the target time into those modulations, so it
keeps the per-step path. A forward pre-hook caches the states per branch, keyed by the identity of the pipeline's
conditioning tensors; it runs before the forward, so a whole-step CUDA graph copies them into its static buffers on
every replay and never bakes a stale tensor. The cache is cleared at the start and end of every pipeline call, so a
LoRA or weight change never meets an old entry. Bit-identical output. Kill switch ``UNSLOTH_DIFFUSION_COND_BLOCKS=0``;
installed only when the diffusers forward source hash matches, else the stock forward.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import weakref
from typing import Any, Optional

COND_BLOCKS_ENV = "UNSLOTH_DIFFUSION_COND_BLOCKS"
COND_KWARG = "unsloth_cond"

# sha256[:16] of inspect.getsource(Flux3Transformer2DModel.forward) the replacement mirrors (diffusers 0.41.0.dev0).
_FLUX3_FORWARD_SHA = frozenset({"f2f7c836416cfd58"})

_PATCHED: dict[type, Any] = {}


def enabled() -> bool:
    raw = (os.environ.get(COND_BLOCKS_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _forward_sha(cls: type) -> Optional[str]:
    try:
        return hashlib.sha256(inspect.getsource(cls.forward).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _tkey(t: Any) -> Any:
    if t is None:
        return None
    return (id(t), t.data_ptr(), tuple(t.shape), t.dtype, str(t.device))


class _Entry:
    __slots__ = ("refs", "states")

    def __init__(self, sources: tuple, states: dict) -> None:
        self.refs = tuple(weakref.ref(t) if t is not None else None for t in sources)
        self.states = states

    def matches(self, sources: tuple) -> bool:
        for ref, t in zip(self.refs, sources):
            if (ref is None) != (t is None):
                return False
            if ref is not None and ref() is not t:
                return False
        return True


def _offloaded(module: Any) -> Optional[str]:
    """Offloaded weights keep the stock path: running conditioning blocks outside the forward reorders prefetch."""
    for name, sub in module.named_modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'} carries an accelerate hook"
        registry = getattr(sub, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) or {}
        if any("offload" in str(k).lower() for k in hooks):
            return f"{name or '<root>'} is group offloaded"
    return None


def flux3_conditioning(
    module: Any,
    *,
    encoder_hidden_states: Any,
    txt_ids: Any,
    image_cond_hidden_states: Any,
    image_cond_ids: Any,
    batch_size: int,
    dtype: Any,
    device: Any,
) -> dict:
    """The conditioning-only part of ``Flux3Transformer2DModel.forward`` for a non-flow-map transformer, op for op:
    the timestep-0 embedding, the text and reference modulations, projections and RoPE, and their per-mode blocks."""
    import torch
    import torch.nn.functional as F

    zero_temb = module.time_proj(torch.zeros(batch_size, device = device, dtype = dtype))
    zero_temb = module.timestep_embedder(zero_temb.to(dtype))[:, None, :]
    zero_temb_act = F.silu(zero_temb)

    txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
    txt_early_mod = module.early_stream_modulations["txt"](txt_temb_act)
    txt_single_mod = module.single_stream_modulations["txt"](txt_temb_act)
    txt_states = module.context_embedder(encoder_hidden_states)
    txt_cos, txt_sin = module.pos_embed(txt_ids)

    out: dict = {}
    if image_cond_hidden_states is not None:
        cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
        cond_early_mod = module.early_stream_modulations["image_cond"](cond_temb_act)
        cond_single_mod = module.single_stream_modulations["image_cond"](cond_temb_act)
        cond_states = module.x_embedders["image_cond"](image_cond_hidden_states)
        cond_cos, cond_sin = module.pos_embed(image_cond_ids)

    for block in module.txt_mode_blocks:
        txt_states = block(txt_states, txt_early_mod, (txt_cos, txt_sin))
    if image_cond_hidden_states is not None:
        for block in module.mode_blocks["image_cond"]:
            cond_states = block(cond_states, cond_early_mod, (cond_cos, cond_sin))
        out.update(
            cond_states = cond_states,
            cond_single_mod = cond_single_mod,
            cond_cos = cond_cos,
            cond_sin = cond_sin,
        )
    out.update(
        txt_states = txt_states, txt_single_mod = txt_single_mod, txt_cos = txt_cos, txt_sin = txt_sin
    )
    return out


def _flux3_cached_forward(
    self: Any,
    hidden_states: Any,
    encoder_hidden_states: Any = None,
    timestep: Any = None,
    img_ids: Any = None,
    txt_ids: Any = None,
    image_cond_hidden_states: Any = None,
    image_cond_ids: Any = None,
    target_timestep: Any = None,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,
    unsloth_cond: Optional[dict] = None,
) -> Any:
    """Stock forward op for op, taking precomputed ``unsloth_cond``; without it, it IS the stock forward."""
    stock = _stock_forward(self)
    if unsloth_cond is None:
        return stock(
            self,
            hidden_states = hidden_states,
            encoder_hidden_states = encoder_hidden_states,
            timestep = timestep,
            img_ids = img_ids,
            txt_ids = txt_ids,
            image_cond_hidden_states = image_cond_hidden_states,
            image_cond_ids = image_cond_ids,
            target_timestep = target_timestep,
            attention_kwargs = attention_kwargs,
            return_dict = return_dict,
        )
    import torch
    import torch.nn.functional as F
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    if target_timestep is not None or bool(self.config.flow_map):
        raise ValueError("cached conditioning states are only valid for a non-flow-map transformer")

    batch_size, image_seq_len = hidden_states.shape[:2]
    dtype = hidden_states.dtype

    image_temb = self.time_proj(timestep.to(dtype) * 1000)
    image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
    image_temb = image_temb.expand(batch_size, image_seq_len, -1)

    encoder_hidden_states = unsloth_cond["txt_states"]
    txt_single_mod = unsloth_cond["txt_single_mod"]
    txt_rotary_emb = (unsloth_cond["txt_cos"], unsloth_cond["txt_sin"])

    image_temb_act = F.silu(image_temb)
    image_early_mod = self.early_stream_modulations["image"](image_temb_act)
    image_single_mod = self.single_stream_modulations["image"](image_temb_act)
    hidden_states = self.x_embedders["image"](hidden_states)
    image_rotary_emb = self.pos_embed(img_ids)

    has_cond = "cond_states" in unsloth_cond
    for block in self.mode_blocks["image"]:
        hidden_states = block(hidden_states, image_early_mod, image_rotary_emb)

    if has_cond:
        cond_rotary_emb = (unsloth_cond["cond_cos"], unsloth_cond["cond_sin"])
        content_states = torch.cat([hidden_states, unsloth_cond["cond_states"]], dim = 1)
        content_mod = torch.cat([image_single_mod, unsloth_cond["cond_single_mod"]], dim = 1)
        joint_rotary_emb = (
            torch.cat([txt_rotary_emb[0], image_rotary_emb[0], cond_rotary_emb[0]], dim = 0),
            torch.cat([txt_rotary_emb[1], image_rotary_emb[1], cond_rotary_emb[1]], dim = 0),
        )
    else:
        content_states = hidden_states
        content_mod = image_single_mod
        joint_rotary_emb = (
            torch.cat([txt_rotary_emb[0], image_rotary_emb[0]], dim = 0),
            torch.cat([txt_rotary_emb[1], image_rotary_emb[1]], dim = 0),
        )

    for block in self.single_transformer_blocks:
        encoder_hidden_states, content_states = block(
            hidden_states = content_states,
            encoder_hidden_states = encoder_hidden_states,
            temb_mod = content_mod,
            temb_mod_context = txt_single_mod,
            image_rotary_emb = joint_rotary_emb,
        )

    hidden_states = content_states[:, :image_seq_len]
    hidden_states = self.norm_out(hidden_states, image_temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample = output)


def _stock_forward(module: Any) -> Any:
    for cls in type(module).__mro__:
        stock = _PATCHED.get(cls)
        if stock is not None:
            return stock
    raise RuntimeError(f"{type(module).__name__} has no stock forward recorded")


def patch_class(cls: type) -> Optional[str]:
    """Install the cached forward on ``cls`` (safe before a CUDA graph or compile binds it). None, else why."""
    if any(c in _PATCHED for c in cls.__mro__):
        return None
    sha = _forward_sha(cls)
    if sha not in _FLUX3_FORWARD_SHA:
        return f"diffusers forward changed (source {sha})"
    # The stock forward carries apply_lora_scale; the replacement declines a per-call scale (the pre-hook passes no
    # states then), so it needs no wrapper of its own and the scale is never applied twice.
    _PATCHED[cls] = cls.forward
    cls.forward = _flux3_cached_forward
    return None


def _stats(module: Any) -> dict:
    stats = module.__dict__.get("_unsloth_cond_stats")
    if stats is None:
        stats = {"hits": 0, "misses": 0, "skipped": 0, "reason": None}
        module.__dict__["_unsloth_cond_stats"] = stats
    return stats


def _pre_hook(module: Any, args: tuple, kwargs: dict) -> Any:
    import torch

    if args or kwargs.get(COND_KWARG) is not None or not module.__dict__.get("_unsloth_cond_live"):
        return None
    if torch.is_grad_enabled() or kwargs.get("target_timestep") is not None:
        return None
    stats = _stats(module)
    attention_kwargs = kwargs.get("attention_kwargs")
    if attention_kwargs and "scale" in attention_kwargs:
        stats["skipped"] += 1
        stats["reason"] = "a per-call LoRA scale"
        return None
    hidden_states = kwargs.get("hidden_states")
    ehs = kwargs.get("encoder_hidden_states")
    txt_ids = kwargs.get("txt_ids")
    ich = kwargs.get("image_cond_hidden_states")
    iid = kwargs.get("image_cond_ids")
    if hidden_states is None or ehs is None or txt_ids is None or (ich is None) != (iid is None):
        return None
    if ich is None and not module.__dict__.get("_unsloth_cond_text_only"):
        # Text alone is ~0.6% of a 1024^2 step (noise on B200); the saving is the ~4k-token reference stream.
        return None
    if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    sources = (ehs, txt_ids, ich, iid)
    key = (
        tuple(_tkey(t) for t in sources),
        int(hidden_states.shape[0]),
        hidden_states.dtype,
        str(hidden_states.device),
    )
    cache = module.__dict__.setdefault("_unsloth_cond_cache", {})
    entry = cache.get(key)
    if entry is None or not entry.matches(sources):
        states = flux3_conditioning(
            module,
            encoder_hidden_states = ehs,
            txt_ids = txt_ids,
            image_cond_hidden_states = ich,
            image_cond_ids = iid,
            batch_size = int(hidden_states.shape[0]),
            dtype = hidden_states.dtype,
            device = hidden_states.device,
        )
        cache[key] = _Entry(sources, states)
        stats["misses"] += 1
    else:
        states = entry.states
        stats["hits"] += 1
    new_kwargs = dict(kwargs)
    new_kwargs[COND_KWARG] = states
    return args, new_kwargs


def _applies(transformer: Any) -> Optional[str]:
    if type(transformer).__name__ != "Flux3Transformer2DModel":
        return "not a FLUX.3 transformer"
    if bool(getattr(getattr(transformer, "config", None), "flow_map", False)):
        return "flow map: each step's target time modulates the conditioning"
    return None


def prepare(transformer: Any, logger: Any = None) -> bool:
    """Call at load, before the speed layer binds the class forward. True when the cache can arm."""
    if transformer is None or not enabled():
        return False
    why = _applies(transformer) or patch_class(type(transformer))
    if why is not None:
        _stats(transformer)["reason"] = why
        if logger is not None and "not a FLUX.3" not in why:
            logger.info("diffusion.cond_blocks: off (%s)", why)
        return False
    return True


def begin_render(transformer: Any, logger: Any = None) -> bool:
    """Arm for one pipeline call. The pre-hook must stay LAST so it runs inside earlier hooks' scope (dynamic text
    dims marks the text length dynamic for the compiled blocks)."""
    if transformer is None:
        return False
    clear(transformer)
    transformer.__dict__["_unsloth_cond_live"] = False
    if not enabled() or _applies(transformer) is not None:
        return False
    if (
        _stock_forward_or_none(transformer) is None
        or type(transformer).forward is not _flux3_cached_forward
    ):
        _stats(transformer)["reason"] = "the class forward was not patched at load"
        return False
    why = _offloaded(transformer)
    if why is not None:
        _stats(transformer)["reason"] = why
        return False
    handle = transformer.__dict__.get("_unsloth_cond_hook")
    hooks = transformer._forward_pre_hooks
    if handle is None or handle.id not in hooks:
        handle = transformer.register_forward_pre_hook(_pre_hook, with_kwargs = True)
        transformer.__dict__["_unsloth_cond_hook"] = handle
        if logger is not None:
            logger.info(
                "diffusion.cond_blocks: FLUX.3 text and reference streams run once per render"
            )
    elif next(reversed(hooks)) != handle.id:
        hooks.move_to_end(handle.id)
    transformer.__dict__["_unsloth_cond_live"] = True
    _stats(transformer)["reason"] = None
    return True


def _stock_forward_or_none(module: Any) -> Any:
    try:
        return _stock_forward(module)
    except RuntimeError:
        return None


def end_render(transformer: Any) -> None:
    if transformer is not None:
        clear(transformer)
        transformer.__dict__["_unsloth_cond_live"] = False


def clear(transformer: Any) -> None:
    if transformer is not None:
        cache = transformer.__dict__.get("_unsloth_cond_cache")
        if cache:
            cache.clear()


def status(transformer: Any) -> Optional[dict]:
    if transformer is None:
        return None
    stats = transformer.__dict__.get("_unsloth_cond_stats")
    if stats is None:
        return None
    return {"armed": transformer.__dict__.get("_unsloth_cond_hook") is not None, **stats}
