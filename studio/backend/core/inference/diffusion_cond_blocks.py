# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Klein Base conditioning blocks run once per render, bit-identical; cache cleared at start
and end of each pipeline call; installed only on matching diffusers forward hash, else stock;
UNSLOTH_DIFFUSION_COND_BLOCKS=0 disables. Offloaded (CUDA): computed inside the hooked forward,
UNSLOTH_DIFFUSION_OFFLOAD_COND_CACHE=0 keeps stock. Modulation heads project one row per sample,
not per token (any FLUX.3), UNSLOTH_DIFFUSION_FLUX3_COMPACT_MOD=0 restores per-token rows"""

from __future__ import annotations

import hashlib
import inspect
import os
import weakref
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

COND_BLOCKS_ENV = "UNSLOTH_DIFFUSION_COND_BLOCKS"
OFFLOAD_COND_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_COND_CACHE"
COMPACT_MOD_ENV = "UNSLOTH_DIFFUSION_FLUX3_COMPACT_MOD"
COND_KWARG = "unsloth_cond"
_COMPACT_ATTR = "_unsloth_compact_mod"
_OFFLOADED_ATTR = "_unsloth_cond_offloaded"

# sha256[:16] of inspect.getsource(Flux3Transformer2DModel.forward) the replacement mirrors (d42ebf9c and 0.42.1).
_FLUX3_FORWARD_SHA = frozenset({"f2f7c836416cfd58"})

_PATCHED: dict[type, Any] = {}


def _switch(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() not in ("0", "off", "false", "no")


def enabled() -> bool:
    return _switch(COND_BLOCKS_ENV)


def offload_cache_enabled() -> bool:
    return _switch(OFFLOAD_COND_ENV)


def compact_enabled() -> bool:
    return _switch(COMPACT_MOD_ENV)


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
    """Why ``module`` is offloaded (None: resident); its conditioning then runs only inside the hooked forward."""
    for name, sub in module.named_modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'} carries an accelerate hook"
        registry = getattr(sub, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) or {}
        if any("offload" in str(k).lower() for k in hooks):
            return f"{name or '<root>'} is group offloaded"
    return None


def _fixed_prefetch_chain(module: Any) -> Optional[str]:
    """diffusers' fixed next-group prefetch chain would leave a skipped block on the device."""
    for name, sub in module.named_modules():
        hooks = getattr(getattr(sub, "_diffusers_hook", None), "hooks", None) or {}
        for key, hook in hooks.items():
            if "lazy_prefetch" in str(key).lower() or getattr(hook, "next_group", None) is not None:
                return f"{name or '<root>'} streams with diffusers' prefetch chain"
    return None


def _offload_device_ok(t: Any) -> bool:
    """Offload cache: CUDA only for now (ROCm, XPU, MPS, CPU keep stock)."""
    import torch
    return bool(t.is_cuda) and not is_rocm_torch(torch)


def _rows(t: Any, batch_size: int, n: int) -> Any:
    return t if t.shape[1] == n else t.expand(batch_size, n, -1)


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
    compact: bool = False,
    target_temb: Any = None,
) -> dict:
    """The conditioning part of ``Flux3Transformer2DModel.forward``, op for op. ``compact``: one modulation row per
    sample (every token of a mode shares its timestep), broadcast by the blocks."""
    import torch
    import torch.nn.functional as F

    zero_temb = module.time_proj(torch.zeros(batch_size, device = device, dtype = dtype))
    zero_temb = module.timestep_embedder(zero_temb.to(dtype))[:, None, :]
    if target_temb is not None:
        zero_temb = zero_temb + target_temb
    zero_temb_act = F.silu(zero_temb)

    txt_temb_act = (
        zero_temb_act
        if compact
        else zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
    )
    txt_early_mod = module.early_stream_modulations["txt"](txt_temb_act)
    txt_single_mod = module.single_stream_modulations["txt"](txt_temb_act)
    txt_states = module.context_embedder(encoder_hidden_states)
    txt_cos, txt_sin = module.pos_embed(txt_ids)

    out: dict = {}
    if image_cond_hidden_states is not None:
        cond_temb_act = (
            zero_temb_act
            if compact
            else zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
        )
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


def _compact_call(module: Any, attention_kwargs: Any) -> bool:
    """Inference without a per-call LoRA scale (stock ``apply_lora_scale`` handles it)."""
    import torch

    if not module.__dict__.get(_COMPACT_ATTR) or torch.is_grad_enabled():
        return False
    return not (attention_kwargs and "scale" in attention_kwargs)


class Pending:
    """An offloaded miss passed as ``unsloth_cond``: a non-tensor, so a CUDA graph runs this call eager and a CFG
    router records no template from it (``pending``)."""

    __slots__ = ("key", "sources")
    pending = True

    def __init__(self, key: tuple, sources: tuple) -> None:
        self.key = key
        self.sources = sources


def _claim_pending(
    module: Any, hidden_states: Any, pending: Pending, sources: tuple
) -> Optional[dict]:
    """Offloaded miss: compute and cache inside the hooked forward, only for the exact call it was recorded for."""
    import torch

    key, recorded = pending.key, pending.sources
    if key[1] != int(hidden_states.shape[0]) or any(a is not b for a, b in zip(sources, recorded)):
        return None
    if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    ehs, txt_ids, ich, iid = sources
    states = flux3_conditioning(
        module,
        encoder_hidden_states = ehs,
        txt_ids = txt_ids,
        image_cond_hidden_states = ich,
        image_cond_ids = iid,
        batch_size = int(hidden_states.shape[0]),
        dtype = hidden_states.dtype,
        device = hidden_states.device,
        compact = bool(module.__dict__.get(_COMPACT_ATTR)),
    )
    module.__dict__.setdefault("_unsloth_cond_cache", {})[key] = _Entry(sources, states)
    stats = _stats(module)
    stats["misses"] += 1
    stats["in_forward"] = stats.get("in_forward", 0) + 1
    return states


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
    """Stock forward op for op, with precomputed ``unsloth_cond`` and / or compact modulation; else stock."""
    if isinstance(unsloth_cond, Pending):
        unsloth_cond = _claim_pending(
            self,
            hidden_states,
            unsloth_cond,
            (encoder_hidden_states, txt_ids, image_cond_hidden_states, image_cond_ids),
        )
    if unsloth_cond is None and not _compact_call(self, attention_kwargs):
        return _stock_forward(self)(
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

    batch_size, image_seq_len = hidden_states.shape[:2]
    dtype = hidden_states.dtype
    compact = bool(self.__dict__.get(_COMPACT_ATTR))

    image_temb = self.time_proj(timestep.to(dtype) * 1000)
    image_temb = self.timestep_embedder(image_temb.to(dtype))[:, None, :]
    if unsloth_cond is not None:
        if target_timestep is not None or bool(self.config.flow_map):
            raise ValueError(
                "cached conditioning states are only valid for a non-flow-map transformer"
            )
    else:
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError(
                "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
            )
        target_temb = None
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
        unsloth_cond = flux3_conditioning(
            self,
            encoder_hidden_states = encoder_hidden_states,
            txt_ids = txt_ids,
            image_cond_hidden_states = image_cond_hidden_states,
            image_cond_ids = image_cond_ids,
            batch_size = batch_size,
            dtype = dtype,
            device = hidden_states.device,
            compact = True,
            target_temb = target_temb,
        )
        if (
            not torch.compiler.is_compiling()
        ):  # a module int Dynamo would guard on: one recompile per call
            _stats(self)["compact"] = _stats(self).get("compact", 0) + 1
    if not compact:
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
        cond_states = unsloth_cond["cond_states"]
        cond_rotary_emb = (unsloth_cond["cond_cos"], unsloth_cond["cond_sin"])
        content_states = torch.cat([hidden_states, cond_states], dim = 1)
        # image and reference spans carry different timesteps: per token here
        content_mod = torch.cat(
            [
                _rows(image_single_mod, batch_size, image_seq_len),
                _rows(unsloth_cond["cond_single_mod"], batch_size, cond_states.shape[1]),
            ],
            dim = 1,
        )
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


def _conditioning_blocks(module: Any, has_cond: bool) -> list:
    blocks = list(module.txt_mode_blocks)
    if has_cond:
        blocks += list(module.mode_blocks["image_cond"])
    return blocks


def _match_prefetch_order(module: Any, has_cond: bool, *, skipping: bool) -> None:
    """Keep the streamed prefetch on order when a hit skips (or a miss runs) the conditioning blocks."""
    from .diffusion_offload_prefetch import match_skipped_order
    try:
        match_skipped_order(module, _conditioning_blocks(module, has_cond), skipping = skipping)
    except Exception:  # noqa: BLE001 - speed hint only
        pass


def _pre_hook(module: Any, args: tuple, kwargs: dict) -> Any:
    import torch

    if args or kwargs.get(COND_KWARG) is not None or not module.__dict__.get("_unsloth_cond_live"):
        return None
    if (
        torch.compiler.is_compiling()
    ):  # a whole-model compile: identity cache and counters stay out of the graph
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
    offloaded = bool(module.__dict__.get(_OFFLOADED_ATTR))
    if offloaded and not _offload_device_ok(hidden_states):
        stats["skipped"] += 1
        stats["reason"] = (
            f"offloaded on {hidden_states.device.type} (the offload cache is CUDA only)"
        )
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
        if offloaded:
            _match_prefetch_order(module, ich is not None, skipping = False)
            new_kwargs = dict(kwargs)
            new_kwargs[COND_KWARG] = Pending(key, sources)
            return args, new_kwargs
        states = flux3_conditioning(
            module,
            encoder_hidden_states = ehs,
            txt_ids = txt_ids,
            image_cond_hidden_states = ich,
            image_cond_ids = iid,
            batch_size = int(hidden_states.shape[0]),
            dtype = hidden_states.dtype,
            device = hidden_states.device,
            compact = bool(module.__dict__.get(_COMPACT_ATTR)),
        )
        cache[key] = _Entry(sources, states)
        stats["misses"] += 1
    else:
        states = entry.states
        stats["hits"] += 1
        if offloaded:
            _match_prefetch_order(module, ich is not None, skipping = True)
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
    """Call at load, before the speed layer binds the class forward. True when the conditioning cache can arm."""
    cond_on = enabled()
    compact = compact_enabled()
    if transformer is None or not (cond_on or compact):
        return False
    applies = _applies(transformer)
    if applies is not None and (not compact or "not a FLUX.3" in applies):
        if cond_on:
            _stats(transformer)["reason"] = applies
            if logger is not None and "not a FLUX.3" not in applies:
                logger.info("diffusion.cond_blocks: off (%s)", applies)
        return False
    why = patch_class(type(transformer))
    if why is not None:
        _stats(transformer)["reason"] = why
        if logger is not None:
            logger.info("diffusion.cond_blocks: off (%s)", why)
        return False
    transformer.__dict__[_COMPACT_ATTR] = compact
    if compact and logger is not None:
        logger.info("diffusion.cond_blocks: FLUX.3 modulation heads project one row per sample")
    if not cond_on:
        return False
    if applies is not None:
        _stats(transformer)["reason"] = applies
        if logger is not None:
            logger.info("diffusion.cond_blocks: off (%s)", applies)
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
    if why is not None and (not offload_cache_enabled() or _fixed_prefetch_chain(transformer)):
        _stats(transformer)["reason"] = _fixed_prefetch_chain(transformer) or why
        return False
    transformer.__dict__[_OFFLOADED_ATTR] = why is not None
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
