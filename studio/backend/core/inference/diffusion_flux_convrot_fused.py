# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 / FLUX.3 int8 ConvRot without the separate rotate + quantize pass over each bf16 activation.

On the int8 GEMM swap (``diffusion_int8_gemm``) every rotated Linear quantizes its own input with ``rotq_h16``: the
producer writes bf16, the kernel reads it back. Here:
  * Linears reading one activation (Q / K / V, and on FLUX.3 also the MLP input) share one rotation + act quant
    (``shared_rotq_linears``: the same op on the same tensor, so bit-identical);
  * the MLP down projection and FLUX.2's single-block output projection read the SwiGLU halves (and the attention
    output) straight from the producers' buffers (``rotq_h16x``), so neither the product nor the concat is written.
Every patch checks its Linears per call and runs the stock code when they are not swapped rotated int8 Linears.
Kill switch ``UNSLOTH_DIFFUSION_FLUX_CONVROT_FUSED=0``; ``UNSLOTH_DIFFUSION_INT8_ROTQUANT_PRODUCER=0`` keeps only the
shared quants."""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Optional

import torch

from . import diffusion_int8_gemm as _gemm

FLUX_CONVROT_FUSED_ENV = "UNSLOTH_DIFFUSION_FLUX_CONVROT_FUSED"
FLUX3_SPAN_FUSED_ENV = "UNSLOTH_DIFFUSION_FLUX3_SPAN_FUSED"
_MARK = "__unsloth_flux_convrot_fused__"
_LOCK = threading.Lock()
_STOCK: dict = {}  # (owner, attr) -> original
# Flux3SingleTransformerBlock class -> the diffusers forward diffusion_cond_blocks records as its stock (span patch)
_SPAN_STOCK: dict = {}
_CALLS = {"shared": [0], "producer": [0], "block": [0]}

# Stock lines each replacement mirrors; a diffusers whose body lacks any keeps the stock function.
_NEEDLES = {
    "f2_projections": (
        "query = attn.to_q(hidden_states)",
        "key = attn.to_k(hidden_states)",
        "value = attn.to_v(hidden_states)",
        "encoder_query = attn.add_q_proj(encoder_hidden_states)",
        "encoder_key = attn.add_k_proj(encoder_hidden_states)",
        "encoder_value = attn.add_v_proj(encoder_hidden_states)",
    ),
    "f3_projections": (
        "return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)",
    ),
    "ff": ("x = self.linear_in(x)", "x = self.act_fn(x)", "x = self.linear_out(x)"),
    "swiglu": ("half = x.shape[-1] // 2", "x = self.gate_fn(x[..., :half]) * x[..., half:]"),
    "f2_parallel": (
        "hidden_states = attn.to_qkv_mlp_proj(hidden_states)",
        "query, key, value = qkv.chunk(3, dim=-1)",
        "query = attn.norm_q(query)",
        "key = attn.norm_k(key)",
        "query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)",
        "key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)",
        "attn_mask=attention_mask,",
        "backend=self._attention_backend,",
        "parallel_config=self._parallel_config,",
        "hidden_states = hidden_states.flatten(2, 3)",
        "hidden_states = hidden_states.to(query.dtype)",
        "mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)",
        "hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)",
        "hidden_states = attn.to_out(hidden_states)",
    ),
    "f3_attn": (
        "query, key, value = _get_qkv_projections(attn, hidden_states)",
        "query = query.unflatten(-1, (attn.heads, -1))",
        "query = attn.norm_q(query.float())",
        "key = attn.norm_k(key.float())",
        "query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)",
        "query, key = query.to(value.dtype), key.to(value.dtype)",
        "attn_mask=attention_mask,",
        "hidden_states = hidden_states.flatten(2, 3)",
        "hidden_states = hidden_states.to(query.dtype)",
        "hidden_states = attn.to_out[0](hidden_states)",
        "hidden_states = attn.to_out[1](hidden_states)",
    ),
    "f3_attn_forward": (
        "return self.processor(self, hidden_states, attention_mask, image_rotary_emb)",
    ),
    "f3_mode": (
        "mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)",
        "norm_hidden_states = self.norm(hidden_states)",
        "norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift",
        "attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)",
        "ff_output = self.ff(norm_hidden_states)",
        "return hidden_states + mod_gate * (attn_output + ff_output)",
    ),
    "f3_single": (
        "mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)",
        "c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)",
        "norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift",
        "norm_encoder_hidden_states = self.norm(encoder_hidden_states)",
        "norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift",
        "combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)",
        "attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)",
        "ff_output = self.ff(combined)",
        "output = attn_output + ff_output",
        "encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output",
        "hidden_states = hidden_states + mod_gate * content_output",
    ),
}


def span_single_enabled() -> bool:
    """FLUX.3 joint blocks under the span-aware forward share the quant too (``UNSLOTH_DIFFUSION_FLUX3_SPAN_FUSED=0``:
    they keep a separate MLP-input quant, as before)."""
    return (os.environ.get(FLUX3_SPAN_FUSED_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def flux_convrot_fused_disabled() -> bool:
    return (os.environ.get(FLUX_CONVROT_FUSED_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _flux_modules() -> tuple:
    try:
        from diffusers.models.transformers import transformer_flux2 as f2
    except Exception:  # noqa: BLE001 - diffusers without this family
        f2 = None
    try:
        from diffusers.models.transformers import transformer_flux3 as f3
    except Exception:  # noqa: BLE001
        f3 = None
    return f2, f3


def _count(kind: str) -> None:
    """Engagement census, eager calls only (a traced forward counts nothing: no side effect in the graph)."""
    if not torch.compiler.is_compiling():
        _CALLS[kind][0] += 1


def _stock(owner: Any, attr: str) -> Any:
    """The original ``owner.attr`` (ours stashed it), else the live one."""
    return _STOCK.get((owner, attr), getattr(owner, attr, None))


def _has(fn: Any, key: str) -> bool:
    from .diffusion_autoopt.patches import body_has
    return fn is not None and body_has(fn, *_NEEDLES[key])


def _rotated(lin: Any) -> bool:
    """``lin`` is a swapped rotated int8 Linear whose act quant the fused rotation covers."""
    rec = _gemm._swapped_rec(lin)
    return rec is not None and bool(rec[2]) and rec[1] == 256


def _plain_swiglu(act: Any, cls_name: str) -> bool:
    """``act`` is the stock FLUX SwiGLU (``silu(x[:half]) * x[half:]``) with an unhooked ``nn.SiLU``."""
    return (
        act is not None
        and type(act) is _CLS.get(cls_name)
        and not _gemm._hooked(act)
        and _gemm._plain_silu(getattr(act, "gate_fn", None))
    )


# Traced gates compare classes, never functions: dynamo resolves ``f.__func__ is g`` to False.
_CLS: dict = {}


def _ff_ready(ff: Any, act_cls: str) -> bool:
    return (
        not _gemm._hooked(ff)
        and _plain_swiglu(getattr(ff, "act_fn", None), act_cls)
        and _rotated(ff.linear_out)
    )


def _ff_tail(ff: Any, h: Any) -> Any:
    """``ff.linear_out(ff.act_fn(h))``: the down projection reads the two halves of ``h`` directly when it can."""
    half = h.shape[-1] // 2
    y = _gemm.producer_linear(ff.linear_out, None, h[..., :half], h[..., half:])
    if y is None:
        return ff.linear_out(ff.act_fn(h))
    _count("producer")
    return y


def _make_ff_forward(stock: Any, act_cls: str) -> Any:
    def forward(self, x):
        if not _ff_ready(self, act_cls):
            return stock(self, x)
        return _ff_tail(self, self.linear_in(x))

    functools.update_wrapper(
        forward, stock, assigned = ("__name__", "__qualname__", "__doc__"), updated = ()
    )
    setattr(forward, _MARK, True)
    return forward


def _make_f2_projections(stock: Any) -> Any:
    def _get_projections(
        attn,
        hidden_states,
        encoder_hidden_states = None,
    ):
        qkv = None
        if _rotated(attn.to_q):
            qkv = _gemm.shared_rotq_linears(hidden_states, (attn.to_q, attn.to_k, attn.to_v))
        if qkv is None:
            return stock(attn, hidden_states, encoder_hidden_states)
        _count("shared")
        query, key, value = qkv
        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            added = None
            if _rotated(attn.add_q_proj):
                added = _gemm.shared_rotq_linears(
                    encoder_hidden_states, (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
                )
            if added is None:
                added = (
                    attn.add_q_proj(encoder_hidden_states),
                    attn.add_k_proj(encoder_hidden_states),
                    attn.add_v_proj(encoder_hidden_states),
                )
            encoder_query, encoder_key, encoder_value = added
        return query, key, value, encoder_query, encoder_key, encoder_value

    functools.update_wrapper(_get_projections, stock, assigned = ("__name__", "__doc__"), updated = ())
    setattr(_get_projections, _MARK, True)
    return _get_projections


def _make_f3_projections(stock: Any) -> Any:
    def _get_projections(attn, hidden_states):
        qkv = None
        if _rotated(attn.to_q):
            qkv = _gemm.shared_rotq_linears(hidden_states, (attn.to_q, attn.to_k, attn.to_v))
        if qkv is None:
            return stock(attn, hidden_states)
        _count("shared")
        return tuple(qkv)

    functools.update_wrapper(_get_projections, stock, assigned = ("__name__", "__doc__"), updated = ())
    setattr(_get_projections, _MARK, True)
    return _get_projections


def _parallel_ready(attn: Any) -> bool:
    """FLUX.2 single-block attention whose output projection can read [attention | SwiGLU] from the producers.
    Traced: plain attribute reads only (no try / except)."""
    proj = getattr(attn, "to_qkv_mlp_proj", None)
    out = getattr(attn, "to_out", None)
    act = getattr(attn, "mlp_act_fn", None)
    if proj is None or out is None or act is None or getattr(attn, "mlp_mult_factor", None) != 2:
        return False
    return (
        getattr(proj, "out_features", None)
        == 3 * attn.inner_dim + 2 * attn.mlp_hidden_dim  # unsharded
        and getattr(out, "in_features", None) == attn.inner_dim + attn.mlp_hidden_dim
        and _plain_swiglu(act, "Flux2SwiGLU")
        and _rotated(out)
    )


def _make_f2_parallel_call(stock: Any, mod: Any) -> Any:
    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask = None,
        image_rotary_emb = None,
    ):
        if not _parallel_ready(attn):
            return stock(self, attn, hidden_states, attention_mask, image_rotary_emb)
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv_dim = 3 * attn.inner_dim
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [qkv_dim, hidden_states.shape[-1] - qkv_dim], dim = -1
        )
        query, key, value = qkv.chunk(3, dim = -1)
        query = query.unflatten(-1, (-1, attn.head_dim))
        key = key.unflatten(-1, (-1, attn.head_dim))
        value = value.unflatten(-1, (-1, attn.head_dim))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            # looked up per call: diffusion_flux2_rope may have replaced it
            query = mod.apply_rotary_emb(query, image_rotary_emb, sequence_dim = 1)
            key = mod.apply_rotary_emb(key, image_rotary_emb, sequence_dim = 1)
        hidden_states = mod.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask = attention_mask,
            backend = self._attention_backend,
            parallel_config = self._parallel_config,
        )
        hidden_states = hidden_states.to(query.dtype)
        half = mlp_hidden_states.shape[-1] // 2
        out = _gemm.producer_linear(
            attn.to_out, hidden_states, mlp_hidden_states[..., :half], mlp_hidden_states[..., half:]
        )
        if out is not None:
            _count("producer")
            return out
        hidden_states = hidden_states.flatten(2, 3)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim = -1)
        return attn.to_out(hidden_states)

    functools.update_wrapper(
        __call__, stock, assigned = ("__name__", "__qualname__", "__doc__"), updated = ()
    )
    setattr(__call__, _MARK, True)
    return __call__


def _f3_block_ready(block: Any) -> bool:
    """A FLUX.3 block whose attention and MLP can share one rotation + act quant of their common input. Traced: plain
    attribute reads only."""
    attn, ff = getattr(block, "attn", None), getattr(block, "ff", None)
    if attn is None or ff is None or getattr(attn, "fused_projections", False):
        return False
    proc = getattr(attn, "processor", None)
    return (
        type(proc) is _CLS.get("Flux3AttnProcessor")
        and type(attn) is _CLS.get("Flux3Attention")
        and not _gemm._hooked(attn)
        and type(ff) is _CLS.get("Flux3FeedForward")
        and _ff_ready(ff, "Flux3SwiGLU")
        and _rotated(attn.to_q)
        and _rotated(attn.to_k)
        and _rotated(attn.to_v)
        and _rotated(ff.linear_in)
    )


def _f3_attn_ff(block: Any, x: Any, image_rotary_emb: Any, mod: Any) -> Optional[tuple]:
    """(attention output, MLP output) of ``x`` with Q / K / V and the MLP input on one rotation + act quant; None:
    the caller runs the stock modules. Mirrors ``Flux3AttnProcessor.__call__`` below the projections."""
    if not _f3_block_ready(block):
        return None
    attn, ff = block.attn, block.ff
    parts = _gemm.shared_rotq_linears(x, (attn.to_q, attn.to_k, attn.to_v, ff.linear_in))
    if parts is None:
        return None
    _count("block")
    query, key, value, h = parts
    proc = attn.processor
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query.float())
    key = attn.norm_k(key.float())
    if image_rotary_emb is not None:
        # looked up per call: diffusion_flux3_rope may have replaced it
        query = mod.apply_rotary_emb(query, image_rotary_emb, sequence_dim = 1)
        key = mod.apply_rotary_emb(key, image_rotary_emb, sequence_dim = 1)
    query, key = query.to(value.dtype), key.to(value.dtype)
    hidden_states = mod.dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask = None,
        backend = proc._attention_backend,
        parallel_config = proc._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states, _ff_tail(ff, h)


def _make_f3_mode_forward(stock: Any, mod: Any) -> Any:
    def forward(self, hidden_states, temb_mod, image_rotary_emb):
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim = -1)
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        both = _f3_attn_ff(self, norm_hidden_states, image_rotary_emb, mod)
        if both is None:
            attn_output = self.attn(
                hidden_states = norm_hidden_states, image_rotary_emb = image_rotary_emb
            )
            ff_output = self.ff(norm_hidden_states)
        else:
            attn_output, ff_output = both
        return hidden_states + mod_gate * (attn_output + ff_output)

    functools.update_wrapper(
        forward, stock, assigned = ("__name__", "__qualname__", "__doc__"), updated = ()
    )
    setattr(forward, _MARK, True)
    return forward


def _make_f3_single_forward(stock: Any, mod: Any) -> Any:
    def forward(
        self, hidden_states, encoder_hidden_states, temb_mod, temb_mod_context, image_rotary_emb
    ):
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim = -1)
        c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim = -1)
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift
        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim = 1)
        both = _f3_attn_ff(self, combined, image_rotary_emb, mod)
        if both is None:
            attn_output = self.attn(hidden_states = combined, image_rotary_emb = image_rotary_emb)
            ff_output = self.ff(combined)
        else:
            attn_output, ff_output = both
        output = attn_output + ff_output
        context_output, content_output = torch.split(
            output, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim = 1
        )
        encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
        hidden_states = hidden_states + mod_gate * content_output
        return encoder_hidden_states, hidden_states

    functools.update_wrapper(
        forward, stock, assigned = ("__name__", "__qualname__", "__doc__"), updated = ()
    )
    setattr(forward, _MARK, True)
    return forward


def _patch(owner: Any, attr: str, key: str, make: Any) -> bool:
    """Swap ``owner.attr`` for ``make(stock)`` when the stock body carries ``_NEEDLES[key]``; idempotent."""
    current = getattr(owner, attr, None)
    if current is None:
        return False
    if getattr(current, _MARK, False):
        return True
    if not _has(current, key):
        return False
    _STOCK[(owner, attr)] = current
    setattr(owner, attr, make(current))
    return True


def _patch_span_single(cls: Any, mod: Any) -> bool:
    """FLUX.3 joint block under ``diffusion_cond_blocks``' span-aware forward (installed first, at load): that forward
    runs the stock diffusers one recorded in ``_PATCHED_SINGLE`` whenever a render has no reference span, so the
    shared quant goes there; with a span its own body reads ``JOINT_ATTN_FF`` (``_f3_attn_ff``). Idempotent."""
    try:
        from . import diffusion_cond_blocks as cond
    except Exception:  # noqa: BLE001
        return False
    span_fwd = getattr(cond, "_flux3_span_single_forward", None)
    registry = getattr(cond, "_PATCHED_SINGLE", None)
    if span_fwd is None or registry is None or getattr(cls, "forward", None) is not span_fwd:
        return False
    stock = registry.get(cls)
    if stock is None:
        return False
    if not getattr(stock, _MARK, False):
        if not _has(stock, "f3_single"):
            return False
        _SPAN_STOCK[cls] = stock
        registry[cls] = _make_f3_single_forward(stock, mod)
    _SPAN_MOD[0] = mod
    cond.JOINT_ATTN_FF = _span_attn_ff
    return True


# the diffusers FLUX.3 module whose apply_rotary_emb / dispatch_attention_fn the span path uses
_SPAN_MOD: list = [None]


def _span_attn_ff(block: Any, x: Any, image_rotary_emb: Any) -> Optional[tuple]:
    """``diffusion_cond_blocks.JOINT_ATTN_FF``: one stable function, so a compiled span block never guards on a new
    object per install."""
    return _f3_attn_ff(block, x, image_rotary_emb, _SPAN_MOD[0])


def _uninstall_span_single() -> None:
    try:
        from . import diffusion_cond_blocks as cond
    except Exception:  # noqa: BLE001
        _SPAN_STOCK.clear()
        return
    registry = getattr(cond, "_PATCHED_SINGLE", None)
    for cls, stock in list(_SPAN_STOCK.items()):
        if registry is not None and getattr(registry.get(cls), _MARK, False):
            registry[cls] = stock
    _SPAN_STOCK.clear()
    _SPAN_MOD[0] = None
    if getattr(cond, "JOINT_ATTN_FF", None) is _span_attn_ff:
        cond.JOINT_ATTN_FF = None


def _install_patches(logger: Any = None) -> dict:
    done = {}
    f2, f3 = _flux_modules()
    with _LOCK:
        for mod, prefix, swiglu, ff_cls in (
            (f2, "f2", "Flux2SwiGLU", "Flux2FeedForward"),
            (f3, "f3", "Flux3SwiGLU", "Flux3FeedForward"),
        ):
            if mod is None:
                continue
            act = getattr(mod, swiglu, None)
            if act is not None and _has(act.forward, "swiglu"):
                _CLS[swiglu] = act
            proj_key = f"{prefix}_projections"
            done[proj_key] = _patch(
                mod,
                "_get_projections",
                proj_key,
                _make_f2_projections if prefix == "f2" else _make_f3_projections,
            )
            ff = getattr(mod, ff_cls, None)
            if ff is not None and swiglu in _CLS:
                done[f"{prefix}_ff"] = _patch(
                    ff, "forward", "ff", lambda s, c = swiglu: _make_ff_forward(s, c)
                )
                if done[f"{prefix}_ff"]:
                    _CLS[ff_cls] = ff
        if f2 is not None and "Flux2SwiGLU" in _CLS:
            proc = getattr(f2, "Flux2ParallelSelfAttnProcessor", None)
            if proc is not None:
                done["f2_parallel"] = _patch(
                    proc, "__call__", "f2_parallel", lambda s: _make_f2_parallel_call(s, f2)
                )
        if f3 is not None and "Flux3FeedForward" in _CLS:
            proc, attn_cls = (
                getattr(f3, "Flux3AttnProcessor", None),
                getattr(f3, "Flux3Attention", None),
            )
            ff = getattr(f3, "Flux3FeedForward", None)
            if (
                proc is not None
                and attn_cls is not None
                and ff is not None
                and _has(proc.__call__, "f3_attn")
                and _has(attn_cls.forward, "f3_attn_forward")
            ):
                _CLS["Flux3AttnProcessor"] = proc
                _CLS["Flux3Attention"] = attn_cls
                for cls_name, key, make in (
                    ("Flux3ModeTransformerBlock", "f3_mode", _make_f3_mode_forward),
                    ("Flux3SingleTransformerBlock", "f3_single", _make_f3_single_forward),
                ):
                    cls = getattr(f3, cls_name, None)
                    if cls is not None:
                        done[key] = _patch(cls, "forward", key, lambda s, m = make: m(s, f3))
                        if key == "f3_single" and not done[key] and span_single_enabled():
                            done[key] = _patch_span_single(cls, f3)
    if logger is not None:
        logger.info("diffusion.flux_convrot_fused: patches %s", done)
    return done


def _rotated_count(transformer: Any) -> int:
    n = 0
    for module in transformer.modules():
        rec = _gemm._swapped_rec(module) if hasattr(module, "weight") else None
        if rec is not None and rec[2] and rec[1] == 256:
            n += 1
    return n


def _rotated_linears(transformer: Any) -> int:
    """ConvRot (group 256) Linears in the denoiser, swapped yet or not."""
    from .diffusion_convrot import is_rotated_linear
    return sum(
        1
        for m in transformer.modules()
        if is_rotated_linear(m) and int(getattr(m, "convrot_groupsize", 0) or 0) == 256
    )


def _arm_cards(transformer: Any, logger: Any = None) -> list:
    """``rotq_h16x`` on every card the denoiser sits on (outside any trace / capture)."""
    from .diffusion_convrot_quant import _arm_producer
    from .diffusion_int8_fused import resident_cuda_cards

    cards = [index for index in resident_cuda_cards(transformer) or () if _arm_producer(index)]
    if logger is not None:
        logger.info(
            "diffusion.flux_convrot_fused: %d swapped rotated int8 Linears; producer-fused rotation on cards %s",
            _rotated_count(transformer),
            cards,
        )
    return cards


def install(transformer: Any, logger: Any = None) -> dict:
    """After the int8 GEMM swap, before the regional compile: patch the FLUX.2 / FLUX.3 helpers (each checks its
    Linears per call) and arm ``rotq_h16x`` on the denoiser's cards, at the first forward when the weights (and so
    the swap) are not on the GPU yet. Nothing without ConvRot Linears."""
    result = {"rotated": 0, "producer_cards": None, "patches": {}}
    if flux_convrot_fused_disabled():
        uninstall()
        return result
    if type(transformer).__name__ not in ("Flux2Transformer2DModel", "Flux3Transformer2DModel"):
        return result
    try:
        result["rotated"] = _rotated_linears(transformer)
        if not result["rotated"]:
            return result
        result["patches"] = _install_patches(logger)
        from .diffusion_int8_fused import resident_cuda_cards, run_on_first_call

        if resident_cuda_cards(transformer) and _rotated_count(transformer):
            result["producer_cards"] = _arm_cards(transformer, logger)
        else:
            run_on_first_call(transformer, "flux_convrot_fused", lambda t: _arm_cards(t, logger))
    except Exception as exc:  # noqa: BLE001 - optimisation only: stock helpers stay
        if logger is not None:
            logger.info("diffusion.flux_convrot_fused: not installed (%s)", exc)
    return result


def uninstall() -> None:
    with _LOCK:
        _uninstall_span_single()
        for key, stock in list(_STOCK.items()):
            owner, attr = key
            if isinstance(owner, str):
                continue
            if getattr(getattr(owner, attr, None), _MARK, False):
                setattr(owner, attr, stock)
        _STOCK.clear()
        _CLS.clear()


def is_installed() -> bool:
    return bool(_SPAN_STOCK) or any(not isinstance(k[0], str) for k in _STOCK)


def call_counts() -> dict:
    return {k: v[0] for k, v in _CALLS.items()}
