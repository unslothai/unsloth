# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Z-Image-Turbo attention inside the regional compile: RoPE as ``addcmul`` in the card's probed fma form
(bit-identical to the complex multiply Inductor cannot lower) and one fused QKV GEMM (exact for int8:
per-row scales). ``to_q/to_k/to_v`` weights become views of the fused one; the fused Linear is used only
while they are still the modules it was built from, so LoRA / later wrappers fall back to stock.

int8 projections (fused QKV, ``to_out``) run through ``diffusion_int8_fused.int8_linear``: torchao's own math without
torchao 0.17's always-on zero-point correction, which re-sums the weight rows on every call.

ConvRot (int8 with a block-Hadamard input rotation): q, k and v rotate the SAME input, and the rotation acts on the
shared input axis, so the three rotated weights concatenate along the output axis like plain ones. The fused Linear is
then itself a ConvRotLinear at the parts' group: one rotation and one activation quant feed the fused GEMM instead of
three of each. Same ops on the same input, so the fused output equals the three separate rotated projections.
Kill switch: ``UNSLOTH_DIFFUSION_ZIMAGE_FUSED=0``.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Optional

ZIMAGE_FUSED_ENV = "UNSLOTH_DIFFUSION_ZIMAGE_FUSED"
_MODULE = "diffusers.models.transformers.transformer_z_image"
_FINGERPRINTS = {"ZSingleStreamAttnProcessor.__call__": frozenset({"c87f9c27d7df695f"})}
_QKV_ATTR = "_unsloth_fused_qkv"

_LOCK = threading.Lock()
_STATE: dict = {}
# device index -> (even lanes, odd lanes) fma form of this card's complex multiply, or None
_FUSION: dict = {}


def zimage_fused_disabled() -> bool:
    return (os.environ.get(ZIMAGE_FUSED_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _stock_digest(cls: Any) -> Optional[str]:
    from .diffusion_qwenimage21_rope import _digest
    return _digest(getattr(cls, "__call__", None))


def _real_rope(x: Any, freqs_cis: Any, fusion: tuple) -> Any:
    """``view_as_real(view_as_complex(x.float()) * freqs_cis.unsqueeze(2)).flatten(3).type_as(x)`` in real ops.
    ``x``: [B, S, H, D]; ``freqs_cis``: [B, S, D/2] complex64."""
    import torch

    xf = x.float()
    f = torch.view_as_real(freqs_cis).unsqueeze(2)
    cos = f[..., :1].expand(*f.shape[:-1], 2).flatten(-2)
    sin = torch.stack([-f[..., 1], f[..., 1]], dim = -1).flatten(-2)
    swapped = xf.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    on_x = torch.addcmul(swapped * sin, xf, cos) if "x" in fusion else None
    on_s = torch.addcmul(xf * cos, swapped, sin) if "s" in fusion else None
    if on_s is None:
        out = on_x
    elif on_x is None:
        out = on_s
    else:
        even = torch.arange(x.shape[-1], device = x.device) % 2 == 0
        out = torch.where(even, on_x, on_s) if fusion[0] == "x" else torch.where(even, on_s, on_x)
    return out.type_as(x)


def _qkv_intact(attn: Any) -> Any:
    """The fused Linear, if ``to_q/to_k/to_v`` are still the exact modules it was built from."""
    rec = attn.__dict__.get(_QKV_ATTR)
    if rec is None:
        return None
    lin, q, k, v = rec
    if attn.to_q is not q or attn.to_k is not k or attn.to_v is not v:
        return None
    return lin


def _make_call(stock: Any) -> Any:
    import torch

    # bound here, not inside the traced __call__
    from .diffusion_int8_fused import int8_linear

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        freqs_cis = None,
    ):
        if not torch.compiler.is_compiling():
            return stock(
                self, attn, hidden_states, encoder_hidden_states, attention_mask, freqs_cis
            )
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        fused = _qkv_intact(attn)
        if fused is not None:
            query, key, value = int8_linear(fused, hidden_states).chunk(3, dim = -1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if freqs_cis is not None:
            fusion = _FUSION.get(query.device.index) if query.is_cuda else None
            if (
                fusion is not None
                and freqs_cis.dtype == torch.complex64
                and query.shape[-1] % 2 == 0
            ):
                query = _real_rope(query, freqs_cis, fusion)
                key = _real_rope(key, freqs_cis, fusion)
            else:
                query = _stock_rope(query, freqs_cis)
                key = _stock_rope(key, freqs_cis)
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask = attention_mask,
            dropout_p = 0.0,
            is_causal = False,
            backend = self._attention_backend,
            parallel_config = self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = int8_linear(attn.to_out[0], hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    functools.update_wrapper(__call__, stock, assigned = ("__name__", "__doc__"), updated = ())
    __call__.__unsloth_zimage_fused__ = True
    return __call__


def _stock_rope(x_in: Any, freqs_cis: Any) -> Any:
    import torch
    with torch.amp.autocast("cuda", enabled = False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
        return x_out.type_as(x_in)


def _rotation_group(linears: list) -> Any:
    """0 when every part is a plain ``nn.Linear``, the shared ConvRot group when every part rotates its input at the
    same group, else None (a mix cannot share one input)."""
    from torch import nn

    if all(type(lin) is nn.Linear for lin in linears):
        return 0
    try:
        from .diffusion_convrot import convrot_linear_class
        cls = convrot_linear_class()
    except Exception:  # noqa: BLE001
        return None
    if not all(type(lin) is cls for lin in linears):
        return None
    groups = {getattr(lin, "convrot_groupsize", None) for lin in linears}
    if len(groups) != 1:
        return None
    group = groups.pop()
    if not isinstance(group, int) or group <= 0 or linears[0].in_features % group:
        return None
    return group


def _fuse_linears(linears: list) -> Any:
    """One Linear over the concatenated outputs, or None; torchao tensors are rebuilt from their data/attribute lists.
    Parts that all rotate their input at one ConvRot group give a fused ConvRotLinear at that group."""
    import torch
    from torch import nn

    ws = [lin.weight for lin in linears]
    group = _rotation_group(linears)
    if group is None:
        return None
    has_bias = [lin.bias is not None for lin in linears]
    if any(has_bias) and not all(has_bias):
        return None
    cls = type(ws[0])
    if any(type(w) is not cls for w in ws) or any(
        w.dtype != ws[0].dtype or w.device != ws[0].device for w in ws
    ):
        return None
    if any(w.shape[1] != ws[0].shape[1] for w in ws):
        return None
    if cls in (torch.Tensor, nn.Parameter):
        fused_w = torch.cat([w.detach() for w in ws], dim = 0)
    else:
        names = list(getattr(cls, "tensor_data_names", ()))
        attrs = list(getattr(cls, "tensor_attribute_names", ()))
        opt_t = list(getattr(cls, "optional_tensor_data_names", ()))
        opt_a = list(getattr(cls, "optional_tensor_attribute_names", ()))
        if not names:
            return None
        kwargs: dict = {}
        for name in names + opt_t:
            parts = [getattr(w, name, None) for w in ws]
            if all(p is None for p in parts):
                continue
            if any(
                p is None or p.dim() == 0 or p.shape[0] != w.shape[0] for p, w in zip(parts, ws)
            ):
                return None
            kwargs[name] = torch.cat(parts, dim = 0)
        for name in attrs + opt_a:
            first = getattr(ws[0], name, None)
            if any(getattr(w, name, None) != first for w in ws[1:]):
                return None
            kwargs[name] = first
        try:
            fused_w = cls(**kwargs)
        except Exception:  # noqa: BLE001 - an unknown constructor keeps the stock projections
            return None
    out = nn.Linear(ws[0].shape[1], sum(w.shape[0] for w in ws), bias = False, device = "meta")
    out.weight = nn.Parameter(fused_w, requires_grad = False)
    if all(has_bias):
        out.bias = nn.Parameter(
            torch.cat([lin.bias.detach() for lin in linears]), requires_grad = False
        )
    if group:
        from .diffusion_convrot import _install_rotation
        _install_rotation(out, group)
    return out


def _share_storage(fused: Any, linears: list) -> bool:
    from torch import nn

    start = 0
    views = []
    try:
        whole = fused.weight.detach()
        for lin in linears:
            n = lin.weight.shape[0]
            view = whole[start : start + n]
            if tuple(view.shape) != tuple(lin.weight.shape) or type(view) is not type(
                lin.weight.detach()
            ):
                return False
            views.append(view)
            start += n
        for lin, view in zip(linears, views):
            lin.weight = nn.Parameter(view, requires_grad = False)
    except Exception:  # noqa: BLE001
        return False
    return True


def install(
    transformer: Any,
    logger: Any = None,
    offload_active: bool = False,
) -> dict:
    """Before the first compile; probe + QKV fusion wait for the first forward if the weights are not on the GPU."""
    if zimage_fused_disabled():
        uninstall()  # a previous load may have patched the process-global class
        return {"real_rope": False, "fused_qkv": 0}
    if type(transformer).__name__ != "ZImageTransformer2DModel":
        return {"real_rope": False, "fused_qkv": 0}
    if not _patch_class(logger):
        return {"real_rope": False, "fused_qkv": 0}
    from .diffusion_int8_fused import resident_cuda_device, run_on_first_call

    if resident_cuda_device(transformer) is not None:
        return install_modules(transformer, logger, fuse_qkv = not offload_active)
    run_on_first_call(
        transformer,
        "zimage_fused",
        lambda t: install_modules(t, logger, fuse_qkv = not offload_active),
    )
    return {"real_rope": None, "fused_qkv": None}


def _patch_class(logger: Any = None) -> bool:
    try:
        import importlib
        cls = importlib.import_module(_MODULE).ZSingleStreamAttnProcessor
    except Exception:  # noqa: BLE001
        return False
    with _LOCK:
        current = cls.__dict__.get("__call__")
        if getattr(current, "__unsloth_zimage_fused__", False):
            return True
        digest = _stock_digest(cls)
        if digest not in _FINGERPRINTS["ZSingleStreamAttnProcessor.__call__"]:
            if logger is not None:
                logger.info(
                    "diffusion.zimage_fused: stock attention kept: processor differs (%s)", digest
                )
            return False
        _STATE["stock"] = current
        cls.__call__ = _make_call(current)
    return True


def install_modules(
    root: Any,
    logger: Any = None,
    fuse_qkv: bool = True,
) -> dict:
    result = {"real_rope": False, "fused_qkv": 0}
    if zimage_fused_disabled() or not _patch_class(logger):
        return result
    import importlib

    import torch

    from .diffusion_int8_fused import resident_cuda_device

    cls = importlib.import_module(_MODULE).ZSingleStreamAttnProcessor
    dev = resident_cuda_device(root)
    if dev is None:
        return result
    with _LOCK:
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        if index not in _FUSION:
            try:
                from .diffusion_qwenimage21_rope import inductor_addcmul_is_fma, probe_fusion
                _FUSION[index] = (
                    probe_fusion(torch.device("cuda", index)) if inductor_addcmul_is_fma() else None
                )
            except Exception:  # noqa: BLE001
                _FUSION[index] = None
        result["real_rope"] = _FUSION[index] is not None
        for _name, module in root.named_modules():
            if not fuse_qkv or type(getattr(module, "processor", None)) is not cls:
                continue
            if _QKV_ATTR in module.__dict__:
                result["fused_qkv"] += 1
                continue
            parts = [getattr(module, n, None) for n in ("to_q", "to_k", "to_v")]
            if any(p is None for p in parts):
                continue
            fused = _fuse_linears(parts)
            if fused is None or not _share_storage(fused, parts):
                continue
            module.__dict__[_QKV_ATTR] = (fused, *parts)
            result["fused_qkv"] += 1
    if logger is not None:
        logger.info(
            "diffusion.zimage_fused: real-arithmetic RoPE %s, fused QKV on %d attention module(s)",
            "on" if result["real_rope"] else "off (card's complex multiply not a known fma form)",
            result["fused_qkv"],
        )
    return result


def uninstall(transformer: Any = None) -> None:
    """Restore the stock processor; drop the fused Linears (the per-row views stay valid) and pending install."""
    if transformer is not None:
        from .diffusion_int8_fused import cancel_first_call
        cancel_first_call(transformer, "zimage_fused")
    with _LOCK:
        if "stock" in _STATE:
            try:
                import importlib
                cls = importlib.import_module(_MODULE).ZSingleStreamAttnProcessor
                if getattr(cls.__dict__.get("__call__"), "__unsloth_zimage_fused__", False):
                    cls.__call__ = _STATE.pop("stock")
            except Exception:  # noqa: BLE001
                pass
        if transformer is not None:
            for module in transformer.modules():
                module.__dict__.pop(_QKV_ATTR, None)
