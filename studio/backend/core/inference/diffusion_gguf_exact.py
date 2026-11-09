# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF dequant bit-identical to llama.cpp's ``dequantize_row_*`` (diffusers' rounds in fp16), rounded once to the
compute dtype. Products are exact in fp32, so only ``+ m`` / ``- m`` rounds and FMA cannot change a bit.
UNSLOTH_DIFFUSION_GGUF_EXACT=0 keeps diffusers'."""

from __future__ import annotations

import os
from typing import Any

EXACT_ENV = "UNSLOTH_DIFFUSION_GGUF_EXACT"
_QK_K = 256
_CAST_ON_STORE = ("cuda", "cpu")
_KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def exact_enabled() -> bool:
    return os.environ.get(EXACT_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _scale(blocks: Any, word: int) -> Any:
    """fp16 at byte ``2 * word`` as fp32 ``(n, 1)``; view only 2 bytes: inductor copies a wider dtype view."""
    import torch
    return blocks[:, 2 * word : 2 * word + 2].view(torch.float16).to(torch.float32)


def _half_math(blocks: Any, dtype: Any) -> Any:
    import torch
    if dtype != torch.float16 or torch.compiler.is_compiling():
        return None
    return blocks[:, :2].view(torch.float16)


def _field(blocks: Any, start: int, stop: int, shape: tuple) -> Any:
    return blocks[:, start:stop].unflatten(1, shape)


def _shifts(like: Any, start: int, stop: int, step: int, shape: tuple) -> Any:
    import torch
    return torch.arange(start, stop, step, device = like.device, dtype = torch.uint8).reshape(shape)


def _emit(scale: Any, q: Any, m: Any, op: str, n: int, block_size: int, dtype: Any) -> Any:
    """``scale * q +/- m`` as ``(n, block_size)``, rounded once to ``dtype``. Only the contiguous result is reshaped
    (a strided reshape is 1.7x slower Q6_K compiled). Traced, not ``addcmul(-m, ...)``: Triton's ``0 - m`` loses -0."""
    import torch

    if (
        not torch.compiler.is_compiling()
        and dtype is not None
        and dtype != torch.float32
        and q.device.type in _CAST_ON_STORE
    ):
        out = torch.empty(
            torch.broadcast_shapes(scale.shape, q.shape), dtype = dtype, device = q.device
        )
        if op:
            torch.addcmul(m if op == "+" else -m, scale, q, out = out)
        else:
            torch.mul(scale, q, out = out)
        return out.reshape(n, block_size)
    y = scale * q
    if op == "+":
        y = y.add_(m)
    elif op == "-":
        y = y.sub_(m)
    y = y.reshape(n, block_size)
    return y if dtype is None or dtype == torch.float32 else y.to(dtype)


def _q4_codes(blocks: Any, at: int) -> Any:
    """``(n, 2, 2, 8)``: element ``16h + 8a + b`` is nibble ``h`` of byte ``8a + b``."""
    return (_field(blocks, at, at + 16, (1, 2, 8)) >> _shifts(blocks, 0, 8, 4, (2, 1, 1))) & 0x0F


def dequantize_blocks_Q4_0(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    q = _q4_codes(blocks, 2).to(torch.int8) - 8
    half = _half_math(blocks, dtype)
    if half is not None:
        return (half.reshape(n, 1, 1, 1) * q).reshape(n, 32)
    return _emit(_scale(blocks, 0).reshape(n, 1, 1, 1), q, None, "", n, 32, dtype)


def dequantize_blocks_Q4_1(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _scale(blocks, 0).reshape(n, 1, 1, 1)
    m = _scale(blocks, 1).reshape(n, 1, 1, 1)
    return _emit(d, _q4_codes(blocks, 4), m, "+", n, 32, dtype)


def _q5_codes(blocks: Any, qh_at: int) -> Any:
    """Code ``k`` takes bit ``k`` of the little-endian 32-bit ``qh``."""
    high = (_field(blocks, qh_at, qh_at + 4, (2, 2, 1)) >> _shifts(blocks, 0, 8, 1, (8,))) & 1
    return _q4_codes(blocks, qh_at + 4) | (high << 4)


def dequantize_blocks_Q5_0(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    q = _q5_codes(blocks, 2).to(torch.int8) - 16
    half = _half_math(blocks, dtype)
    if half is not None:
        return (half.reshape(n, 1, 1, 1) * q).reshape(n, 32)
    return _emit(_scale(blocks, 0).reshape(n, 1, 1, 1), q, None, "", n, 32, dtype)


def dequantize_blocks_Q5_1(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _scale(blocks, 0).reshape(n, 1, 1, 1)
    m = _scale(blocks, 1).reshape(n, 1, 1, 1)
    return _emit(d, _q5_codes(blocks, 4), m, "+", n, 32, dtype)


def dequantize_blocks_Q8_0(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    q = blocks[:, 2:].view(torch.int8)
    half = _half_math(blocks, dtype)
    if half is not None:
        return half * q
    return _emit(_scale(blocks, 0), q, None, "", n, 32, dtype)


def _k_codes(blocks: Any, at: int) -> Any:
    """``(n, chunk c, shift j, half h, 16)`` = ``(q[32c + 16h + l] >> 2j) & 3``; scale group ``8c + 2j + h``."""
    return (_field(blocks, at, at + 64, (2, 1, 2, 16)) >> _shifts(blocks, 0, 8, 2, (4, 1, 1))) & 3


def dequantize_blocks_Q2_K(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    scales = blocks[:, :16]
    d = _scale(blocks, 40)
    dmin = _scale(blocks, 41)
    dl = (d * (scales & 0x0F)).reshape(n, 2, 4, 2, 1)
    ml = (dmin * (scales >> 4)).reshape(n, 2, 4, 2, 1)
    return _emit(dl, _k_codes(blocks, 16), ml, "-", n, _QK_K, dtype)


def dequantize_blocks_Q3_K(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    scales = blocks[:, 96:108]
    d = _scale(blocks, 54)
    # ggml's aux[] unpack: nibble of byte g % 8, 2 high bits from byte 8 + g % 4
    low = (scales[:, :8].unsqueeze(1) >> _shifts(blocks, 0, 8, 4, (2, 1))) & 0x0F
    high = (scales[:, 8:].unsqueeze(1) >> _shifts(blocks, 0, 8, 2, (4, 1))) & 3
    sc = (low.reshape(n, 16) | (high.reshape(n, 16) << 4)).to(torch.int8) - 32
    dl = (d * sc).reshape(n, 2, 4, 2, 1)
    # hmask bit 4c + j clear -> subtract 4
    hb = (_field(blocks, 0, 32, (1, 1, 2, 16)) >> _shifts(blocks, 0, 8, 1, (2, 4, 1, 1))) & 1
    q = _k_codes(blocks, 32).to(torch.int8) - ((hb ^ 1) << 2).to(torch.int8)
    return _emit(dl, q, None, "", n, _QK_K, dtype)


def _scale_min_k4(scales: Any) -> tuple:
    """ggml's get_scale_min_k4 for j = 0..7."""
    import torch

    a, b, c = scales[:, :4], scales[:, 4:8], scales[:, 8:12]
    sc = torch.cat((a & 63, (c & 0x0F) | ((a >> 6) << 4)), dim = 1)
    m = torch.cat((b & 63, (c >> 4) | ((b >> 6) << 4)), dim = 1)
    return sc, m


def _k4_scales(blocks: Any) -> tuple:
    n = blocks.shape[0]
    d = _scale(blocks, 0)
    dmin = _scale(blocks, 1)
    sc, m = _scale_min_k4(blocks[:, 4:16])
    return (d * sc).reshape(n, 4, 2, 1), (dmin * m).reshape(n, 4, 2, 1)


def _k4_codes(blocks: Any, at: int) -> Any:
    return (_field(blocks, at, at + 128, (4, 1, 32)) >> _shifts(blocks, 0, 8, 4, (2, 1))) & 0x0F


def dequantize_blocks_Q4_K(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d1, m1 = _k4_scales(blocks)
    return _emit(d1, _k4_codes(blocks, 16), m1, "-", n, _QK_K, dtype)


def dequantize_blocks_Q5_K(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d1, m1 = _k4_scales(blocks)
    qh = (_field(blocks, 16, 48, (1, 1, 32)) >> _shifts(blocks, 0, 8, 1, (4, 2, 1))) & 1
    q = _k4_codes(blocks, 48) | (qh << 4)
    return _emit(d1, q, m1, "-", n, _QK_K, dtype)


def dequantize_blocks_Q6_K(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    d = _scale(blocks, 104)
    sc = blocks[:, 192:208].view(torch.int8)
    # (n, chunk, nibble, half, 2, 16): quarter t = 2 * nibble + half, ggml's q1..q4
    ql = (_field(blocks, 0, 128, (2, 1, 2, 2, 16)) >> _shifts(blocks, 0, 8, 4, (2, 1, 1, 1))) & 0x0F
    qh = (_field(blocks, 128, 192, (2, 1, 1, 2, 16)) >> _shifts(blocks, 0, 8, 2, (2, 2, 1, 1))) & 3
    q = (ql | (qh << 4)).to(torch.int8) - 32
    return _emit((d * sc).reshape(n, 2, 2, 2, 2, 1), q, None, "", n, _QK_K, dtype)


def dequantize_blocks_IQ4_XS(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    d = _scale(blocks, 0)
    scales_h = blocks[:, 2:3].to(torch.int32) | (blocks[:, 3:4].to(torch.int32) << 8)
    shifts = torch.arange(0, 16, 2, device = blocks.device, dtype = torch.int32)
    high = ((scales_h >> shifts) & 3).to(torch.uint8)
    low = ((blocks[:, 4:8].unsqueeze(-1) >> _shifts(blocks, 0, 8, 4, (2,))) & 0x0F).reshape(n, 8)
    ls = (low | (high << 4)).to(torch.int8) - 32
    dl = (d * ls).reshape(n, 8, 1, 1)
    kvalues = torch.tensor(_KVALUES_IQ4NL, device = blocks.device, dtype = torch.float32)
    index = (_field(blocks, 8, 136, (8, 1, 16)) >> _shifts(blocks, 0, 8, 4, (2, 1))) & 0x0F
    return _emit(dl, kvalues[index.long()], None, "", n, _QK_K, dtype)


_FUNCTIONS = {
    "Q4_0": dequantize_blocks_Q4_0,
    "Q4_1": dequantize_blocks_Q4_1,
    "Q5_0": dequantize_blocks_Q5_0,
    "Q5_1": dequantize_blocks_Q5_1,
    "Q8_0": dequantize_blocks_Q8_0,
    "Q2_K": dequantize_blocks_Q2_K,
    "Q3_K": dequantize_blocks_Q3_K,
    "Q4_K": dequantize_blocks_Q4_K,
    "Q5_K": dequantize_blocks_Q5_K,
    "Q6_K": dequantize_blocks_Q6_K,
    "IQ4_XS": dequantize_blocks_IQ4_XS,
}
EXACT_FUNCTIONS = tuple(_FUNCTIONS.values())
_EXACT = frozenset(EXACT_FUNCTIONS)
_TAG = "_unsloth_gguf_exact"
for _fn in EXACT_FUNCTIONS:
    # an attribute test traces inside a compiled block; set membership graph-breaks
    setattr(_fn, _TAG, True)
del _fn
_replaced: dict = {}
# "forward" (diffusers' forward_native) outlives uninstall: a later patch may fall back to _exact_forward
_STATE: dict = {}


def _table() -> Any:
    table = _STATE.get("table")
    if table is None:
        try:
            from diffusers.quantizers.gguf import utils as gguf_utils
        except Exception:  # noqa: BLE001
            return None
        table = _STATE["table"] = getattr(gguf_utils, "dequantize_functions", None)
    return table


def _sizes() -> Any:
    sizes = _STATE.get("sizes")
    if sizes is None:
        from gguf import GGML_QUANT_SIZES
        sizes = _STATE["sizes"] = GGML_QUANT_SIZES
    return sizes


def exact_function(quant_type: Any) -> Any:
    table = _table()
    fn = table.get(quant_type) if isinstance(table, dict) else None
    return fn if getattr(fn, _TAG, False) else None


def row_aligned(raw: Any, quant_type: Any) -> bool:
    return raw.dim() >= 1 and raw.shape[-1] % _sizes()[quant_type][1] == 0


def dequantize_cast(
    raw: Any,
    quant_type: Any,
    dtype: Any,
    blocks_fn: Any = None,
) -> Any:
    import torch

    block_size, type_size = _sizes()[quant_type]
    blocks = raw.view(torch.uint8).reshape(-1, type_size)
    if blocks_fn is None:
        blocks_fn = exact_function(quant_type)
    elif not torch.compiler.is_compiling():
        torch._dynamo.maybe_mark_dynamic(blocks, 0)
    return blocks_fn(blocks, block_size, type_size, dtype).reshape(*raw.shape[:-1], -1)


def _stock_forward(linear_cls: Any) -> Any:
    """diffusers' own ``forward_native`` under any wrappers, by code file (``functools.wraps`` copies ``__module__``)."""
    import sys

    home = getattr(sys.modules.get(linear_cls.__module__), "__file__", None)
    seen = []
    stash = getattr(linear_cls, "_original_utils_GGUFLinear_forward_native", None)
    for fn in (linear_cls.__dict__.get("forward_native"), stash):
        while callable(fn) and fn not in seen:
            seen.append(fn)
            if (
                home is not None
                and getattr(getattr(fn, "__code__", None), "co_filename", None) == home
            ):
                return fn
            fn = getattr(fn, "__wrapped__", None)
    return None


def set_compiled_dequant(on: bool) -> None:
    _STATE["compiled_dequant"] = bool(on)


def _exact_forward(self: Any, inputs: Any) -> Any:
    weight = self.weight
    dtype = self.compute_dtype
    quant_type = getattr(weight, "quant_type", None)
    if (
        dtype is None
        or quant_type is None
        or weight.dim() != 2
        or _STATE.get("compiled_dequant")
        or exact_function(quant_type) is None
        or not row_aligned(weight, quant_type)
    ):
        return _STATE["forward"](self, inputs)
    import torch

    dense = dequantize_cast(weight.as_tensor(), quant_type, dtype)
    bias = self.bias.to(dtype) if self.bias is not None else None
    return torch.nn.functional.linear(inputs, dense, bias)


def stock_forward() -> Any:
    return _STATE.get("forward")


def install_exact_dequant(logger: Any = None) -> list:
    if not exact_enabled():
        return []
    try:
        import gguf
        from diffusers.quantizers.gguf import utils as gguf_utils
    except Exception:  # noqa: BLE001 - no GGUF support in this diffusers
        return []
    table = getattr(gguf_utils, "dequantize_functions", None)
    if not isinstance(table, dict):
        return []
    _STATE["table"] = table
    _STATE["sizes"] = gguf.GGML_QUANT_SIZES
    replaced = []
    for name, fn in _FUNCTIONS.items():
        qtype = getattr(gguf.GGMLQuantizationType, name, None)
        if qtype is None or qtype not in table or table[qtype] is fn:
            continue
        _replaced.setdefault(qtype, table[qtype])
        table[qtype] = fn
        replaced.append(name)
    linear_cls = getattr(gguf_utils, "GGUFLinear", None)
    stock = _stock_forward(linear_cls) if linear_cls is not None else None
    if stock is not None:
        _STATE.setdefault("forward", stock)
        if linear_cls.__dict__.get("forward_native") is stock:
            linear_cls.forward_native = _exact_forward
    if replaced and logger is not None:
        logger.info("diffusion.gguf: llama.cpp-exact fp32 dequant for %s", ", ".join(replaced))
    return replaced


def uninstall_exact_dequant() -> None:
    table = _STATE.get("table")
    if isinstance(table, dict):
        for qtype, original in _replaced.items():
            if table.get(qtype) in _EXACT:
                table[qtype] = original
    _replaced.clear()
    try:
        from diffusers.quantizers.gguf import utils as gguf_utils
    except Exception:  # noqa: BLE001
        return
    linear_cls = getattr(gguf_utils, "GGUFLinear", None)
    if linear_cls is not None and linear_cls.__dict__.get("forward_native") is _exact_forward:
        linear_cls.forward_native = _STATE["forward"]
