# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF dequant bit-identical to llama.cpp's ggml-quants.c ``dequantize_row_*`` for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS.

diffusers' functions for these types compute in fp16 (IQ4_XS rounds its sub-block scale to fp16), so a bf16 compute
dtype rounds twice and 3-10% of the weights land one bf16 ulp away from llama.cpp's. These compute in fp32 with
ggml's formula and operation order (fp16 d / dmin -> fp32, integer codes, ``d * sc`` then ``* q``, then ``- m``) and
round once to the compute dtype. Every product is exact in fp32 (an 11-bit fp16 mantissa times at most 13 bits of
integer scale and code), so only the final ``+ m`` / ``- m`` rounds and FMA contraction cannot change a bit.

``dtype=None`` (diffusers' call) returns fp32; ``dequantize_cast`` returns the compute dtype in one step, which the
compiled GGUFLinear forward (diffusion_gguf_compile) fuses into one kernel. Installed into diffusers'
``dequantize_functions`` in place; UNSLOTH_DIFFUSION_GGUF_EXACT=0 keeps diffusers' functions."""

from __future__ import annotations

import os
from typing import Any

EXACT_ENV = "UNSLOTH_DIFFUSION_GGUF_EXACT"
_QK_K = 256
# devices whose eager kernels round an fp32 result once on store into a narrower out= tensor
_CAST_ON_STORE = ("cuda", "cpu")
_KVALUES_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def exact_enabled() -> bool:
    return os.environ.get(EXACT_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _scale(blocks: Any, word: int) -> Any:
    """The fp16 field at byte ``2 * word`` of each block as fp32, shape ``(n, 1)``: GGML_FP16_TO_FP32 is exact."""
    import torch

    # the 2-byte slice, not the whole block, is reinterpreted: inductor materialises a dtype view of a different width
    return blocks[:, 2 * word : 2 * word + 2].view(torch.float16).to(torch.float32)


def _field(blocks: Any, start: int, stop: int, shape: tuple) -> Any:
    """Bytes ``start:stop`` of each block split into ``shape`` (a view: only the last dim is split)."""
    return blocks[:, start:stop].unflatten(1, shape)


def _shifts(like: Any, start: int, stop: int, step: int, shape: tuple) -> Any:
    import torch
    return torch.arange(start, stop, step, device = like.device, dtype = torch.uint8).reshape(shape)


def _emit(scale: Any, q: Any, m: Any, op: str, n: int, block_size: int, dtype: Any) -> Any:
    """ggml's last step, ``scale * q`` then ``+ m`` / ``- m`` (``op`` "", "+", "-"), as ``(n, block_size)`` in ``dtype``
    (fp32 when None), rounded once from the fp32 value.

    ``scale`` / ``q`` / ``m`` broadcast to a shape laid out in output order, so the only reshape is of the contiguous
    result: inductor then indexes the packed bytes directly (a reshape of a strided intermediate makes it divide by
    the dynamic block count per element, 1.7x slower for Q6_K on a T4). Traced, it is plain fp32 ops and one cast,
    fused into the dequant kernel. Eager with a narrower ``dtype``, the result is written straight into a ``dtype``
    buffer: TensorIterator computes in fp32 (the common dtype) and rounds on store, so no fp32 weight is
    materialised (CUDA / ROCm and CPU, where that is verified; other devices take the plain ops). The traced
    difference is ``(scale * q) - m``, not ``addcmul(-m, ...)``: Triton lowers that negation as ``0 - m``, which turns
    ``-0 - +0`` into ``+0``."""
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
    """Q4_0 / Q4_1 / Q5_x 4-bit codes as ``(n, 2, 2, 8)``: element ``16 * h + 8 * a + b`` is nibble ``h`` of byte
    ``8 * a + b`` (ggml's ``qs[j] & 0xF`` for j < 16, ``qs[j - 16] >> 4`` after)."""
    return (_field(blocks, at, at + 16, (1, 2, 8)) >> _shifts(blocks, 0, 8, 4, (2, 1, 1))) & 0x0F


def dequantize_blocks_Q4_0(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    d = _scale(blocks, 0).reshape(n, 1, 1, 1)
    q = _q4_codes(blocks, 2).to(torch.int8) - 8
    return _emit(d, q, None, "", n, 32, dtype)


def dequantize_blocks_Q4_1(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _scale(blocks, 0).reshape(n, 1, 1, 1)
    m = _scale(blocks, 1).reshape(n, 1, 1, 1)
    # ggml: x0*d + m
    return _emit(d, _q4_codes(blocks, 4), m, "+", n, 32, dtype)


def _q5_codes(blocks: Any, qh_at: int) -> Any:
    """Q5_0 / Q5_1 5-bit codes 0..31 as ``(n, 2, 2, 8)``: code ``k`` takes bit ``k`` of the little-endian 32-bit ``qh``,
    which is bit ``k % 8`` of byte ``k // 8``."""
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
    d = _scale(blocks, 0).reshape(n, 1, 1, 1)
    q = _q5_codes(blocks, 2).to(torch.int8) - 16
    return _emit(d, q, None, "", n, 32, dtype)


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
    d = _scale(blocks, 0)
    return _emit(d, blocks[:, 2:].view(torch.int8), None, "", n, 32, dtype)


def _k_codes(blocks: Any, at: int) -> Any:
    """Q2_K / Q3_K 2-bit codes as ``(n, 2, 4, 2, 16)`` (128-chunk c, shift j, half h, l):
    ``(q[32c + 16h + l] >> 2j) & 3``, the order ggml writes them; group ``8c + 2j + h`` of 16 shares a scale."""
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
    # ggml's aux[] unpack: scale g = low nibble of byte g (g < 8) or high nibble of byte g - 8,
    # plus 2 bits of byte 8 + g % 4
    low = (scales[:, :8].unsqueeze(1) >> _shifts(blocks, 0, 8, 4, (2, 1))) & 0x0F
    high = (scales[:, 8:].unsqueeze(1) >> _shifts(blocks, 0, 8, 2, (4, 1))) & 3
    sc = (low.reshape(n, 16) | (high.reshape(n, 16) << 4)).to(torch.int8) - 32
    dl = (d * sc).reshape(n, 2, 4, 2, 1)
    # hmask bit 4c + j of byte 16h + l set -> 0, clear -> 4
    hb = (_field(blocks, 0, 32, (1, 1, 2, 16)) >> _shifts(blocks, 0, 8, 1, (2, 4, 1, 1))) & 1
    q = _k_codes(blocks, 32).to(torch.int8) - ((hb ^ 1) << 2).to(torch.int8)
    return _emit(dl, q, None, "", n, _QK_K, dtype)


def _scale_min_k4(scales: Any) -> tuple:
    """ggml's get_scale_min_k4 for j = 0..7 on the 12 packed bytes: (sc, m), each ``(n, 8)`` uint8."""
    import torch

    a, b, c = scales[:, :4], scales[:, 4:8], scales[:, 8:12]
    sc = torch.cat((a & 63, (c & 0x0F) | ((a >> 6) << 4)), dim = 1)
    m = torch.cat((b & 63, (c >> 4) | ((b >> 6) << 4)), dim = 1)
    return sc, m


def _k4_scales(blocks: Any) -> tuple:
    """Q4_K / Q5_K ``d * sc`` and ``dmin * m`` per group of 32, as ``(n, 4, 2, 1)`` (64-chunk, nibble)."""
    n = blocks.shape[0]
    d = _scale(blocks, 0)
    dmin = _scale(blocks, 1)
    sc, m = _scale_min_k4(blocks[:, 4:16])
    return (d * sc).reshape(n, 4, 2, 1), (dmin * m).reshape(n, 4, 2, 1)


def _k4_codes(blocks: Any, at: int) -> Any:
    """Q4_K / Q5_K low 4 bits as ``(n, 4, 2, 32)``: chunk c's 32 bytes, low nibbles (group 2c) then high (2c + 1)."""
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
    # group g = 2c + h takes bit g of qh[l]
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
    # (n, 128-chunk c, nibble, half, 2, 16): quarter t = 2 * nibble + half takes nibble t >> 1 of
    # ql[64c + 32(t & 1) + l] and bits 2t of qh[32c + l]; the 16-element group 8c + 2t + l // 16 shares a scale
    ql = (_field(blocks, 0, 128, (2, 1, 2, 2, 16)) >> _shifts(blocks, 0, 8, 4, (2, 1, 1, 1))) & 0x0F
    qh = (_field(blocks, 128, 192, (2, 1, 1, 2, 16)) >> _shifts(blocks, 0, 8, 2, (2, 2, 1, 1))) & 3
    q = (ql | (qh << 4)).to(torch.int8) - 32
    # ggml: d * sc[is] * q, left to right
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
    # sub-block ib's 16 bytes: low nibbles are elements 0..15, high nibbles 16..31
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
    # an attribute test traces without a graph break inside a compiled block, unlike set membership
    setattr(_fn, _TAG, True)
del _fn
# qtype -> the function this module replaced
_replaced: dict = {}
# set on install: diffusers' table, gguf's (block_size, type_size) table, diffusers' GGUFLinear.forward_native (kept:
# a later patch may hold _exact_forward as its fallback after uninstall)
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
    """This module's function for ``quant_type`` while it is the installed one, else None."""
    table = _table()
    fn = table.get(quant_type) if isinstance(table, dict) else None
    return fn if getattr(fn, _TAG, False) else None


def row_aligned(raw: Any, quant_type: Any) -> bool:
    """Whole blocks per row, the layout ``dequantize_cast`` reshapes."""
    return raw.dim() >= 1 and raw.shape[-1] % _sizes()[quant_type][1] == 0


def dequantize_cast(
    raw: Any,
    quant_type: Any,
    dtype: Any,
    blocks_fn: Any = None,
) -> Any:
    """GGUF bytes ``(rows, n_blocks * type_size)`` -> ``(rows, n_blocks * block_size)`` in ``dtype``, rounded once from
    the fp32 dequant. ``blocks_fn`` is the (compiled) block function, ``exact_function(quant_type)`` by default."""
    import torch

    block_size, type_size = _sizes()[quant_type]
    blocks = raw.view(torch.uint8).reshape(-1, type_size)
    if blocks_fn is None:
        blocks_fn = exact_function(quant_type)
    elif not torch.compiler.is_compiling():
        # only the block count varies between weights: a static block width keeps the kernel's indexing constant,
        # and one graph serves every weight of the type without a recompile
        torch._dynamo.maybe_mark_dynamic(blocks, 0)
    return blocks_fn(blocks, block_size, type_size, dtype).reshape(*raw.shape[:-1], -1)


def _exact_forward(self: Any, inputs: Any) -> Any:
    """GGUFLinear.forward_native with the weight dequantised straight to the compute dtype (one rounding, no fp32
    weight in eager); other types and shapes take the replaced forward."""
    weight = self.weight
    dtype = self.compute_dtype
    quant_type = getattr(weight, "quant_type", None)
    if (
        dtype is None
        or quant_type is None
        or weight.dim() != 2
        or exact_function(quant_type) is None
        or not row_aligned(weight, quant_type)
    ):
        return _STATE["forward"](self, inputs)
    import torch

    dense = dequantize_cast(weight.as_tensor(), quant_type, dtype)
    bias = self.bias.to(dtype) if self.bias is not None else None
    return torch.nn.functional.linear(inputs, dense, bias)


def install_exact_dequant(logger: Any = None) -> list:
    """Replace diffusers' functions for the types above in its table in place (the loader and the forward read the
    same dict) and route GGUFLinear's eager forward through ``dequantize_cast``. Idempotent; returns the types
    replaced by this call."""
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
        # only a type diffusers already supports: the loader's support list stays diffusers'
        if qtype is None or qtype not in table or table[qtype] is fn:
            continue
        _replaced.setdefault(qtype, table[qtype])
        table[qtype] = fn
        replaced.append(name)
    linear_cls = getattr(gguf_utils, "GGUFLinear", None)
    forward = getattr(linear_cls, "forward_native", None)
    if callable(forward) and forward is not _exact_forward:
        _STATE.setdefault("forward", forward)
        # only over diffusers' own forward, before diffusion_gguf_compile patches it (its fallback is then this one)
        if linear_cls.__dict__.get("forward_native") is _STATE["forward"]:
            _STATE["patched_cls"] = linear_cls
            linear_cls.forward_native = _exact_forward
    if replaced and logger is not None:
        logger.info("diffusion.gguf: llama.cpp-exact fp32 dequant for %s", ", ".join(replaced))
    return replaced


def uninstall_exact_dequant() -> None:
    """Put back what ``install_exact_dequant`` replaced, where its own is still installed. Idempotent."""
    table = _STATE.get("table")
    if isinstance(table, dict):
        for qtype, original in _replaced.items():
            if table.get(qtype) in _EXACT:
                table[qtype] = original
    _replaced.clear()
    linear_cls = _STATE.pop("patched_cls", None)
    if linear_cls is not None and linear_cls.__dict__.get("forward_native") is _exact_forward:
        linear_cls.forward_native = _STATE["forward"]
