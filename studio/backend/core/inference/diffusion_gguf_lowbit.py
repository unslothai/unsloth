# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch dequant for the GGUF types diffusers' table lacks (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), so a
GGUF holding them loads on the diffusers path instead of being refused, and its weights stay packed on the device.
Each function is gguf-py's numpy ``dequantize_blocks`` op for op (the same fp32 order, so the same bits) on gguf-py's
own grids; types diffusers already handles are never replaced. UNSLOTH_DIFFUSION_GGUF_LOWBIT_IQ=0 keeps the refusal."""

from __future__ import annotations

import os
from typing import Any

LOWBIT_IQ_ENV = "UNSLOTH_DIFFUSION_GGUF_LOWBIT_IQ"
_QK_K = 256
# table -> gguf-py's grid / sign table on the CPU
_HOST: dict = {}
# (table, device) -> that table on the device
_CONSTS: dict = {}
_TABLES = {
    "IQ1_S": ("IQ1_S",),
    "IQ1_M": ("IQ1_S",),
    "IQ2_XXS": ("ksigns", "IQ2_XXS"),
    "IQ2_XS": ("ksigns", "IQ2_XS"),
    "IQ2_S": ("IQ2_S",),
    "IQ3_XXS": ("ksigns", "IQ3_XXS"),
    "IQ3_S": ("IQ3_S",),
}


def lowbit_iq_enabled() -> bool:
    return os.environ.get(LOWBIT_IQ_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _host(name: str) -> Any:
    value = _HOST.get(name)
    if value is not None:
        return value
    import numpy as np
    import torch
    from gguf import quants

    if name == "ksigns":
        table = np.frombuffer(quants.IQ2_XXS.ksigns, dtype = np.uint8).astype(np.int64)
    else:
        cls = getattr(quants, name)
        cls.init_grid()
        table = np.ascontiguousarray(cls.grid.reshape(cls.grid_shape), dtype = np.float32)
    value = torch.from_numpy(table.copy())
    _HOST[name] = value
    return value


def _build(name: str, device: Any) -> Any:
    value = _host(name).to(device)
    _CONSTS[(name, str(device))] = value
    return value


def _const(name: str, device: Any) -> Any:
    value = _CONSTS.get((name, str(device)))
    if value is not None:
        return value
    import torch

    if torch.compiler.is_compiling():
        host = _HOST.get(name)
        if host is not None:
            # unprepared device: a traceable copy, no graph break
            return host.to(device)
        # only without install_lowbit_iq
        return torch.compiler.disable(_build)(name, device)
    return _build(name, device)


def prepare_lowbit_iq_tables(device: Any, *modules: Any) -> list:
    """Build on ``device`` before compile so graphs hold a device constant; with ``modules``, only what their IQ weights need."""
    try:
        import torch

        if modules:
            types = set()
            for module in modules:
                for param in getattr(module, "parameters", lambda: ())():
                    qtype = getattr(param, "quant_type", None)
                    name = getattr(qtype, "name", None)
                    if name in _TABLES:
                        types.add(name)
            names = sorted({t for name in types for t in _TABLES[name]})
        else:
            names = sorted({t for tables in _TABLES.values() for t in tables})
        if not names:
            return []
        # "cuda" -> "cuda:0", the key a weight's device gives
        device = torch.empty(0, device = device).device
        for name in names:
            if (name, str(device)) not in _CONSTS:
                _build(name, device)
        return names
    except Exception:  # noqa: BLE001 - optimisation only: the lookup still builds traceably
        return []


def _shifts(values: tuple, like: Any) -> Any:
    import torch
    return torch.tensor(values, device = like.device, dtype = like.dtype)


def _words(raw: Any, width: int) -> Any:
    """Little-endian ``width``-byte unsigned words of a uint8 tensor, as int64 (last dim / width words)."""
    b = raw.reshape(*raw.shape[:-1], -1, width).long()
    out = b[..., 0]
    for i in range(1, width):
        out = out | (b[..., i] << (8 * i))
    return out


def _f16(raw: Any) -> Any:
    import torch
    return raw.contiguous().view(torch.float16).to(torch.float32)


def _sign_bits(byte: Any) -> Any:
    """Each byte's 8 bits, low first, as +1 (bit clear) / -1 (bit set), fp32."""
    import torch

    bit = (byte.unsqueeze(-1) >> _shifts(tuple(range(8)), byte)) & 1
    return torch.where(bit == 0, 1.0, -1.0).to(torch.float32)


def _ksign_bits(index: Any) -> Any:
    return _sign_bits(_const("ksigns", index.device)[index])


def _grid(name: str, index: Any) -> Any:
    return _const(name, index.device)[index]


def dequantize_blocks_IQ2_XXS(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _f16(blocks[:, :2])
    qs = blocks[:, 2:].reshape(
        n, -1, 2, 4
    )  # per 32 values: 4 grid bytes, then a word of 4 sign indices + scale
    aux = _words(qs[:, :, 1], 4)[..., 0]
    db = d * (0.5 + (aux >> 28).float()) * 0.25
    db = db.reshape(n, -1, 1, 1)
    signs = (aux.unsqueeze(-1) >> _shifts((0, 7, 14, 21), aux)) & 0x7F
    signs = _ksign_bits(signs).reshape(n, -1, 4, 8)
    grid = _grid("IQ2_XXS", qs[:, :, 0].long()).reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def dequantize_blocks_IQ2_XS(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _f16(blocks[:, :2])
    qs = _words(blocks[:, 2 : 2 + 2 * _QK_K // 8], 2)
    scales = blocks[:, 2 + 2 * _QK_K // 8 :]
    scales = ((scales.reshape(n, -1, 1) >> _shifts((0, 4), scales)) & 0x0F).reshape(n, -1)
    db = d * (0.5 + scales.float()) * 0.25
    db = db.reshape(n, -1, 1, 1)
    signs = _ksign_bits(qs >> 9).reshape(n, -1, 2, 8)
    grid = _grid("IQ2_XS", qs & 511).reshape(n, -1, 2, 8)
    return (db * grid * signs).reshape(n, -1)


def dequantize_blocks_IQ2_S(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    q = _QK_K // 8
    d = _f16(blocks[:, :2])
    qs = blocks[:, 2 : 2 + q]
    signs = blocks[:, 2 + q : 2 + 2 * q]
    qh = blocks[:, 2 + 2 * q : 2 + 2 * q + _QK_K // 32]
    scales = blocks[:, 2 + 2 * q + _QK_K // 32 :]
    scales = ((scales.reshape(n, -1, 1) >> _shifts((0, 4), scales)) & 0x0F).reshape(n, -1)
    db = d * (0.5 + scales.float()) * 0.25
    db = db.reshape(n, -1, 1, 1)
    signs = _sign_bits(signs).reshape(n, -1, 2, 8)
    qh = qh.reshape(n, -1, 1) >> _shifts((0, 2, 4, 6), qh)
    index = qs.long() | ((qh & 0x03).long() << 8).reshape(n, -1)
    grid = _grid("IQ2_S", index).reshape(n, -1, 2, 8)
    return (db * grid * signs).reshape(n, -1)


def dequantize_blocks_IQ3_XXS(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _f16(blocks[:, :2])
    qs = blocks[:, 2 : 2 + _QK_K // 4]
    scales = _words(blocks[:, 2 + _QK_K // 4 :], 4)
    db = d * (0.5 + (scales >> 28).float()) * 0.5
    db = db.reshape(n, -1, 1, 1)
    signs = (scales.unsqueeze(-1) >> _shifts((0, 7, 14, 21), scales)) & 0x7F
    signs = _ksign_bits(signs).reshape(n, -1, 4, 8)
    grid = _grid("IQ3_XXS", qs.long()).reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def dequantize_blocks_IQ3_S(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    n = blocks.shape[0]
    d = _f16(blocks[:, :2])
    o = 2
    qs = blocks[:, o : o + _QK_K // 4]
    o += _QK_K // 4
    qh = blocks[:, o : o + _QK_K // 32]
    o += _QK_K // 32
    signs = blocks[:, o : o + _QK_K // 8]
    scales = blocks[:, o + _QK_K // 8 :]
    scales = ((scales.reshape(n, -1, 1) >> _shifts((0, 4), scales)) & 0x0F).reshape(n, -1)
    db = d * (1 + 2 * scales.long()).float()
    db = db.reshape(n, -1, 1, 1)
    signs = _sign_bits(signs).reshape(n, -1, 4, 8)
    qh = ((qh.reshape(n, -1, 1) >> _shifts(tuple(range(8)), qh)) & 0x01).long().reshape(n, -1)
    index = qs.long() | (qh << 8)
    grid = _grid("IQ3_S", index).reshape(n, -1, 4, 8)
    return (db * grid * signs).reshape(n, -1)


def _iq1_delta() -> float:
    value = _HOST.get("iq1_delta")
    if value is None:
        from gguf import quants
        value = _HOST["iq1_delta"] = float(quants.IQ1_S.delta)
    return value


def dequantize_blocks_IQ1_S(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    d = _f16(blocks[:, :2])
    qs = blocks[:, 2 : 2 + _QK_K // 8]
    qh = _words(blocks[:, 2 + _QK_K // 8 :], 2)
    dl = d * (2 * ((qh >> 12) & 7) + 1).float()
    dl = dl.reshape(n, -1, 1, 1)
    delta = _iq1_delta()
    delta = torch.where((qh & 0x8000) == 0, delta, -delta).to(torch.float32).reshape(n, -1, 1, 1)
    qh = qh.reshape(n, -1, 1) >> _shifts((0, 3, 6, 9), qh)
    index = qs.long() | ((qh & 7) << 8).reshape(n, -1)
    grid = _grid("IQ1_S", index).reshape(n, -1, 4, 8)
    return (dl * (grid + delta)).reshape(n, -1)


def dequantize_blocks_IQ1_M(
    blocks: Any,
    block_size: Any,
    type_size: Any,
    dtype: Any = None,
) -> Any:
    import torch

    n = blocks.shape[0]
    qs = blocks[:, : _QK_K // 8]
    qh = blocks[:, _QK_K // 8 : _QK_K // 8 + _QK_K // 16]
    scales = _words(blocks[:, _QK_K // 8 + _QK_K // 16 :], 2)
    # the fp16 super-scale is spread over the top nibble of the four scale words
    d = (scales.reshape(n, 4) & 0xF000) >> _shifts((12, 8, 4, 0), scales)
    d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
    d = d.to(torch.int16).view(torch.float16).to(torch.float32).reshape(n, 1)
    scales = ((scales.reshape(n, -1, 1) >> _shifts((0, 3, 6, 9), scales)) & 0x07).reshape(n, -1)
    dl = d * (2 * scales + 1).float()
    dl = dl.reshape(n, -1, 2, 1, 1)
    qh = qh.reshape(n, -1, 1) >> _shifts((0, 4), qh)
    index = qs.long() | ((qh & 0x07).long() << 8).reshape(n, -1)
    delta = _iq1_delta()
    delta = torch.where((qh & 0x08) == 0, delta, -delta).to(torch.float32).reshape(n, -1, 2, 2, 1)
    grid = _grid("IQ1_S", index).reshape(n, -1, 2, 2, 8)
    return (dl * (grid + delta)).reshape(n, -1)


_FUNCTIONS = {
    "IQ1_S": dequantize_blocks_IQ1_S,
    "IQ1_M": dequantize_blocks_IQ1_M,
    "IQ2_XXS": dequantize_blocks_IQ2_XXS,
    "IQ2_XS": dequantize_blocks_IQ2_XS,
    "IQ2_S": dequantize_blocks_IQ2_S,
    "IQ3_XXS": dequantize_blocks_IQ3_XXS,
    "IQ3_S": dequantize_blocks_IQ3_S,
}
_added: list = []


def install_lowbit_iq(logger: Any = None) -> list:
    """Add the missing IQ types to diffusers' table in place (the loader reads the same dict); never replaces one."""
    if not lowbit_iq_enabled():
        return []
    try:
        import gguf
        from diffusers.quantizers.gguf import utils as gguf_utils
    except Exception:  # noqa: BLE001 - no GGUF support in this diffusers: nothing to extend
        return []
    table = getattr(gguf_utils, "dequantize_functions", None)
    supported = getattr(gguf_utils, "SUPPORTED_GGUF_QUANT_TYPES", None)
    if not isinstance(table, dict) or not isinstance(supported, list):
        return []
    added = []
    for name, fn in _FUNCTIONS.items():
        qtype = getattr(gguf.GGMLQuantizationType, name, None)
        if qtype is None or qtype in table:
            continue
        table[qtype] = fn
        if qtype not in supported:
            supported.append(qtype)
        added.append(name)
        _added.append(qtype)
    if _added:
        try:
            for name in sorted({t for tables in _TABLES.values() for t in tables}):
                _host(name)
            _iq1_delta()
        except Exception:  # noqa: BLE001 - the eager lookup builds them on first use
            pass
    if added and logger is not None:
        logger.info("diffusion.gguf: dequantising %s on the device", ", ".join(added))
    return added


def uninstall_lowbit_iq() -> None:
    """Take back exactly the types ``install_lowbit_iq`` added. Idempotent."""
    try:
        from diffusers.quantizers.gguf import utils as gguf_utils
    except Exception:  # noqa: BLE001
        _added.clear()
        return
    for qtype in _added:
        if gguf_utils.dequantize_functions.get(qtype) in _FUNCTIONS.values():
            del gguf_utils.dequantize_functions[qtype]
        if qtype in gguf_utils.SUPPORTED_GGUF_QUANT_TYPES:
            gguf_utils.SUPPORTED_GGUF_QUANT_TYPES.remove(qtype)
    _added.clear()
