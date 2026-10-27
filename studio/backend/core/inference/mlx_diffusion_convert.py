# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion GGUF -> mflux pre-quantized transformer layout (numpy + gguf only, streamed per tensor).

Q4_0 / Q4_1 / Q8_0 map exactly onto MLX affine group-32 (dequant = q * scale + bias):
  Q4_0: scale d, bias -8d;  Q4_1: scale d, bias m;  Q8_0: q = qs + 128, scale d, bias -128d.
Q5_0 / Q5_1 likewise onto 5-bit (Q5_0: bias -16d; Q5_1: bias m), the fifth bit coming from the block's qh word.
Q4_K / Q5_K keep their codes too: each 32-wide sub-block is affine with scale d * sc and bias -dmin * m, so they land
on MLX 4 / 5-bit group-32; only the scale product is rounded to the 16-bit scale dtype, as the d of Q4_0 already is.
Q6_K uses 16-wide groups, which MLX does not have: its exact dequantized values are re-quantized to 8-bit group-32,
whose step is at most a quarter of Q6_K's own. Q2_K / Q3_K (16-wide, 2-3 bits) are refused.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

CONVERTER_VERSION = "1"
MFLUX_VERSION = "0.22.0"
GROUP_SIZE = 32
OUTPUT_FILENAME = "0.safetensors"
MARKER_FILENAME = "unsloth_mlx_convert.json"

CONVERTIBLE_TYPES = frozenset(
    {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "BF16", "F16", "F32"}
)
# File-name tokens whose mixes hold only the types above (a Q4_K_M also carries Q5_K / Q6_K, a Q5_K_M Q6_K).
CONVERTIBLE_QUANT_TOKENS = frozenset(
    {
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "BF16",
        "F16",
        "F32",
    }
)
_SUPPORTED_NAMES = "Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/BF16"

_BLOCK_BYTES = {
    "Q4_0": 18,
    "Q4_1": 20,
    "Q5_0": 22,
    "Q5_1": 24,
    "Q8_0": 34,
    "Q4_K": 144,
    "Q5_K": 176,
    "Q6_K": 210,
}
# Bits of the MLX layer each block type becomes.
_BLOCK_BITS = {
    "Q4_0": 4,
    "Q4_1": 4,
    "Q5_0": 5,
    "Q5_1": 5,
    "Q8_0": 8,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q6_K": 8,
}
_QK_K = 256
_DENSE_TYPES = frozenset({"BF16", "F16", "F32"})
_PAD_TOKENS = ("cap_pad_token", "x_pad_token")
_QKV_PARTS = ("to_q", "to_k", "to_v")


class UnsupportedGGUFError(ValueError):
    """The GGUF holds a tensor type this converter cannot map onto MLX affine weights."""


# ── Filename-level verdict (router) ─────────────────────────────────────────────────────────────

_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:I?Q\d+_[A-Z0-9_]+|Q\d+|BF16|F16|F32|MXFP4|TQ\d_\d))(?:\.gguf)?$",
    re.IGNORECASE,
)


def _quant_token(gguf_filename: str) -> Optional[str]:
    try:
        from hub.utils.gguf import extract_quant_token
        return extract_quant_token(gguf_filename)
    except Exception:  # noqa: BLE001 -- the regex below answers when the hub helper cannot
        match = _QUANT_RE.search(Path(gguf_filename).name)
        return match.group(1) if match else None


def quant_convertible(gguf_filename: Optional[str]) -> tuple[bool, Optional[str]]:
    """Filename verdict only; conversion re-checks every tensor."""
    if not gguf_filename:
        return (
            False,
            f"GGUF quant unknown (no GGUF filename); the MLX engine needs {_SUPPORTED_NAMES}",
        )
    token = _quant_token(str(gguf_filename))
    if not token:
        return False, (
            f"GGUF quant unknown for {Path(str(gguf_filename)).name!r}; the MLX engine needs "
            f"{_SUPPORTED_NAMES}"
        )
    upper = str(token).upper()
    if upper.startswith("UD-"):
        upper = upper[3:]
    if upper in CONVERTIBLE_QUANT_TOKENS:
        return True, None
    return False, (f"GGUF quant {token} is not convertible to MLX yet (only {_SUPPORTED_NAMES})")


# ── Block math ──────────────────────────────────────────────────────────────────────────────────


_ROW_CHUNK = 256


def _pack(q: np.ndarray, bits: int) -> np.ndarray:
    """(R, C) integers in [0, 2**bits) -> (R, C * bits / 32) uint32: MLX's layout, a little-endian bit stream per row
    (for 4 and 8 bits, element i at bits * (i % per_word) of word i // per_word)."""
    rows, cols = q.shape
    if (cols * bits) % 32:
        raise ValueError(f"{cols} columns do not pack into {bits}-bit words")
    if 32 % bits == 0:
        per_word = 32 // bits
        q = q.reshape(rows, cols // per_word, per_word).astype(np.uint32)
        shifts = (bits * np.arange(per_word, dtype = np.uint32)).astype(np.uint32)
        return np.bitwise_or.reduce(q << shifts, axis = -1).astype(np.uint32)
    out = np.empty((rows, cols * bits // 32), dtype = np.uint32)
    shifts = np.arange(bits, dtype = np.uint8)
    for start in range(0, rows, _ROW_CHUNK):
        chunk = np.asarray(q[start : start + _ROW_CHUNK]).astype(np.uint8)
        stream = ((chunk[..., None] >> shifts) & 1).reshape(chunk.shape[0], -1)
        packed = np.packbits(stream, axis = -1, bitorder = "little")
        out[start : start + _ROW_CHUNK] = np.ascontiguousarray(packed).view("<u4")
    return out


def _unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    rows, words = packed.shape
    if 32 % bits == 0:
        per_word = 32 // bits
        shifts = (bits * np.arange(per_word, dtype = np.uint32)).astype(np.uint32)
        mask = np.uint32((1 << bits) - 1)
        q = (packed.astype(np.uint32)[..., None] >> shifts) & mask
        return q.reshape(rows, words * per_word)
    stream = np.unpackbits(
        np.ascontiguousarray(packed.astype("<u4")).view(np.uint8), axis = -1, bitorder = "little"
    )
    cols = words * 32 // bits
    weights = (1 << np.arange(bits, dtype = np.uint32)).astype(np.uint32)
    return (stream.reshape(rows, cols, bits).astype(np.uint32) * weights).sum(axis = -1)


def _f16_field(blocks: np.ndarray, start: int) -> np.ndarray:
    return blocks[..., start : start + 2].copy().view(np.float16)[..., 0].astype(np.float32)


def affine_from_ggml(
    kind: str, raw: np.ndarray, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """GGUF block bytes -> (packed uint32, scales f32, biases f32, bits), MLX affine g32 (see the module docstring)."""
    kind = str(kind).upper()
    if kind not in _BLOCK_BYTES:
        raise UnsupportedGGUFError(f"{kind} blocks have no exact MLX affine mapping")
    if kind in _KQUANT_TYPES:
        return _affine_from_kquant(kind, raw, rows, cols)
    if cols % GROUP_SIZE:
        raise ValueError(f"{cols} columns are not a whole number of {GROUP_SIZE}-wide blocks")
    nb = cols // GROUP_SIZE
    raw = np.ascontiguousarray(np.asarray(raw)).view(np.uint8).reshape(-1)
    expected = rows * nb * _BLOCK_BYTES[kind]
    if raw.size != expected:
        raise ValueError(f"{kind}: {raw.size} bytes for {rows}x{cols}, expected {expected}")
    blocks = raw.reshape(rows, nb, _BLOCK_BYTES[kind])
    if kind == "Q4_0":
        d = _f16_field(blocks, 0)
        qs = blocks[..., 2:]
        q = np.concatenate([qs & 0xF, qs >> 4], axis = -1).reshape(rows, cols)
        return _pack(q, 4), d, -8.0 * d, 4
    if kind == "Q4_1":
        d = _f16_field(blocks, 0)
        m = _f16_field(blocks, 2)
        qs = blocks[..., 4:]
        q = np.concatenate([qs & 0xF, qs >> 4], axis = -1).reshape(rows, cols)
        return _pack(q, 4), d, m, 4
    if kind in ("Q5_0", "Q5_1"):
        d = _f16_field(blocks, 0)
        start = 2 if kind == "Q5_0" else 4
        qh = blocks[..., start : start + 4].copy().view("<u4")
        qs = blocks[..., start + 4 :]
        hi = ((qh >> np.arange(32, dtype = np.uint32)) & 1).astype(np.uint8)
        q = np.concatenate([qs & 0xF, qs >> 4], axis = -1) | (hi << 4)
        bias = -16.0 * d if kind == "Q5_0" else _f16_field(blocks, 2)
        return _pack(q.reshape(rows, cols), 5), d, bias, 5
    d = _f16_field(blocks, 0)
    qs = blocks[..., 2:].copy().view(np.int8).astype(np.int16) + 128
    return _pack(qs.astype(np.uint8).reshape(rows, cols), 8), d, -128.0 * d, 8


_KQUANT_TYPES = frozenset({"Q4_K", "Q5_K", "Q6_K"})


def _kquant_blocks(kind: str, raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if cols % _QK_K:
        raise ValueError(
            f"{kind}: {cols} columns are not a whole number of {_QK_K}-wide super-blocks"
        )
    raw = np.ascontiguousarray(np.asarray(raw)).view(np.uint8).reshape(-1)
    expected = rows * (cols // _QK_K) * _BLOCK_BYTES[kind]
    if raw.size != expected:
        raise ValueError(f"{kind}: {raw.size} bytes for {rows}x{cols}, expected {expected}")
    return raw.reshape(-1, _BLOCK_BYTES[kind])


def _affine_from_kquant(
    kind: str, raw: np.ndarray, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same block layout as gguf-py's dequantizers (``gguf.quants``), which the tests hold this to."""
    blocks = _kquant_blocks(kind, raw, rows, cols)
    n = blocks.shape[0]
    if kind == "Q6_K":
        from gguf.quants import Q6_K

        values = Q6_K.dequantize_blocks(blocks).reshape(rows, cols)
        packed, scales, biases = quantize_dense_affine(values, 8)
        return packed, scales, biases, 8
    from gguf.quants import Q4_K

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, m = Q4_K.get_scale_min(blocks[:, 4:16])
    scale = (d * sc.astype(np.float32)).reshape(rows, cols // GROUP_SIZE)
    bias = -(dmin * m.astype(np.float32)).reshape(rows, cols // GROUP_SIZE)
    if kind == "Q4_K":
        qs = blocks[:, 16:]
        q = (
            qs.reshape(n, -1, 1, 32) >> np.array([0, 4], dtype = np.uint8).reshape(1, 1, 2, 1)
        ) & 0x0F
        return _pack(q.reshape(rows, cols), 4), scale, bias, 4
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:]
    ql = (qs.reshape(n, -1, 1, 32) >> np.array([0, 4], dtype = np.uint8).reshape(1, 1, 2, 1)) & 0x0F
    hi = (qh.reshape(n, -1, 1, 32) >> np.arange(8, dtype = np.uint8).reshape(1, 1, 8, 1)) & 0x01
    q = ql.reshape(n, -1, 32) | (hi.reshape(n, -1, 32) << 4)
    return _pack(q.reshape(rows, cols), 5), scale, bias, 5


def _f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round to nearest even; NaN stays NaN."""
    a = np.ascontiguousarray(a, dtype = np.float32)
    u = a.view(np.uint32)
    rounded = ((u + (np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1)))) >> 16).astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():
        rounded = np.where(nan, ((u >> 16) | np.uint32(0x40)).astype(np.uint16), rounded)
    return rounded


def _bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b, dtype = np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_to_bf16(a: np.ndarray) -> np.ndarray:
    """float32 values as they read back after a bf16 store."""
    return _bf16_bits_to_f32(_f32_to_bf16_bits(a))


def quantize_dense_affine(
    w: np.ndarray,
    bits: int,
    group_size: int = GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max affine quantization of a dense (R, C) weight; scale / bias already bf16-rounded."""
    if bits not in (4, 8):
        raise ValueError(f"dense quantization supports 4 or 8 bits, not {bits}")
    w = np.asarray(w, dtype = np.float32)
    rows, cols = w.shape
    if cols % group_size:
        raise ValueError(f"{cols} columns are not a whole number of {group_size}-wide groups")
    g = w.reshape(rows, cols // group_size, group_size)
    lo = g.min(axis = -1)
    hi = g.max(axis = -1)
    levels = float((1 << bits) - 1)
    bias = round_to_bf16(lo)
    scale = round_to_bf16((hi - bias) / levels)
    flat = scale == 0
    safe_scale = np.where(flat, np.float32(1.0), scale).astype(np.float32)
    q = np.rint((g - bias[..., None]) / safe_scale[..., None])
    q = np.where(flat[..., None], 0.0, q)
    q = np.clip(q, 0, levels).astype(np.uint8).reshape(rows, cols)
    return _pack(q, bits), scale.astype(np.float32), bias.astype(np.float32)


def dequantize_affine(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    bits: int,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    q = _unpack(np.asarray(packed), bits).astype(np.float32)
    rows, cols = q.shape
    q = q.reshape(rows, cols // group_size, group_size)
    s = np.asarray(scales, dtype = np.float32)[..., None]
    b = np.asarray(biases, dtype = np.float32)[..., None]
    return (q * s + b).reshape(rows, cols)


# ── Names ───────────────────────────────────────────────────────────────────────────────────────


def zimage_rename(name: str) -> str:
    """Single-file Z-Image tensor name -> mflux's (the qkv split happens in the converter)."""
    k = re.sub(r"^final_layer\.", "all_final_layer.2-1.", name)
    k = re.sub(r"^x_embedder\.", "all_x_embedder.2-1.", k)
    k = k.replace(
        "all_final_layer.2-1.adaLN_modulation.1.", "all_final_layer.2-1.adaLN_modulation.0."
    )
    k = k.replace("t_embedder.mlp.0.", "t_embedder.linear1.").replace(
        "t_embedder.mlp.2.", "t_embedder.linear2."
    )
    k = k.replace(".attention.out.", ".attention.to_out.0.")
    k = k.replace(".attention.q_norm.", ".attention.norm_q.").replace(
        ".attention.k_norm.", ".attention.norm_k."
    )
    return k


# ── Planning ────────────────────────────────────────────────────────────────────────────────────

_ST_DTYPE_BYTES = {"U32": 4, "BF16": 2, "F16": 2}


def _type_name(tensor: Any) -> str:
    return str(tensor.tensor_type.name).upper()


def _gguf_dims(tensor: Any) -> list[int]:
    """numpy-order shape (outermost first); GGUF stores the innermost dimension first."""
    return [int(x) for x in reversed([int(v) for v in tensor.shape])]


def _dense_f32(tensor: Any) -> np.ndarray:
    kind = _type_name(tensor)
    raw = np.asarray(tensor.data)
    shape = _gguf_dims(tensor)
    if kind == "BF16":
        return _bf16_bits_to_f32(np.ascontiguousarray(raw).view(np.uint16)).reshape(shape)
    if kind == "F16":
        return np.ascontiguousarray(raw).view(np.float16).astype(np.float32).reshape(shape)
    return np.ascontiguousarray(raw).view(np.float32).reshape(shape)


def _dense_bf16_bits(tensor: Any) -> np.ndarray:
    """Dense tensor as bf16 bit patterns; a BF16 source is copied bit for bit."""
    if _type_name(tensor) == "BF16":
        return (
            np.ascontiguousarray(np.asarray(tensor.data))
            .view(np.uint16)
            .reshape(_gguf_dims(tensor))
        )
    return _f32_to_bf16_bits(_dense_f32(tensor))


class _Job:
    """One source tensor -> its output tensors, planned from shapes alone and produced on demand."""

    def __init__(self, tensor: Any, outputs: list[tuple[str, str, list[int]]], produce: Callable):
        self.tensor = tensor
        self.outputs = outputs
        self.produce = produce


def _scale_store(a: np.ndarray, scale_dtype: str) -> np.ndarray:
    if scale_dtype == "f16":
        return np.ascontiguousarray(a, dtype = np.float32).astype(np.float16)
    return _f32_to_bf16_bits(a)


def _plan(reader: Any, *, dense_bits: int, scale_dtype: str) -> tuple[list[_Job], dict]:
    sdt = "F16" if scale_dtype == "f16" else "BF16"
    jobs: list[_Job] = []
    bits_count = {4: 0, 5: 0, 8: 0}
    for tensor in reader.tensors:
        kind = _type_name(tensor)
        name = zimage_rename(str(tensor.name))
        dims = _gguf_dims(tensor)
        if kind in _BLOCK_BYTES:
            if not name.endswith(".weight"):
                raise UnsupportedGGUFError(f"quantized tensor {tensor.name} is not a .weight")
            cols = dims[-1]
            rows = int(np.prod(dims[:-1])) if len(dims) > 1 else 1
            bits = _BLOCK_BITS[kind]
            bits_count[bits] += 1
            base = name[: -len(".weight")]
            parts = [(base, 0, rows)]
            if base.endswith(".attention.qkv"):
                if rows % 3:
                    raise UnsupportedGGUFError(
                        f"{tensor.name}: {rows} rows do not split into q/k/v"
                    )
                n = rows // 3
                stem = base[: -len("qkv")]
                parts = [(stem + p, i * n, (i + 1) * n) for i, p in enumerate(_QKV_PARTS)]
            outputs = []
            for nm, a, b in parts:
                r = b - a
                outputs += [
                    (nm + ".weight", "U32", [r, cols * bits // 32]),
                    (nm + ".scales", sdt, [r, cols // GROUP_SIZE]),
                    (nm + ".biases", sdt, [r, cols // GROUP_SIZE]),
                ]

            def produce(
                tensor = tensor,
                kind = kind,
                rows = rows,
                cols = cols,
                parts = parts,
            ):
                w, s, b, _bits = affine_from_ggml(kind, tensor.data, rows, cols)
                for _nm, lo, hi in parts:
                    yield np.ascontiguousarray(w[lo:hi])
                    yield _scale_store(s[lo:hi], scale_dtype)
                    yield _scale_store(b[lo:hi], scale_dtype)

            jobs.append(_Job(tensor, outputs, produce))
            continue
        if name in _PAD_TOKENS:
            dims = [1, int(np.prod(dims))]
        quantize = (
            dense_bits
            and len(dims) == 2
            and name.endswith(".weight")
            and dims[-1] % GROUP_SIZE == 0
            and name not in _PAD_TOKENS
        )
        if name.endswith(".attention.qkv.weight"):
            if dims[0] % 3:
                raise UnsupportedGGUFError(f"{tensor.name}: {dims[0]} rows do not split into q/k/v")
            n = dims[0] // 3
            stem = name[: -len("qkv.weight")]
            parts = [(stem + p, i * n, (i + 1) * n) for i, p in enumerate(_QKV_PARTS)]
        else:
            parts = [(name[: -len(".weight")] if quantize else name, 0, dims[0] if dims else 1)]
        outputs = []
        for nm, a, b in parts:
            r = b - a
            if quantize:
                bits_count[dense_bits] += 1
                outputs += [
                    (nm + ".weight", "U32", [r, dims[-1] * dense_bits // 32]),
                    (nm + ".scales", sdt, [r, dims[-1] // GROUP_SIZE]),
                    (nm + ".biases", sdt, [r, dims[-1] // GROUP_SIZE]),
                ]
            elif len(parts) > 1:
                outputs.append((nm + ".weight", "BF16", [r, *dims[1:]]))
            else:
                outputs.append((nm, "BF16", list(dims)))

        def produce_dense(
            tensor = tensor,
            dims = dims,
            parts = parts,
            quantize = quantize,
        ):
            if quantize:
                full = _dense_f32(tensor).reshape(dims)
                for _nm, lo, hi in parts:
                    w, s, b = quantize_dense_affine(round_to_bf16(full[lo:hi]), dense_bits)
                    yield w
                    yield _scale_store(s, scale_dtype)
                    yield _scale_store(b, scale_dtype)
                return
            bits16 = _dense_bf16_bits(tensor).reshape(dims)
            for _nm, lo, hi in parts:
                yield np.ascontiguousarray(bits16[lo:hi] if len(parts) > 1 else bits16)

        jobs.append(_Job(tensor, outputs, produce_dense))
    return jobs, bits_count


def _check_types(reader: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in reader.tensors:
        kind = _type_name(tensor)
        counts[kind] = counts.get(kind, 0) + 1
    bad = {k: v for k, v in counts.items() if k not in CONVERTIBLE_TYPES}
    if bad:
        listed = ", ".join(f"{v} x {k}" for k, v in sorted(bad.items()))
        raise UnsupportedGGUFError(
            f"GGUF holds tensor types MLX conversion does not support yet: {listed} "
            "(supported: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, BF16, F16, F32)"
        )
    return counts


# ── Writer ──────────────────────────────────────────────────────────────────────────────────────


def _nbytes(dtype: str, shape: list[int]) -> int:
    return int(np.prod(shape)) * _ST_DTYPE_BYTES[dtype] if shape else _ST_DTYPE_BYTES[dtype]


def _header_bytes(entries: list[tuple[str, str, list[int]]], metadata: dict[str, str]) -> bytes:
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape in entries:
        size = _nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)
    return blob


def planned_output_bytes(
    gguf_path: os.PathLike | str, dense_bits_options: tuple[int, ...] = (0, 8, 4)
) -> dict[int, int]:
    """Exact size of the converted transformer per ``dense_bits``, from the GGUF header alone (no tensor data read)."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    _check_types(reader)
    sizes: dict[int, int] = {}
    for bits in dense_bits_options:
        jobs, _ = _plan(reader, dense_bits = bits, scale_dtype = "bf16")
        sizes[bits] = sum(_nbytes(dtype, shape) for job in jobs for _n, dtype, shape in job.outputs)
    return sizes


def convert_gguf_to_mflux(
    gguf_path: os.PathLike | str,
    out_dir: os.PathLike | str,
    *,
    dense_bits: int = 0,
    scale_dtype: str = "bf16",
    cancel_event: Any = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Refuses unsupported types before writing; lands the file atomically, marker last."""
    from gguf import GGUFReader

    if dense_bits not in (0, 4, 8):
        raise ValueError(f"dense_bits must be 0, 4 or 8, not {dense_bits}")
    if scale_dtype not in ("bf16", "f16"):
        raise ValueError(f"scale_dtype must be 'bf16' or 'f16', not {scale_dtype!r}")
    started = time.perf_counter()
    gguf_path = Path(gguf_path)
    reader = GGUFReader(str(gguf_path))
    counts = _check_types(reader)
    jobs, bits_count = _plan(reader, dense_bits = dense_bits, scale_dtype = scale_dtype)
    entries = [entry for job in jobs for entry in job.outputs]
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise UnsupportedGGUFError("tensor names collide after the mflux rename")
    if any(bits_count.values()):
        # mflux reads each layer's bits from its shapes; the level only marks the file as pre-quantized.
        level = str(max(bits_count, key = lambda b: (bits_count[b], b)))
    else:
        level = "None"
    metadata = {
        "quantization_level": level,
        "mflux_version": MFLUX_VERSION,
        "source_gguf": gguf_path.name,
        "unsloth_converter": CONVERTER_VERSION,
    }
    header = _header_bytes(entries, metadata)

    target_dir = Path(out_dir) / "transformer"
    target_dir.mkdir(parents = True, exist_ok = True)
    final = target_dir / OUTPUT_FILENAME
    marker = target_dir / MARKER_FILENAME
    # A stale marker must not vouch for the file this run is about to replace.
    with _suppress_missing():
        marker.unlink()
    tmp = target_dir / f".{OUTPUT_FILENAME}.{os.getpid()}.tmp"
    total = len(jobs)
    written = 0
    try:
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            for index, job in enumerate(jobs):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("cancelled")
                produced = list(job.produce())
                if len(produced) != len(job.outputs):
                    raise RuntimeError(f"{job.tensor.name}: produced {len(produced)} tensors")
                for (name, dtype, shape), array in zip(job.outputs, produced):
                    data = np.ascontiguousarray(array)
                    if list(data.shape) != list(shape) or data.nbytes != _nbytes(dtype, shape):
                        raise RuntimeError(
                            f"{name}: produced {data.dtype}{list(data.shape)}, planned {dtype}{shape}"
                        )
                    fh.write(data.tobytes())
                    written += data.nbytes
                if progress is not None:
                    progress(index + 1, total)
        os.replace(tmp, final)
    except BaseException:
        with _suppress_missing():
            tmp.unlink()
        raise
    out_bytes = final.stat().st_size
    stat = gguf_path.stat()
    summary = {
        "converter_version": CONVERTER_VERSION,
        "source_gguf": gguf_path.name,
        "source_size": int(stat.st_size),
        "source_mtime": float(stat.st_mtime),
        "counts": counts,
        "dense_bits": dense_bits,
        "scale_dtype": scale_dtype,
        "quantization_level": level,
        "n_out_tensors": len(entries),
        "output_bytes": int(out_bytes),
        "bytes": int(written),
        "seconds": round(time.perf_counter() - started, 3),
    }
    marker_tmp = target_dir / f".{MARKER_FILENAME}.{os.getpid()}.tmp"
    marker_tmp.write_text(json.dumps(summary, indent = 1), encoding = "utf-8")
    os.replace(marker_tmp, marker)
    return summary


class _suppress_missing:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, FileNotFoundError)


# ── Cache layout ────────────────────────────────────────────────────────────────────────────────


def _safe(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._@-]+", "_", str(text).replace("\\", "/").replace("/", "--"))
    return "_" if out in ("", ".", "..") else out


def cache_dir_for(
    repo_id: str,
    revision: Optional[str],
    gguf_filename: str,
    *,
    dense_bits: int,
    scale_dtype: str = "bf16",
) -> Path:
    """``<studio cache>/mlx_diffusion/<repo>@<revision>/<gguf stem>-d<bits>-<scale dtype>``."""
    from utils.paths.storage_roots import cache_root

    stem = str(gguf_filename)
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    return (
        cache_root()
        / "mlx_diffusion"
        / f"{_safe(repo_id)}@{_safe(revision or 'local')}"
        / f"{_safe(stem)}-d{int(dense_bits)}-{_safe(scale_dtype)}"
    )


def conversion_complete(
    out_dir: os.PathLike | str, *, source_path: Optional[os.PathLike | str] = None
) -> bool:
    target_dir = Path(out_dir) / "transformer"
    try:
        marker = json.loads((target_dir / MARKER_FILENAME).read_text(encoding = "utf-8"))
        size = (target_dir / OUTPUT_FILENAME).stat().st_size
    except (OSError, ValueError):
        return False
    if not isinstance(marker, dict) or marker.get("converter_version") != CONVERTER_VERSION:
        return False
    if int(marker.get("output_bytes", -1)) != size:
        return False
    if source_path is not None:
        try:
            stat = Path(source_path).stat()
        except OSError:
            return False
        if int(marker.get("source_size", -1)) != int(stat.st_size):
            return False
        if abs(float(marker.get("source_mtime", -1.0)) - float(stat.st_mtime)) > 1e-3:
            return False
    return True
