# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused GEMM for packed W4A8 / W6A8 layers: int8 activations times the packed weight, decoded to the int8 grid in
registers inside the K loop (never written to memory), int8 tensor-core MMA into int32, then Studio's int8 GEMM
epilogue. Output is bit-identical to ``decode_int8_torch`` + ``diffusion_int8_gemm.reference``, checked by a per-card probe.
On sm8x under Triton >= 3.8 (triton-lang/triton#12486, int8 tiles spill) only zero-spill tiles are accepted; anything
else returns None and the caller keeps decode + int8 GEMM.
``UNSLOTH_DIFFUSION_GROUPED_FUSED=0`` disables, ``=1`` also probes unlisted sm80+ archs with the fallback tiles."""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Optional

GROUPED_FUSED_ENV = "UNSLOTH_DIFFUSION_GROUPED_FUSED"
_OFF = ("0", "off", "false", "no")
_ON = ("1", "on", "true", "yes", "force")
_LOCK = threading.Lock()
_STATE: dict = {}
_OP: dict = {}
_CALLS = [0]
# torch._int_mm's row floor; the caller's int8 path owns smaller batches
_MIN_ROWS = 17

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages, loop, split): loop 2 = column phases (split-K over
# an int32 buffer; split 0 = automatic), 0 = byte tiles + join, 1 = per-weight byte gathers. The fused GEMM decodes
# every weight once per M tile, so it only pays at small row counts.
_PHASE = (64, 64, 128, 8, 4, 3, 2, 0)
_JOIN = (128, 64, 64, 8, 4, 3, 0)
# (major, minor) -> ((rows upper bound inclusive, tile, bit depths), ...); first match wins, larger batches decode.
# Measured: sm120 wins only at <= 64 rows and 6 bits; sm89 _JOIN at <= 128 rows (0 spills on L4). sm80 is not listed:
# every tile spills under Triton 3.8; sm90 not measured.
_ARCH_TILES: dict = {
    (8, 9): ((128, _JOIN, (4, 6)),),
    (12, 0): ((64, _PHASE, (6,)),),
}
_FALLBACK_TILES = ((128, _JOIN, (4, 6)),)

_DEVICE: dict = {}
_SPILLS: dict = {}


def fused_mode() -> str:
    raw = (os.environ.get(GROUPED_FUSED_ENV) or "").strip().lower()
    if raw in _OFF:
        return "off"
    if raw in _ON:
        return "force"
    return "auto"


def _triton_version() -> Optional[tuple]:
    try:
        import triton
    except Exception:  # noqa: BLE001
        return None
    match = re.match(r"(\d+)\.(\d+)", str(triton.__version__))
    return (int(match.group(1)), int(match.group(2))) if match else None


def spill_check_needed(capability: tuple, version: Optional[tuple] = None) -> bool:
    """sm8x under Triton >= 3.8: int8 tiles may spill (triton-lang/triton#12486), so a tile needs zero spills."""
    version = _triton_version() if version is None else version
    return int(capability[0]) == 8 and version is not None and version >= (3, 8)


def arch_tiles(capability: Optional[tuple], mode: Optional[str] = None) -> tuple:
    """Candidate ``(rows bound, tile)`` rules for this arch, () where the fused GEMM stays off."""
    mode = fused_mode() if mode is None else mode
    if mode == "off" or capability is None:
        return ()
    cap = (int(capability[0]), int(capability[1]))
    rules = _ARCH_TILES.get(cap)
    if rules is None and mode == "force" and cap >= (8, 0):
        rules = _FALLBACK_TILES
    return tuple(rules or ())


def fused_call_count() -> int:
    return _CALLS[0]


def _interpreting() -> bool:
    return (os.environ.get("TRITON_INTERPRET") or "").strip() == "1"


def _kernel() -> Any:
    with _LOCK:
        if "kernel" in _STATE:
            return _STATE["kernel"]
        try:
            import triton

            from .diffusion_grouped_int8_gemm_kernels import grouped_mm_dq
            _STATE["kernel"] = (triton, grouped_mm_dq)
        except Exception as exc:  # noqa: BLE001 -- no Triton: decode + GEMM
            _STATE["kernel"] = None
            _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return _STATE["kernel"]


def covered(
    xq: Any, packed: Any, s_rel: Any, codebook: Any, bits: int, group: int, tile: tuple
) -> bool:
    """Shape / dtype / layout gate of one launch with ``tile``."""
    import torch
    try:
        bm, bn, bk = tile[0], tile[1], tile[2]
        if (
            bits not in (4, 6)
            or group < 1
            or group % 4
            or bk % group
            or bk % 8
            or (bk // group) & (bk // group - 1)
        ):
            return False
        if len(tile) > 6 and tile[6] == 2 and bk % 64:
            return False  # phase loop: K / 4 per dot, at least the int8 MMA depth
        if xq.dim() != 2 or packed.dim() != 2 or s_rel.dim() != 2 or xq.dtype != torch.int8:
            return False
        m, k = int(xq.shape[0]), int(xq.shape[1])
        n = int(packed.shape[0])
        if m < 1 or n < 1 or k % bk or int(packed.shape[1]) * 8 != k * bits:
            return False
        if int(s_rel.shape[0]) != n or int(s_rel.shape[1]) * group != k:
            return False
        if packed.dtype not in (torch.int8, torch.uint8) or s_rel.dtype not in (
            torch.uint8,
            torch.float8_e4m3fn,
        ):
            return False
        if not (xq.is_contiguous() and packed.is_contiguous() and s_rel.is_contiguous()):
            return False
        if codebook is not None and (
            bits != 4 or codebook.numel() != 16 or codebook.dtype != torch.float32
        ):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def launch(
    xq: Any,
    packed: Any,
    s_rel: Any,
    codebook: Any,
    xs: Any,
    ws: Any,
    bias: Any,
    bits: int,
    group: int,
    tile: tuple,
    rn: Optional[bool] = None,
) -> tuple:
    """``(bf16 [M, N], compiled kernel handle)`` with ``tile``; raises on a launch failure."""
    import torch

    built = _kernel()
    if built is None:
        raise RuntimeError(_STATE.get("error") or "triton unavailable")
    triton, kernel = built
    m, k = int(xq.shape[0]), int(xq.shape[1])
    n = int(packed.shape[0])
    bm, bn, bk, gm, warps, stages = tile[:6]
    gather = int(tile[6]) if len(tile) > 6 else 2
    split = split_k(m, n, k, tile, xq.device) if gather == 2 else 1
    out = torch.empty((m, n), device = xq.device, dtype = torch.bfloat16)
    part = torch.zeros((m, n), device = xq.device, dtype = torch.int32) if split > 1 else out
    cb = codebook.reshape(-1).contiguous() if codebook is not None else ws
    xs = xs.reshape(-1)
    ws = ws.reshape(-1)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), split)
    rn = (not _interpreting()) if rn is None else rn
    hw = bool(rn) and xq.is_cuda and tuple(torch.cuda.get_device_capability(xq.device)) >= (8, 9)
    kw = (
        {}
        if _interpreting()
        else {"num_warps": warps, "num_stages": stages, "enable_fp_fusion": False}
    )
    import contextlib

    guard = torch.cuda.device(xq.device) if xq.is_cuda else contextlib.nullcontext()
    with guard:
        handle = kernel[grid](
            xq,
            packed.view(torch.uint8),
            s_rel.view(torch.uint8),
            cb,
            xs,
            ws,
            bias if bias is not None else ws,
            out,
            part,
            m,
            n,
            k,
            xq.stride(0),
            packed.stride(0),
            int(s_rel.shape[1]),
            out.stride(0),
            BITS = int(bits),
            GROUP = int(group),
            HAS_CB = codebook is not None,
            HAS_BIAS = bias is not None,
            XS_FP32 = xs.dtype == torch.float32,
            WS_FP32 = ws.dtype == torch.float32,
            RN = bool(rn),
            HW = hw,
            GATHER = gather,
            BLOCK_M = bm,
            BLOCK_N = bn,
            BLOCK_K = bk,
            GROUP_M = gm,
            SPLIT_K = split,
            **kw,
        )
        if split > 1:
            from .diffusion_grouped_int8_gemm_kernels import grouped_epilogue
            ekw = {} if _interpreting() else {"num_warps": 4, "enable_fp_fusion": False}
            grouped_epilogue[(triton.cdiv(m, 32), triton.cdiv(n, 128))](
                part,
                xs,
                ws,
                bias if bias is not None else ws,
                out,
                m,
                n,
                out.stride(0),
                HAS_BIAS = bias is not None,
                XS_FP32 = xs.dtype == torch.float32,
                WS_FP32 = ws.dtype == torch.float32,
                RN = bool(rn),
                BLOCK_M = 32,
                BLOCK_N = 128,
                **ekw,
            )
    return out, handle


_SMS: dict = {}


def _sm_count(device: Any) -> int:
    if getattr(device, "type", "") != "cuda":
        return 4  # the interpreter: small, so the tests reach the split path
    index = device.index if device.index is not None else 0
    if index not in _SMS:
        import torch
        _SMS[index] = int(torch.cuda.get_device_properties(index).multi_processor_count)
    return _SMS[index]


def split_k(m: int, n: int, k: int, tile: tuple, device: Any) -> int:
    """K splits for the phase loop: double while the output tiles fill under two waves of the card and every split
    keeps at least two K blocks (an explicit 8th tile entry > 0 pins it)."""
    if len(tile) > 7 and int(tile[7]) > 0:
        return int(tile[7])
    bm, bn, bk = tile[0], tile[1], tile[2]
    tiles = -(-m // bm) * -(-n // bn)
    steps = k // bk
    target = 2 * _sm_count(device)
    split = 1
    while split < 16 and tiles * split < target and (split * 2) * 2 <= steps:
        split *= 2
    return split


def n_spills(handle: Any) -> Optional[int]:
    """Spilled registers the compiled kernel reports (None when this Triton does not say)."""
    for obj in (handle, getattr(handle, "kernel", None)):
        val = getattr(obj, "n_spills", None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


def reference(xq: Any, q8: Any, xs: Any, ws: Any, bias: Any) -> Any:
    """The decoded-weight path's maths: int32 GEMM, then the epilogue in torchao's eager order."""
    import torch

    if xq.is_cuda:
        from .diffusion_int8_gemm import reference as gemm_reference
        return gemm_reference(xq, q8, xs, ws, bias)
    c = (xq.to(torch.int64) @ q8.to(torch.int64).t()).to(torch.int32)
    y = (c * xs.reshape(-1, 1)).to(torch.bfloat16)
    y = y * ws.reshape(-1)
    if bias is not None:
        y = y + bias
    return y.to(torch.bfloat16)


def probe_cases(device: Any) -> list:
    """Packed operands covering both bit depths, the 4-bit codebook, both group sizes, ragged rows, bias / none,
    exact .5 ties, subnormal and zero steps."""
    import torch

    from .diffusion_grouped_int8 import pack_codes

    g = torch.Generator().manual_seed(0)
    cases = []
    for bits, group, book, m, n, k, bias in (
        (6, 16, False, 33, 256, 512, True),
        (6, 32, False, 130, 384, 768, False),
        (4, 16, True, 70, 256, 512, True),
        (4, 32, False, 17, 128, 1024, False),
    ):
        codes = torch.randint(0, 1 << bits, (n, k), generator = g)
        raw = torch.randint(0, 256, (n, k // group), generator = g)
        raw[(raw & 0x7F) == 0x7F] = 0x38
        raw[0, :4] = torch.tensor([0x00, 0x01, 0x80, 0x7E])
        raw[1, :] = 0x30
        cb = None
        if book:
            cb = torch.sort(torch.randn(16, generator = g)).values.float()
            cb = (cb / cb.abs().max()).to(device)
        cases.append(
            dict(
                xq = torch.randint(-128, 128, (m, k), generator = g, dtype = torch.int32)
                .to(torch.int8)
                .to(device),
                packed = pack_codes(codes, bits).to(device),
                s_rel = raw.to(torch.uint8).to(device),
                codebook = cb,
                xs = (torch.rand(m, generator = g) * 1e-2 + 1e-4).float().to(device),
                ws = (torch.rand(n, generator = g) * 1e-2 + 1e-4).float().to(device),
                bias = (torch.randn(n, generator = g)).to(torch.bfloat16).to(device) if bias else None,
                bits = bits,
                group = group,
            )
        )
    return cases


def check_tile(device: Any, tile: tuple, need_no_spills: bool) -> tuple:
    """``(ok, spills)``: every probe case bit-identical to decode + reference with ``tile`` (and zero spills when
    required; an unreported spill count fails then)."""
    from .diffusion_grouped_int8 import decode_int8_torch

    spills = None
    for case in probe_cases(device):
        if not covered(
            case["xq"],
            case["packed"],
            case["s_rel"],
            case["codebook"],
            case["bits"],
            case["group"],
            tile,
        ):
            continue
        out, handle = launch(
            case["xq"],
            case["packed"],
            case["s_rel"],
            case["codebook"],
            case["xs"],
            case["ws"],
            case["bias"],
            case["bits"],
            case["group"],
            tile,
        )
        got = n_spills(handle)
        if got is not None:
            spills = max(spills or 0, got)
        if need_no_spills and (got is None or got > 0):
            return False, spills
        q8 = decode_int8_torch(
            case["packed"], case["s_rel"], case["codebook"], case["group"], case["bits"]
        )
        ref = reference(case["xq"], q8, case["xs"], case["ws"], case["bias"])
        import torch

        if not torch.equal(out, ref):
            return False, spills
    return True, spills


def fused_config(index: int) -> Optional[tuple]:
    """Once per card: the accepted ``(rows bound, tile)`` rules, or None (the caller keeps decode + int8 GEMM)."""
    with _LOCK:
        if index in _DEVICE:
            return _DEVICE[index]
    rules = None
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
        if torch.cuda.is_available() and not is_rocm_torch(torch) and _kernel() is not None:
            cap = torch.cuda.get_device_capability(index)
            cands = arch_tiles(cap)
            strict = spill_check_needed(cap)
            ok_tiles: dict = {}
            dev = torch.device("cuda", index)
            for rule in cands:
                tile = rule[1]
                if tile not in ok_tiles:
                    try:
                        ok, spills = check_tile(dev, tile, strict)
                    except Exception:  # noqa: BLE001 -- a tile that cannot build here drops out
                        ok, spills = False, None
                    ok_tiles[tile] = ok
                    _SPILLS[(index, tile)] = spills
            kept = [rule for rule in cands if ok_tiles.get(rule[1])]
            rules = tuple(kept) or None
    except Exception:  # noqa: BLE001
        rules = None
    with _LOCK:
        _DEVICE[index] = rules
    return rules


def tile_for(
    rules: Optional[tuple],
    rows: int,
    bits: Optional[int] = None,
) -> Optional[tuple]:
    """The first rule's tile covering ``rows`` (and ``bits`` when the rule names depths), else None."""
    for rule in rules or ():
        bound, tile = rule[0], rule[1]
        if (bound is None or rows <= bound) and (bits is None or len(rule) < 3 or bits in rule[2]):
            return tile
    return None


def grouped_mm_dequant(
    xq: Any,
    packed: Any,
    s_rel: Any,
    codebook: Any,
    xs: Any,
    ws: Any,
    bias: Any,
    bits: int,
    group: int,
) -> Optional[Any]:
    """bf16 ``[M, N]`` of int8 ``xq`` times the packed weight with the int8 GEMM epilogue, or None when this card /
    shape is not covered (the caller decodes and runs its int8 GEMM)."""
    try:
        if (
            fused_mode() == "off"
            or not getattr(xq, "is_cuda", False)
            or int(xq.shape[0]) < _MIN_ROWS
        ):
            return None
        index = xq.device.index
        if index is None:
            import torch
            index = torch.cuda.current_device()
        tile = tile_for(fused_config(index), int(xq.shape[0]), int(bits))
        if tile is None or not covered(xq, packed, s_rel, codebook, bits, group, tile):
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        out, _ = launch(xq, packed, s_rel, codebook, xs, ws, bias, bits, group, tile)
    except Exception as exc:  # noqa: BLE001 -- a failed launch turns the fused path off on this card
        with _LOCK:
            _DEVICE[index] = None
            _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return None
    _CALLS[0] += 1
    return out


def _mm_eager(xq, packed, s_rel, codebook, xs, ws, bias, bits, group):
    out = grouped_mm_dequant(xq, packed, s_rel, codebook, xs, ws, bias, bits, group)
    if out is not None:
        return out
    from .diffusion_grouped_int8 import decode_int8

    return reference(xq, decode_int8(packed, s_rel, codebook, group, bits), xs, ws, bias)


def grouped_mm_op() -> Any:
    """``torch.ops.unsloth_grouped_int8.mm_dequant``: opaque to the compiler (shape-only fake). Call ``fused_config``
    for the card in eager first; inside a graph the op only reads that verdict."""
    with _LOCK:
        if "op" in _OP:
            return _OP["op"]
    import torch

    @torch.library.custom_op(
        "unsloth_grouped_int8::mm_dequant",
        mutates_args = (),
        schema = "(Tensor xq, Tensor packed, Tensor s_rel, Tensor? codebook, Tensor xs, Tensor ws, Tensor? bias, "
        "int bits, int group) -> Tensor",
    )
    def mm_dequant(xq, packed, s_rel, codebook, xs, ws, bias, bits, group):
        return _mm_eager(xq, packed, s_rel, codebook, xs, ws, bias, bits, group)

    @mm_dequant.register_fake
    def _(xq, packed, s_rel, codebook, xs, ws, bias, bits, group):
        return xq.new_empty((xq.shape[0], packed.shape[0]), dtype = torch.bfloat16)

    with _LOCK:
        _OP.setdefault("op", mm_dequant)
        return _OP["op"]
