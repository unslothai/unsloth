# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Round-to-nearest NVFP4 / MXFP4 quantizers for 16-bit Linear weights, in the layout of `unsloth.kernels.nvfp4`.

Both return (packed uint8 [N, K / 2], block scale [N, K / 16], global scale fp32 [1]) with decode
`e2m1 * scale / global_scale`, the compressed-tensors NVFP4 convention, so the W4A16 kernels and
`_UnslothNVFP4Linear` run either format unchanged.

MXFP4 (OCP: E2M1 values, one power-of-two scale per 32) is exact in that layout: each 32-group scale is
written to both of its 16-groups, as e4m3 under a power-of-two global scale when the tensor's block exponents
span at most 17 octaves (4.5 bits per weight), else as bfloat16 (5 bits per weight).
"""

import os

import torch

__all__ = [
    "NVFP4_WEIGHT_QUANT_ENV",
    "nvfp4_weight_quant_method",
    "nvfp4_quantize_weight",
    "mxfp4_quantize_weight",
    "fp4_quantize_weight",
]

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_E4M3_MAX_4OVER6 = 256.0  # largest max / 4 block scale (256 * 6 / 4 = 384) that still fits 448
NVFP4_GROUP = 16
MXFP4_GROUP = 32
# e4m3 holds 2^-9 (subnormal) .. 2^8 exactly: 17 octaves of power-of-two MX scales under one global scale.
_E4M3_POW2_SPAN = 17
_ROW_CHUNK = 4096  # rows quantized per step: bounds the fp32 temporaries to a few times one chunk

NVFP4_WEIGHT_QUANT_ENV = (
    "UNSLOTH_NVFP4_WEIGHT_QUANT"  # same name and values as unsloth_zoo / Studio
)
_NVFP4_METHODS = {
    "max6": "max6",
    "max/6": "max6",
    "rtn": "max6",
    "torchao": "max6",
    "off": "max6",
    "0": "max6",
    "4over6": "4over6",
    "4/6": "4over6",
    "four_over_six": "4over6",
    "fouroversix": "4over6",
    "on": "4over6",
    "1": "4over6",
}


def nvfp4_weight_quant_method(method = None):
    """`max6` (block max mapped to 6) or `4over6` (per block the lower-error of max -> 6 and max -> 4); the
    default is 4over6, as unsloth_zoo's expert quantizer."""
    raw = method if method is not None else os.environ.get(NVFP4_WEIGHT_QUANT_ENV, "")
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "")
    if value == "":
        return "4over6"
    if value not in _NVFP4_METHODS:
        raise ValueError(f"Unsloth: {NVFP4_WEIGHT_QUANT_ENV}={raw!r} is not one of max6, 4over6")
    return _NVFP4_METHODS[value]


def _round_e2m1(y):
    """Nearest E2M1 code (0..15, bit 3 = sign) of y, saturating at 6, ties to the even code."""
    a = y.abs().clamp(max = _E2M1_MAX)
    grid = torch.tensor(_E2M1, dtype = a.dtype, device = a.device)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(a, mid, right = False)
    tie = (idx < 7) & (a == mid[idx.clamp(max = 6)]) & (idx % 2 == 1)
    idx = torch.where(tie, idx + 1, idx)
    sign = (y < 0) & (idx > 0)
    return (idx | (sign.to(idx.dtype) << 3)).to(torch.uint8)


def _e2m1_to_float(codes):
    lut = torch.tensor(_E2M1, dtype = torch.float32, device = codes.device)
    values = lut[(codes & 7).long()]
    return torch.where((codes & 8).bool(), -values, values)


def _pack_nibbles(codes):
    # Even element in the low nibble, as compressed-tensors' NVFP4PackedCompressor.
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _check_weight(weight, group):
    if weight.dim() != 2:
        raise ValueError(f"Unsloth: FP4 quantization needs a 2-D weight, got {tuple(weight.shape)}")
    if weight.shape[1] % group:
        raise ValueError(
            f"Unsloth: FP4 quantization needs in_features divisible by {group}, got {weight.shape[1]}"
        )


def _nvfp4_candidate(blocks, bmax, g, top):
    s = (bmax / top * g).clamp(max = _E4M3_MAX).to(torch.float8_e4m3fn)
    decode = s.to(torch.float32) / g
    y = torch.where(decode[..., None] > 0, blocks / decode[..., None], torch.zeros_like(blocks))
    return _round_e2m1(y), s, decode


@torch.no_grad()
def nvfp4_quantize_weight(weight, method = None):
    """NVFP4 of an (N, K) weight: one fp32 global scale (`top * 6 / amax`), e4m3 scale per 16, E2M1 codes."""
    _check_weight(weight, NVFP4_GROUP)
    method = nvfp4_weight_quant_method(method)
    top = _E4M3_MAX_4OVER6 if method == "4over6" else _E4M3_MAX
    N, K = weight.shape
    amax = weight.abs().amax().to(torch.float32)
    g = torch.where(amax > 0, (top * _E2M1_MAX) / amax, torch.ones_like(amax))
    packed = torch.empty((N, K // 2), dtype = torch.uint8, device = weight.device)
    scales = torch.empty((N, K // NVFP4_GROUP), dtype = torch.float8_e4m3fn, device = weight.device)
    for r0 in range(0, N, _ROW_CHUNK):
        r1 = min(N, r0 + _ROW_CHUNK)
        blocks = weight[r0:r1].to(torch.float32).view(r1 - r0, K // NVFP4_GROUP, NVFP4_GROUP)
        bmax = blocks.abs().amax(dim = -1)
        q, s, decode = _nvfp4_candidate(blocks, bmax, g, _E2M1_MAX)
        if method == "4over6":
            q4, s4, decode4 = _nvfp4_candidate(blocks, bmax, g, 4.0)
            err6 = ((_e2m1_to_float(q) * decode[..., None] - blocks) ** 2).sum(dim = -1)
            err4 = ((_e2m1_to_float(q4) * decode4[..., None] - blocks) ** 2).sum(dim = -1)
            take4 = err4 < err6
            q = torch.where(take4[..., None], q4, q)
            s = torch.where(take4, s4.view(torch.uint8), s.view(torch.uint8)).view(
                torch.float8_e4m3fn
            )
            del q4, s4, decode4, err6, err4, take4
        packed[r0:r1] = _pack_nibbles(q.reshape(r1 - r0, K))
        scales[r0:r1] = s
        del blocks, bmax, q, s, decode
    return packed, scales, g.reshape(1)


@torch.no_grad()
def mxfp4_quantize_weight(weight):
    """OCP MXFP4 of an (N, K) weight (shared exponent floor(log2(block amax)) - 2, E2M1 RTN), written in the
    NVFP4 kernel layout (see the module docstring)."""
    _check_weight(weight, MXFP4_GROUP)
    N, K = weight.shape
    G = K // MXFP4_GROUP
    packed = torch.empty((N, K // 2), dtype = torch.uint8, device = weight.device)
    exps = torch.empty((N, G), dtype = torch.int16, device = weight.device)
    nonzero = torch.empty((N, G), dtype = torch.bool, device = weight.device)
    for r0 in range(0, N, _ROW_CHUNK):
        r1 = min(N, r0 + _ROW_CHUNK)
        w = weight[r0:r1].to(torch.float32).view(r1 - r0, G, MXFP4_GROUP)
        bmax = w.abs().amax(dim = -1)
        e = torch.floor(torch.log2(bmax.clamp(min = 2.0**-126))) - 2
        e = torch.where(bmax > 0, e, torch.full_like(e, -127.0)).clamp(-127, 127)
        q = _round_e2m1(w / torch.exp2(e)[..., None])
        packed[r0:r1] = _pack_nibbles(q.reshape(r1 - r0, K))
        exps[r0:r1] = e.to(torch.int16)
        nonzero[r0:r1] = bmax > 0
        del w, bmax, e, q
    live = exps[nonzero]
    e_max = int(live.max()) if live.numel() else 0
    e_min = int(live.min()) if live.numel() else 0
    # All-zero blocks decode to 0 under any scale: give them the top one so they never widen the span.
    exps = torch.where(nonzero, exps, torch.full_like(exps, e_max))
    if e_max - e_min <= _E4M3_POW2_SPAN:
        shift = 8 - e_max  # the largest block scale lands on 2^8
        scale = torch.exp2((exps + shift).to(torch.float32)).to(torch.float8_e4m3fn)
        global_scale = torch.tensor([2.0**shift], dtype = torch.float32, device = weight.device)
    else:
        scale = torch.exp2(exps.to(torch.float32)).to(torch.bfloat16)
        global_scale = torch.ones(1, dtype = torch.float32, device = weight.device)
    return packed, scale.repeat_interleave(2, dim = 1).contiguous(), global_scale


def fp4_quantize_weight(
    weight,
    fmt,
    method = None,
):
    if fmt == "nvfp4":
        return nvfp4_quantize_weight(weight, method)
    if fmt == "mxfp4":
        return mxfp4_quantize_weight(weight)
    raise ValueError(f"Unsloth: unknown FP4 format {fmt!r}; use 'nvfp4' or 'mxfp4'")
