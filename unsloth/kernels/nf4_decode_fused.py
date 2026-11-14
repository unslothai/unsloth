# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import functools
from typing import List, Union

import torch
import triton
import triton.language as tl

from .nf4 import _HAS_MUL_RN, _mul, _nf4_lut
from .nf4_gemv import _lut_mode
from .triton_launch import launch, tag_compile_cache

__all__ = [
    "EPI_ROPE_KV",
    "EPI_SWIGLU",
    "EPI_RESNORM",
    "PRE_NONE",
    "PRE_SCALED",
    "PRE_INLOOP",
    "FusedSite",
    "BatchSite",
    "batch_fused_site",
    "batch_final_norm",
    "site_config",
    "fused_site",
    "final_norm",
    "supported_capability",
]

EPI_PLAIN, EPI_ROPE_KV, EPI_SWIGLU, EPI_RESNORM = 0, 1, 2, 3
PRE_NONE, PRE_SCALED, PRE_INLOOP = 0, 1, 2


@torch.library.custom_op("unsloth::nf4_decode_fused_launch", mutates_args = ("outputs",))
def _opaque_launch(
    kind: int,
    grid: List[int],
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    tensor_order: List[int],
    ints: List[int],
    constant_names: str,
    constant_types: str,
    integer_constants: List[int],
    float_constants: List[float],
    option_names: str,
    options: List[Union[int, float, bool]],
) -> None:
    kernels = (
        _nf4_decode_fused_kernel,
        _lora_shrink_kernel,
        _final_norm_kernel,
        None,
        _batch_main_kernel,
        _batch_shrink_kernel,
        _batch_final_kernel,
    )
    tensors = tuple(inputs[i] if i >= 0 else outputs[-i - 1] for i in tensor_order)
    integers, floats = iter(integer_constants), iter(float_constants)
    constants = [
        next(floats) if kind == "f" else bool(next(integers)) if kind == "b" else next(integers)
        for kind in constant_types
    ]
    constexprs = dict(zip(constant_names.split(","), constants))
    launch_options = dict(zip(option_names.split(","), options))
    # Inductor re-emission changes rounding; keep the eager binary inside the graph.
    with torch.cuda.device(tensors[0].device):
        launch(
            kernels[kind],
            tuple(grid),
            tensors + tuple(ints),
            len(tensors),
            constexprs,
            tensors[0].device.index,
            **launch_options,
        )


@_opaque_launch.register_fake
def _opaque_launch_fake(
    kind,
    grid,
    inputs,
    outputs,
    tensor_order,
    ints,
    constant_names,
    constant_types,
    integer_constants,
    float_constants,
    option_names,
    options,
):
    return None


def _launch(kind, kernel, grid, args, n_tensors, constexprs, device_index, **options):
    if not torch.compiler.is_compiling():
        launch(kernel, grid, args, n_tensors, constexprs, device_index, **options)
        return
    if kind in (0, 4):
        epi = constexprs["EPI"]
        mutable = (2, 3, 5) if epi == EPI_RESNORM else (17, 18, 19) if epi == EPI_ROPE_KV else (17,)
    elif kind in (1, 5):
        mutable = (6, 7, 8)
    elif kind in (2, 6):
        mutable = (3,)
    else:
        mutable = (7, 8, 9)
    inputs, outputs, tensor_order = [], [], []
    for i in range(n_tensors):
        if i in mutable:
            outputs.append(args[i])
            tensor_order.append(-len(outputs))
        else:
            tensor_order.append(len(inputs))
            inputs.append(args[i])
    constant_types, integer_constants, float_constants = [], [], []
    for value in constexprs.values():
        if isinstance(value, float):
            constant_types.append("f")
            float_constants.append(value)
        else:
            constant_types.append("b" if isinstance(value, bool) else "i")
            integer_constants.append(int(value))
    _opaque_launch(
        kind,
        list(grid),
        inputs,
        outputs,
        tensor_order,
        list(args[n_tensors:]),
        ",".join(constexprs),
        "".join(constant_types),
        integer_constants,
        float_constants,
        ",".join(options),
        list(options.values()),
    )


@triton.jit
def _rstd_from_partials(
    SS, n_ss, K,
    EPS: tl.constexpr,  # norm epsilon
    SS_BLOCK: tl.constexpr,  # partial reduction tile
):  # fmt: skip
    j = tl.arange(0, SS_BLOCK)
    ss = tl.sum(tl.load(SS + j, mask = j < n_ss, other = 0.0), axis = 0)
    return tl.math.rsqrt(ss / K + EPS)


@triton.jit
def _nf4_decode_fused_kernel(
    X, G, R, XG_OUT, SS_IN, SS_OUT, G_NEXT,
    W, ABSMAX, CODE2, ABSMAX2, SEG_OFFSET, LUT, LB0,
    LB1, LB2, LORA_H, OUT, K_CACHE, V_CACHE, COS,
    SIN, POS, SLOT, W1, W2, ABSMAX1, ABSMAX_SEG2,
    NESTED1, NESTED2, lora_rank, N, K, HALF, seg1,
    seg2, n_ss, stride_cache_h, stride_cache_l, stride_cos, head_dim,
    EPS: tl.constexpr,  # norm epsilon
    BLOCK_P: tl.constexpr,  # output row pairs
    BLOCK_K: tl.constexpr,  # input tile
    BLOCKSIZE: tl.constexpr,  # NF4 block size
    BLOCKSIZE2: tl.constexpr,  # nested scale block size
    NESTED: tl.constexpr,  # double quantization
    USE_MUL_RN: tl.constexpr,  # reduction product rounding
    EVEN_K: tl.constexpr,  # input divisibility
    LUT_MODE: tl.constexpr,  # code lookup mode
    PRE: tl.constexpr,  # input norm mode
    LORA: tl.constexpr,  # adapter presence
    RANK_PAD: tl.constexpr,  # padded adapter rank
    EPI: tl.constexpr,  # output epilogue
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    SEPARATE: tl.constexpr,  # separate projections
):  # fmt: skip
    W = W.to(tl.pointer_type(tl.uint8))
    pid = tl.program_id(0)
    R2: tl.constexpr = 2 * BLOCK_P  # output rows per program
    i = tl.arange(0, R2)
    p = pid * BLOCK_P + i % BLOCK_P
    rows = (p // HALF) * (2 * HALF) + p % HALF + (i // BLOCK_P) * HALF
    rmask = rows < N
    rows = tl.where(rmask, rows, 0).to(tl.int64)
    s0 = rows < seg1
    s1 = (rows >= seg1) & (rows < seg2)
    storage_rows = rows
    wbase = W + tl.zeros([R2], tl.int32)
    abase = ABSMAX + tl.zeros([R2], tl.int32)
    a2base = ABSMAX2 + tl.zeros([R2], tl.int32)
    if SEPARATE:
        storage_rows = rows - tl.where(s0, 0, tl.where(s1, seg1, seg2))
        wbase = tl.where(
            s0,
            W,
            tl.where(s1, W1.to(tl.pointer_type(tl.uint8)), W2.to(tl.pointer_type(tl.uint8))),
        )
        abase = tl.where(s0, ABSMAX, tl.where(s1, ABSMAX1, ABSMAX_SEG2))
        a2base = tl.where(s0, ABSMAX2, tl.where(s1, NESTED1, NESTED2))
    if NESTED:
        offset = tl.load(SEG_OFFSET + tl.where(s0, 0, tl.where(s1, 1, 2)))
    HALF_K: tl.constexpr = BLOCK_K // 2  # packed input bytes
    NB: tl.constexpr = BLOCK_K // BLOCKSIZE  # quantization blocks per tile
    half_k = tl.arange(0, HALF_K)
    ks = tl.arange(0, BLOCK_K)
    nbs = tl.arange(0, NB)
    blocks_per_row = K // BLOCKSIZE
    acc = tl.zeros([R2], dtype = tl.float32)
    if PRE == 2:
        ss = tl.zeros([BLOCK_K], dtype = tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = ((k0 + 2 * half_k) < K) | EVEN_K
        m = rmask[:, None] & kmask[None, :]
        b = tl.load(
            wbase[:, None] + storage_rows[:, None] * (K // 2) + (k0 // 2 + half_k)[None, :],
            mask = m,
            other = 0,
        )
        b = b.to(tl.int32)
        hi = _nf4_lut(LUT, b >> 4, LUT_MODE)
        lo = _nf4_lut(LUT, b & 15, LUT_MODE)
        w = tl.reshape(tl.join(hi, lo), (R2, BLOCK_K))
        xk = k0 + ks
        xm = (xk < K) | EVEN_K
        x = tl.load(X + xk, mask = xm, other = 0.0).to(tl.float32)
        if PRE == 2:
            ss += _mul(x, x, USE_MUL_RN)
            x = x * tl.load(G + xk, mask = xm, other = 0.0).to(tl.float32)
        # Each quantization block is summed before it is scaled, as gemv_nf4 does.
        # RN products keep butterfly lanes bit-identical before shared stores (triton#7264).
        part = tl.sum(tl.reshape(_mul(w, x[None, :], USE_MUL_RN), (R2, NB, BLOCKSIZE)), axis = 2)
        bmask = rmask[:, None] & (((k0 + nbs * BLOCKSIZE) < K) | EVEN_K)[None, :]
        blk = storage_rows[:, None] * blocks_per_row + (k0 // BLOCKSIZE + nbs)[None, :]
        if NESTED:
            q = tl.load(abase[:, None] + blk, mask = bmask, other = 0).to(tl.int32)
            c2 = tl.load(CODE2 + q, mask = bmask, other = 0.0)
            s2 = tl.load(a2base[:, None] + blk // BLOCKSIZE2, mask = bmask, other = 0.0)
            a = _mul(c2, s2, USE_MUL_RN)
            a = a + offset[:, None]
        else:
            a = tl.load(abase[:, None] + blk, mask = bmask, other = 0.0)
        acc += tl.sum(_mul(part, a, USE_MUL_RN), axis = 1)
    if PRE == 1:
        acc = acc * _rstd_from_partials(SS_IN, n_ss, K, EPS, SS_BLOCK)
    if PRE == 2:
        acc = acc * tl.math.rsqrt(tl.sum(ss, axis = 0) / K + EPS)
    if LORA:
        seg = tl.where(s0, 0, tl.where(s1, 1, 2))
        lr = rows - tl.where(s0, 0, tl.where(s1, seg1, seg2)).to(tl.int64)
        j = tl.arange(0, RANK_PAD)
        jm = j < lora_rank
        brow = tl.where(s0, LB0, tl.where(s1, LB1, LB2)) + lr * lora_rank
        bw = tl.load(brow[:, None] + j[None, :], mask = rmask[:, None] & jm[None, :], other = 0.0)
        hv = tl.load(LORA_H + seg[:, None] * RANK_PAD + j[None, :], mask = jm[None, :], other = 0.0)
        acc += tl.sum(_mul(bw.to(tl.float32), hv, USE_MUL_RN), axis = 1)

    out_ty = OUT.dtype.element_ty
    if EPI == 3:
        # Each residual element is read and written by one thread, so the in-place update is safe.
        ri = pid * R2 + tl.arange(0, R2)
        rm = ri < N
        h = acc.to(out_ty).to(tl.float32) + tl.load(R + ri, mask = rm, other = 0.0).to(tl.float32)
        h = h.to(R.dtype.element_ty)
        tl.store(R + ri, h, mask = rm)
        hf = tl.where(rm, h.to(tl.float32), 0.0)
        tl.store(SS_OUT + pid, tl.sum(_mul(hf, hf, USE_MUL_RN), axis = 0))
        g = tl.load(G_NEXT + ri, mask = rm, other = 0.0).to(tl.float32)
        tl.store(XG_OUT + ri, (hf * g).to(XG_OUT.dtype.element_ty), mask = rm)
    else:
        a_val, b_val = tl.split(tl.trans(tl.reshape(acc, (2, BLOCK_P))))
        pA = pid * BLOCK_P + tl.arange(0, BLOCK_P)
        rowA = (pA // HALF) * (2 * HALF) + pA % HALF
        rowB = rowA + HALF
        if EPI == 0:
            tl.store(OUT + rowA, a_val.to(out_ty), mask = rowA < N)
            tl.store(OUT + rowB, b_val.to(out_ty), mask = rowB < N)
        elif EPI == 1:
            pm = rowA < N
            pos = tl.load(POS).to(tl.int64)
            slot = tl.load(SLOT).to(tl.int64)
            jj = pA % HALF
            c = tl.load(COS + pos * stride_cos + jj, mask = pm, other = 1.0).to(tl.float32)
            s = tl.load(SIN + pos * stride_cos + jj, mask = pm, other = 0.0).to(tl.float32)
            rot = rowA < seg2  # Q and K rows rotate, V rows do not
            a_rot = tl.where(rot, a_val * c - b_val * s, a_val)
            b_rot = tl.where(rot, b_val * c + a_val * s, b_val)
            is_q = rowA < seg1
            tl.store(OUT + rowA, a_rot.to(out_ty), mask = pm & is_q)
            tl.store(OUT + rowB, b_rot.to(out_ty), mask = pm & is_q)
            is_k = (rowA >= seg1) & (rowA < seg2)
            kr = tl.where(is_k, rowA - seg1, 0)
            kptr = (
                K_CACHE + slot * stride_cache_l + (kr // head_dim) * stride_cache_h + kr % head_dim
            )
            tl.store(kptr, a_rot.to(K_CACHE.dtype.element_ty), mask = pm & is_k)
            tl.store(kptr + HALF, b_rot.to(K_CACHE.dtype.element_ty), mask = pm & is_k)
            is_v = rowA >= seg2
            vr = tl.where(is_v, rowA - seg2, 0)
            vptr = (
                V_CACHE + slot * stride_cache_l + (vr // head_dim) * stride_cache_h + vr % head_dim
            )
            tl.store(vptr, a_rot.to(V_CACHE.dtype.element_ty), mask = pm & is_v)
            tl.store(vptr + HALF, b_rot.to(V_CACHE.dtype.element_ty), mask = pm & is_v)
        else:
            pm = pA < HALF
            act = a_val * tl.sigmoid(a_val) * b_val
            tl.store(OUT + pA, act.to(out_ty), mask = pm)


@triton.jit
def _lora_shrink_kernel(
    X, G, SS_IN, A0, A1, A2, P,
    H, COUNTER, n_rows, K, lora_rank, n_ss, n_splits,
    SCALING: tl.constexpr,  # adapter scale
    EPS: tl.constexpr,  # norm epsilon
    RANK_PAD: tl.constexpr,  # padded adapter rank
    BLOCK_R: tl.constexpr,  # adapter row tile
    SPLIT_K: tl.constexpr,  # adapter input tile
    PRE: tl.constexpr,  # input norm mode
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    USE_MUL_RN: tl.constexpr,  # reduction product rounding
):  # fmt: skip
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    rm = rows < n_rows
    ks = pid * SPLIT_K + tl.arange(0, SPLIT_K)
    km = ks < K
    x = tl.load(X + ks, mask = km, other = 0.0).to(tl.float32)
    if PRE == 2:
        tl.store(P + n_splits * BLOCK_R + pid, tl.sum(_mul(x, x, USE_MUL_RN), axis = 0))
        x = x * tl.load(G + ks, mask = km, other = 0.0).to(tl.float32)
    sg = rows // lora_rank
    arow = tl.where(sg == 0, A0, tl.where(sg == 1, A1, A2)) + (rows % lora_rank).to(tl.int64) * K
    a = tl.load(arow[:, None] + ks[None, :], mask = rm[:, None] & km[None, :], other = 0.0)
    tl.store(
        P + pid * BLOCK_R + rows,
        tl.sum(_mul(a.to(tl.float32), x[None, :], USE_MUL_RN), axis = 1),
    )
    count = tl.atomic_add(COUNTER, 1, sem = "acq_rel")
    if count == n_splits - 1:
        acc = tl.zeros([BLOCK_R], dtype = tl.float32)
        for s in range(0, n_splits):
            acc += tl.load(P + s * BLOCK_R + rows, cache_modifier = ".cg")
        if PRE == 1:
            acc = acc * _rstd_from_partials(SS_IN, n_ss, K, EPS, SS_BLOCK)
        if PRE == 2:
            ss = 0.0
            for s in range(0, n_splits):
                ss += tl.load(P + n_splits * BLOCK_R + s, cache_modifier = ".cg")
            acc = acc * tl.math.rsqrt(ss / K + EPS)
        seg = rows // lora_rank
        j = rows % lora_rank
        tl.store(H + seg * RANK_PAD + j, acc * SCALING, mask = rm)
        tl.atomic_xchg(COUNTER, 0)


@triton.jit
def _final_norm_kernel(
    R, SS, W, OUT, n_ss, K,
    EPS: tl.constexpr,  # norm epsilon
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    BLOCK: tl.constexpr,  # hidden tile
):  # fmt: skip
    k = tl.arange(0, BLOCK)
    km = k < K
    rstd = _rstd_from_partials(SS, n_ss, K, EPS, SS_BLOCK)
    h = tl.load(R + k, mask = km, other = 0.0).to(tl.float32)
    out_ty = OUT.dtype.element_ty
    xn = (h * rstd).to(out_ty).to(tl.float32)
    tl.store(
        OUT + k,
        (xn * tl.load(W + k, mask = km, other = 0.0).to(tl.float32)).to(out_ty),
        mask = km,
    )


# Tuned on Llama 1B / 8B; other GPUs use gemv_nf4's config (lost on T4) and need =force.
_TABLE = {
    (8, 0): dict(qkv = (2, 1024, 2), o = (2, 2048, 4), gate_up = (2, 1024, 2), down = (2, 2048, 4)),  # A100
    (8, 9): dict(qkv = (4, 512, 2), o = (2, 1024, 2), gate_up = (4, 1024, 2), down = (2, 2048, 4)),  # L4
    (7, 5): dict(qkv = (4, 1024, 2), o = (4, 1024, 2), gate_up = (4, 1024, 2), down = (4, 2048, 4)),  # T4
}
_NARROW_CAPS = ((7, 5), (8, 9))


def supported_capability(capability):
    return tuple(capability) in _TABLE


@functools.lru_cache(maxsize = None)
def site_config(kind, N, K, blocksize, capability):
    entry = _TABLE.get(tuple(capability), {}).get(kind)
    if entry is not None:
        block_p, block_k, num_warps = entry
        block_k = max(blocksize, min(block_k, triton.next_power_of_2(K)))
        return block_p, block_k, num_warps, _lut_mode(block_k)
    block_k = max(blocksize, min(2048, triton.next_power_of_2(K)))
    if tuple(capability) in _NARROW_CAPS and N * K > 1 << 25 and K >= 4096:
        return 2, max(blocksize, 1024), 2, _lut_mode(max(blocksize, 1024))
    return 2, block_k, 4, _lut_mode(block_k)


class FusedSite:
    """One call site's NF4 buffers and per-projection LoRA A / B."""

    def __init__(
        self,
        kind,
        packed,
        absmax,
        code,
        bounds,
        K,
        blocksize,
        code2 = None,
        absmax2 = None,
        seg_offset = None,
        blocksize2 = 256,
        lora = None,
        scaling = 1.0,
        segments = None,
    ):
        pad = lambda xs: tuple(xs) + (xs[-1],) * (3 - len(xs))
        self.segments = tuple(segments) if segments is not None else None
        self.separate = segments is not None and len(segments) > 1
        if segments is None:
            segments = [(packed, absmax, absmax2)]
        else:
            packed, absmax, absmax2 = segments[0]
        self.W3 = pad([segment[0] for segment in segments])
        self.scales3 = pad([segment[1] for segment in segments])
        self.state2scales3 = pad(
            [segment[2] if segment[2] is not None else segment[1] for segment in segments]
        )
        self.kind = kind
        self.packed = packed
        self.absmax = absmax
        self.code = code
        self.bounds = list(bounds)
        self.N = self.bounds[-1]
        self.K = K
        self.blocksize = blocksize
        self.blocksize2 = blocksize2
        self.nested = code2 is not None
        self.code2 = code2 if self.nested else absmax
        self.absmax2 = absmax2 if self.nested else absmax
        self.seg_offset = seg_offset if self.nested else absmax
        self.lora = lora is not None
        if self.lora:
            r = lora[0][0].shape[0]
            self.rank = r
            self.rank_pad = max(16, triton.next_power_of_2(r))
            self.n_seg = len(lora)
            self.A3 = pad([a for a, _ in lora])
            self.B3 = pad([b for _, b in lora])
            self.scaling = float(scaling)
            device = packed.device
            n_rows = self.n_seg * r
            self.block_r = max(16, triton.next_power_of_2(n_rows))
            self.split_k = max(16, min(256, triton.next_power_of_2(K), 16384 // self.block_r))
            self.n_splits = triton.cdiv(K, self.split_k)
            # Workspaces outlive the generate() inference_mode they are built in (as BatchSite's).
            with torch.inference_mode(False), torch.no_grad():
                self.H = torch.zeros(3 * self.rank_pad, device = device, dtype = torch.float32)
                self.partials = torch.zeros(
                    self.n_splits * (self.block_r + 1), device = device, dtype = torch.float32
                )
                self.counter = torch.zeros(1, device = device, dtype = torch.int32)

    def seg_bounds(self):
        b = self.bounds + [self.N] * (3 - len(self.bounds))
        return b[0], b[1]

    def n_programs(self, block_p, epi):
        if epi in (EPI_PLAIN, EPI_RESNORM):
            return triton.cdiv(self.N, 2 * block_p)
        return triton.cdiv(self.N // 2, block_p)


def _lora_shrink(site, x, pre, norm_weight, ss_in, n_ss, eps, device_index):
    d = x
    args = (
        x,
        norm_weight if norm_weight is not None else d,
        ss_in if ss_in is not None else d,
        *site.A3,
        site.partials,
        site.H,
        site.counter,
        site.n_seg * site.rank,
        site.K,
        site.rank,
        n_ss,
        site.n_splits,
    )
    constexprs = dict(
        SCALING = site.scaling,
        EPS = eps,
        RANK_PAD = site.rank_pad,
        BLOCK_R = site.block_r,
        SPLIT_K = site.split_k,
        PRE = pre,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        USE_MUL_RN = _HAS_MUL_RN,
    )
    _launch(
        1, _lora_shrink_kernel, (site.n_splits,), args, 9, constexprs, device_index, num_warps = 4
    )


def fused_site(
    site,
    x,
    out = None,
    *,
    epi,
    pre = PRE_NONE,
    capability,
    norm_weight = None,
    eps = 1e-6,
    ss_in = None,
    n_ss = 0,
    residual = None,
    ss_out = None,
    xg_out = None,
    next_norm_weight = None,
    k_cache = None,
    v_cache = None,
    cache_strides = (0, 0),
    cos = None,
    sin = None,
    pos = None,
    slot = None,
    head_dim = None,
    config = None,
):
    """One fused call site; returns its ss partial / program count.
    pre: PRE_NONE (x is the GEMV input), PRE_SCALED (x = h * gamma; rstd from ss_in[:n_ss]),
         PRE_INLOOP (x = h, gamma = norm_weight; the norm runs in the main loop).
    EPI_RESNORM: residual += out (in place), xg_out = residual * next_norm_weight, ss_out[pid] partials.
    EPI_ROPE_KV: Q (rotated) -> out, K (rotated) / V -> k_cache / v_cache at row slot[0]; cache_strides
    = (head stride, row stride) in elements; pos[0] is the RoPE position."""
    K, N = site.K, site.N
    block_p, block_k, num_warps, lut = config or site_config(
        site.kind, N, K, site.blocksize, capability
    )
    if epi in (EPI_PLAIN, EPI_RESNORM):
        half = block_p
    elif epi == EPI_ROPE_KV:
        half = head_dim // 2
    else:
        half = site.bounds[0]
    nprog = site.n_programs(block_p, epi)
    device_index = x.device.index
    if site.lora:
        _lora_shrink(site, x, pre, norm_weight, ss_in, n_ss, eps, device_index)
    seg1, seg2 = site.seg_bounds()
    d = x
    B3 = site.B3 if site.lora else (d, d, d)
    tensors = (
        x,
        norm_weight if norm_weight is not None else d,
        residual if residual is not None else d,
        xg_out if xg_out is not None else d,
        ss_in if ss_in is not None else d,
        ss_out if ss_out is not None else d,
        next_norm_weight if next_norm_weight is not None else d,
        site.packed,
        site.absmax,
        site.code2,
        site.absmax2,
        site.seg_offset,
        site.code,
        B3[0],
        B3[1],
        B3[2],
        site.H if site.lora else d,
        out if out is not None else d,
        k_cache if k_cache is not None else d,
        v_cache if v_cache is not None else d,
        cos if cos is not None else d,
        sin if sin is not None else d,
        pos if pos is not None else d,
        slot if slot is not None else d,
        site.W3[1],
        site.W3[2],
        site.scales3[1],
        site.scales3[2],
        site.state2scales3[1],
        site.state2scales3[2],
    )
    ints = (
        site.rank if site.lora else 0,
        N,
        K,
        half,
        seg1,
        seg2,
        n_ss,
        cache_strides[0],
        cache_strides[1],
        cos.stride(0) if cos is not None else 0,
        head_dim or 1,
    )
    constexprs = dict(
        EPS = eps,
        BLOCK_P = block_p,
        BLOCK_K = block_k,
        BLOCKSIZE = site.blocksize,
        BLOCKSIZE2 = site.blocksize2 if site.nested else 1,
        NESTED = site.nested,
        USE_MUL_RN = _HAS_MUL_RN,
        EVEN_K = K % block_k == 0,
        LUT_MODE = lut,
        PRE = pre,
        LORA = site.lora,
        RANK_PAD = site.rank_pad if site.lora else 16,
        EPI = epi,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        SEPARATE = site.separate,
    )
    _launch(
        0,
        _nf4_decode_fused_kernel,
        (nprog,),
        tensors + ints,
        len(tensors),
        constexprs,
        device_index,
        num_warps = num_warps,
        enable_fp_fusion = True,
    )
    return nprog


def final_norm(residual, ss, n_ss, weight, eps, out):
    """out = RMSNorm(residual) * weight with rstd from the last EPI_RESNORM site's partials."""
    K = residual.numel()
    args = (residual, ss, weight, out, n_ss, K)
    constexprs = dict(
        EPS = eps,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        BLOCK = triton.next_power_of_2(K),
    )
    _launch(2, _final_norm_kernel, (1,), args, 4, constexprs, residual.device.index, num_warps = 8)
    return out


@triton.jit
def _batch_main_kernel(
    X, G, R, XG_OUT, SS_IN, SS_OUT, G_NEXT,
    W, ABSMAX, CODE2, ABSMAX2, SEG_OFFSET, LUT, LB0,
    LB1, LB2, LORA_H, OUT, K_CACHE, V_CACHE, COS,
    SIN, POS, SLOT, W1, W2, ABSMAX1, ABSMAX_SEG2,
    NESTED1, NESTED2, lora_rank, N, K, HALF, seg1,
    seg2, n_ss, stride_cache_h, stride_cache_l, stride_cos, head_dim, ss_stride,
    out_stride, residual_stride, xg_stride, cache_batch_stride,
    EPS: tl.constexpr,  # norm epsilon
    BLOCK_P: tl.constexpr,  # output row pairs
    BLOCK_K: tl.constexpr,  # input tile
    BLOCKSIZE: tl.constexpr,  # NF4 block size
    BLOCKSIZE2: tl.constexpr,  # nested scale block size
    NESTED: tl.constexpr,  # double quantization
    USE_MUL_RN: tl.constexpr,  # reduction product rounding
    EVEN_K: tl.constexpr,  # input divisibility
    LUT_MODE: tl.constexpr,  # code lookup mode
    PRE: tl.constexpr,  # input norm mode
    LORA: tl.constexpr,  # adapter presence
    RANK_PAD: tl.constexpr,  # padded adapter rank
    EPI: tl.constexpr,  # output epilogue
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    SEPARATE: tl.constexpr,  # separate projections
):  # fmt: skip
    row = tl.program_id(1).to(tl.int64)
    X = X + row * K
    R = R + row * residual_stride
    XG_OUT = XG_OUT + row * xg_stride
    SS_IN = SS_IN + row * ss_stride
    SS_OUT = SS_OUT + row * ss_stride
    LORA_H = LORA_H + row * (3 * RANK_PAD)
    OUT = OUT + row * out_stride
    K_CACHE = K_CACHE + row * cache_batch_stride
    V_CACHE = V_CACHE + row * cache_batch_stride
    POS = POS + row
    _nf4_decode_fused_kernel(
        X,
        G,
        R,
        XG_OUT,
        SS_IN,
        SS_OUT,
        G_NEXT,
        W,
        ABSMAX,
        CODE2,
        ABSMAX2,
        SEG_OFFSET,
        LUT,
        LB0,
        LB1,
        LB2,
        LORA_H,
        OUT,
        K_CACHE,
        V_CACHE,
        COS,
        SIN,
        POS,
        SLOT,
        W1,
        W2,
        ABSMAX1,
        ABSMAX_SEG2,
        NESTED1,
        NESTED2,
        lora_rank,
        N,
        K,
        HALF,
        seg1,
        seg2,
        n_ss,
        stride_cache_h,
        stride_cache_l,
        stride_cos,
        head_dim,
        EPS = EPS,
        BLOCK_P = BLOCK_P,
        BLOCK_K = BLOCK_K,
        BLOCKSIZE = BLOCKSIZE,
        BLOCKSIZE2 = BLOCKSIZE2,
        NESTED = NESTED,
        USE_MUL_RN = USE_MUL_RN,
        EVEN_K = EVEN_K,
        LUT_MODE = LUT_MODE,
        PRE = PRE,
        LORA = LORA,
        RANK_PAD = RANK_PAD,
        EPI = EPI,
        SS_BLOCK = SS_BLOCK,
        SEPARATE = SEPARATE,
    )


@triton.jit
def _batch_shrink_kernel(
    X, G, SS_IN, A0, A1, A2, P,
    H, COUNTER, n_rows, K, lora_rank, n_ss, n_splits,
    ss_stride,
    SCALING: tl.constexpr,  # adapter scale
    EPS: tl.constexpr,  # norm epsilon
    RANK_PAD: tl.constexpr,  # padded adapter rank
    BLOCK_R: tl.constexpr,  # adapter row tile
    SPLIT_K: tl.constexpr,  # adapter input tile
    PRE: tl.constexpr,  # input norm mode
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    USE_MUL_RN: tl.constexpr,  # reduction product rounding
):  # fmt: skip
    row = tl.program_id(1).to(tl.int64)
    X = X + row * K
    SS_IN = SS_IN + row * ss_stride
    P = P + row * n_splits * (BLOCK_R + 1)
    H = H + row * (3 * RANK_PAD)
    COUNTER = COUNTER + row
    _lora_shrink_kernel(
        X,
        G,
        SS_IN,
        A0,
        A1,
        A2,
        P,
        H,
        COUNTER,
        n_rows,
        K,
        lora_rank,
        n_ss,
        n_splits,
        SCALING = SCALING,
        EPS = EPS,
        RANK_PAD = RANK_PAD,
        BLOCK_R = BLOCK_R,
        SPLIT_K = SPLIT_K,
        PRE = PRE,
        SS_BLOCK = SS_BLOCK,
        USE_MUL_RN = USE_MUL_RN,
    )


@triton.jit
def _batch_final_kernel(
    R, SS, W, OUT, n_ss, K, ss_stride,
    EPS: tl.constexpr,  # norm epsilon
    SS_BLOCK: tl.constexpr,  # partial reduction tile
    BLOCK: tl.constexpr,  # hidden tile
):  # fmt: skip
    row = tl.program_id(1).to(tl.int64)
    R = R + row * K
    SS = SS + row * ss_stride
    OUT = OUT + row * K
    _final_norm_kernel(R, SS, W, OUT, n_ss, K, EPS = EPS, SS_BLOCK = SS_BLOCK, BLOCK = BLOCK)


def _batch_lora_shrink(site, x, pre, norm_weight, ss_in, n_ss, eps, device_index):
    d = x
    args = (
        x,
        norm_weight if norm_weight is not None else d,
        ss_in if ss_in is not None else d,
        *site.A3,
        site.partials,
        site.H,
        site.counter,
        site.n_seg * site.rank,
        site.K,
        site.rank,
        n_ss,
        site.n_splits,
        ss_in.stride(0) if ss_in is not None else 0,
    )
    constexprs = dict(
        SCALING = site.scaling,
        EPS = eps,
        RANK_PAD = site.rank_pad,
        BLOCK_R = site.block_r,
        SPLIT_K = site.split_k,
        PRE = pre,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        USE_MUL_RN = _HAS_MUL_RN,
    )
    _launch(
        5,
        _batch_shrink_kernel,
        (site.n_splits, x.shape[0]),
        args,
        9,
        constexprs,
        device_index,
        num_warps = 4,
    )


def batch_fused_site(
    site,
    x,
    out = None,
    *,
    epi,
    pre = PRE_NONE,
    capability,
    norm_weight = None,
    eps = 1e-06,
    ss_in = None,
    n_ss = 0,
    residual = None,
    ss_out = None,
    xg_out = None,
    next_norm_weight = None,
    k_cache = None,
    v_cache = None,
    cache_strides = (0, 0),
    cos = None,
    sin = None,
    pos = None,
    slot = None,
    head_dim = None,
    config = None,
):
    K, N = (site.K, site.N)
    if x.dim() != 2 or x.shape[1] != K or x.stride() != (K, 1):
        x = x.reshape(-1, K).contiguous()
    block_p, block_k, num_warps, lut = config or site_config(
        site.kind, N, K, site.blocksize, capability
    )
    if epi in (EPI_PLAIN, EPI_RESNORM):
        half = block_p
    elif epi == EPI_ROPE_KV:
        half = head_dim // 2
    else:
        half = site.bounds[0]
    nprog = site.n_programs(block_p, epi)
    device_index = x.device.index
    if site.lora:
        _batch_lora_shrink(site, x, pre, norm_weight, ss_in, n_ss, eps, device_index)
    seg1, seg2 = site.seg_bounds()
    d = x
    B3 = site.B3 if site.lora else (d, d, d)
    tensors = (
        x,
        norm_weight if norm_weight is not None else d,
        residual if residual is not None else d,
        xg_out if xg_out is not None else d,
        ss_in if ss_in is not None else d,
        ss_out if ss_out is not None else d,
        next_norm_weight if next_norm_weight is not None else d,
        site.packed,
        site.absmax,
        site.code2,
        site.absmax2,
        site.seg_offset,
        site.code,
        B3[0],
        B3[1],
        B3[2],
        site.H if site.lora else d,
        out if out is not None else d,
        k_cache if k_cache is not None else d,
        v_cache if v_cache is not None else d,
        cos if cos is not None else d,
        sin if sin is not None else d,
        pos if pos is not None else d,
        slot if slot is not None else d,
        site.W3[1],
        site.W3[2],
        site.scales3[1],
        site.scales3[2],
        site.state2scales3[1],
        site.state2scales3[2],
    )
    ints = (
        site.rank if site.lora else 0,
        N,
        K,
        half,
        seg1,
        seg2,
        n_ss,
        cache_strides[0],
        cache_strides[1],
        cos.stride(0) if cos is not None else 0,
        head_dim or 1,
        (ss_in if ss_in is not None else ss_out).stride(0)
        if ss_in is not None or ss_out is not None
        else 0,
        (out if out is not None else x).stride(0),
        (residual if residual is not None else x).stride(0),
        (xg_out if xg_out is not None else x).stride(0),
        k_cache.stride(1) if k_cache is not None else 0,
    )
    constexprs = dict(
        EPS = eps,
        BLOCK_P = block_p,
        BLOCK_K = block_k,
        BLOCKSIZE = site.blocksize,
        BLOCKSIZE2 = site.blocksize2 if site.nested else 1,
        NESTED = site.nested,
        USE_MUL_RN = _HAS_MUL_RN,
        EVEN_K = K % block_k == 0,
        LUT_MODE = lut,
        PRE = pre,
        LORA = site.lora,
        RANK_PAD = site.rank_pad if site.lora else 16,
        EPI = epi,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        SEPARATE = site.separate,
    )
    _launch(
        4,
        _batch_main_kernel,
        (nprog, x.shape[0]),
        tensors + ints,
        len(tensors),
        constexprs,
        device_index,
        num_warps = num_warps,
        enable_fp_fusion = True,
    )
    return nprog


def batch_final_norm(residual, ss, n_ss, weight, eps, out):
    K = residual.shape[-1]
    args = (residual, ss, weight, out, n_ss, K, ss.stride(0))
    constexprs = dict(
        EPS = eps,
        SS_BLOCK = max(16, triton.next_power_of_2(max(n_ss, 1))),
        BLOCK = triton.next_power_of_2(K),
    )
    _launch(
        6,
        _batch_final_kernel,
        (1, residual.shape[0]),
        args,
        4,
        constexprs,
        residual.device.index,
        num_warps = 8,
    )
    return out


class BatchSite(FusedSite):
    def __init__(self, original, batch):
        if batch not in (2, 4):
            raise ValueError("batch must be 2 or 4")
        self.__dict__.update(vars(original))
        self.batch_size = batch
        if self.lora:
            with torch.inference_mode(False), torch.no_grad():
                self.H = torch.zeros(
                    batch * 3 * self.rank_pad,
                    device = self.packed.device,
                    dtype = torch.float32,
                )
                self.partials = torch.zeros(
                    batch * self.n_splits * (self.block_r + 1),
                    device = self.packed.device,
                    dtype = torch.float32,
                )
                self.counter = torch.zeros(batch, device = self.packed.device, dtype = torch.int32)


tag_compile_cache(__file__)
