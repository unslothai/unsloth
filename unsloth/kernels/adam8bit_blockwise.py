# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Triton Adam / AdamW steps for Unsloth's bitsandbytes stand-in (unsloth/bnb_shim), bit-identical
to bitsandbytes' CUDA kernels on NVIDIA: blockwise 8-bit state (kOptimizerStatic8bit2StateBlockwise)
and 32-bit state (kOptimizer32bit2State), float32 / bfloat16 / float16 weights.

bitsandbytes builds with --use_fast_math, so its kernels compute with flush-to-zero float32 ops,
approximate lg2 / ex2 (powf), sqrt and division (rcp * x), and nvcc contracts some a * b + c into
fma differently per kernel and weight dtype. These kernels issue the same PTX through inline asm
in the same order (fp fusion cannot change them), so weights, codes and absmax match bit for bit.
ROCm and TRITON_INTERPRET=1 run the same algorithm with IEEE ops.

Per step the update for one weight element is (bitsandbytes' order, Adam and AdamW alike):
    s1 = beta1 * s1 + (1 - beta1) * g
    s2 = beta2 * s2 + (1 - beta2) * g * g
    p  = p + (-lr * sqrt(1 - beta2**t) / (1 - beta1**t)) * s1 / (sqrt(s2) + sqrt(1 - beta2**t) * eps)
    p  = p * (1 - lr * weight_decay)                     (only when weight_decay > 0)
with p rounded to the weight dtype after each of the two lines. 8-bit state dequantizes s1 / s2 as
code[q] * absmax[block] (signed / unsigned dynamic map, 256 values per block), skips elements with
a non-finite gradient, and requantizes against the new per-block absmax by bitsandbytes' binary
search, keeping the first moment's sign.

The step count and the learning rate are scalar arguments (as in bitsandbytes), or, for CUDA graph
capture, read from device tensors (CAPTURABLE). Imported lazily on the first step on a GPU."""

import os

import torch
import triton
import triton.language as tl

from .triton_launch import tag_compile_cache

__all__ = [
    "BLOCKSIZE",
    "adam_8bit_blockwise_step",
    "adam_32bit_step",
    "compiled_safe_8bit_step",
    "compiled_safe_32bit_step",
    "MultiTensorPlan",
    "MAX_GROUPS",
]

BLOCKSIZE = 256  # bitsandbytes' blockwise optimizer block (Optimizer2State.init_state)
_EXACT = torch.version.hip is None and os.environ.get("TRITON_INTERPRET", "0") != "1"
_ROWS = 4  # 256-element blocks per program
_ELEMS_32 = 1024  # elements per program, 32-bit state


@triton.jit
def _mul(a, b, EXACT: tl.constexpr):
    if EXACT:
        return tl.inline_asm_elementwise(
            "mul.rn.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )
    else:
        return a * b


@triton.jit
def _add(a, b, EXACT: tl.constexpr):
    if EXACT:
        return tl.inline_asm_elementwise(
            "add.rn.ftz.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )
    else:
        return a + b


@triton.jit
def _fma(a, b, c, EXACT: tl.constexpr):
    if EXACT:
        return tl.inline_asm_elementwise(
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=r,r,r,r",
            [a, b, c],
            dtype = tl.float32,
            is_pure = True,
            pack = 1,
        )
    else:
        return a * b + c


@triton.jit
def _div(a, b, EXACT: tl.constexpr):
    # __fdividef and fast-math `/`: rcp.approx(b) * a
    if EXACT:
        return tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=r,r,r",
            [a, b],
            dtype = tl.float32,
            is_pure = True,
            pack = 1,
        )
    else:
        return a / b


@triton.jit
def _sqrt(x, EXACT: tl.constexpr):
    if EXACT:
        return tl.inline_asm_elementwise(
            "sqrt.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype = tl.float32, is_pure = True, pack = 1
        )
    else:
        return tl.sqrt(x)


@triton.jit
def _pow(x, y, EXACT: tl.constexpr):
    # __powf / fast-math powf: ex2.approx(y * lg2.approx(x))
    if EXACT:
        lg = tl.inline_asm_elementwise(
            "lg2.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype = tl.float32, is_pure = True, pack = 1
        )
        return tl.inline_asm_elementwise(
            "ex2.approx.ftz.f32 $0, $1;",
            "=r,r",
            [_mul(y, lg, EXACT)],
            dtype = tl.float32,
            is_pure = True,
            pack = 1,
        )
    else:
        return tl.exp2(y * tl.log2(x))


@triton.jit
def _scalars(
    STEP, LR, BETA1, BETA2, WEIGHT_DECAY, shape, CAPTURABLE: tl.constexpr, EXACT: tl.constexpr
):
    # The per-step constants as bitsandbytes' kernels form them, broadcast to `shape`.
    if CAPTURABLE:
        step = tl.load(STEP).to(tl.float32)
        lr = tl.load(LR).to(tl.float32)
    else:
        step = STEP.to(tl.float32)
        lr = LR
    one = tl.full(shape, 1.0, tl.float32)
    lr = lr + tl.zeros(shape, tl.float32)
    beta1 = BETA1 + tl.zeros(shape, tl.float32)
    beta2 = BETA2 + tl.zeros(shape, tl.float32)
    step = step + tl.zeros(shape, tl.float32)
    correction1 = _add(one, -_pow(beta1, step, EXACT), EXACT)
    correction2 = _sqrt(_add(one, -_pow(beta2, step, EXACT), EXACT), EXACT)
    step_size = _div(_mul(correction2, -lr, EXACT), correction1, EXACT)
    decay = _fma(-lr, WEIGHT_DECAY + tl.zeros(shape, tl.float32), one, EXACT)
    return beta1, beta2, correction2, step_size, decay


@triton.jit
def _quantize(Q, x, SIGNED: tl.constexpr, EXACT: tl.constexpr):
    # bitsandbytes' quantize_2D: binary search over the sorted 256-entry code, then the nearer
    # side by midpoint (its quadrant shortcut reads the same entries as code[pivot]).
    pivot = tl.full(x.shape, 127, tl.int32)
    upper_pivot = tl.full(x.shape, 255, tl.int32)
    lower_pivot = tl.zeros(x.shape, tl.int32)
    lower = tl.full(x.shape, -1.0 if SIGNED else 0.0, tl.float32)
    upper = tl.full(x.shape, 1.0, tl.float32)
    val = tl.load(Q + pivot)
    for k in tl.static_range(7):
        up = x > val
        lower_pivot = tl.where(up, pivot, lower_pivot)
        lower = tl.where(up, val, lower)
        upper_pivot = tl.where(up, upper_pivot, pivot)
        upper = tl.where(up, upper, val)
        pivot = tl.where(up, pivot + (64 >> k), pivot - (64 >> k))
        val = tl.load(Q + pivot)
    half = tl.full(x.shape, 0.5, tl.float32)
    up = x > val
    return tl.where(
        up,
        tl.where(x > _mul(_add(upper, val, EXACT), half, EXACT), upper_pivot, pivot),
        tl.where(x < _mul(_add(lower, val, EXACT), half, EXACT), lower_pivot, pivot),
    )


# n and the step only bound masks / feed arithmetic: one compiled variant per (weight dtype, mode).
@triton.jit(do_not_specialize = ["N", "N_BLOCKS", "STEP"])
def _adam8bit_blockwise_kernel(
    P,
    G,
    S1,
    S2,
    Q1,
    Q2,
    A1,
    A2,
    N,
    N_BLOCKS,
    STEP,  # int, or an int32 / float32 device scalar when CAPTURABLE
    LR,  # float, or a float32 device scalar when CAPTURABLE
    BETA1,
    BETA2,
    EPS,
    WEIGHT_DECAY,
    GNORM_SCALE,
    HAS_DECAY: tl.constexpr,  # weight_decay > 0: bitsandbytes skips the decay multiply at 0
    S2_PRODUCT_FMA: tl.constexpr,  # where nvcc contracted s2's update for this weight dtype
    CAPTURABLE: tl.constexpr,  # step / lr from device memory (CUDA graphs)
    ROWS: tl.constexpr,  # 256-element blocks per program
    BLOCK: tl.constexpr,  # 256: one absmax per block
    EXACT: tl.constexpr,  # PTX fast-math ops (NVIDIA) vs IEEE (ROCm, interpreter): per process
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    offs = rows[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    valid = offs < N
    row_valid = rows < N_BLOCKS
    beta1, beta2, correction2, step_size, decay = _scalars(
        STEP, LR, BETA1, BETA2, WEIGHT_DECAY, (ROWS, 1), CAPTURABLE, EXACT
    )
    one = tl.full((ROWS, 1), 1.0, tl.float32)
    # Padding as bitsandbytes' block loads: gradient 0, codes 128 / 0, so the tail absmax agrees.
    g = tl.load(G + offs, valid, 0.0).to(tl.float32)
    c1 = tl.load(S1 + offs, valid, 128).to(tl.int32)
    c2 = tl.load(S2 + offs, valid, 0).to(tl.int32)
    finite = (g == g) & (tl.abs(g) != float("inf"))
    g = _mul(g, GNORM_SCALE + tl.zeros(g.shape, tl.float32), EXACT)
    a1 = tl.load(A1 + rows, row_valid, 0.0)[:, None]
    a2 = tl.load(A2 + rows, row_valid, 0.0)[:, None]
    s2 = _mul(tl.load(Q2 + c2), a2, EXACT)
    w2g = _mul(_add(one, -beta2, EXACT), g, EXACT)
    if S2_PRODUCT_FMA:
        s2 = _fma(w2g, g, _mul(s2, beta2, EXACT), EXACT)
    else:
        s2 = _fma(s2, beta2, _mul(g, w2g, EXACT), EXACT)
    s1 = _mul(tl.load(Q1 + c1), a1, EXACT)
    s1 = _fma(_add(one, -beta1, EXACT), g, _mul(s1, beta1, EXACT), EXACT)
    s1 = tl.where(finite, s1, 0.0)
    s2 = tl.where(finite, s2, 0.0)
    new1 = tl.max(tl.abs(s1), axis = 1)
    new2 = tl.max(tl.abs(s2), axis = 1)
    tl.store(A1 + rows, new1, row_valid)
    tl.store(A2 + rows, new2, row_valid)
    p_in = tl.load(P + offs, valid, 0.0)
    p = p_in.to(tl.float32)
    denom = _fma(correction2, EPS + tl.zeros((ROWS, 1), tl.float32), _sqrt(s2, EXACT), EXACT)
    p = _fma(step_size, _div(s1, denom, EXACT), p, EXACT).to(p_in.dtype)
    if HAS_DECAY:
        p = _mul(decay, p.to(tl.float32), EXACT).to(p_in.dtype)
    tl.store(P + offs, tl.where(finite, p, p_in), valid)
    q1 = _quantize(Q1, _div(s1, new1[:, None], EXACT), True, EXACT)
    # Keep the first moment's sign (bitsandbytes nudges the code one step towards it).
    code_neg = tl.load(Q1 + q1).to(tl.int32, bitcast = True) < 0
    s1_neg = s1.to(tl.int32, bitcast = True) < 0
    q1 = tl.where(code_neg != s1_neg, tl.where(s1 > 0.0, q1 + 1, q1 - 1), q1)
    q2 = _quantize(Q2, _div(s2, new2[:, None], EXACT), False, EXACT)
    tl.store(S1 + offs, (q1 & 255).to(tl.uint8), valid)
    tl.store(S2 + offs, q2.to(tl.uint8), valid)


@triton.jit(do_not_specialize = ["N", "STEP"])
def _adam32bit_kernel(
    P,
    G,
    S1,
    S2,
    N,
    STEP,
    LR,
    BETA1,
    BETA2,
    EPS,
    WEIGHT_DECAY,
    GNORM_SCALE,
    HAS_DECAY: tl.constexpr,
    CAPTURABLE: tl.constexpr,
    BLOCK: tl.constexpr,  # elements per program
    EXACT: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N
    beta1, beta2, correction2, step_size, decay = _scalars(
        STEP, LR, BETA1, BETA2, WEIGHT_DECAY, (BLOCK,), CAPTURABLE, EXACT
    )
    one = tl.full((BLOCK,), 1.0, tl.float32)
    g = tl.load(G + offs, valid, 0.0).to(tl.float32)
    g = _mul(g, GNORM_SCALE + tl.zeros(g.shape, tl.float32), EXACT)
    s1 = tl.load(S1 + offs, valid, 0.0)
    s2 = tl.load(S2 + offs, valid, 0.0)
    s1 = _fma(s1, beta1, _mul(_add(one, -beta1, EXACT), g, EXACT), EXACT)
    s2 = _fma(s2, beta2, _mul(_add(one, -beta2, EXACT), _mul(g, g, EXACT), EXACT), EXACT)
    p_in = tl.load(P + offs, valid, 0.0)
    p = p_in.to(tl.float32)
    denom = _fma(correction2, EPS + tl.zeros((BLOCK,), tl.float32), _sqrt(s2, EXACT), EXACT)
    p = _fma(step_size, _div(s1, denom, EXACT), p, EXACT).to(p_in.dtype)
    if HAS_DECAY:
        p = _mul(decay, p.to(tl.float32), EXACT).to(p_in.dtype)
    tl.store(P + offs, p, valid)
    tl.store(S1 + offs, s1, valid)
    tl.store(S2 + offs, s2, valid)


@triton.jit
def _pick(g, V0, V1, V2, V3):
    # a per-group scalar (at most 4 param groups per multi-tensor launch)
    return tl.where(g == 0, V0, tl.where(g == 1, V1, tl.where(g == 2, V2, V3)))


@triton.jit
def _tensor_args(t, GROUPS, STEPS, LR0, LR1, LR2, LR3, B10, B11, B12, B13, B20, B21, B22, B23,
                 EPS0, EPS1, EPS2, EPS3, WD0, WD1, WD2, WD3):  # fmt: skip
    g = tl.load(GROUPS + t)
    step = tl.load(STEPS + t)
    return (
        g,
        step,
        _pick(g, LR0, LR1, LR2, LR3),
        _pick(g, B10, B11, B12, B13),
        _pick(g, B20, B21, B22, B23),
        _pick(g, EPS0, EPS1, EPS2, EPS3),
        _pick(g, WD0, WD1, WD2, WD3),
    )


# Multi-tensor launches: one program per chunk of a weight (ROWS blocks of 256 for 8-bit state, as the
# per-tensor kernel, which keeps large weights as fast; 1024 elements for 32-bit), driven by
# device tables of data pointers / sizes / (tensor, block) per program. Same op sequence per element
# as the per-tensor kernels above, so the bits are bitsandbytes' too.
@triton.jit(do_not_specialize = ["T"])
def _adam8bit_multi_kernel(
    PTRS,  # int64 [6, T]: weight, gradient, state1, state2, absmax1, absmax2 data pointers
    NUMELS,  # int64 [T]
    GROUPS,  # int32 [T]: param group of each weight (0..3)
    STEPS,  # int32 [T]: step count, already advanced for this step
    BLOCK_T,  # int32 [programs]: weight of each program
    BLOCK_J,  # int32 [programs]: chunk (ROWS blocks of 256) of that weight
    T,
    Q1,
    Q2,
    LR0, LR1, LR2, LR3, B10, B11, B12, B13, B20, B21, B22, B23,
    EPS0, EPS1, EPS2, EPS3, WD0, WD1, WD2, WD3,
    GNORM_SCALE,
    DTYPE: tl.constexpr,  # weight / gradient dtype of the whole launch
    S2_PRODUCT_FMA: tl.constexpr,
    ALIGNED: tl.constexpr,  # every pointer 16 byte aligned
    ROWS: tl.constexpr,  # 256-element blocks per program, as the per-tensor kernel
    BLOCK: tl.constexpr,  # 256
    EXACT: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    t = tl.load(BLOCK_T + b)
    j = tl.load(BLOCK_J + b).to(tl.int64)
    n = tl.load(NUMELS + t)
    grp, step, lr, beta1, beta2, eps, wd = _tensor_args(
        t, GROUPS, STEPS, LR0, LR1, LR2, LR3, B10, B11, B12, B13, B20, B21, B22, B23,
        EPS0, EPS1, EPS2, EPS3, WD0, WD1, WD2, WD3,
    )  # fmt: skip
    P = tl.load(PTRS + t).to(tl.pointer_type(DTYPE))
    G = tl.load(PTRS + T + t).to(tl.pointer_type(DTYPE))
    S1 = tl.load(PTRS + 2 * T + t).to(tl.pointer_type(tl.uint8))
    S2 = tl.load(PTRS + 3 * T + t).to(tl.pointer_type(tl.uint8))
    A1 = tl.load(PTRS + 4 * T + t).to(tl.pointer_type(tl.float32))
    A2 = tl.load(PTRS + 5 * T + t).to(tl.pointer_type(tl.float32))
    if ALIGNED:
        P = tl.multiple_of(P, 16)
        G = tl.multiple_of(G, 16)
        S1 = tl.multiple_of(S1, 16)
        S2 = tl.multiple_of(S2, 16)
    n_blocks = (n + BLOCK - 1) // BLOCK
    rows = j * ROWS + tl.arange(0, ROWS)
    offs = rows[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    valid = offs < n
    row_valid = rows < n_blocks
    beta1, beta2, correction2, step_size, decay = _scalars(
        step, lr, beta1, beta2, wd, (ROWS, 1), False, EXACT
    )
    one = tl.full((ROWS, 1), 1.0, tl.float32)
    g = tl.load(G + offs, valid, 0.0).to(tl.float32)
    c1 = tl.load(S1 + offs, valid, 128).to(tl.int32)
    c2 = tl.load(S2 + offs, valid, 0).to(tl.int32)
    finite = (g == g) & (tl.abs(g) != float("inf"))
    g = _mul(g, GNORM_SCALE + tl.zeros(g.shape, tl.float32), EXACT)
    a1 = tl.load(A1 + rows, row_valid, 0.0)[:, None]
    a2 = tl.load(A2 + rows, row_valid, 0.0)[:, None]
    s2 = _mul(tl.load(Q2 + c2), a2, EXACT)
    w2g = _mul(_add(one, -beta2, EXACT), g, EXACT)
    if S2_PRODUCT_FMA:
        s2 = _fma(w2g, g, _mul(s2, beta2, EXACT), EXACT)
    else:
        s2 = _fma(s2, beta2, _mul(g, w2g, EXACT), EXACT)
    s1 = _mul(tl.load(Q1 + c1), a1, EXACT)
    s1 = _fma(_add(one, -beta1, EXACT), g, _mul(s1, beta1, EXACT), EXACT)
    s1 = tl.where(finite, s1, 0.0)
    s2 = tl.where(finite, s2, 0.0)
    new1 = tl.max(tl.abs(s1), axis = 1)
    new2 = tl.max(tl.abs(s2), axis = 1)
    tl.store(A1 + rows, new1, row_valid)
    tl.store(A2 + rows, new2, row_valid)
    p_in = tl.load(P + offs, valid, 0.0)
    p = p_in.to(tl.float32)
    denom = _fma(correction2, eps + tl.zeros((ROWS, 1), tl.float32), _sqrt(s2, EXACT), EXACT)
    p = _fma(step_size, _div(s1, denom, EXACT), p, EXACT).to(DTYPE)
    if wd > 0.0:
        p = _mul(decay, p.to(tl.float32), EXACT).to(DTYPE)
    tl.store(P + offs, tl.where(finite, p, p_in), valid)
    q1 = _quantize(Q1, _div(s1, new1[:, None], EXACT), True, EXACT)
    code_neg = tl.load(Q1 + q1).to(tl.int32, bitcast = True) < 0
    s1_neg = s1.to(tl.int32, bitcast = True) < 0
    q1 = tl.where(code_neg != s1_neg, tl.where(s1 > 0.0, q1 + 1, q1 - 1), q1)
    q2 = _quantize(Q2, _div(s2, new2[:, None], EXACT), False, EXACT)
    tl.store(S1 + offs, (q1 & 255).to(tl.uint8), valid)
    tl.store(S2 + offs, q2.to(tl.uint8), valid)


@triton.jit(do_not_specialize = ["T"])
def _adam32bit_multi_kernel(
    PTRS,  # int64 [4, T]: weight, gradient, state1, state2
    NUMELS,
    GROUPS,
    STEPS,
    BLOCK_T,
    BLOCK_J,
    T,
    LR0, LR1, LR2, LR3, B10, B11, B12, B13, B20, B21, B22, B23,
    EPS0, EPS1, EPS2, EPS3, WD0, WD1, WD2, WD3,
    GNORM_SCALE,
    DTYPE: tl.constexpr,
    ALIGNED: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    t = tl.load(BLOCK_T + b)
    j = tl.load(BLOCK_J + b).to(tl.int64)
    n = tl.load(NUMELS + t)
    grp, step, lr, beta1, beta2, eps, wd = _tensor_args(
        t, GROUPS, STEPS, LR0, LR1, LR2, LR3, B10, B11, B12, B13, B20, B21, B22, B23,
        EPS0, EPS1, EPS2, EPS3, WD0, WD1, WD2, WD3,
    )  # fmt: skip
    P = tl.load(PTRS + t).to(tl.pointer_type(DTYPE))
    G = tl.load(PTRS + T + t).to(tl.pointer_type(DTYPE))
    S1 = tl.load(PTRS + 2 * T + t).to(tl.pointer_type(tl.float32))
    S2 = tl.load(PTRS + 3 * T + t).to(tl.pointer_type(tl.float32))
    if ALIGNED:
        P = tl.multiple_of(P, 16)
        G = tl.multiple_of(G, 16)
        S1 = tl.multiple_of(S1, 16)
        S2 = tl.multiple_of(S2, 16)
    offs = j * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n
    beta1, beta2, correction2, step_size, decay = _scalars(
        step, lr, beta1, beta2, wd, (1,), False, EXACT
    )
    one = tl.full((1,), 1.0, tl.float32)
    g = tl.load(G + offs, valid, 0.0).to(tl.float32)
    g = _mul(g, GNORM_SCALE + tl.zeros(g.shape, tl.float32), EXACT)
    s1 = tl.load(S1 + offs, valid, 0.0)
    s2 = tl.load(S2 + offs, valid, 0.0)
    s1 = _fma(s1, beta1, _mul(_add(one, -beta1, EXACT), g, EXACT), EXACT)
    s2 = _fma(s2, beta2, _mul(_add(one, -beta2, EXACT), _mul(g, g, EXACT), EXACT), EXACT)
    p_in = tl.load(P + offs, valid, 0.0)
    p = p_in.to(tl.float32)
    denom = _fma(correction2, eps + tl.zeros((1,), tl.float32), _sqrt(s2, EXACT), EXACT)
    p = _fma(step_size, _div(s1, denom, EXACT), p, EXACT).to(DTYPE)
    if wd > 0.0:
        p = _mul(decay, p.to(tl.float32), EXACT).to(DTYPE)
    tl.store(P + offs, p, valid)
    tl.store(S1 + offs, s1, valid)
    tl.store(S2 + offs, s2, valid)


tag_compile_cache(__file__)

# Compiled kernels by specialization: one step launches a kernel per weight (hundreds per step for
# LoRA adapters), so later launches skip JITFunction's argument binding and specialization.
_LAUNCHERS = {}
_FAST_LAUNCH = None  # None: untried, True: works, False: this Triton's CompiledKernel differs


def _spec(arg):
    # what Triton specializes on: tensor dtype + 16 byte alignment; int width (do_not_specialize ints)
    if torch.is_tensor(arg):
        return (arg.dtype, arg.data_ptr() % 16 == 0)
    if isinstance(arg, int):
        # everything Triton may specialize an int on (== 1, % 16, width), do_not_specialize or not
        return (int, -(1 << 31) <= arg < (1 << 31), arg == 1, arg % 16 == 0)
    return (type(arg),)


def _run(kernel, grid, args, **constexprs):
    global _FAST_LAUNCH
    if _FAST_LAUNCH is not False:
        key = (
            kernel,
            torch.cuda.current_device(),
            tuple(constexprs.items()),
            tuple(map(_spec, args)),
        )
        compiled = _LAUNCHERS.get(key)
        if compiled is None:
            compiled = _LAUNCHERS[key] = kernel.warmup(
                *args, grid = grid, **constexprs, enable_fp_fusion = True
            )
        try:
            compiled[tuple(grid) + (1,) * (3 - len(grid))](*args)  # CompiledKernel takes a 3-D grid
            _FAST_LAUNCH = True
            return
        except TypeError:
            if _FAST_LAUNCH:
                raise
            _FAST_LAUNCH = False
    kernel[grid](*args, **constexprs, enable_fp_fusion = True)


# Where nvcc fused s2 = s2 * beta2 + (1 - beta2) * g * g in the 8-bit kernel, per weight dtype:
# float32 folds the g * g product into the fma, half / bfloat16 the s2 * beta2 one.
_S2_PRODUCT_FMA = {torch.float32: True, torch.float16: False, torch.bfloat16: False}


def _check(p, g):
    if p.dtype not in _S2_PRODUCT_FMA or g.dtype != p.dtype:
        raise ValueError(
            f"Unsloth: 8-bit Adam needs float32 / float16 / bfloat16 weights and gradients of the "
            f"same dtype, got weight {p.dtype} and gradient {g.dtype}"
        )
    if not (p.is_contiguous() and g.is_contiguous()) or g.numel() != p.numel():
        raise ValueError("Unsloth: 8-bit Adam needs contiguous weights and gradients of one size")


def adam_8bit_blockwise_step(
    p,
    g,
    state1,
    state2,
    qmap1,
    qmap2,
    absmax1,
    absmax2,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    gnorm_scale = 1.0,
    s2_product_fma = None,
):
    """bitsandbytes optimizer_update_8bit_blockwise("adam", ...) in place. step / lr: Python
    numbers, or device tensors (CUDA graph capture)."""
    _check(p, g)
    n = p.numel()
    if n == 0:
        return
    blocks = -(n // -BLOCKSIZE)
    capturable = torch.is_tensor(step) or torch.is_tensor(lr)
    if capturable:
        step = step if torch.is_tensor(step) else torch.tensor(step, device = p.device)
        lr = lr if torch.is_tensor(lr) else torch.tensor(lr, dtype = torch.float32, device = p.device)
    if s2_product_fma is None:
        s2_product_fma = _S2_PRODUCT_FMA[p.dtype]
    with torch.cuda.device(p.device):
        _run(
            _adam8bit_blockwise_kernel,
            (-(blocks // -_ROWS),),
            (
                p,
                g,
                state1,
                state2,
                qmap1,
                qmap2,
                absmax1,
                absmax2,
                n,
                blocks,
                step,
                lr,
                float(beta1),
                float(beta2),
                float(eps),
                float(weight_decay),
                float(gnorm_scale),
            ),
            HAS_DECAY = float(weight_decay) > 0.0,
            S2_PRODUCT_FMA = bool(s2_product_fma),
            CAPTURABLE = capturable,
            ROWS = _ROWS,
            BLOCK = BLOCKSIZE,
            EXACT = _EXACT,
        )


def adam_32bit_step(
    p,
    g,
    state1,
    state2,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    gnorm_scale = 1.0,
):
    """bitsandbytes optimizer_update_32bit("adam", ...) in place (no max_unorm, no skip_zeros)."""
    _check(p, g)
    n = p.numel()
    if n == 0:
        return
    capturable = torch.is_tensor(step) or torch.is_tensor(lr)
    if capturable:
        step = step if torch.is_tensor(step) else torch.tensor(step, device = p.device)
        lr = lr if torch.is_tensor(lr) else torch.tensor(lr, dtype = torch.float32, device = p.device)
    with torch.cuda.device(p.device):
        _run(
            _adam32bit_kernel,
            (-(n // -_ELEMS_32),),
            (
                p,
                g,
                state1,
                state2,
                n,
                step,
                lr,
                float(beta1),
                float(beta2),
                float(eps),
                float(weight_decay),
                float(gnorm_scale),
            ),
            HAS_DECAY = float(weight_decay) > 0.0,
            CAPTURABLE = capturable,
            BLOCK = _ELEMS_32,
            EXACT = _EXACT,
        )


_MULTI_BLOCK_32 = 1024  # elements per program, 32-bit state multi-tensor launch
MAX_GROUPS = 4  # param groups one multi-tensor launch can serve


class MultiTensorPlan:
    """Device tables for one multi-tensor launch: weights of one dtype and one state kind (8-bit or
    32-bit) on one device. Weight / state pointers are fixed at build; the gradient row is refreshed
    when a gradient moved; the step counts live on the device and advance in place."""

    def __init__(self, params, states, groups, eight):
        device = params[0].device
        self.eight, self.dtype, self.device = eight, params[0].dtype, device
        self.T = len(params)
        block = BLOCKSIZE * _ROWS if eight else _MULTI_BLOCK_32  # elements per program
        self.block = block
        keys = ("state1", "state2", "absmax1", "absmax2") if eight else ("state1", "state2")
        rows = [[p.data_ptr() for p in params], [0] * self.T]
        rows += [[st[k].data_ptr() for st in states] for k in keys]
        self.grad_ptrs = None
        tensors = list(params) + [st[k] for st in states for k in keys]
        self.aligned = all(t.data_ptr() % 16 == 0 for t in tensors)
        with torch.cuda.device(device):
            self.ptrs = torch.tensor(rows, dtype = torch.int64).to(device)
            numels = torch.tensor([p.numel() for p in params], dtype = torch.int64)
            nblocks = -(numels // -block)
            self.blocks = int(nblocks.sum())
            starts = torch.cumsum(nblocks, 0) - nblocks
            self.numels = numels.to(device)
            self.groups = torch.tensor(groups, dtype = torch.int32).to(device)
            self.steps = torch.tensor([int(st["step"]) for st in states], dtype = torch.int32).to(
                device
            )
            ids = torch.arange(self.T, dtype = torch.int64, device = device)
            nb = nblocks.to(device)
            self.block_t = torch.repeat_interleave(ids, nb).to(torch.int32)
            self.block_j = (
                torch.arange(self.blocks, dtype = torch.int64, device = device)
                - torch.repeat_interleave(starts.to(device), nb)
            ).to(torch.int32)
        self.qmap1 = states[0].get("qmap1") if eight else None
        self.qmap2 = states[0].get("qmap2") if eight else None

    def set_grads(self, grads):
        ptrs = [g.data_ptr() for g in grads]
        if ptrs != self.grad_ptrs:
            # pageable host -> device copy, stream ordered; no device sync
            self.ptrs[1].copy_(torch.tensor(ptrs, dtype = torch.int64), non_blocking = True)
            self.grad_ptrs = ptrs
            self.aligned_grads = all(p % 16 == 0 for p in ptrs)

    def launch(
        self,
        group_scalars,
        gnorm_scale = 1.0,
    ):
        """group_scalars: up to MAX_GROUPS (lr, beta1, beta2, eps, weight_decay) tuples."""
        if self.blocks == 0:
            return
        cols = list(zip(*group_scalars))
        pad = lambda xs: [float(x) for x in xs] + [0.0] * (MAX_GROUPS - len(xs))
        scalars = []
        for col in cols:
            scalars += pad(col)
        self.steps.add_(1)
        aligned = self.aligned and self.aligned_grads
        with torch.cuda.device(self.device):
            if self.eight:
                _run(
                    _adam8bit_multi_kernel,
                    (self.blocks,),
                    (self.ptrs, self.numels, self.groups, self.steps, self.block_t, self.block_j,
                     self.T, self.qmap1, self.qmap2, *scalars, float(gnorm_scale)),
                    DTYPE = _TL_DTYPE[self.dtype],
                    S2_PRODUCT_FMA = _S2_PRODUCT_FMA[self.dtype],
                    ALIGNED = aligned,
                    ROWS = _ROWS,
                    BLOCK = BLOCKSIZE,
                    EXACT = _EXACT,
                )  # fmt: skip
            else:
                _run(
                    _adam32bit_multi_kernel,
                    (self.blocks,),
                    (self.ptrs, self.numels, self.groups, self.steps, self.block_t, self.block_j,
                     self.T, *scalars, float(gnorm_scale)),
                    DTYPE = _TL_DTYPE[self.dtype],
                    ALIGNED = aligned,
                    BLOCK = _MULTI_BLOCK_32,
                    EXACT = _EXACT,
                )  # fmt: skip


_TL_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


# Opaque ops for steps inside torch.compile'd code (Dynamo would otherwise trace the launcher).
# A second import of this file under another name reuses the registered ops.
_OPS_NS = "unsloth_bnb_shim"


def _register_ops():
    ops = getattr(torch.ops, _OPS_NS, None)
    if ops is not None and hasattr(ops, "adam8bit_blockwise_step"):
        return
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return

    @custom_op(
        f"{_OPS_NS}::adam8bit_blockwise_step",
        mutates_args = ("p", "state1", "state2", "absmax1", "absmax2"),
        device_types = "cuda",
    )
    def _adam8bit_op(
        p: torch.Tensor,
        g: torch.Tensor,
        state1: torch.Tensor,
        state2: torch.Tensor,
        qmap1: torch.Tensor,
        qmap2: torch.Tensor,
        absmax1: torch.Tensor,
        absmax2: torch.Tensor,
        step: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        adam_8bit_blockwise_step(
            p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, step, lr, beta1, beta2, eps,
            weight_decay,
        )  # fmt: skip

    @_adam8bit_op.register_fake
    def _adam8bit_fake(
        p,
        g,
        state1,
        state2,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
    ):
        return None

    @custom_op(
        f"{_OPS_NS}::adam32bit_step",
        mutates_args = ("p", "state1", "state2"),
        device_types = "cuda",
    )
    def _adam32bit_op(
        p: torch.Tensor,
        g: torch.Tensor,
        state1: torch.Tensor,
        state2: torch.Tensor,
        step: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        adam_32bit_step(p, g, state1, state2, step, lr, beta1, beta2, eps, weight_decay)

    @_adam32bit_op.register_fake
    def _adam32bit_fake(p, g, state1, state2, step, lr, beta1, beta2, eps, weight_decay):
        return None


try:
    _register_ops()
except Exception:  # an older torch without custom_op support: compiled callers graph-break instead
    pass


def compiled_safe_8bit_step(*args):
    """adam_8bit_blockwise_step through the opaque op when traced (scalar step / lr only)."""
    if torch.compiler.is_compiling() and not (torch.is_tensor(args[8]) or torch.is_tensor(args[9])):
        ops = getattr(torch.ops, _OPS_NS, None)
        if ops is not None and hasattr(ops, "adam8bit_blockwise_step"):
            return ops.adam8bit_blockwise_step(*args)
    return adam_8bit_blockwise_step(*args)


def compiled_safe_32bit_step(*args):
    if torch.compiler.is_compiling() and not (torch.is_tensor(args[4]) or torch.is_tensor(args[5])):
        ops = getattr(torch.ops, _OPS_NS, None)
        if ops is not None and hasattr(ops, "adam32bit_step"):
            return ops.adam32bit_step(*args)
    return adam_32bit_step(*args)
