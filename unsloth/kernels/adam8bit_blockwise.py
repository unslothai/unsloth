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


tag_compile_cache(__file__)

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
        _adam8bit_blockwise_kernel[(-(blocks // -_ROWS),)](
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
            HAS_DECAY = float(weight_decay) > 0.0,
            S2_PRODUCT_FMA = bool(s2_product_fma),
            CAPTURABLE = capturable,
            ROWS = _ROWS,
            BLOCK = BLOCKSIZE,
            EXACT = _EXACT,
            enable_fp_fusion = True,
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
        _adam32bit_kernel[(-(n // -_ELEMS_32),)](
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
            HAS_DECAY = float(weight_decay) > 0.0,
            CAPTURABLE = capturable,
            BLOCK = _ELEMS_32,
            EXACT = _EXACT,
            enable_fp_fusion = True,
        )


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
