# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trained bfloat16 weights (full finetuning, or LoRA's modules_to_save / trained norms): step them in float32,
round back stochastically (UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 disables).

bfloat16 keeps 8 mantissa bits, so round-to-nearest drops updates far smaller than a weight's spacing.

On CUDA with Triton, torch AdamW weights take one kernel per weight: read the bfloat16 weight, gradient and
moments, do AdamW's float32 arithmetic in registers, write all three back with stochastic rounding (Philox
seeded by (param index, step), so resumes and DDP replicas draw alike). bitsandbytes Adam / AdamW takes one
kernel too; other bitsandbytes optimizers step a float32 copy, rounded back in one pass.
UNSLOTH_BF16_SR_FUSED_BNB=0 keeps bitsandbytes on the float32-copy route; UNSLOTH_BF16_SR_FUSED=0 keeps the float32-copy route with torch.randint draws everywhere.
"""

import contextlib
import functools
import os
import types
import weakref

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

__all__ = [
    "enable_bf16_stochastic_rounding",
    "stochastic_round_to_bf16_",
    "bf16_stochastic_rounding_enabled",
]

_MARK = "_unsloth_bf16_stochastic_rounding"
_CHUNK = 1 << 22


def bf16_stochastic_rounding_enabled():
    value = os.environ.get("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _fused_enabled():
    """UNSLOTH_BF16_SR_FUSED=0 keeps the float32-copy path (torch.randint draws) on CUDA too."""
    value = os.environ.get("UNSLOTH_BF16_SR_FUSED", "1").strip().lower()
    return triton is not None and value not in ("0", "false", "no", "off")


if triton is not None:

    @triton.jit
    def _round_bf16(x, random):
        # Up with probability low / 2**16: E[result] = x. Same law as (bits + U[0, 2**16)) & -65536.
        bits = x.to(tl.uint32, bitcast = True)
        low = bits & 65535
        upper = bits & 4294901760
        special = (bits & 2139095040) == 2139095040
        rounded = tl.where((random & 65535) < low, upper + 65536, upper)
        return tl.where(special, bits, rounded).to(tl.float32, bitcast = True).to(tl.bfloat16)

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _round_kernel(
        DST,
        SRC,
        N,
        SEED,
        BLOCK: tl.constexpr,  # elements per program, fixed at _BLOCK
    ):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        x = tl.load(SRC + i, valid, 0).to(tl.float32)
        # Lane 0 of Philox(SEED, element index): what the fused AdamW kernel uses for the weight.
        r0, _, _, _ = tl.randint4x(SEED, i)
        tl.store(DST + i, _round_bf16(x, r0), valid)

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _adamw_kernel(
        P,
        G,
        M,
        V,
        N,
        SEED,
        DECAY,  # 1 - lr * weight_decay
        W1,  # 1 - beta1
        B2,
        W2,  # 1 - beta2
        NEG_STEP,  # -lr / (1 - beta1**t)
        BC2_SQRT,  # sqrt(1 - beta2**t)
        EPS,
        BLOCK: tl.constexpr,  # elements per program, fixed at _BLOCK
    ):
        # torch.optim.AdamW's foreach float32 arithmetic, op for op, then one rounding per output.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        p = tl.load(P + i, valid, 0).to(tl.float32)
        g = tl.load(G + i, valid, 0).to(tl.float32)
        m = tl.load(M + i, valid, 0).to(tl.float32)
        v = tl.load(V + i, valid, 0).to(tl.float32)
        p = p * DECAY
        # torch.lerp: from the far end once the weight reaches 0.5, so beta1 = 0 gives exactly g
        m = tl.where(W1 < 0.5, m + W1 * (g - m), g - (g - m) * (1.0 - W1))
        v = v * B2
        v = v + W2 * g * g
        denom = tl.div_rn(tl.sqrt_rn(v), BC2_SQRT) + EPS
        p = p + NEG_STEP * tl.div_rn(m, denom)
        r0, r1, r2, _ = tl.randint4x(SEED, i)
        tl.store(P + i, _round_bf16(p, r0), valid)
        tl.store(M + i, _round_bf16(m, r1), valid)
        tl.store(V + i, _round_bf16(v, r2), valid)

    @triton.jit
    def _adamw_multi_kernel(
        META,  # int64 [T, 7]: weight, grad, exp_avg, exp_avg_sq addresses, numel, Philox seed, then 2 float32:
        SCALE,  # META as float32 [T, 14]: NEG_STEP, BC2_SQRT per weight at columns 12, 13 (each its own step)
        CHUNKS,  # int32 [C, 2]: (weight row, chunk index) per program
        DECAY,
        W1,
        B2,
        W2,
        EPS,
        G_DTYPE: tl.constexpr,  # the bucket's gradient dtype
        BLOCK: tl.constexpr,  # _BLOCK, so element indices (and Philox offsets) match _adamw_kernel
        SUB: tl.constexpr,  # blocks per chunk
    ):
        # _adamw_kernel over many weights in one launch, through an address table (never a base pointer plus an
        # offset, which ROCm buffer ops cannot follow into another allocation). Same element index, seed and
        # float32 ops per weight, so the bits are the same as one launch per weight.
        c = tl.program_id(0).to(tl.int64)
        t = tl.load(CHUNKS + 2 * c).to(tl.int64)
        k = tl.load(CHUNKS + 2 * c + 1).to(tl.int64)
        row = META + 7 * t
        P = tl.load(row).to(tl.pointer_type(tl.bfloat16))
        G = tl.load(row + 1).to(tl.pointer_type(G_DTYPE))
        M = tl.load(row + 2).to(tl.pointer_type(tl.bfloat16))
        V = tl.load(row + 3).to(tl.pointer_type(tl.bfloat16))
        N = tl.load(row + 4)
        SEED = tl.load(row + 5)
        NEG_STEP = tl.load(SCALE + 14 * t + 12)
        BC2_SQRT = tl.load(SCALE + 14 * t + 13)
        for s in range(SUB):
            i = (k * SUB + s) * BLOCK + tl.arange(0, BLOCK)
            valid = i < N
            p = tl.load(P + i, valid, 0).to(tl.float32)
            g = tl.load(G + i, valid, 0).to(tl.float32)
            m = tl.load(M + i, valid, 0).to(tl.float32)
            v = tl.load(V + i, valid, 0).to(tl.float32)
            p = p * DECAY
            m = tl.where(W1 < 0.5, m + W1 * (g - m), g - (g - m) * (1.0 - W1))
            v = v * B2
            v = v + W2 * g * g
            denom = tl.div_rn(tl.sqrt_rn(v), BC2_SQRT) + EPS
            p = p + NEG_STEP * tl.div_rn(m, denom)
            r0, r1, r2, _ = tl.randint4x(SEED, i)
            tl.store(P + i, _round_bf16(p, r0), valid)
            tl.store(M + i, _round_bf16(m, r1), valid)
            tl.store(V + i, _round_bf16(v, r2), valid)

    @triton.jit
    def _bnb_quantize(
        Q,
        x,
        SIGNED: tl.constexpr,  # signed code (first moment) starts the search from -1, unsigned from 0
    ):
        # bitsandbytes' quantize_2D: binary search, then the nearer side by midpoint.
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
        up = x > val
        code = tl.where(
            up,
            tl.where(x > (upper + val) * 0.5, upper_pivot, pivot),
            tl.where(x < (lower + val) * 0.5, lower_pivot, pivot),
        )
        return code

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _bnb_adam8bit_kernel(
        P,
        G,
        S1,
        S2,
        Q1,
        Q2,
        A1,
        A2,
        N,
        SEED,
        BETA1,
        BETA2,
        EPS_C2,  # correction2 * eps
        STEP_SIZE,  # -lr * correction2 / correction1
        DECAY,  # 1 - lr * weight_decay
        HAS_DECAY: tl.constexpr,  # weight_decay > 0: bitsandbytes skips the decay multiply at 0
        BLOCK: tl.constexpr,  # 256: one absmax per program
    ):
        # bitsandbytes' kOptimizerStatic8bit2StateBlockwise on the bfloat16 weight, rounded stochastically.
        b = tl.program_id(0).to(tl.int64)
        i = b * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        # Padding matches bitsandbytes' block loads (gradient 0, codes 128 / 0) so absmax agrees.
        g = tl.load(G + i, valid, 0).to(tl.float32)
        c1 = tl.load(S1 + i, valid, 128).to(tl.int32)
        c2 = tl.load(S2 + i, valid, 0).to(tl.int32)
        finite = (g == g) & (tl.abs(g) != float("inf"))
        s2 = tl.load(Q2 + c2) * tl.load(A2 + b)
        s2 = (s2 * BETA2) + ((1.0 - BETA2) * g * g)
        s1 = tl.load(Q1 + c1) * tl.load(A1 + b)
        s1 = (s1 * BETA1) + ((1.0 - BETA1) * g)
        s1 = tl.where(finite, s1, 0.0)
        s2 = tl.where(finite, s2, 0.0)
        new1 = tl.max(tl.abs(s1), axis = 0)
        new2 = tl.max(tl.abs(s2), axis = 0)
        tl.store(A1 + b, new1)
        tl.store(A2 + b, new2)
        p = tl.load(P + i, valid, 0).to(tl.float32)
        stepped = p + STEP_SIZE * (s1 / (tl.sqrt(s2) + EPS_C2))
        if HAS_DECAY:
            stepped = stepped * DECAY
        p = tl.where(finite, stepped, p)
        # Same draws as the float32-copy route's round kernel.
        r0, _, _, _ = tl.randint4x(SEED, i)
        tl.store(P + i, _round_bf16(p, r0), valid)
        q1 = _bnb_quantize(Q1, s1 / new1, True)
        code_neg = tl.load(Q1 + q1).to(tl.int32, bitcast = True) < 0
        s1_neg = s1.to(tl.int32, bitcast = True) < 0
        q1 = tl.where(code_neg != s1_neg, tl.where(s1 > 0.0, q1 + 1, q1 - 1), q1)
        q2 = _bnb_quantize(Q2, s2 / new2, False)
        tl.store(S1 + i, q1.to(tl.uint8), valid)
        tl.store(S2 + i, q2.to(tl.uint8), valid)

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _bnb_adam32bit_kernel(
        P,
        G,
        S1,
        S2,
        N,
        SEED,
        BETA1,
        BETA2,
        EPS_C2,  # correction2 * eps
        STEP_SIZE,  # -lr * correction2 / correction1
        DECAY,  # 1 - lr * weight_decay
        HAS_DECAY: tl.constexpr,  # weight_decay > 0: bitsandbytes skips the decay multiply at 0
        BLOCK: tl.constexpr,  # elements per program, fixed at _BLOCK
    ):
        # bitsandbytes' kOptimizer32bit2State on the bfloat16 weight, rounded stochastically.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        g = tl.load(G + i, valid, 0).to(tl.float32)
        s1 = tl.load(S1 + i, valid, 0)
        s2 = tl.load(S2 + i, valid, 0)
        s1 = s1 * BETA1 + ((1.0 - BETA1) * g)
        s2 = s2 * BETA2 + ((1.0 - BETA2) * (g * g))
        p = tl.load(P + i, valid, 0).to(tl.float32)
        p = p + STEP_SIZE * (s1 / (tl.sqrt(s2) + EPS_C2))
        if HAS_DECAY:
            p = p * DECAY
        r0, _, _, _ = tl.randint4x(SEED, i)
        tl.store(P + i, _round_bf16(p, r0), valid)
        tl.store(S1 + i, s1, valid)
        tl.store(S2 + i, s2, valid)


_BLOCK = 1024
_BNB_BLOCK = 256


def _grid(n):
    return ((n + _BLOCK - 1) // _BLOCK,)


_TRITON_OK = {}


def _triton_ok(device):
    """Compile and run both kernels once per device; any failure keeps the float32-copy route there."""
    key = str(device)
    if key not in _TRITON_OK:
        try:
            with torch.cuda.device(device):
                p, g, m, v = (torch.zeros(2, dtype = torch.bfloat16, device = device) for _ in range(4))
                seed = 1 << 40  # the int64 specialisation real seeds use
                _adamw_kernel[(1,)](
                    p, g, m, v, 2, seed, 1.0, 0.1, 0.999, 0.001, -1e-3, 1.0, 1e-8,
                    BLOCK = _BLOCK, enable_fp_fusion = False,
                )  # fmt: skip
                _round_kernel[(1,)](
                    p, torch.ones(2, device = device), 2, seed, BLOCK = _BLOCK, enable_fp_fusion = False
                )
                torch.cuda.synchronize(device)
            _TRITON_OK[key] = bool(p.float().eq(1).all().item())
        except torch.cuda.OutOfMemoryError:
            return False  # transient: the float32-copy route serves this step, the next one checks again
        except Exception as exc:
            print(
                f"Unsloth: one-pass bfloat16 stochastic rounding is unavailable on {device} ({exc})."
            )
            _TRITON_OK[key] = False
    return _TRITON_OK[key]


def _round_cuda_(dst, src, seed):
    n = src.numel()
    if n:
        with torch.cuda.device(dst.device):
            _round_kernel[_grid(n)](dst, src, n, seed, BLOCK = _BLOCK, enable_fp_fusion = False)
    return dst


def _bnb_fused_enabled():
    """UNSLOTH_BF16_SR_FUSED_BNB=0 keeps bitsandbytes 8-bit AdamW on the float32-copy route."""
    value = os.environ.get("UNSLOTH_BF16_SR_FUSED_BNB", "1").strip().lower()
    return _fused_enabled() and value not in ("0", "false", "no", "off")


_BNB_OK = {}


def _f32(x):
    return torch.tensor(x, dtype = torch.float32)


def _bnb_launch(p, g, state, step, config, seed):
    beta1, beta2 = (float(b) for b in config["betas"][:2])
    lr, eps, weight_decay = float(config["lr"]), float(config["eps"]), float(config["weight_decay"])
    # in float32 as bitsandbytes' kernel forms them
    correction1 = 1.0 - _f32(beta1) ** step
    correction2 = torch.sqrt(1.0 - _f32(beta2) ** step)
    step_size = (-_f32(lr) * correction2) / correction1
    decay = 1.0 - _f32(lr) * _f32(weight_decay)
    n = p.numel()
    if state["state1"].dtype == torch.float32:
        with torch.cuda.device(p.device):
            _bnb_adam32bit_kernel[_grid(n)](
                p,
                g,
                state["state1"],
                state["state2"],
                n,
                seed,
                beta1,
                beta2,
                float(correction2 * _f32(eps)),
                float(step_size),
                float(decay),
                HAS_DECAY = weight_decay > 0.0,
                BLOCK = _BLOCK,
                # bitsandbytes builds with --use_fast_math (fmad on): contract as it does
                enable_fp_fusion = True,
            )
        return
    with torch.cuda.device(p.device):
        _bnb_adam8bit_kernel[((n + _BNB_BLOCK - 1) // _BNB_BLOCK,)](
            p,
            g,
            state["state1"],
            state["state2"],
            state["qmap1"],
            state["qmap2"],
            state["absmax1"],
            state["absmax2"],
            n,
            seed,
            beta1,
            beta2,
            float(correction2 * _f32(eps)),
            float(step_size),
            float(decay),
            HAS_DECAY = weight_decay > 0.0,
            BLOCK = _BNB_BLOCK,
            # bitsandbytes builds with --use_fast_math (fmad on): contract as it does, so states match bitwise
            enable_fp_fusion = True,
        )


def _bnb_triton_ok(device):
    """Compile and run the fused 8-bit kernel once per device; a failure keeps the float32-copy route there."""
    key = str(device)
    if key not in _BNB_OK:
        try:
            p = torch.zeros(300, dtype = torch.bfloat16, device = device)
            g = torch.full((300,), 1e-3, dtype = torch.bfloat16, device = device)
            code = torch.linspace(-1, 1, 256, device = device)
            state = dict(
                state1 = torch.full((300,), 128, dtype = torch.uint8, device = device),
                state2 = torch.zeros(300, dtype = torch.uint8, device = device),
                qmap1 = code,
                qmap2 = code.abs().sort().values,
                absmax1 = torch.zeros(2, device = device),
                absmax2 = torch.zeros(2, device = device),
            )
            config = dict(betas = (0.9, 0.999), lr = 1.0, eps = 1e-8, weight_decay = 0.0)
            _bnb_launch(p, g, state, 1, config, 1 << 40)
            torch.cuda.synchronize(device)
            # Adam's first step moves every weight by -lr * sign(g)
            _BNB_OK[key] = bool((p.float() + 1).abs().le(2**-8).all().item())
        except torch.cuda.OutOfMemoryError:
            return False
        except Exception as exc:
            print(
                f"Unsloth: one-pass bfloat16 stochastic rounding for 8-bit AdamW is unavailable on {device} ({exc})."
            )
            _BNB_OK[key] = False
    return _BNB_OK[key]


def _bnb_fused_step(
    optimizer,
    group,
    p,
    gindex,
    pindex,
    seed,
    owner = None,
):
    """bitsandbytes Adam / AdamW on a bfloat16 weight in one kernel. `owner`: the DTensor whose local shard `p`
    is (it seeds the draws). Returns False (nothing changed) when the weight or optimizer is not one it reproduces."""
    owner = p if owner is None else owner
    if not _bnb_fused_enabled() or p.device.type != "cuda" or not p.is_contiguous():
        return False
    if getattr(optimizer, "optimizer_name", None) != "adam":
        return False
    g = p.grad
    if (
        g is None
        or g.is_sparse
        or g.layout != torch.strided
        or g.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or g.shape != p.shape
        or g.device != p.device
    ):
        return False
    state = optimizer.state.get(p)
    if not state:
        return False
    n = p.numel()
    blocks = (n + _BNB_BLOCK - 1) // _BNB_BLOCK
    if torch.is_tensor(state.get("state1")) and state["state1"].dtype == torch.float32:
        # 32-bit state: small weights and Trainer-overridden embeddings
        expected = (("state1", torch.float32, n), ("state2", torch.float32, n))
    else:
        expected = (
            ("state1", torch.uint8, n),
            ("state2", torch.uint8, n),
            ("absmax1", torch.float32, blocks),
            ("absmax2", torch.float32, blocks),
            ("qmap1", torch.float32, 256),
            ("qmap2", torch.float32, 256),
        )
    for key, dtype, size in expected:
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != dtype
            or value.numel() != size
            or value.device != p.device
            or not value.is_contiguous()
        ):
            return False
    config = optimizer.get_config(gindex, pindex, group)
    if (
        len(config["betas"]) != 2
        or config.get("skip_zeros")
        or float(config.get("max_unorm", 0.0) or 0.0) > 0.0
        or float(config.get("percentile_clipping", 100) or 100) < 100
    ):
        return False
    if not (_triton_ok(p.device) and _bnb_triton_ok(p.device)):
        return False
    if not g.is_contiguous():  # as bitsandbytes' update_step does
        p.grad = g = g.contiguous()
    state["step"] += 1
    step = state["step"]
    try:
        _bnb_launch(
            p.detach(), g, state, int(step), config, _seed_for(optimizer, owner, seed, int(step))
        )
    except torch.cuda.OutOfMemoryError:
        state["step"] -= 1
        raise
    except (
        Exception
    ) as error:  # a launch failing after the device probe passed: nothing was written
        state["step"] -= 1
        _BNB_OK[str(p.device)] = False
        print(
            f"Unsloth: one-pass bfloat16 stochastic rounding for 8-bit AdamW failed on {p.device} ({error}); "
            "using the float32 route there."
        )
        return False
    increment = getattr(torch.autograd.graph, "increment_version", None)
    if increment is not None:
        increment(p)
        if owner is not p:
            increment(owner)
    _count(optimizer, "_unsloth_sr_bnb_fused_steps")
    return True


def stochastic_round_to_bf16_(
    dst,
    src,
    generator = None,
):
    """dst (bfloat16) <- src (float32) rounded stochastically (unbiased: E[round(x)] = x).
    DTensors (FSDP2 / TP shards) round their local shards: the rounding is elementwise."""
    dst_local, src = _local(dst), _local(src)
    flat_src = src.reshape(-1)
    flat_dst = dst_local.view(-1)
    for start in range(0, flat_src.numel(), _CHUNK):
        bits = flat_src[start : start + _CHUNK].contiguous().view(torch.int32)
        noise = torch.randint(
            0, 1 << 16, bits.shape, dtype = torch.int32, device = bits.device, generator = generator
        )
        rounded = (bits + noise) & -65536
        special = (bits & 0x7F800000) == 0x7F800000
        rounded = torch.where(special, bits, rounded)
        flat_dst[start : start + _CHUNK].copy_(rounded.view(torch.float32))
    return dst


def _is_dtensor(p):
    return type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor")


def _local(t):
    """A DTensor's local shard (same storage), else the tensor itself."""
    local = getattr(t, "_local_tensor", None) if t is not None else None
    return t if local is None else local


def _tracked(p, dtensor = False):
    """bfloat16 weights this optimizer can round. DTensors (FSDP2 / TP) round their local shard: torch AdamW /
    Adam's update and the rounding are elementwise, and bitsandbytes steps each shard on its own."""
    if _is_dtensor(p) and not dtensor:
        return False
    local = _local(p)
    return (
        p.dtype == torch.bfloat16
        and p.requires_grad
        and local.dtype == torch.bfloat16
        and local.layout == torch.strided
        and local.is_contiguous()
        and local.device.type in ("cuda", "cpu", "xpu")
    )


def _dtensor_steppable(p):
    """A DTensor weight steps on its shard only when its gradient is laid out like it (FSDP2's reduce-scattered
    shard, TP's matching placement); anything else stays on the optimizer's own step."""
    if not _is_dtensor(p):
        return True
    g = p.grad
    if not _is_dtensor(g):
        return False
    try:
        same = tuple(g.placements) == tuple(p.placements) and g.device_mesh == p.device_mesh
    except Exception:
        return False
    return same and _local(g).shape == _local(p).shape


def _shard_offset(optimizer, p):
    """Flat global index where a DTensor's local shard starts: 0 for a plain tensor and the same on every replica
    of one shard, so replicas round identically and different shards draw different noise."""
    if not _is_dtensor(p):
        return 0
    offsets = optimizer.__dict__.setdefault("_unsloth_sr_shard_offsets", {})
    key = id(p)
    if key not in offsets:
        try:
            try:
                from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
            except ImportError:  # torch < 2.5
                from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
            _, offset = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
            offsets[key] = sum(int(o) * int(s) for o, s in zip(offset, p.stride()))
        except Exception:
            # Fallback: the coordinate along the sharded mesh dims only, so replicas still agree.
            coordinate = p.device_mesh.get_coordinate() or ()
            offsets[key] = 1 + sum(
                int(c) * 7919**i
                for i, (c, placement) in enumerate(zip(coordinate, p.placements))
                if placement.is_shard()
            )
    return offsets[key]


def _generator(optimizer, device, seed):
    generators = optimizer.__dict__.setdefault("_unsloth_sr_generators", {})
    key = str(device)
    if key not in generators:
        g = torch.Generator(device = device)
        # The same seed on every rank: DDP replicas round identically and stay in sync.
        g.manual_seed(int(seed) + 0x5EED)
        generators[key] = g
    return generators[key]


def _seed_for(optimizer, p, seed, step):
    """Seeded by (param index, optimizer step): a resumed run draws what an uninterrupted one would.
    A DTensor shard also mixes in where it starts (0 for plain weights, so their draws are unchanged)."""
    index = optimizer._unsloth_sr_index[id(p)]
    return (
        int(seed)
        + 0x5EED
        + index * 1000003
        + int(step) * 2654435761
        + _shard_offset(optimizer, p) * 0x9E3779B97F4A7C15
    ) & (0x7FFFFFFFFFFFFFFF)


def _step_of(optimizer, p):
    step = (optimizer.state.get(p) or {}).get("step", 0)
    return int(step.item()) if torch.is_tensor(step) else int(step)


def _reseed(
    optimizer,
    p,
    seed,
    owner = None,
):
    g = _generator(optimizer, p.device, seed)
    g.manual_seed(_seed_for(optimizer, p if owner is None else owner, seed, _step_of(optimizer, p)))
    return g


class _Float32Step:
    """Step a float32 copy of one bfloat16 weight, then round it back into the weight in place.

    The weight's .data is never swapped: changing its dtype resets the weight's AccumulateGrad node, and
    DDP's allreduce hooks live on that node, so DDP would silently stop syncing that gradient from the
    next backward. The optimizer steps the copy under the weight's own state (moved to the copy's key for
    the call), and only the copy is float32."""

    def __init__(
        self,
        optimizer,
        p,
        seed,
        triton_round = False,
        owner = None,
    ):
        # owner: the DTensor whose local shard p is (bitsandbytes steps shards): it seeds the draws.
        self.optimizer, self.p, self.seed = optimizer, p, seed
        self.owner = p if owner is None else owner
        self.triton_round = (
            triton_round and p.device.type == "cuda" and _fused_enabled() and _triton_ok(p.device)
        )

    def __enter__(self):
        p = self.p
        copy = torch.nn.Parameter(p.detach().float())
        copy.grad = p.grad.float()
        state = self.optimizer.state
        if p in state:
            state[copy] = state.pop(p)
        self.copy = copy
        return copy

    def __exit__(self, *exc):
        p, copy = self.p, self.copy
        self.copy = None
        state = self.optimizer.state
        if copy in state:
            state[p] = state.pop(copy)
        if exc[0] is None:
            stepped = copy.detach()
            if self.triton_round:
                try:
                    seed = _seed_for(
                        self.optimizer, self.owner, self.seed, _step_of(self.optimizer, p)
                    )
                    _round_cuda_(_local(p.detach()), _local(stepped), seed)
                    _count(self.optimizer, "_unsloth_sr_triton_rounds")
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as error:  # a launch failing after the device probe passed
                    _TRITON_OK[str(p.device)] = False
                    print(
                        f"Unsloth: one-pass bfloat16 stochastic rounding failed on {p.device} ({error}); "
                        "using the float32 route there."
                    )
                    self.triton_round = False
            if not self.triton_round:
                stochastic_round_to_bf16_(
                    p.detach(), stepped, _reseed(self.optimizer, p, self.seed, self.owner)
                )
            if _is_dtensor(self.owner):
                # The rounding wrote the local shard: bump the DTensor's own version counter too, so a graph
                # saved before the step refuses to backpropagate, as for a plain weight.
                increment = getattr(torch.autograd.graph, "increment_version", None)
                if increment is not None:
                    increment(self.owner)
                _count(self.optimizer, "_unsloth_sr_dtensor_rounds")
        return False


_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _moments(state, p, dtype):
    # By name: a scalar parameter's fp32 `step` has its shape too.
    return [
        key
        for key in _MOMENT_KEYS
        if torch.is_tensor(state.get(key))
        and state[key].dtype == dtype
        and state[key].shape == p.shape
    ]


def _upcast_state(state, p):
    for key in _moments(state, p, torch.bfloat16):
        state[key] = state[key].float()


def _store_state(optimizer, state, p, seed):
    for key in _moments(state, p, torch.float32):
        value = state[key]
        stored = torch.empty_like(value, dtype = torch.bfloat16)
        stochastic_round_to_bf16_(stored, value, _generator(optimizer, value.device, seed))
        state[key] = stored


def _bnb_rounded_step(optimizer, seed, owner, plain, group, p, gindex, pindex):
    """One bitsandbytes update of bfloat16 weight p (owner: the DTensor p is the local shard of), rounded
    stochastically; `plain` is bitsandbytes' own update_step."""
    if _bnb_fused_step(optimizer, group, p, gindex, pindex, seed, owner):
        if owner is not p:
            _count(optimizer, "_unsloth_sr_dtensor_rounds")
        return None
    # bitsandbytes needs p and grad in one dtype: step a float32 copy, round back in one pass.
    with _Float32Step(optimizer, p, seed, triton_round = True, owner = owner) as copy:
        return plain(group, copy, gindex, pindex)


def _local_shard_bnb(optimizer):
    """bitsandbytes Adam / AdamW stepping DTensor weights on their local shards (models/fsdp2_bnb_optim.py)."""
    return getattr(type(optimizer), "_unsloth_fsdp2_local_shards", False) is True and callable(
        getattr(optimizer, "_shard_call", None)
    )


def _wrap_bitsandbytes(optimizer, tracked, seed):
    # Read at call time: FSDP2 / TP may swap the class to its local-shard form after this wraps.
    previous = vars(optimizer).get("update_step")
    # DTensors tracked by identity: accelerate's FSDP2 prepare frees the plain weights this saw, and a shard
    # built then can reuse a freed weight's id.
    shards = {
        id(p): weakref.ref(p)
        for group in optimizer.param_groups
        for p in group["params"]
        if id(p) in tracked and _is_dtensor(p)
    }

    def update_step(self, group, p, gindex, pindex):
        original = (
            previous if previous is not None else types.MethodType(type(self).update_step, self)
        )
        if id(p) not in tracked or p.grad is None or p.dtype != torch.bfloat16:
            return original(group, p, gindex, pindex)
        if _is_dtensor(p):
            ref = shards.get(id(p))
            if ref is None or ref() is not p or not _local_shard_bnb(self):
                return original(group, p, gindex, pindex)
            # The local-shard class hands bitsandbytes a plain alias of this rank's shard: round that one.
            return self._shard_call(
                "update_step",
                group,
                p,
                gindex,
                pindex,
                True,
                step = functools.partial(_bnb_rounded_step, self, seed, p),
            )
        return _bnb_rounded_step(self, seed, p, original, group, p, gindex, pindex)

    optimizer.update_step = types.MethodType(update_step, optimizer)


def _step_hooks(optimizer):
    return bool(
        getattr(optimizer, "_optimizer_step_pre_hooks", None)
        or getattr(optimizer, "_optimizer_step_post_hooks", None)
    )


def _wrap_torch(optimizer, tracked, seed):
    original = optimizer.step
    # The class step without torch's hook dispatch: hooks run once around `update`, with the real weights.
    cls_step = type(optimizer).step
    unhooked = (
        getattr(cls_step, "__wrapped__", None) if getattr(cls_step, "hooked", False) else None
    )
    if unhooked is None or getattr(original, "__func__", None) is not cls_step:
        unhooked = None

    def update(self, closure = None):
        """Step self.param_groups' single group of tracked weights inside torch's step hook dispatch (so state
        offload, gradient and DP hooks see the real weights): the one-pass kernel where it applies, else a float32
        copy per weight, rounded back in place."""
        if closure is not None:  # e.g. state offload's restore, swapped in by its pre hook
            with torch.enable_grad():
                closure()
        (group,) = self.param_groups
        params = [p for p in group["params"] if p.grad is not None]  # pre hooks may drop gradients
        if params and _fused_group(self, group):
            fused = [p for p in params if _fused_param(self, p)]
            left = {id(p) for p in _fused_adamw_step(self, group, fused, seed)} if fused else set()
            done = {id(p) for p in fused if id(p) not in left}
            params = [p for p in params if id(p) not in done]
        try:
            for p in params:
                state = self.state.get(p)
                if state:
                    _upcast_state(state, p)
                with _Float32Step(self, p, seed) as copy:
                    self.param_groups = [dict(group, params = [copy])]
                    if unhooked is not None:
                        unhooked(self)
                    else:
                        original()
                _store_state(self, self.state[p], p, seed)
        finally:
            self.param_groups = [group]

    profile_hook_step = getattr(torch.optim.Optimizer, "profile_hook_step", None)
    hooked = profile_hook_step(update) if profile_hook_step is not None else update

    # A bound method: torch's LR scheduler wraps `optimizer.step.__func__`.
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            for group in groups:
                rest = [
                    p
                    for p in group["params"]
                    if id(p) not in tracked or (p.grad is not None and not _dtensor_steppable(p))
                ]
                mine = [
                    p
                    for p in group["params"]
                    if id(p) in tracked and p.grad is not None and _dtensor_steppable(p)
                ]
                if rest:
                    self.param_groups = [dict(group, params = rest)]
                    original()
                # Step hooks fire per weight when the optimizer has its own (state offload, idle experts,
                # DP sync), as torch's per-weight steps did before; else once for the group.
                batches = [[p] for p in mine] if _step_hooks(self) else ([mine] if mine else [])
                # State offload copies the next weights' state in while each one steps.
                pipelined = getattr(
                    getattr(self, "_unsloth_state_offloader", None), "pipelined", None
                )
                with pipelined(batches) if callable(pipelined) else contextlib.nullcontext():
                    for batch in batches:
                        self.param_groups = [dict(group, params = batch)]
                        hooked(self, None)
        finally:
            self.param_groups = groups
        return loss

    optimizer.step = types.MethodType(step, optimizer)


def _count(
    optimizer,
    name,
    n = 1,
):
    optimizer.__dict__[name] = optimizer.__dict__.get(name, 0) + n


def _float(x):
    if torch.is_tensor(x):
        if x.device.type != "cpu" or x.numel() != 1:
            return None
        x = x.item()
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fused_group(optimizer, group):
    """Groups the one-pass kernel reproduces: AdamW (or Adam without L2 decay), float hyperparameters."""
    if not _fused_enabled():
        return False
    if group.get("amsgrad") or group.get("maximize") or group.get("differentiable"):
        return False
    if group.get("capturable"):
        return False
    # GradScaler hands a fused optimizer its scale and overflow flag instead of unscaling: torch's kernel uses them.
    if (
        getattr(optimizer, "grad_scale", None) is not None
        or getattr(optimizer, "found_inf", None) is not None
    ):
        return False
    betas = group.get("betas", (0.9, 0.999))
    values = [
        _float(group.get("lr")),
        _float(group.get("eps")),
        _float(group.get("weight_decay", 0)),
    ]
    values += [_float(b) for b in betas]
    if len(betas) != 2 or any(v is None for v in values):
        return False
    decoupled = group.get("decoupled_weight_decay", None)
    if decoupled is None:  # torch < 2.7: AdamW groups carry no flag, the class decides
        decoupled = isinstance(optimizer, torch.optim.AdamW)
    return bool(decoupled) or values[2] == 0


def _fused_param(optimizer, p):
    if not _dtensor_steppable(p):
        return False
    g = _local(p.grad)
    if (
        p.device.type != "cuda"
        or g is None
        or g.is_sparse
        or g.layout != torch.strided
        or g.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or g.device != _local(p).device
        or g.shape != _local(p).shape
        or not g.is_contiguous()
        or not _triton_ok(p.device)
    ):
        return False
    state = optimizer.state.get(p)
    if not state:
        return True
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.bfloat16
            or value.shape != p.shape
            or _is_dtensor(value) != _is_dtensor(p)
            or _local(value).shape != _local(p).shape
            or _local(value).device != _local(p).device
            or not _local(value).is_contiguous()
        ):
            return False  # e.g. float32 moments from a checkpoint: the float32 route converts them
    return torch.is_tensor(state.get("step")) and state["step"].numel() == 1


def _scalar_dtype():
    try:
        from torch.optim.optimizer import _get_scalar_dtype
        return _get_scalar_dtype()
    except Exception:
        return torch.float32


def _multi_tensor_enabled():
    """UNSLOTH_BF16_SR_MULTI_TENSOR=0 launches the one-pass kernel once per weight instead of once per bucket."""
    value = os.environ.get("UNSLOTH_BF16_SR_MULTI_TENSOR", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


_SUB = 16  # _BLOCK-element blocks per program in the multi-tensor kernel: 16384 elements
_MULTI_OK = {}
_G_DTYPES = {}
if triton is not None:
    _G_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def _adamw_scalars(lr, beta1, beta2, t):
    """NEG_STEP and BC2_SQRT exactly as _fused_adamw_step passes them to _adamw_kernel (Python floats)."""
    return (lr / (1 - beta1**t)) * -1, (1 - beta2**t) ** 0.5


def _chunk_table(optimizer, device, numels):
    """(weight row, chunk index) per program, cached per device and weight sizes (grads move, sizes do not)."""
    cache = optimizer.__dict__.setdefault("_unsloth_sr_chunk_tables", {})
    key = (str(device), tuple(numels))
    table = cache.get(key)
    if table is None:
        span = _BLOCK * _SUB
        counts = (torch.tensor(numels, dtype = torch.int64) + span - 1) // span
        row = torch.repeat_interleave(torch.arange(len(numels)), counts)
        first = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        chunk = torch.arange(row.numel()) - first
        table = torch.stack((row, chunk), 1).to(torch.int32).to(device)
        if len(cache) > 8:  # e.g. a model whose trainable set keeps changing
            cache.clear()
        cache[key] = table
    return table


def _multi_launch(device, g_dtype, tensors, scalars, chunks, group_scalars):
    """One _adamw_multi_kernel launch. tensors: (weight, grad, exp_avg, exp_avg_sq, seed) per weight, all local;
    scalars: (NEG_STEP, BC2_SQRT) per weight. The table goes over in one copy: 6 int64 columns, then the two
    float32 scalars packed into a 7th."""
    rows = len(tensors)
    table = torch.empty((rows, 7), dtype = torch.int64, pin_memory = True)
    table[:, :6] = torch.tensor(
        [
            [p.data_ptr(), g.data_ptr(), m.data_ptr(), v.data_ptr(), p.numel(), s]
            for p, g, m, v, s in tensors
        ],
        dtype = torch.int64,
    )
    table[:, 6:].view(torch.float32).copy_(torch.tensor(scalars, dtype = torch.float32))
    table = table.to(device, non_blocking = True)
    decay, w1, b2, w2, eps = group_scalars
    with torch.cuda.device(device):
        _adamw_multi_kernel[(chunks.shape[0],)](
            table, table.view(torch.float32), chunks, decay, w1, b2, w2, eps,
            G_DTYPE = _G_DTYPES[g_dtype], BLOCK = _BLOCK, SUB = _SUB, enable_fp_fusion = False,
        )  # fmt: skip


def _multi_ok(device):
    """Run the multi-tensor kernel once per device against _adamw_kernel on weights with tails and mixed steps;
    any failure or bit difference keeps the per-weight launches there."""
    key = str(device)
    if key not in _MULTI_OK:
        try:
            gen = torch.Generator(device = "cpu").manual_seed(0)
            sizes = (3, 1025, _BLOCK * _SUB + 7)
            make = (
                lambda n, s = 1.0: (torch.randn(n, generator = gen) * s).to(torch.bfloat16).to(device)
            )
            weights = [
                [make(n, 0.05), make(n, 1e-3), make(n, 1e-3), make(n, 1e-6).abs()] for n in sizes
            ]
            single = [[x.clone() for x in w] for w in weights]
            seeds = [(1 << 40) + 977 * j for j in range(len(sizes))]
            scalars = [_adamw_scalars(1e-3, 0.9, 0.999, t) for t in (1, 7, 300)]
            group_scalars = (1 - 1e-5, 0.1, 0.999, 0.001, 1e-8)
            with torch.cuda.device(device):
                for (p, g, m, v), seed, (neg, bc2) in zip(single, seeds, scalars):
                    _adamw_kernel[_grid(p.numel())](
                        p, g, m, v, p.numel(), seed, *group_scalars[:4], neg, bc2, group_scalars[4],
                        BLOCK = _BLOCK, enable_fp_fusion = False,
                    )  # fmt: skip
                chunks = _chunk_table(_PROBE, device, sizes)
                _multi_launch(
                    device, torch.bfloat16,
                    [(p, g, m, v, s) for (p, g, m, v), s in zip(weights, seeds)],
                    scalars, chunks, group_scalars,
                )  # fmt: skip
                torch.cuda.synchronize(device)
            _MULTI_OK[key] = all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for w, sw in zip(weights, single)
                for a, b in zip(w, sw)
            )
            if not _MULTI_OK[key]:
                print(
                    f"Unsloth: multi-tensor bfloat16 stochastic rounding differs on {device}; one launch per weight."
                )
        except torch.cuda.OutOfMemoryError:
            return False
        except Exception as exc:
            print(
                f"Unsloth: multi-tensor bfloat16 stochastic rounding is unavailable on {device} ({exc})."
            )
            _MULTI_OK[key] = False
    return _MULTI_OK[key]


class _Probe:
    pass


_PROBE = _Probe()


def _multi_adamw_step(optimizer, group, params, states, ts, seed, group_scalars):
    """Step params (fused-eligible, state created, step counted) in one _adamw_multi_kernel launch per
    (device, gradient dtype). Returns the ids it stepped; the rest take one launch per weight."""
    if len(params) < 2 or not _multi_tensor_enabled():
        return set()
    try:
        if torch.cuda.is_current_stream_capturing():
            return set()  # host-side tables would replay stale: keep the per-weight launches
    except Exception:
        pass
    lr = _float(group["lr"])
    beta1, beta2 = (_float(b) for b in group["betas"])
    buckets = {}
    for p, state, t in zip(params, states, ts):
        local = _local(p)
        if local.numel() == 0:
            continue
        g = _local(p.grad)
        buckets.setdefault((local.device, g.dtype), []).append((p, state, t))
    done = set()
    for (device, g_dtype), items in buckets.items():
        if len(items) < 2 or g_dtype not in _G_DTYPES or not _multi_ok(device):
            continue
        tensors, scalars = [], []
        for p, state, t in items:
            tensors.append(
                (
                    _local(p.data),
                    _local(p.grad),
                    _local(state["exp_avg"]),
                    _local(state["exp_avg_sq"]),
                    _seed_for(optimizer, p, seed, int(t)),
                )
            )
            scalars.append(_adamw_scalars(lr, beta1, beta2, t))
        try:
            chunks = _chunk_table(optimizer, device, [x[0].numel() for x in tensors])
            _multi_launch(device, g_dtype, tensors, scalars, chunks, group_scalars)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            _MULTI_OK[str(device)] = False
            print(
                f"Unsloth: multi-tensor bfloat16 stochastic rounding failed on {device} ({exc}); "
                "one launch per weight there."
            )
            continue
        done.update(id(p) for p, _, _ in items)
        _count(optimizer, "_unsloth_sr_multi_tensor_launches")
    return done


def _fused_adamw_step(optimizer, group, params, seed):
    """One kernel per weight: AdamW in float32 registers, weight and both bfloat16 moments rounded stochastically.
    Returns the weights it could not step (a failed launch), for the float32 route."""
    lr, eps, weight_decay = (_float(group[k]) for k in ("lr", "eps", "weight_decay"))
    beta1, beta2 = (_float(b) for b in group["betas"])
    on_device = bool(group.get("fused") or group.get("capturable"))
    states = []
    for p in params:
        state = optimizer.state[p]
        if not state:  # what torch.optim.AdamW creates on a first step
            state["step"] = (
                torch.zeros((), dtype = _scalar_dtype(), device = p.device)
                if on_device
                else torch.tensor(0.0, dtype = _scalar_dtype())
            )
            state["exp_avg"] = torch.zeros_like(p, memory_format = torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format = torch.preserve_format)
        states.append(state)
    steps = [state["step"] for state in states]
    cpu = [s for s in steps if s.device.type == "cpu"]
    dev = [s for s in steps if s.device.type != "cpu"]
    if cpu:
        torch._foreach_add_(cpu, 1)
    if dev:
        torch._foreach_add_(dev, 1)
    # One host read per device for its step counters (torch's fused AdamW keeps them on the GPU).
    values = {}
    for device in {s.device for s in dev}:
        here = [s for s in dev if s.device == device]
        values.update(zip(map(id, here), torch.stack([s.float() for s in here]).tolist()))
    decay = 1 - lr * weight_decay
    stepped, left = [], []
    ts = [step.item() if step.device.type == "cpu" else values[id(step)] for step in steps]
    multi = _multi_adamw_step(
        optimizer, group, params, states, ts, seed, (decay, 1 - beta1, beta2, 1 - beta2, eps)
    )
    for i, (p, state, step) in enumerate(zip(params, states, steps)):
        t = ts[i]
        if id(p) in multi:
            stepped.append(p)
            continue
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        # A DTensor (FSDP2 / TP) steps its local shard: every op in the kernel is elementwise.
        n = _local(p).numel()
        if n:
            try:
                with torch.cuda.device(p.device):
                    _adamw_kernel[_grid(n)](
                        _local(p.data),
                        _local(p.grad),
                        _local(state["exp_avg"]),
                        _local(state["exp_avg_sq"]),
                        n,
                        _seed_for(optimizer, p, seed, int(t)),
                        decay,
                        1 - beta1,
                        beta2,
                        1 - beta2,
                        (lr / bc1) * -1,
                        bc2**0.5,
                        eps,
                        BLOCK = _BLOCK,
                        enable_fp_fusion = False,
                    )
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                # A launch failed after the device probe passed (a specialisation the probe did not compile).
                # Nothing was written for this weight or the ones after it: undo their step count and hand them
                # to the float32 route, which every later step on this device takes too.
                _TRITON_OK[str(p.device)] = False
                print(
                    f"Unsloth: one-pass bfloat16 stochastic rounding failed on {p.device} ({exc}); "
                    "using the float32 route there."
                )
                for q, s in zip(params[i:], steps[i:]):
                    if id(q) not in multi:
                        s.sub_(1)
                left = [q for q in params[i:] if id(q) not in multi]
                break
        stepped.append(p)
    # The kernel writes the storage directly: bump the parameters' own version counters (not a .data alias, which
    # has its own), one tensor at a time since torch < 2.5 takes a single tensor.
    increment = getattr(torch.autograd.graph, "increment_version", None)
    if increment is not None:
        for p in stepped:
            for t in (p, _local(p)) if _is_dtensor(p) else (p,):
                try:
                    increment(t)
                except Exception:
                    pass
    _count(optimizer, "_unsloth_sr_fused_steps", len(stepped))
    _count(optimizer, "_unsloth_sr_dtensor_rounds", sum(1 for p in stepped if _is_dtensor(p)))
    return left


def _kind(optimizer):
    try:
        from bitsandbytes.optim.optimizer import Optimizer8bit
        if isinstance(optimizer, Optimizer8bit) and hasattr(optimizer, "update_step"):
            return "bitsandbytes"
    except Exception:
        pass
    if type(optimizer) in (torch.optim.AdamW, torch.optim.Adam):
        return "torch"
    return None


def enable_bf16_stochastic_rounding(
    optimizer,
    seed = 3407,
    verbose = True,
):
    """Step this optimizer's bfloat16 weights in float32 with stochastic rounding back to bfloat16.
    Returns the number of weights it covers (0: nothing to do, unsupported optimizer, or turned off).
    Call before an LR scheduler is built over the optimizer. Idempotent."""
    if optimizer is None or not bf16_stochastic_rounding_enabled():
        return 0
    # accelerate's AcceleratedOptimizer, and under it the in-backward step's wrapper (optimizers/auto_fit.py):
    # transformers 5 builds the scheduler after accelerate wrapped the optimizer.
    seen = set()
    inner = getattr(optimizer, "optimizer", None)
    while inner is not None and hasattr(inner, "param_groups") and id(inner) not in seen:
        seen.add(id(optimizer))
        optimizer = inner
        inner = getattr(optimizer, "optimizer", None)
    if getattr(optimizer, _MARK, None) is not None:
        return getattr(optimizer, _MARK)
    kind = _kind(optimizer)
    if kind is None:
        return 0
    params = [
        p for group in optimizer.param_groups for p in group["params"] if _tracked(p, dtensor = True)
    ]
    if not params:
        return 0
    # A routed weight may first receive a gradient on either side of a checkpoint.
    tracked = {id(p): index for index, p in enumerate(params)}
    optimizer._unsloth_sr_index = tracked
    if kind == "bitsandbytes":
        _wrap_bitsandbytes(optimizer, tracked, seed)
    else:
        _wrap_torch(optimizer, tracked, seed)
    setattr(optimizer, _MARK, len(params))
    if verbose:
        print(
            f"Unsloth: steps {len(params)} trained bfloat16 weights in float32 and "
            "rounds them back stochastically, so updates smaller than bfloat16's spacing are not lost. "
            "Set UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 to turn this off."
        )
    return len(params)


def _unsloth_model(model):
    """True for a model Unsloth loaded, full finetuning or LoRA (every load stamps _unsloth_full_finetuning). Under
    LoRA the adapters are float32, but trained bfloat16 weights beside them (modules_to_save embed_tokens / lm_head,
    trained norms) step in bfloat16 and need the rounding as much as full finetuning does; a LoRA run with none
    covers nothing and its optimizer is left untouched."""
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        if getattr(model, "_unsloth_full_finetuning", None) is not None:
            return True
        model = getattr(model, "module", None)
    return False


def patch_trainer_bf16_stochastic_rounding(Trainer):
    """create_scheduler: the optimizer exists and the LR scheduler has not yet wrapped its step."""
    original = Trainer.create_scheduler
    if getattr(original, _MARK, False):
        return

    def create_scheduler(
        self,
        num_training_steps,
        optimizer = None,
        *args,
        **kwargs,
    ):
        target = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        if getattr(self, "lr_scheduler", None) is None and _unsloth_model(
            getattr(self, "model", None)
        ):
            try:
                enable_bf16_stochastic_rounding(target, seed = getattr(self.args, "seed", 3407))
            except Exception as exc:  # never block training on this
                print(f"Unsloth: bfloat16 stochastic rounding was not enabled ({exc}).")
        return original(self, num_training_steps, optimizer, *args, **kwargs)

    setattr(create_scheduler, _MARK, True)
    create_scheduler.__wrapped__ = original
    create_scheduler.__doc__ = original.__doc__
    Trainer.create_scheduler = create_scheduler
