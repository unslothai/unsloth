# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes' Adam / AdamW updates in plain torch (CPU tensors, GPUs without Triton, a failed
Triton launch). Same algorithm and state layout as the Triton kernels in
unsloth/kernels/adam8bit_blockwise.py, with IEEE float32 ops instead of bitsandbytes' fast-math
ones, so a value can differ from bitsandbytes in the last bit (and a code sitting on a midpoint
by one step). `compute_dtype=torch.float64` evaluates the same update in float64 (the oracle the
tests compare both against)."""

import torch

__all__ = ["adam_8bit_blockwise_torch", "adam_32bit_torch", "quantize_2d_torch"]

BLOCKSIZE = 256
# Blocks per chunk, bounding the float temporaries on large weights.
_CHUNK_BLOCKS = 1 << 14


def _scalars(step, lr, beta1, beta2, weight_decay, dtype, device):
    t = lambda x: torch.as_tensor(x, dtype = dtype, device = device)
    step = t(step.item() if torch.is_tensor(step) else step)
    lr = t(lr.item() if torch.is_tensor(lr) else lr)
    beta1, beta2 = t(beta1), t(beta2)
    correction1 = 1.0 - beta1**step
    correction2 = torch.sqrt(1.0 - beta2**step)
    step_size = (-lr * correction2) / correction1
    decay = 1.0 - lr * t(weight_decay)
    return beta1, beta2, correction2, step_size, decay


def quantize_2d_torch(code, x, signed):
    """bitsandbytes' quantize_2D: binary search over the sorted 256-entry code, nearer side by
    midpoint. Returns int64 codes."""
    pivot = torch.full(x.shape, 127, dtype = torch.long, device = x.device)
    upper_pivot = torch.full_like(pivot, 255)
    lower_pivot = torch.zeros_like(pivot)
    lower = torch.full_like(x, -1.0 if signed else 0.0)
    upper = torch.ones_like(x)
    val = code[pivot]
    for k in range(7):
        up = x > val
        lower_pivot = torch.where(up, pivot, lower_pivot)
        lower = torch.where(up, val, lower)
        upper_pivot = torch.where(up, upper_pivot, pivot)
        upper = torch.where(up, upper, val)
        pivot = torch.where(up, pivot + (64 >> k), pivot - (64 >> k))
        val = code[pivot]
    up = x > val
    return torch.where(
        up,
        torch.where(x > (upper + val) * 0.5, upper_pivot, pivot),
        torch.where(x < (lower + val) * 0.5, lower_pivot, pivot),
    )


def _step_p(p, s1, s2, correction2, step_size, decay, eps, has_decay, dtype):
    stepped = (p.to(s1.dtype) + step_size * (s1 / (torch.sqrt(s2) + correction2 * eps))).to(dtype)
    if has_decay:
        stepped = (stepped.to(s1.dtype) * decay).to(dtype)
    return stepped


@torch.no_grad()
def adam_8bit_blockwise_torch(
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
    compute_dtype = torch.float32,
):
    """bitsandbytes optimizer_update_8bit_blockwise("adam", ...) in place."""
    n = p.numel()
    if n == 0:
        return
    cd, dev = compute_dtype, p.device
    beta1, beta2, correction2, step_size, decay = _scalars(
        step, lr, beta1, beta2, weight_decay, cd, dev
    )
    eps = torch.as_tensor(eps, dtype = torch.float32).to(cd)
    has_decay = float(weight_decay) > 0.0
    code1 = qmap1.to(device = dev, dtype = cd)
    code2 = qmap2.to(device = dev, dtype = cd)
    flat_p, flat_g = p.view(-1), g.reshape(-1)
    flat_1, flat_2 = state1.view(-1), state2.view(-1)
    blocks = absmax1.numel()
    for b0 in range(0, blocks, _CHUNK_BLOCKS):
        b1 = min(blocks, b0 + _CHUNK_BLOCKS)
        lo, hi = b0 * BLOCKSIZE, min(n, b1 * BLOCKSIZE)
        pad = (b1 - b0) * BLOCKSIZE - (hi - lo)
        rows = (b1 - b0, BLOCKSIZE)
        # Padding as bitsandbytes' block loads: gradient 0, codes 128 / 0.
        gv = torch.nn.functional.pad(flat_g[lo:hi].to(cd), (0, pad)).view(rows)
        c1 = torch.nn.functional.pad(flat_1[lo:hi].long(), (0, pad), value = 128).view(rows)
        c2 = torch.nn.functional.pad(flat_2[lo:hi].long(), (0, pad), value = 0).view(rows)
        finite = torch.isfinite(gv)
        gv = gv * gnorm_scale
        s2 = code2[c2] * absmax2[b0:b1, None].to(cd)
        s2 = s2 * beta2 + (1.0 - beta2) * gv * gv
        s1 = code1[c1] * absmax1[b0:b1, None].to(cd)
        s1 = s1 * beta1 + (1.0 - beta1) * gv
        s1 = torch.where(finite, s1, 0.0)
        s2 = torch.where(finite, s2, 0.0)
        new1 = s1.abs().amax(1)
        new2 = s2.abs().amax(1)
        absmax1[b0:b1] = new1.to(absmax1.dtype)
        absmax2[b0:b1] = new2.to(absmax2.dtype)
        pv = flat_p[lo:hi]
        valid = hi - lo
        stepped = _step_p(
            pv,
            s1.reshape(-1)[:valid],
            s2.reshape(-1)[:valid],
            correction2,
            step_size,
            decay,
            eps,
            has_decay,
            p.dtype,
        )
        pv.copy_(torch.where(finite.reshape(-1)[:valid], stepped, pv))
        q1 = quantize_2d_torch(code1, s1 / new1[:, None], True)
        # Keep the first moment's sign: one code step towards it.
        flip = torch.signbit(code1[q1]) != torch.signbit(s1)
        q1 = torch.where(flip, torch.where(s1 > 0, q1 + 1, q1 - 1), q1) & 255
        q2 = quantize_2d_torch(code2, s2 / new2[:, None], False)
        flat_1[lo:hi] = q1.reshape(-1)[:valid].to(torch.uint8)
        flat_2[lo:hi] = q2.reshape(-1)[:valid].to(torch.uint8)


@torch.no_grad()
def adam_32bit_torch(
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
    compute_dtype = torch.float32,
):
    """bitsandbytes optimizer_update_32bit("adam", ...) in place (no max_unorm, no skip_zeros)."""
    if p.numel() == 0:
        return
    cd = compute_dtype
    beta1, beta2, correction2, step_size, decay = _scalars(
        step, lr, beta1, beta2, weight_decay, cd, p.device
    )
    eps = torch.as_tensor(eps, dtype = torch.float32).to(cd)
    gv = g.to(cd) * gnorm_scale
    s1 = state1.to(cd) * beta1 + (1.0 - beta1) * gv
    s2 = state2.to(cd) * beta2 + (1.0 - beta2) * (gv * gv)
    p.copy_(
        _step_p(p, s1, s2, correction2, step_size, decay, eps, float(weight_decay) > 0.0, p.dtype)
    )
    state1.copy_(s1)
    state2.copy_(s2)
