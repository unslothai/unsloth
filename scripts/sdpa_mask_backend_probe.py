#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Which SDPA backends tolerate a dense bool attn_mask, and at what cost, at Hunyuan's real
joint shape (B=1, H=16, N=50345, D=128, bf16)? Decides whether nulling the all-True mask is
the real win on the PRODUCTION cuDNN path (not just the native math fallback)."""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, N, D = 1, 16, 50345, 128
dev, dt = "cuda:0", torch.bfloat16


def mk():
    return torch.randn(B, H, N, D, device=dev, dtype=dt)


def timed(fn, iters=20):
    torch.cuda.synchronize()
    for _ in range(3):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return f"UNSUPPORTED ({type(e).__name__})"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


q, k, v = mk(), mk(), mk()
dense = torch.ones(B, 1, N, N, dtype=torch.bool, device=dev)

backends = {
    "default(dispatch)": None,
    "MATH": [SDPBackend.MATH],
    "FLASH": [SDPBackend.FLASH_ATTENTION],
    "EFFICIENT": [SDPBackend.EFFICIENT_ATTENTION],
    "CUDNN": [SDPBackend.CUDNN_ATTENTION],
}

print(f"shape B={B} H={H} N={N} D={D} {dt}\n")
print(f"{'backend':<20}{'mask=dense(ms)':>18}{'mask=None(ms)':>18}")
for name, bk in backends.items():
    def run_dense():
        if bk is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=dense)
        with sdpa_kernel(bk):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=dense)

    def run_none():
        if bk is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        with sdpa_kernel(bk):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None)

    dms = timed(run_dense)
    nms = timed(run_none)
    d_s = f"{dms:.2f}" if isinstance(dms, float) else dms
    n_s = f"{nms:.2f}" if isinstance(nms, float) else nms
    print(f"{name:<20}{d_s:>18}{n_s:>18}")
