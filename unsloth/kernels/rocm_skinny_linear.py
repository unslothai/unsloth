# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Decode linears on RDNA3.5 (gfx1151) ROCm: out[M, N] = X @ W.T + s * (X @ A.T) @ B.T for
# M <= 16 rows of a 16-bit W, in one Triton launch. rocBLAS runs these skinny GEMMs with a 128x128
# macro tile on gfx1151, so the base GEMM, X @ A.T and the LoRA addmm each cost a full tile pass;
# here W is streamed once and the rank-r term is accumulated in fp32 beside it (one rounding to
# W's dtype at the end instead of three). Deterministic: split-K partials are summed in a fixed
# order. On by default only where measured (gfx1151, Radeon 8060S / Strix Halo); UNSLOTH_ROCM_SKINNY_LINEAR=all
# enables it on every gfx11 part, =0 disables it.

import os

import torch

__all__ = [
    "skinny_linear",
    "skinny_linear_supported",
    "skinny_decode_linear",
    "pick_config",
    "routed",
    "STATS",
]

MAX_ROWS = 16
STATS = {"calls": 0}
_SUPPORTED = {}
_ENV = os.environ.get("UNSLOTH_ROCM_SKINNY_LINEAR", "1")
_CONFIG_ENV = os.environ.get("UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG", "")
triton = None
if _ENV != "0" and torch.version.hip is not None:
    try:
        import triton
        import triton.language as tl
    except Exception:
        triton = None

if triton is not None:

    @triton.jit
    def _skinny_linear_kernel(
        X,
        W,
        A,
        B,
        Y,
        P,
        M,
        N,
        K,
        R,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ar,
        stride_ak,
        stride_bn,
        stride_br,
        stride_ym,
        stride_yn,
        k_per_split,
        scale,
        HAS_LORA: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = rm < M
        n_mask = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        rr = tl.arange(0, BLOCK_R)
        r_mask = rr < R
        xa = tl.zeros((BLOCK_M, BLOCK_R), dtype = tl.float32)
        k_start = pid_k * k_per_split
        for k0 in range(k_start, k_start + k_per_split, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            k_mask = rk < K
            x = tl.load(
                X + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
                mask = m_mask[:, None] & k_mask[None, :],
                other = 0.0,
            )
            w = tl.load(
                W + rn[None, :] * stride_wn + rk[:, None] * stride_wk,
                mask = n_mask[None, :] & k_mask[:, None],
                other = 0.0,
            )
            acc = tl.dot(x, w, acc)
            if HAS_LORA:
                a = tl.load(
                    A + rr[None, :] * stride_ar + rk[:, None] * stride_ak,
                    mask = r_mask[None, :] & k_mask[:, None],
                    other = 0.0,
                )
                xa = tl.dot(x, a, xa)
        if HAS_LORA:
            b = tl.load(
                B + rn[None, :] * stride_bn + rr[:, None] * stride_br,
                mask = n_mask[None, :] & r_mask[:, None],
                other = 0.0,
            ).to(tl.float32)
            acc += scale * tl.dot(xa, b, input_precision = "ieee")
        if SPLIT_K == 1:
            tl.store(
                Y + rm[:, None] * stride_ym + rn[None, :] * stride_yn,
                acc.to(Y.dtype.element_ty),
                mask = m_mask[:, None] & n_mask[None, :],
            )
        else:
            tl.store(
                P + pid_k * (BLOCK_M * N) + rm[:, None] * N + rn[None, :],
                acc,
                mask = n_mask[None, :] & (rm[:, None] < BLOCK_M),
            )

    @triton.jit
    def _splitk_reduce_kernel(
        P,
        Y,
        M,
        N,
        stride_ym,
        stride_yn,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        offs = rm[:, None] * N + rn[None, :]
        mask = (rm[:, None] < BLOCK_M) & n_mask[None, :]
        acc = tl.load(P + offs, mask = mask, other = 0.0)
        for s in tl.static_range(1, SPLIT_K):
            acc += tl.load(P + s * (BLOCK_M * N) + offs, mask = mask, other = 0.0)
        tl.store(
            Y + rm[:, None] * stride_ym + rn[None, :] * stride_yn,
            acc.to(Y.dtype.element_ty),
            mask = (rm[:, None] < M) & n_mask[None, :],
        )


def _parse_config(text):
    try:
        values = tuple(int(x) for x in text.split(","))
        return values if len(values) in (4, 5) else None
    except Exception:
        return None


_FORCED = _parse_config(_CONFIG_ENV) if _CONFIG_ENV else None


def pick_config(N, K):
    """(BLOCK_N, BLOCK_K, num_warps, SPLIT_K, num_stages) for an [M <= 16, K] x [N, K]^T decode linear.
    Measured on gfx1151 (Radeon 8060S): 64 x 64 tiles, 2 warps and about 256 of K per split were the best or
    within 5% of it for every Qwen3-0.6B projection; the split is capped so wide outputs keep one pass."""
    if _FORCED is not None:
        return _FORCED
    split = max(1, min(8, round(K / 256)))
    tiles = -(-N // 64)
    while split > 1 and tiles * split > 512:
        split //= 2
    return 64, 64, 2, split, 2


def routed(N, K):
    """Shapes the kernel takes. Measured on gfx1151 (Linux and Windows, M = 1 to 16, LoRA r32): faster than
    rocBLAS for every Qwen3-0.6B / Qwen3-4B projection with K <= 3072 and for Llama-3-8B k / v (N = 1024);
    slower for K = 4096 with N >= 2560 (Qwen3-4B o_proj, Llama-3-8B q / gate_up), which keep rocBLAS."""
    return K <= 3072 or N <= 1024


# Archs with a measured decode win; others keep rocBLAS / hipBLASLt unless UNSLOTH_ROCM_SKINNY_LINEAR=all.
_ARCHS = ("gfx1151",)


def skinny_linear_supported(device):
    """Triton on a measured gfx11 ROCm device (any gfx11 with UNSLOTH_ROCM_SKINNY_LINEAR=all)."""
    if triton is None or device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    ok = _SUPPORTED.get(index)
    if ok is None:
        try:
            arch = torch.cuda.get_device_properties(index).gcnArchName.split(":")[0]
        except Exception:
            arch = ""
        ok = _SUPPORTED[index] = arch in _ARCHS or (_ENV == "all" and arch.startswith("gfx11"))
    return ok


def skinny_linear(
    X,
    W,
    A = None,
    B = None,
    scale = 1.0,
    out = None,
    config = None,
):
    """X [M, K] @ W[N, K].T (+ scale * (X @ A[R, K].T) @ B[N, R].T) into out [M, N] (strided ok)."""
    M, K = X.shape
    N = W.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype = X.dtype, device = X.device)
    config = config or pick_config(N, K)
    bn, bk, nw, sk = config[:4]
    stages = config[4] if len(config) > 4 else 2
    has_lora = A is not None
    R = A.shape[0] if has_lora else 0
    block_r = max(16, triton.next_power_of_2(R)) if has_lora else 16
    if not has_lora:
        A = B = W
    # Every split covers whole BLOCK_K steps; the last split's tail is masked.
    k_per_split = triton.cdiv(triton.cdiv(K, sk), bk) * bk
    sk = triton.cdiv(K, k_per_split)
    partials = (
        torch.empty((sk, MAX_ROWS, N), dtype = torch.float32, device = X.device) if sk > 1 else out
    )
    grid = (triton.cdiv(N, bn), sk)
    _skinny_linear_kernel[grid](
        X,
        W,
        A,
        B,
        out,
        partials,
        M,
        N,
        K,
        R,
        X.stride(0),
        X.stride(1),
        W.stride(0),
        W.stride(1),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        out.stride(0),
        out.stride(1),
        k_per_split,
        float(scale),
        HAS_LORA = has_lora,
        SPLIT_K = sk,
        BLOCK_M = MAX_ROWS,
        BLOCK_N = bn,
        BLOCK_K = bk,
        BLOCK_R = block_r,
        num_warps = nw,
        num_stages = stages,
    )
    if sk > 1:
        _splitk_reduce_kernel[(triton.cdiv(N, 128),)](
            partials,
            out,
            M,
            N,
            out.stride(0),
            out.stride(1),
            SPLIT_K = sk,
            BLOCK_M = MAX_ROWS,
            BLOCK_N = 128,
            num_warps = 4,
        )
    return out


def skinny_decode_linear(X, W, A, B, scale, out):
    """fast_linear_forward's 16-bit decode step (X [bsz, 1, K]) through skinny_linear, or None when
    it does not apply: the caller then runs the rocBLAS path."""
    if triton is None:
        return None
    bsz = X.shape[0]
    if not (1 <= bsz <= MAX_ROWS) or X.dim() != 3 or X.shape[1] != 1:
        return None
    dtype = X.dtype
    if dtype not in (torch.bfloat16, torch.float16) or W.dtype != dtype or W.dim() != 2:
        return None
    if not routed(W.shape[0], W.shape[1]):
        return None
    if W.device != X.device or not skinny_linear_supported(X.device):
        return None
    if X.device.index is not None and X.device.index != torch.cuda.current_device():
        return None
    if torch.is_grad_enabled() and (X.requires_grad or W.requires_grad):
        return None
    if A is None:
        # No active LoRA (a base layer, or an adapter folded for this generate()): lora_S is None.
        B, scale = None, 1.0
    else:
        if (
            A.dtype != dtype
            or B.dtype != dtype
            or A.device != X.device
            or B.device != X.device
            or A.dim() != 2
            or B.dim() != 2
            or A.shape[1] != W.shape[1]
            or B.shape[0] != W.shape[0]
            or B.shape[1] != A.shape[0]
            or A.shape[0] > 128
        ):
            return None
        if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
            return None
    N = W.shape[0]
    if out is None:
        out = torch.empty((bsz, 1, N), dtype = dtype, device = X.device)
    elif (
        out.shape != (bsz, 1, N)
        or out.dtype != dtype
        or out.device != X.device
        or type(out) is not torch.Tensor
    ):
        return None
    if type(X) is not torch.Tensor:
        return None
    skinny_linear(X[:, 0, :], W, A, B, scale, out[:, 0, :])
    STATS["calls"] += 1
    return out
