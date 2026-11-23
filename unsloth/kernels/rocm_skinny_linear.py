# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Decode linears on RDNA3.5 (gfx1151) ROCm: out[M, N] = X @ W.T + s * (X @ A.T) @ B.T for
# M <= 16 rows of a 16-bit W, in one Triton pass over W. rocBLAS runs these skinny GEMMs with a 128x128
# macro tile on gfx1151, so the base GEMM, X @ A.T and the LoRA addmm each cost a full tile pass;
# here W is streamed once and the rank-r term is accumulated in fp32 beside it (one rounding to
# W's dtype at the end instead of three). When K is split (SPLIT_K > 1, the default for K >= 384), the
# fp32 partials go to a [SPLIT_K, 16, N] scratch buffer and a second small launch sums them in a fixed
# order, so the result is deterministic. On by default only where measured (gfx1151, Radeon 8060S / Strix Halo); UNSLOTH_ROCM_SKINNY_LINEAR=all
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
    # M (rows, 1 to MAX_ROWS) is not specialized: Triton would otherwise compile M == 1 and M % 16 == 0 as their
    # own variants, three per call site instead of one.
    @triton.jit(do_not_specialize = ["M"])
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

    @triton.jit(do_not_specialize = ["M"])
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


def _pow2(value, low, high):
    return low <= value <= high and value & (value - 1) == 0


def _parse_config(text):
    """BLOCK_N,BLOCK_K,num_warps,SPLIT_K[,num_stages]; anything the kernel cannot launch is ignored."""
    try:
        values = tuple(int(x) for x in text.split(","))
    except Exception:
        return None
    if len(values) not in (4, 5):
        return None
    block_n, block_k, warps, split = values[:4]
    if not (_pow2(block_n, 16, 256) and _pow2(block_k, 16, 256) and _pow2(warps, 1, 16)):
        return None
    if not 1 <= split <= 64 or (len(values) == 5 and not 1 <= values[4] <= 8):
        return None
    return values


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
    """X [M, K] @ W[N, K].T (+ scale * (X @ A[R, K].T) @ B[N, R].T) into out [M, N] (strided ok).
    Raw pointers: plain dense tensors on one device only (skinny_decode_linear checks the rest)."""
    M, K = X.shape
    N = W.shape[0]
    if not (1 <= M <= MAX_ROWS) or K < 1 or N < 1 or W.shape[1] != K:
        raise ValueError(
            f"skinny_linear: X {tuple(X.shape)} @ W {tuple(W.shape)}.T needs 1 <= M <= 16"
        )
    if A is not None and (
        B is None or A.shape[1] != K or B.shape[0] != N or B.shape[1] != A.shape[0]
    ):
        raise ValueError("skinny_linear: A must be [R, K] and B [N, R]")
    if out is not None and tuple(out.shape) != (M, N):
        raise ValueError(f"skinny_linear: out must be [{M}, {N}]")
    return _launch(X, W, A, B, scale, out, config)


def _launch(
    X,
    W,
    A,
    B,
    scale,
    out,
    config = None,
):
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


_INT32 = 2**31
_BROKEN = []
_PLAIN = (torch.Tensor, torch.nn.Parameter)
_STRIDED = torch.strided
_functorch_active = getattr(torch._C, "_are_functorch_transforms_active", lambda: False)
_forward_ad = torch.autograd.forward_ad
_increment_version = torch.autograd.graph.increment_version


def _autocast_changes(dtype):
    # The rocBLAS path's matmul / mm follow a different autocast dtype; this kernel would not.
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != dtype
    except Exception:
        return torch.is_autocast_enabled()


def skinny_decode_linear(X, W, A, B, scale, out):
    """fast_linear_forward's 16-bit decode step (X [bsz, 1, K]) through skinny_linear, or None when
    it does not apply: the caller then runs the rocBLAS path. Never raises for a rejected call.

    The kernel reads X / W / A / B through raw pointers with int32 offsets, so every operand must be a plain
    dense tensor (the exact type: a wrapper subclass such as TorchAO's NF4Tensor reports a float dtype and a
    2D shape over packed storage and only dequantizes through its own dispatch), and out must not overlap
    itself or any operand. Lazily negated views (is_neg) keep the dispatched ops: the raw storage holds the
    un-negated values. Forward-mode AD, torch.func transforms and a different autocast dtype need the
    dispatched ops. A supplied out gets its version counter bumped like an out= op, so a tensor autograd saved
    for backward still raises when backward reads it. This runs per projection per token in eager decode, so
    the checks stay arithmetic."""
    if triton is None or _BROKEN:
        return None
    if type(X) is not torch.Tensor or X.dim() != 3 or X.shape[1] != 1:
        return None
    bsz, _, K = X.shape
    if not (1 <= bsz <= MAX_ROWS):
        return None
    dtype = X.dtype
    if dtype not in (torch.bfloat16, torch.float16) or type(W) not in _PLAIN or W.dim() != 2:
        return None
    N = W.shape[0]
    if W.dtype != dtype or K < 1 or N < 1 or W.shape[1] != K or N * MAX_ROWS * 64 >= _INT32:
        return None
    if X.layout != _STRIDED or W.layout != _STRIDED or X.is_neg() or W.is_neg():
        return None
    if not routed(N, K):
        return None
    device = X.device
    if W.device != device or not skinny_linear_supported(device):
        return None
    if device.index is not None and device.index != torch.cuda.current_device():
        return None
    grad = torch.is_grad_enabled()
    if grad and (X.requires_grad or W.requires_grad):
        return None
    if (
        getattr(_forward_ad, "_current_level", -1) >= 0
        or _functorch_active()
        or _autocast_changes(dtype)
    ):
        return None
    xs0, _, xs1 = X.stride()
    ws0, ws1 = W.stride()
    # [first, last + 1) element spans; strides are never negative.
    spans = [(X, (bsz - 1) * xs0 + (K - 1) * xs1 + 1), (W, (N - 1) * ws0 + (K - 1) * ws1 + 1)]
    if A is None:
        # No active LoRA (a base layer, or an adapter folded for this generate()): lora_S is None.
        B, scale = None, 1.0
    else:
        if (
            type(A) not in _PLAIN
            or type(B) not in _PLAIN
            or type(scale) not in (int, float)
            or A.dtype != dtype
            or B.dtype != dtype
            or A.dim() != 2
            or B.dim() != 2
            or A.layout != _STRIDED
            or B.layout != _STRIDED
            or A.is_neg()
            or B.is_neg()
            or A.device != device
            or B.device != device
        ):
            return None
        R = A.shape[0]
        if A.shape[1] != K or B.shape[0] != N or B.shape[1] != R or not 1 <= R <= 128:
            return None
        if grad and (A.requires_grad or B.requires_grad):
            return None
        as0, as1 = A.stride()
        bs0, bs1 = B.stride()
        spans.append((A, (R - 1) * as0 + (K - 1) * as1 + 1))
        spans.append((B, (N - 1) * bs0 + (R - 1) * bs1 + 1))
    supplied = False
    if out is None:
        out = torch.empty((bsz, 1, N), dtype = dtype, device = device)
    else:
        if (
            type(out) is not torch.Tensor
            or out.shape != (bsz, 1, N)
            or out.dtype != dtype
            or out.layout != _STRIDED
            or out.is_neg()
            or out.device != device
            or (grad and out.requires_grad)
            or (out.is_inference() and not torch.is_inference_mode_enabled())
        ):
            return None
        os0, _, os1 = out.stride()
        # out[:, 0, :] must not write an element twice (expanded / as_strided views).
        if (bsz > 1 and os0 == 0) or (N > 1 and os1 == 0):
            return None
        if bsz > 1 and N > 1 and os0 < N * os1 and os1 < bsz * os0:
            return None
        o_span = (bsz - 1) * os0 + (N - 1) * os1 + 1
        if o_span >= _INT32:
            return None
        # Byte ranges, conservative for interleaved views: the kernel reads the operands while it writes out.
        size = X.element_size()
        o_start = out.data_ptr()
        o_end = o_start + o_span * size
        for t, span in spans:
            t_start = t.data_ptr()
            if t_start < o_end and o_start < t_start + span * size:
                return None
        # Inference tensors have no version counter.
        supplied = not out.is_inference()
    for _, span in spans:
        if span >= _INT32:
            # The kernel's pointer offsets are int32.
            return None
    try:
        _launch(X[:, 0, :], W, A, B, scale, out[:, 0, :])
    except torch.cuda.OutOfMemoryError:
        return None
    except Exception:
        # A launch that cannot compile (an unusual UNSLOTH_ROCM_SKINNY_LINEAR_CONFIG, a Triton change) fails
        # before it writes out: the rocBLAS path takes this and every later call.
        _BROKEN.append(1)
        return None
    if supplied:
        # The raw-pointer write is invisible to autograd: bump the version as torch.matmul(out = out) does.
        _increment_version(out)
    STATS["calls"] += 1
    return out
