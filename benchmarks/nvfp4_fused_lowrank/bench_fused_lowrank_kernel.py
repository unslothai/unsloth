#!/usr/bin/env python3
"""Fused NVFP4 GEMM + SVDQuant low-rank correction: portable benchmark and correctness check.

SELF-CONTAINED ON PURPOSE. This file imports nothing from its own directory so it can be dropped
onto any machine (RTX 5090, RTX 6000 PRO, DGX Spark, B200, B300, a Colab box) and run. It carries
its own copies of the two Triton kernels; ``--selftest`` compares those copies against the
development modules by ABSTRACT SYNTAX TREE when they happen to be importable, so the duplication
cannot drift silently while the prose is free to differ (the copies here are documented for a
reader with no campaign context, and comparing raw text made every kernel report as drifted over
its docstring).

WHAT IT MEASURES
----------------
SVDQuant computes ``y = alpha * (Q_A(x) Q_W(R)^T) + (x L1^T) L2^T``. The rank-r branch is cheap in
FLOPs and expensive in MEMORY: unfused it re-reads the M x K activation for the down projection and
does a read-modify-write of the M x N output for the up projection, which is why its cost is nearly
independent of r. Four ways of paying for it are timed here:

  ``unfused``    ``mm_fp4`` then ``torch.mm`` then ``torch.addmm``, which is what every public
                 datacentre-Blackwell path does today (vllm-omni, AMD Quark, diffusers Nunchaku
                 Lite).
  ``kaug``       THE PORTABLE FUSED PATH. The branch is appended to the SAME NVFP4 GEMM as extra K
                 columns: ``x' = [x | t]`` and ``W' = [W | L2]``, both NVFP4, K' = K + 64. One GEMM,
                 one output write, no kernel modification of any kind, so it runs wherever an NVFP4
                 GEMM runs. The correction is carried at fp4 precision (see ``--refine``).
  ``triton``     A hand-written Triton ``tl.dot_scaled`` GEMM that issues the rank correction as one
                 extra bf16 ``tl.dot`` into the SAME fp32 accumulator. Full bf16 precision for the
                 branch, but the Triton NVFP4 GEMM itself is well off the vendor kernel.
  ``tk``         Triton's own production library (``triton_kernels``), NVFP4 matmul, unfused
                 branch. Included as the reference for how fast Triton can be at all.

and the two activation-side kernels: the quantiser alone versus one kernel that reads x once and
emits the quantised x AND ``t = x L1^T`` (and, for ``kaug``, the quantised ``t`` in the same
buffer).

RUNNING
-------
    python bench_fused_lowrank_kernel.py --out result.json
    python bench_fused_lowrank_kernel.py --shapes zimage --ranks 32 --quick
    python bench_fused_lowrank_kernel.py --dry-run          # no GPU needed, compile only
    python bench_fused_lowrank_kernel.py --window-check     # SAFETY gate, run this first

``--window-check`` is not a benchmark. K-augmentation is only sound when a scalar ``c`` exists
with ``amax(L2)/amax(W) <= c <= amax(x)/amax(t)``; below the lower bound the builder is forced to
shrink the weight global scale, which requantises the FIRST K COLUMNS and silently replaces the
base GEMM with a different one that still runs and still looks right. The check fits both plain
form A and a Hessian-whitened A_H from a real quantisation residual, reports the window, the
achieved ``s`` and first-K bit-identity for each, and runs two controls on the guard: a gauge
rescaling that must leave the bounds unchanged, and an oversized ``L2`` that must be refused.
Run it before trusting any fused number on new hardware or a new factorisation.

Every path that is unavailable on the box is SKIPPED WITH A REASON in the JSON, never silently
replaced by a different path. Windows-safe: no fork, no POSIX-only calls, no shell-outs except an
optional ``nvidia-smi`` whose absence is tolerated.

GETTING FLASHINFER TO WORK ON sm_120 (5090, RTX 6000 PRO, Spark) -- READ THIS FIRST
-----------------------------------------------------------------------------------
On sm_120 flashinfer JITs its NVFP4 kernels as ``compute_120f``/``sm_120a``, which needs nvcc
>= 12.9. Nearly every consumer stack today (including Colab's, torch 2.11+cu128) ships nvcc 12.8,
and the failure is MISREPORTED: flashinfer logs ``Failed to get device capability: SM 12.x
requires CUDA >= 12.9`` at INFO level, then hands an EMPTY arch list to its own check and raises
the completely misleading ``RuntimeError: FlashInfer requires GPUs with sm75 or higher``. If you
see that message on a Blackwell card, the card is fine and the toolchain is the problem.

What does NOT fix it: ``pip install nvidia-cuda-nvcc-cu12``. That wheel ships ptxas, headers and
libnvvm but NO ``nvcc`` driver binary, so nothing on ``PATH`` changes. Upgrading torch to a cu130
build does not fix it either; flashinfer reads the TOOLCHAIN nvcc, not torch's.

What does fix it, verified on an RTX PRO 6000 Blackwell (sm_120) Colab VM, Ubuntu 22.04:

    apt-get install -y cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 \
        cuda-cccl-12-9 cuda-nvrtc-dev-12-9 cuda-crt-12-9 libcurand-dev-12-9 \
        libcusparse-dev-12-9 libcusolver-dev-12-9 libnvjitlink-dev-12-9
    export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
    export PATH=/usr/local/cuda-12.9/bin:$PATH
    nvcc --version        # must say 12.9 or newer
    pip install flashinfer-python==0.6.18.post1

``cuda-nvcc-12-9`` alone is not enough: it gives you the compiler but not the headers and import
libraries the JIT links against, and each missing one surfaces as a different confusing error
(``cuda_runtime.h`` not found, then ``curand_kernel.h``, then a link failure on cublasLt). Install
the whole list. ``cuda-toolkit-12-9`` also works and is simpler if you have the disk.

THE PORTABILITY POINT THAT FOLLOWS FROM THIS: the Triton paths in this file need NO such fix. On
the same VM, with nvcc 12.8 and no flashinfer at all, ``tl.dot_scaled`` compiled and ran on sm_120
(it lowers to ``mma.sync.aligned.m16n8k64...kind::mxf4nvf4``, not to the sm_100 ``tcgen05``), so
the Triton NVFP4 GEMM and the Triton quantiser are the only NVFP4 paths that work out of the box
on a stock consumer Blackwell install. They are slower than the vendor kernel where the vendor
kernel is available, and they are the difference between a working 4-bit path and none where it
is not. That is what the ``triton`` fallback rung in the dispatch chain is for.

WINDOWS. Triton is available on Windows only through the community ``triton-windows`` wheels
(``pip install triton-windows``); upstream ``triton`` publishes no ``win_amd64`` wheel. flashinfer
publishes no ``win_amd64`` wheel either, and its JIT additionally needs MSVC plus nvcc >= 12.9, so
on Windows expect the torch fallback unless you build it yourself. ``nvidia-cutlass-dsl`` is
Linux-only (manylinux wheels only). WSL2 behaves exactly like Linux for all three. Nothing in this
file is Unix-specific, so the dispatch resolves correctly on all of them; it just resolves to a
lower rung on Windows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

FP4_MAX, FP8_MAX = 6.0, 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# =============================================================================== shape catalogue
SHAPES = {
    # (label, M, K, N)
    "legacy": [
        ("legacy mlp-up   M4096", 4096, 3072, 12288),
        ("legacy mlp-dn   M4096", 4096, 12288, 3072),
        ("flux qkv        M4096", 4096, 4096, 4096),
        ("legacy mlp-up  M16384", 16384, 3072, 12288),
    ],
    "zimage": [
        ("zimg attn       M4128", 4128, 3840, 3840),
        ("zimg ffn-up     M4128", 4128, 3840, 10240),
        ("zimg ffn-dn     M4128", 4128, 10240, 3840),
        ("zimg ffn-up       M64", 64, 3840, 10240),
    ],
    "wan5b": [
        ("wan attn       M27280", 27280, 3072, 3072),
        ("wan ffn-up     M27280", 27280, 3072, 14336),
        ("wan ffn-dn     M27280", 27280, 14336, 3072),
        ("wan ffn-up      M5070", 5070, 3072, 14336),
    ],
    # A deliberately small set for consumer cards, where a 27280 x 14336 bf16 output is 782 MB and
    # four of them at once will not fit next to the operands on a 32 GB 5090.
    "small": [
        ("zimg attn       M4128", 4128, 3840, 3840),
        ("zimg ffn-up     M4128", 4128, 3840, 10240),
        ("legacy mlp-up   M4096", 4096, 3072, 12288),
        ("wan ffn-up      M5070", 5070, 3072, 14336),
    ],
}


# ===================================================================================== env report
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output = True, text = True, timeout = 30).stdout.strip()
    except Exception:                                                   # noqa: BLE001
        return None


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "is_windows": os.name == "nt",
        "is_wsl": "microsoft" in platform.release().lower(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = p.name
            env["compute_capability"] = f"sm_{p.major}{p.minor}"
            env["sm_count"] = p.multi_processor_count
            env["total_mem_gib"] = round(p.total_memory / 2**30, 1)
            env["torch_cuda"] = torch.version.cuda
    except Exception as exc:                                            # noqa: BLE001
        env["torch"] = f"unavailable: {exc}"
    for mod in ("triton", "flashinfer", "cutlass", "triton_kernels"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "present")
        except Exception as exc:                                        # noqa: BLE001
            env[mod] = f"unavailable: {type(exc).__name__}"
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    if smi:
        env["nvidia_smi"] = smi
    return env


# =========================================================================== Triton kernels (copy)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                                       # noqa: BLE001
    _HAS_TRITON = False
    triton = None

    class _TLStub:
        def jit(self, f):
            return f
    tl = None


if _HAS_TRITON:

    @triton.jit
    def _sf_offsets(row, kblk, k_pad_blocks, SWIZZLED: tl.constexpr, stride_row):
        """Byte offset of the block scale for logical (row, kblk) in either layout.

        The swizzled form is cutlass' 128x4 tiling: rows are ordered
        ``(row % 32, (row % 128) // 32)`` inside a 128-row by 4-column tile.
        """
        if SWIZZLED:
            mi = row // 128
            rem = row % 128
            r4 = rem // 32
            r32 = rem % 32
            ki = kblk // 4
            c4 = kblk % 4
            return mi * (k_pad_blocks // 4) * 512 + ki * 512 + r32 * 16 + r4 * 4 + c4
        return row * stride_row + kblk

    @triton.jit
    def _nvfp4_gemm_lowrank(
        A, B, ASF, BSF, BIAS, C, ALPHA, T, L2T,
        M, N, K,
        stride_am, stride_bn, stride_cm, stride_tm, stride_l2r,
        stride_asf, stride_bsf,
        a_k_pad_blocks, b_k_pad_blocks,
        HAS_BIAS: tl.constexpr, RANK: tl.constexpr, FOLD_ALPHA: tl.constexpr,
        A_SWIZZLED: tl.constexpr, B_SWIZZLED: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """``C = alpha * (A_fp4 @ B_fp4^T) + T @ L2T + bias`` in one launch.

        alpha belongs to the fp4 term only, so it is applied to the accumulator BEFORE the bf16
        rank dot is accumulated on top of it (``FOLD_ALPHA=False``); the alternative folds 1/alpha
        into L2 on the host and applies one trailing multiply, which is a different function
        because the folded factor is re-rounded to bf16.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

        offs_k2 = tl.arange(0, BLOCK_K // 2)
        offs_ks = tl.arange(0, BLOCK_K // 16)
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k2[None, :]
        b_ptrs = B + offs_n[None, :] * stride_bn + offs_k2[:, None]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask = mask_m[:, None], other = 0)
            b = tl.load(b_ptrs, mask = mask_n[None, :], other = 0)
            kblk = k0 * (BLOCK_K // 16) + offs_ks
            asf = tl.load(ASF + _sf_offsets(offs_m[:, None], kblk[None, :], a_k_pad_blocks,
                                            A_SWIZZLED, stride_asf),
                          mask = mask_m[:, None], other = 0)
            bsf = tl.load(BSF + _sf_offsets(offs_n[:, None], kblk[None, :], b_k_pad_blocks,
                                            B_SWIZZLED, stride_bsf),
                          mask = mask_n[:, None], other = 0)
            acc = tl.dot_scaled(a, asf.to(tl.float8e4nv, bitcast = True), "e2m1",
                                b, bsf.to(tl.float8e4nv, bitcast = True), "e2m1", acc)
            a_ptrs += BLOCK_K // 2
            b_ptrs += BLOCK_K // 2

        alpha = tl.load(ALPHA)
        if RANK > 0:
            offs_r = tl.arange(0, RANK)
            t = tl.load(T + offs_m[:, None] * stride_tm + offs_r[None, :],
                        mask = mask_m[:, None], other = 0.0)
            l2t = tl.load(L2T + offs_r[:, None] * stride_l2r + offs_n[None, :],
                          mask = mask_n[None, :], other = 0.0)
            if FOLD_ALPHA:
                acc = tl.dot(t, l2t, acc)
                acc = acc * alpha
            else:
                acc = acc * alpha
                acc = tl.dot(t, l2t, acc)
        else:
            acc = acc * alpha
        if HAS_BIAS:
            acc = acc + tl.load(BIAS + offs_n, mask = mask_n, other = 0.0).to(tl.float32)[None, :]
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :]
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _e2m1_code(v):
        """fp32 -> the 4-bit e2m1 code, round-to-nearest-even, saturating at +-6.

        Magnitude grid {0,.5,1,1.5,2,3,4,6} = codes 0..7; bit 3 is the sign. The seven thresholds are
        the midpoints; ``>`` gives ties-down (to the even code below) and ``>=`` ties-up (to the even
        code above), which is the pattern round-to-nearest-EVEN produces on this grid.
        """
        a = tl.abs(v)
        c = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
             + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
             + (a > 5.0).to(tl.int32))
        return c | tl.where(v < 0.0, 8, 0).to(tl.int32)


    @triton.jit
    def _sf_swizzled_offset(row, kblk, k_pad_blocks):
        """Flat byte index of block scale (``row``, ``kblk``) in the cutlass 128x4 swizzle."""
        mi = row // 128
        rem = row % 128
        r4 = rem // 32
        r32 = rem % 32
        ki = kblk // 4
        c4 = kblk % 4
        return mi * (k_pad_blocks // 4) * 512 + ki * 512 + r32 * 16 + r4 * 4 + c4


    @triton.jit
    def _quant_down(
        X, XQ, XSF, T, L1T, GSF,
        M, K, k_pad_blocks,
        stride_xm, stride_qm, stride_tm, stride_lk,
        RANK: tl.constexpr, HAS_LR: tl.constexpr, EMIT_T_COLS: tl.constexpr,
        KPAD_COLS: tl.constexpr, T_REPS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """One pass over ``X``: emit NVFP4 (nibbles + swizzled scales) and ``T = X @ L1T``.

        K is required to be a multiple of BLOCK_K (every 4-bit-routed layer in z-image and Wan has
        K in {1536, 3072, 3840, 4096, 12288, ...}, all multiples of 256), so only M is masked.
        """
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        gsf = tl.load(GSF)

        offs_r = tl.arange(0, RANK)
        tacc = tl.zeros((BLOCK_M, RANK), dtype = tl.float32)

        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
        l_ptrs = L1T + offs_k[:, None] * stride_lk + offs_r[None, :]
        offs_k2 = tl.arange(0, BLOCK_K // 2)
        q_ptrs = XQ + offs_m[:, None] * stride_qm + offs_k2[None, :]
        offs_kb = tl.arange(0, BLOCK_K // 16)

        NBLK: tl.constexpr = BLOCK_K // 16
        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(x_ptrs, mask = mask_m[:, None], other = 0.0)

            # --- the low-rank down projection, off the SAME registers the encode is about to use ----
            if HAS_LR:
                tacc = tl.dot(x, tl.load(l_ptrs), tacc)

            # --- the NVFP4 encode ---------------------------------------------------------------
            x3 = tl.reshape(x.to(tl.float32), (BLOCK_M, NBLK, 16))
            amax = tl.max(tl.abs(x3), axis = 2)
            # sf = e4m3(amax * gsf / 6); the cast is the hardware's round-to-nearest-even.
            sf = ((amax / 6.0) * gsf).to(tl.float8e4nv)
            step = sf.to(tl.float32) / gsf
            # An all-zero block gives step 0; flashinfer emits code 0 there, and 0/0 would be NaN.
            # DIVIDE by the step; multiplying by its reciprocal is a different rounding and
            # disagrees with flashinfer on every element that lands on an e2m1 tie.
            safe = tl.where(step > 0.0, step, 1.0)
            code = _e2m1_code(tl.where(step[:, :, None] > 0.0, x3 / safe[:, :, None], 0.0))

            lo, hi = tl.split(tl.reshape(code, (BLOCK_M, BLOCK_K // 2, 2)))
            packed = (lo | (hi << 4)).to(tl.uint8)
            tl.store(q_ptrs, packed, mask = mask_m[:, None])

            kb = k0 * NBLK + offs_kb
            tl.store(XSF + _sf_swizzled_offset(offs_m[:, None], kb[None, :], k_pad_blocks),
                     sf.to(tl.uint8, bitcast = True), mask = mask_m[:, None])

            x_ptrs += BLOCK_K
            q_ptrs += BLOCK_K // 2
            l_ptrs += BLOCK_K * stride_lk

        if HAS_LR:
            tl.store(T + offs_m[:, None] * stride_tm + offs_r[None, :],
                     tacc.to(T.dtype.element_ty), mask = mask_m[:, None])
        if EMIT_T_COLS:
            # K-AUGMENTED MODE. ``t`` is finished only now, at the end of the K loop, so its NVFP4
            # encoding is a second phase of the SAME kernel rather than a second kernel: the augmented
            # operand [x | t | 0] leaves this launch complete, having read x exactly once.
            #
            # The appended segment uses the SAME global scale factor as x, which is what keeps the
            # x half of the operand bit-identical to flashinfer's own output; ``g844_kaug.balance_factors``
            # has already scaled L1 so that t lands under that ceiling.
            _emit_t_cols(XQ, XSF, tacc, gsf, offs_m, mask_m, K, k_pad_blocks, stride_qm,
                         RANK, KPAD_COLS, T_REPS, BLOCK_M)


    @triton.jit
    def _emit_t_cols(XQ, XSF, tacc, gsf, offs_m, mask_m, K, k_pad_blocks, stride_qm,
                     RANK: tl.constexpr, KPAD_COLS: tl.constexpr, T_REPS: tl.constexpr,
                     BLOCK_M: tl.constexpr):
        """Encode ``tacc`` (and, with ``T_REPS=2``, its own fp4 residual) into the appended columns."""
        tl.static_assert(RANK * T_REPS <= KPAD_COLS, "rank * reps must fit the appended pad")
        # Zero the WHOLE appended pad first; t is written over the front of it below.
        tl.store(XQ + offs_m[:, None] * stride_qm + (K // 2 + tl.arange(0, KPAD_COLS // 2))[None, :],
                 tl.zeros((BLOCK_M, KPAD_COLS // 2), dtype = tl.uint8), mask = mask_m[:, None])
        tl.store(XSF + _sf_swizzled_offset(offs_m[:, None],
                                           (K // 16 + tl.arange(0, KPAD_COLS // 16))[None, :],
                                           k_pad_blocks),
                 tl.zeros((BLOCK_M, KPAD_COLS // 16), dtype = tl.uint8), mask = mask_m[:, None])
        NB: tl.constexpr = RANK // 16
        tp = tl.reshape(tacc, (BLOCK_M, NB, 16))
        tam = tl.max(tl.abs(tp), axis = 2)
            # SATURATE, never overflow. e4m3's maximum is 448 and this expression reaches it exactly
        # when a block of t attains the calibrated amax; a scored render on an unseen prompt can run
        # slightly hotter, and an e4m3 overflow here is a NaN block scale, i.e. a NaN correction on
        # a layer that looked fine during calibration. ``balance_factors`` already leaves two octaves
        # of headroom, so this clamp should never bind; it is here so that if it ever does, the layer
        # degrades by one clipped block instead of poisoning the render.
        tsf = tl.minimum((tam / 6.0) * gsf, 448.0).to(tl.float8e4nv)
        tstep = tsf.to(tl.float32) / gsf
        tsafe = tl.where(tstep > 0.0, tstep, 1.0)
        tcode = _e2m1_code(tl.where(tstep[:, :, None] > 0.0, tp / tsafe[:, :, None], 0.0))
        tlo, thi = tl.split(tl.reshape(tcode, (BLOCK_M, RANK // 2, 2)))
        tl.store(XQ + offs_m[:, None] * stride_qm + (K // 2 + tl.arange(0, RANK // 2))[None, :],
                 (tlo | (thi << 4)).to(tl.uint8), mask = mask_m[:, None])
        tl.store(XSF + _sf_swizzled_offset(offs_m[:, None],
                                           (K // 16 + tl.arange(0, RANK // 16))[None, :],
                                           k_pad_blocks),
                 tsf.to(tl.uint8, bitcast = True), mask = mask_m[:, None])
        if T_REPS >= 2:
            # Second-level fp4: the residual of the first encode, paired on the weight side with a
            # second copy of L2, so the two appended terms sum to (Q(t) + Q(t - Q(t))) L2^T.
            deq = _decode_e2m1(tcode) * tstep[:, :, None]
            res = tp - deq
            ram = tl.max(tl.abs(res), axis = 2)
            rsf = ((ram / 6.0) * gsf).to(tl.float8e4nv)
            rstep = rsf.to(tl.float32) / gsf
            rsafe = tl.where(rstep > 0.0, rstep, 1.0)
            rcode = _e2m1_code(tl.where(rstep[:, :, None] > 0.0, res / rsafe[:, :, None], 0.0))
            rlo, rhi = tl.split(tl.reshape(rcode, (BLOCK_M, RANK // 2, 2)))
            tl.store(XQ + offs_m[:, None] * stride_qm
                     + ((K + RANK) // 2 + tl.arange(0, RANK // 2))[None, :],
                     (rlo | (rhi << 4)).to(tl.uint8), mask = mask_m[:, None])
            tl.store(XSF + _sf_swizzled_offset(offs_m[:, None],
                                               ((K + RANK) // 16 + tl.arange(0, RANK // 16))[None, :],
                                               k_pad_blocks),
                     rsf.to(tl.uint8, bitcast = True), mask = mask_m[:, None])
        if T_REPS >= 4:
            # THE FULL TWO-LEVEL PRODUCT. Two appended blocks fix only the activation half and leave
            # the weight half at fp4, which measured ~16% relative rms on real z-image layers -- more
            # than a rank-32 correction removes, so the arm loses. Four blocks carry
            # ``(Q(t) + Q(res_t)) (Q(L2) + Q(res_L2))^T``: this side emits ``[t, res_t, t, res_t]``
            # and the weight side ``[L2, L2, res_L2, res_L2]``, so the four diagonal pairings are the
            # four terms of that product and the error becomes second order.
            tl.store(XQ + offs_m[:, None] * stride_qm
                     + ((K + 2 * RANK) // 2 + tl.arange(0, RANK // 2))[None, :],
                     (tlo | (thi << 4)).to(tl.uint8), mask = mask_m[:, None])
            tl.store(XSF + _sf_swizzled_offset(
                offs_m[:, None], ((K + 2 * RANK) // 16 + tl.arange(0, RANK // 16))[None, :],
                k_pad_blocks), tsf.to(tl.uint8, bitcast = True), mask = mask_m[:, None])
            tl.store(XQ + offs_m[:, None] * stride_qm
                     + ((K + 3 * RANK) // 2 + tl.arange(0, RANK // 2))[None, :],
                     (rlo | (rhi << 4)).to(tl.uint8), mask = mask_m[:, None])
            tl.store(XSF + _sf_swizzled_offset(
                offs_m[:, None], ((K + 3 * RANK) // 16 + tl.arange(0, RANK // 16))[None, :],
                k_pad_blocks), rsf.to(tl.uint8, bitcast = True), mask = mask_m[:, None])
        # NOTE the zeroing happens FIRST, over the whole pad, and t is written on top of it. Zeroing
        # only the tail would need ``tl.arange(0, (KPAD_COLS - RANK) // 2)``, and at rank 16 in a
        # 64-column pad that is arange(0, 24), which Triton rejects outright: arange's length must be a
        # power of two. Zeroing the full pad keeps every extent a power of two for every rank.


    @triton.jit
    def _decode_e2m1(code):
        """e2m1 code (0..15) -> its value, as a chain of selects. No lookup table, no smem."""
        a = code & 7
        v = tl.where(a == 0, 0.0, tl.where(a == 1, 0.5, tl.where(a == 2, 1.0,
            tl.where(a == 3, 1.5, tl.where(a == 4, 2.0, tl.where(a == 5, 3.0,
            tl.where(a == 6, 4.0, 6.0)))))))
        return tl.where((code & 8) != 0, -v, v)


    @triton.jit
    def _assemble(XQ_SRC, XQ_DST, SF_SRC, SF_DST, T, GSF, M, K, kpb_src, kpb_dst,
                  stride_src, stride_dst, stride_tm,
                  RANK: tl.constexpr, KPAD_COLS: tl.constexpr, T_REPS: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_K2: tl.constexpr):
        """Copy the vendor quantiser's operand into the wider augmented layout and encode the tail.

        One kernel, one pass: the fp4 nibbles and the block scales are re-laid-out and ``t`` is encoded
        into the appended columns, so the whole augmented operand costs one read and one write of the
        4-bit activation (1/4 of the bf16 activation's bytes) on top of the vendor quantiser.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_k2 = pid_k * BLOCK_K2 + tl.arange(0, BLOCK_K2)
        mask_k = offs_k2 < (K // 2)
        v = tl.load(XQ_SRC + offs_m[:, None] * stride_src + offs_k2[None, :],
                    mask = mask_m[:, None] & mask_k[None, :], other = 0)
        tl.store(XQ_DST + offs_m[:, None] * stride_dst + offs_k2[None, :], v,
                 mask = mask_m[:, None] & mask_k[None, :])

        # the block scales: only the row-block stride changes, so this is a coordinate remap
        NQ: tl.constexpr = BLOCK_K2 // 32                 # 4 k-blocks per 64 logical elements
        mi = offs_m // 128
        rem = offs_m % 128
        r4 = rem // 32
        r32 = rem % 32
        kq = pid_k * NQ + tl.arange(0, NQ)
        c4 = tl.arange(0, 4)
        mask_kq = kq < (K // 64)
        src_off = (mi[:, None, None] * (kpb_src // 4) * 512 + kq[None, :, None] * 512
                   + r32[:, None, None] * 16 + r4[:, None, None] * 4 + c4[None, None, :])
        dst_off = (mi[:, None, None] * (kpb_dst // 4) * 512 + kq[None, :, None] * 512
                   + r32[:, None, None] * 16 + r4[:, None, None] * 4 + c4[None, None, :])
        m3 = mask_m[:, None, None] & mask_kq[None, :, None]
        sv = tl.load(SF_SRC + src_off, mask = m3, other = 0)
        tl.store(SF_DST + dst_off, sv, mask = m3)

        if pid_k == 0:
            offs_r = tl.arange(0, RANK)
            t = tl.load(T + offs_m[:, None] * stride_tm + offs_r[None, :],
                        mask = mask_m[:, None], other = 0.0).to(tl.float32)
            _emit_t_cols(XQ_DST, SF_DST, t, tl.load(GSF), offs_m, mask_m, K, kpb_dst,
                         stride_dst, RANK, KPAD_COLS, T_REPS, BLOCK_M)


# ================================================================================== host helpers
def dev_guard(t):
    """Enter the tensor's own CUDA device before launching anything on it.

    flashinfer's ``fp4_gemm_cutlass.cu`` reads the STREAM from the tensor but installs no
    ``CUDADeviceGuard``. Call ``mm_fp4`` or ``nvfp4_quantize`` while the current device is a
    different one and the kernel is launched onto a stream from another context: it hangs, the card
    enters "GPU requires reset", and only a root-level reset recovers it. Three cards on the
    development host were lost to exactly this. Triton launches want the same guard, less
    catastrophically. Every launch below is wrapped, so a multi-GPU caller does not have to know.
    """
    import torch
    return torch.cuda.device(t.device)


def preflight(device = None) -> dict:
    """A tiny GUARDED mm_fp4 on each device before any real work. Cheap, and it is the check that
    would have caught the multi-device hang before it took a card down rather than after."""
    import torch
    out = {}
    if not torch.cuda.is_available():
        return {"cuda": False}
    devs = [device] if device is not None else list(range(torch.cuda.device_count()))
    for d in devs:
        dd = torch.device(f"cuda:{d}") if not isinstance(d, torch.device) else d
        rec = {"name": torch.cuda.get_device_name(dd),
               "capability": "sm_%d%d" % torch.cuda.get_device_capability(dd)}
        try:
            with torch.cuda.device(dd):
                x = torch.randn(128, 256, device = dd, dtype = torch.bfloat16) * 0.05
                w = torch.randn(128, 256, device = dd, dtype = torch.bfloat16) * 0.02
                import flashinfer as _fi
                ag, wg = global_scale(x), global_scale(w)
                aq, asf = _fi.nvfp4_quantize(x, ag, do_shuffle = False)
                wq, wsf = _fi.nvfp4_quantize(w, wg, do_shuffle = False)
                y = _fi.mm_fp4(aq, wq.T, asf, wsf.T, (1.0 / (ag * wg)).float(), torch.bfloat16,
                               out = torch.zeros(128, 128, device = dd, dtype = torch.bfloat16),
                               backend = "cutlass")
                torch.cuda.synchronize(dd)
            rec["mm_fp4"] = "ok"
            rec["checksum"] = float(y.float().abs().sum())
        except Exception as exc:                                        # noqa: BLE001
            rec["mm_fp4"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        out[str(dd)] = rec
    return out


def unswizzle_sf(sf, m: int, k: int, vec: int = 16):
    """cutlass 128x4 swizzled block scales -> the plain ``[m, k // vec]`` e4m3 matrix."""
    import torch
    kb = k // vec
    m_pad = (m + 127) // 128 * 128
    k_pad = (kb + 3) // 4 * 4
    flat = sf.reshape(-1).view(torch.float8_e4m3fn)
    v = flat.reshape(m_pad // 128, k_pad // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(m_pad, k_pad)
    return v[:m, :kb].contiguous()


def global_scale(t):
    return (FP4_MAX * FP8_MAX / t.float().abs().amax().clamp(min = 1e-8)).reshape(1).to(t.device)


def swizzle_sf(sf_lin, m: int, k: int):
    """Plain ``[m, k//16]`` e4m3 block scales -> the cutlass 128x4 swizzled flat buffer."""
    import torch
    kb = k // 16
    m_pad = (m + 127) // 128 * 128
    k_pad = (kb + 3) // 4 * 4
    full = torch.zeros(m_pad, k_pad, device = sf_lin.device, dtype = torch.uint8)
    full[:m, :kb] = sf_lin.view(torch.uint8)
    v = full.reshape(m_pad, k_pad // 4, 4).reshape(m_pad // 128, 4, 32, k_pad // 4, 4)
    return v.permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


def torch_nvfp4_quantize(x, gsf):
    """A pure-torch ``flashinfer.nvfp4_quantize(x, gsf, do_shuffle=False)``.

    Exists so this benchmark can produce a full column on a box where flashinfer's NVFP4 JIT will
    not build -- which is the normal case on sm_120 today, because flashinfer needs CUDA >= 12.9 to
    emit ``compute_120f`` and most consumer stacks still ship 12.8. It reproduces the same
    definition the Triton kernel does: ``sf = e4m3((amax/6) * gsf)``, ``step = sf/gsf``,
    ``code = e2m1_rne(x/step)``, low nibble first.

    HOW CLOSE IT ACTUALLY IS, measured (``--selftest``, group C1), not assumed. An earlier version
    of this docstring called it "a verified stand-in and not a second opinion". That was wrong, and
    nothing checked it, because ``--selftest`` was advertised but unimplemented. Measured against
    ``flashinfer.nvfp4_quantize`` on four tensors on sm_100:

      * block scales: bit-identical, always. The scale path is exact in both.
      * nibbles: 0.02% to 0.06% of them differ. EVERY differing element sits exactly on an e2m1
        round-to-nearest-even tie point under this function's fp32 arithmetic, the code moves by
        exactly one step, the sign never flips, and the reconstruction rms error against the
        original tensor is equal to seven significant figures.
      * the tie RULE is not the cause and is not in doubt: fed a block containing each of the seven
        exact midpoints {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0} with ``step == 1`` exactly, this
        function and flashinfer agree on all seven and both round half to even.

    What is left is ULP noise in the per-element ratio, and it is not removable by reassociating:
    ``x/(sf/gsf)``, ``x*(gsf/sf)``, ``(x*gsf)/sf`` and ``x*reciprocal(sf/gsf)`` were all tried and
    none is uniformly closer than another (totals 9236, 9614, 10004 and 7163 differing nibbles over
    the same four tensors).

    So: a FAITHFUL stand-in, equal in reconstruction error, NOT a bit-exact one. If you need
    bit-exactness against the vendor operand, use the Triton quantiser in this file, which IS
    bit-identical to flashinfer (0 differing bytes on the same four tensors here, and 0 of
    6291456 nibbles per case including a tie-stress case in outputs/g844_quant_verify.json).
    """
    import torch
    M, K = x.shape
    xb = x.float().reshape(M, K // 16, 16)
    sf = (((xb.abs().amax(-1)) / 6.0) * gsf.float()).to(torch.float8_e4m3fn)
    step = (sf.float() / gsf.float()).unsqueeze(-1)
    ratio = torch.where(step > 0, xb / torch.where(step > 0, step, torch.ones_like(step)),
                        torch.zeros_like(xb))
    a = ratio.abs()
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())
    code = code | torch.where(ratio < 0, 8, 0)
    code = code.reshape(M, K // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return packed, swizzle_sf(sf, M, K)


def sf_buffer_numel(m: int, k: int) -> int:
    kb = k // 16
    return ((m + 127) // 128 * 128) * ((kb + 3) // 4 * 4)


def triton_quant_down(x, gsf, l1t = None, *, kaug_cols: int = 0, t_reps: int = 1, cfg = None):
    """``(xq, x_sf, t)``. With ``kaug_cols`` the output is the AUGMENTED operand, K' = K+kaug_cols."""
    import torch
    assert x.is_contiguous() and x.dim() == 2
    M, K = x.shape
    cfg = dict(dict(BLOCK_M = 32, BLOCK_K = 256, num_warps = 4, num_stages = 3)
               if cfg is None else cfg)
    assert K % cfg["BLOCK_K"] == 0, f"K={K} must be a multiple of BLOCK_K={cfg['BLOCK_K']}"
    rank = 0 if l1t is None else l1t.shape[1]
    Kout = K + kaug_cols
    xq = torch.empty((M, Kout // 2), device = x.device, dtype = torch.uint8)
    x_sf = torch.zeros(sf_buffer_numel(M, Kout), device = x.device, dtype = torch.uint8)
    t = (torch.empty((M, rank), device = x.device, dtype = torch.bfloat16) if rank else None)
    k_pad = ((Kout // 16) + 3) // 4 * 4
    grid = (triton.cdiv(M, cfg["BLOCK_M"]),)
    with dev_guard(x):
        _quant_down[grid](
            x, xq, x_sf, t if t is not None else xq, l1t if l1t is not None else xq, gsf,
            M, K, k_pad,
            x.stride(0), xq.stride(0), t.stride(0) if t is not None else 0,
            l1t.stride(0) if l1t is not None else 0,
            RANK = max(rank, 16), HAS_LR = rank > 0,
            EMIT_T_COLS = kaug_cols > 0, KPAD_COLS = kaug_cols, T_REPS = t_reps,
            BLOCK_M = cfg["BLOCK_M"], BLOCK_K = cfg["BLOCK_K"],
            num_warps = cfg["num_warps"], num_stages = cfg["num_stages"])
    return xq, x_sf, t


def triton_assemble_kaug(xq, x_sf, t, gsf, M: int, K: int, *, kaug_cols: int = 64,
                         block_m: int = 128, t_reps: int = 1):
    """Widen a VENDOR-quantised activation into the K-augmented operand, in one Triton pass.

    This is the shipped activation path, and it exists because fusing the down projection into a
    Triton quantiser lost: flashinfer's hand-written quantiser is 3-5x faster than anything Triton
    emits for these shapes, and that deficit was larger than the up projection the fusion saves.
    Here the vendor quantiser keeps its own speed, ``t = x @ L1^T`` is a plain bf16 ``mm``, and this
    kernel only re-lays-out the 4-bit payload (the swizzle differs from the unaugmented one solely
    in the row-block stride) and encodes ``t`` into the appended columns. The x half is therefore
    bit-identical to the shipped operand BY CONSTRUCTION, not by verification.
    """
    import torch
    import triton
    kb, kb2 = K // 16, (K + kaug_cols) // 16
    kpad, kpad2 = (kb + 3) // 4 * 4, (kb2 + 3) // 4 * 4
    m_pad = (M + 127) // 128 * 128
    xq = xq.view(torch.uint8)
    xq_aug = torch.empty((M, (K + kaug_cols) // 2), device = xq.device, dtype = torch.uint8)
    sf_aug = torch.zeros(m_pad * kpad2, device = xq.device, dtype = torch.uint8)
    BK2 = 128
    grid = (triton.cdiv(M, block_m), triton.cdiv(K // 2, BK2))
    with dev_guard(xq):
        _assemble[grid](xq, xq_aug, x_sf.view(torch.uint8).reshape(-1), sf_aug, t, gsf, M, K,
                        kpad, kpad2, xq.stride(0), xq_aug.stride(0), t.stride(0),
                        RANK = t.shape[1], KPAD_COLS = kaug_cols, T_REPS = t_reps,
                        BLOCK_M = block_m, BLOCK_K2 = BK2, num_warps = 8, num_stages = 3)
    return xq_aug, sf_aug


def triton_mm_lowrank(aq, a_sf, wq, w_sf, alpha, *, bias = None, t = None, l2t = None,
                      a_swizzled = True, b_swizzled = False, fold_alpha = False,
                      out = None, cfg = None):
    import torch
    M, K2 = aq.shape
    N = wq.shape[0]
    K = K2 * 2
    cfg = dict(dict(BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256, GROUP_M = 8,
                    num_warps = 4, num_stages = 3) if cfg is None else cfg)
    rank = 0 if t is None else t.shape[1]
    if out is None:
        out = torch.empty((M, N), device = aq.device, dtype = torch.bfloat16)

    def pad_blocks(swz):
        kb = K // 16
        return ((kb + 3) // 4 * 4) if swz else kb
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    d = aq
    with dev_guard(aq):
        _nvfp4_gemm_lowrank[grid](
            aq, wq, a_sf.view(torch.uint8).reshape(-1), w_sf.view(torch.uint8).reshape(-1),
            bias if bias is not None else d, out, alpha,
            t if t is not None else d, l2t if l2t is not None else d,
            M, N, K, aq.stride(0), wq.stride(0), out.stride(0),
            t.stride(0) if t is not None else 0, l2t.stride(0) if l2t is not None else 0,
            0 if a_swizzled else K // 16, 0 if b_swizzled else K // 16,
            pad_blocks(a_swizzled), pad_blocks(b_swizzled),
            HAS_BIAS = bias is not None, RANK = rank, FOLD_ALPHA = fold_alpha,
            A_SWIZZLED = a_swizzled, B_SWIZZLED = b_swizzled,
            BLOCK_M = cfg["BLOCK_M"], BLOCK_N = cfg["BLOCK_N"], BLOCK_K = cfg["BLOCK_K"],
            GROUP_M = cfg["GROUP_M"], num_warps = cfg["num_warps"],
            num_stages = cfg["num_stages"])
    return out


S_FLOOR = 1e-3
# Headroom on the appended activation block's e4m3 scale. ``(block_amax/6) * a_gsf`` reaches
# e4m3's maximum of 448 exactly when a block of t attains the calibrated amax, so calibrating t to
# fill the range leaves none, and a scored render on an unseen prompt that runs a hair hotter
# turns the block scale into a NaN. ``p`` only positions the per-block scale; the precision inside
# a block comes from that block's own scale, so two octaves of headroom cost nothing.
T_HEADROOM = 4.0


def balance_factors(l1, l2, amax_x: float, amax_t: float, amax_w: float,
                    s_floor: float = S_FLOOR):
    """Rebalance ``(L1, L2)`` so both appended halves fit their NVFP4 ceilings. Exact in real math.

    ``mm_fp4`` computes ``alpha * sum(code_a sf_a)(code_b sf_b)`` with ``alpha = 1/(a_gsf w_gsf)``,
    which is exactly ``sum(deq_a deq_b)``: alpha is not free, it is the inverse of the scale already
    inside the block scales. So the appended columns contribute ``sum(deq_t deq_L2)`` and the only
    way to make that equal ``t L2^T`` is for the STORED factors to already be the right size. The
    ceilings are ``|deq| <= 6*448/gsf``, i.e. ``amax(x)`` on the activation side and ``amax(W)`` on
    the weight side. A rank-r factorisation is defined only up to ``(L2 c)(L1 / c)`` and that
    freedom is exactly what is spent here: ``p`` puts ``t`` just under the activation ceiling, and
    ``s`` lowers the WEIGHT global scale to buy whatever headroom L2 still needs. Lowering
    ``w_gsf`` is free in relative precision (e4m3 has 2.3e5 of dynamic range) and only costs
    bit-identity of the weight half, which is rebuilt offline anyway; ``a_gsf`` never moves, so the
    ACTIVATION half -- the half produced on every call -- stays bit-identical.
    """
    p = float(amax_x / max(amax_t * T_HEADROOM, 1e-30))
    l1p = (l1.float() * p).to(l1.dtype)
    l2p = (l2.float() / p).to(l2.dtype)
    s = min(1.0, float(amax_w) / max(float(l2p.float().abs().amax()), 1e-30))
    # e4m3's smallest subnormal is 2^-9. A block scale is amax_block*gsf*s/6, so s may drop three
    # orders of magnitude before the smallest weight blocks flush to ZERO -- which deletes them
    # from the GEMM rather than rounding them. Below the floor the layer is NOT correctable this
    # way and the caller must leave it uncorrected. Observed live: one z-image layer asked for
    # s = 9.4e-08 while every timing looked normal.
    return l1p, l2p, s, bool(s >= s_floor)


def balance_factors_percol(l1, l2, amax_x: float, amax_t_cols, amax_w: float,
                           s_floor: float = S_FLOOR, headroom: float = T_HEADROOM):
    """``(l1p, l2p, s, ok)`` with a PER-COLUMN split of the factors instead of one global scalar.

    A rank-r factorisation's columns follow the singular values and span one to two orders of
    magnitude on real layers, while NVFP4 gives the appended block ONE e4m3 scale per 16 columns.
    A single global rebalance therefore leaves that whole spread inside one block and the trailing
    columns round to the coarsest fp4 levels. Folding a diagonal into L1 and its inverse into L2
    changes no function; choosing ``d_j = sqrt(amax(L2_j)/amax(t_j))`` leaves each side carrying
    the SQUARE ROOT of the spread. ``amax(t_j)`` is a per-column calibration statistic.
    """
    import torch
    r = l1.shape[0]
    at = amax_t_cols.float().reshape(r).clamp(min = 1e-30)
    al2 = l2.float().abs().amax(dim = 0).reshape(r).clamp(min = 1e-30)
    d = (al2 / at).sqrt()
    c = float(amax_x / headroom) / float((at * d).max().clamp(min = 1e-30))
    d = d * c
    l1p = (l1.float() * d.reshape(r, 1)).to(l1.dtype)
    l2p = (l2.float() / d.reshape(1, r)).to(l2.dtype)
    s = min(1.0, float(amax_w) / max(float(l2p.float().abs().amax()), 1e-30))
    return l1p, l2p, s, bool(s >= s_floor)


def balance_windowed_percol(l1, l2, amax_x: float, amax_t_cols, amax_w: float,
                            s_floor: float = S_FLOOR, headroom: float = T_HEADROOM):
    """The builder that RESPECTS the window. ``(l1p, l2p, s, ok, info)``.

    ``balance_factors_percol`` above places ``c`` at the activation bound divided by the headroom
    and then accepts whatever ``s`` falls out. That is a BUG, and it is the one this campaign
    shipped for a while: on z-image it produced ``s = 0.25``, which requantises the first K columns
    of the augmented weight and silently replaces the base GEMM with a different one.

    The fix is to treat the two bounds as a window and refuse when it is empty:

        lo = G / amax(w)      with   G = max_j sqrt(amax(L2_j) amax(t_j))
        hi = amax(x) / G

    ``c`` is then placed at ``max(min(hi / headroom, hi), lo)`` -- as much activation headroom as
    the window allows, but never below ``lo``, because bit-identity of the base GEMM outranks
    headroom. When ``lo > hi`` there is NO scalar that keeps both halves in range and the function
    returns ``ok = False`` with a reason instead of a plausible-looking factorisation.
    """
    import torch
    r = l1.shape[0]
    at = amax_t_cols.float().reshape(r).clamp(min = 1e-30)
    al2 = l2.float().abs().amax(dim = 0).reshape(r).clamp(min = 1e-30)
    g = float((al2 * at).sqrt().max())
    lo = g / max(float(amax_w), 1e-30)
    hi = float(amax_x) / max(g, 1e-30)
    info = {"c_lo": lo, "c_hi": hi, "G": g,
            "log10_width": (math.log10(hi / lo) if lo > 0 and hi > lo else None)}
    if lo > hi:
        info["reason"] = ("empty c window: amax(L2)*amax(t) > amax(w)*amax(x), so no scalar keeps "
                          "the appended weight columns inside w_gsf AND the appended activation "
                          "columns inside a_gsf. Take the unfused branch.")
        return None, None, 0.0, False, info
    c = max(min(hi / headroom, hi), lo)
    d = (al2 / at).sqrt() * c
    l1p = (l1.float() * d.reshape(r, 1)).to(l1.dtype)
    l2p = (l2.float() / d.reshape(1, r)).to(l2.dtype)
    s = min(1.0, float(amax_w) / max(float(l2p.float().abs().amax()), 1e-30))
    info["c"] = c
    info["s"] = s
    if s < 1.0:
        info["reason"] = f"s={s:.4g} < 1 would requantise the first K columns"
    return l1p, l2p, s, bool(s >= s_floor), info


def build_kaug_weight(w, l2p, w_gsf, s: float, pad: int, quantise, refine: int = 1):
    """``W' = [W | L2' | 0]`` quantised with ``w_gsf * s``. Returns (wq, wsf, w_gsf_aug, 1/s).

    ``refine=2`` writes ``[L2', L2', res, res]`` instead, pairing with an activation side of
    ``[t, res_t, t, res_t]`` so the four diagonal products sum to the full two-level fp4 product
    ``(Q(t)+Q(res_t))(Q(L2)+Q(res_L2))^T``. It needs ``4r <= pad``.
    """
    import torch
    N, K = w.shape
    r = l2p.shape[1]
    w_gsf_aug = (w_gsf.float() * s).to(w_gsf.dtype)
    wa = torch.zeros(N, K + pad, device = w.device, dtype = w.dtype)
    wa[:, :K] = w
    wa[:, K:K + r] = l2p.to(w.dtype)
    if refine >= 2:
        assert 4 * r <= pad, f"refine=2 needs 4r={4 * r} <= pad={pad}"
        q1, sf1 = quantise(wa[:, K:K + r].contiguous(), w_gsf_aug)
        res = (l2p.float() - dequant_nvfp4(q1, sf1, w_gsf_aug, N, r)).to(w.dtype)
        wa[:, K + r:K + 2 * r] = l2p.to(w.dtype)
        wa[:, K + 2 * r:K + 3 * r] = res
        wa[:, K + 3 * r:K + 4 * r] = res
    wq, wsf = quantise(wa, w_gsf_aug)
    return wq, wsf, w_gsf_aug, 1.0 / s


def dequant_nvfp4(q, sf_swz, gsf, rows: int, k: int):
    """flashinfer's operand -> fp32, through the verified 128x4 decoder."""
    import torch
    lut = torch.tensor(list(E2M1_GRID) + [-v for v in E2M1_GRID], device = q.device,
                       dtype = torch.float32)
    kb = k // 16
    sf = unswizzle_sf(sf_swz, rows, k).view(torch.float8_e4m3fn).float()
    qq = q.view(torch.uint8)
    vals = torch.stack((lut[(qq & 0x0F).long()], lut[(qq >> 4).long()]), -1).reshape(rows, kb, 16)
    return (vals * (sf / gsf.float()).reshape(rows, kb, 1)).reshape(rows, k)


# ======================================================================================= timing
def timed(fn, warm: int, iters: int, graph: bool = True):
    """Time ``fn``, by CUDA-GRAPH REPLAY where possible, with the eager number kept beside it.

    One event pair around one eager call does not measure a small Triton kernel: it measures the
    Python launch path in front of it. That path costs 34-46 us on this build, and it does not
    depend on the problem size, so a kernel whose real cost runs from 4 us to 20 us across a
    256-fold change in rows reports a FLAT 41 us and looks like a fixed overhead that dwarfs the
    GEMM. Measured directly on the assemble kernel:

        M=    64   eager 0.0455 ms   graph 0.0042 ms
        M=  4128   eager 0.0459 ms   graph 0.0082 ms
        M= 27280   eager 0.0538 ms   graph 0.0205 ms

    Only the graph column scales with M, and only the graph column is what deployment pays: the
    transformer is captured with g833_graph.GraphedForward, so the Python launch path is executed
    once at capture and never again. Back-to-back eager launches do NOT fix this -- the CPU still
    cannot enqueue faster than ~34 us, so the amortised number is just as flat.

    ``graph_ms`` is the headline; ``eager_ms`` is retained because a caller that does NOT capture
    graphs really does pay it, and because a large gap between the two is the signature of a
    launch-bound kernel.
    """
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing = True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing = True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b) for a, b in zip(st, en))
    out = {"eager_min_ms": ms[0], "eager_p50_ms": ms[len(ms) // 2],
           "min_ms": ms[0], "p50_ms": ms[len(ms) // 2], "p90_ms": ms[int(len(ms) * 0.9)],
           "timed_by": "eager"}
    if not graph:
        return out
    try:
        g = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(g):
            fn()
        for _ in range(max(3, warm)):
            g.replay()
        torch.cuda.synchronize()
        gv = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            for _ in range(10):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            gv.append(a.elapsed_time(b) / 10.0)
        gv.sort()
        out.update({"graph_min_ms": gv[0], "graph_p50_ms": gv[len(gv) // 2],
                    "min_ms": gv[0], "p50_ms": gv[len(gv) // 2],
                    "p90_ms": gv[int(len(gv) * 0.9)], "timed_by": "cuda_graph",
                    "eager_over_graph": (out["eager_p50_ms"] / gv[len(gv) // 2]
                                         if gv[len(gv) // 2] > 0 else None)})
    except Exception as exc:                                            # noqa: BLE001
        out["graph_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return out


# ======================================================================================= dry run
def _dry_run_jobs():
    """(name, kernel, signature, constexprs) for every kernel this file can compile."""
    gemm_sig = {
        "A": "*u8", "B": "*u8", "ASF": "*u8", "BSF": "*u8", "BIAS": "*u8", "C": "*bf16",
        "ALPHA": "*fp32", "T": "*bf16", "L2T": "*bf16", "M": "i32", "N": "i32", "K": "i32",
        "stride_am": "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_tm": "i32",
        "stride_l2r": "i32", "stride_asf": "i32", "stride_bsf": "i32",
        "a_k_pad_blocks": "i32", "b_k_pad_blocks": "i32",
        "HAS_BIAS": "constexpr", "RANK": "constexpr", "FOLD_ALPHA": "constexpr",
        "A_SWIZZLED": "constexpr", "B_SWIZZLED": "constexpr", "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "GROUP_M": "constexpr"}
    quant_sig = {
        "X": "*bf16", "XQ": "*u8", "XSF": "*u8", "T": "*bf16", "L1T": "*bf16", "GSF": "*fp32",
        "M": "i32", "K": "i32", "k_pad_blocks": "i32", "stride_xm": "i32", "stride_qm": "i32",
        "stride_tm": "i32", "stride_lk": "i32", "RANK": "constexpr", "HAS_LR": "constexpr",
        "EMIT_T_COLS": "constexpr", "KPAD_COLS": "constexpr", "T_REPS": "constexpr",
        "BLOCK_M": "constexpr", "BLOCK_K": "constexpr"}
    return {
        "gemm_rank0": (_nvfp4_gemm_lowrank, gemm_sig,
                       dict(HAS_BIAS = False, RANK = 0, FOLD_ALPHA = False, A_SWIZZLED = True,
                            B_SWIZZLED = False, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256,
                            GROUP_M = 8)),
        "gemm_rank32": (_nvfp4_gemm_lowrank, gemm_sig,
                        dict(HAS_BIAS = True, RANK = 32, FOLD_ALPHA = False, A_SWIZZLED = True,
                             B_SWIZZLED = False, BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 256,
                             GROUP_M = 8)),
        "quant_down_rank32": (_quant_down, quant_sig,
                              dict(RANK = 32, HAS_LR = True, EMIT_T_COLS = False,
                                   KPAD_COLS = 0, T_REPS = 1, BLOCK_M = 32, BLOCK_K = 256)),
        "quant_down_kaug32": (_quant_down, quant_sig,
                              dict(RANK = 32, HAS_LR = True, EMIT_T_COLS = True,
                                   KPAD_COLS = 64, T_REPS = 1, BLOCK_M = 32, BLOCK_K = 256)),
        "assemble_kaug32": (_assemble, {
            "XQ_SRC": "*u8", "XQ_DST": "*u8", "SF_SRC": "*u8", "SF_DST": "*u8", "T": "*bf16",
            "GSF": "*fp32", "M": "i32", "K": "i32", "kpb_src": "i32", "kpb_dst": "i32",
            "stride_src": "i32", "stride_dst": "i32", "stride_tm": "i32",
            "RANK": "constexpr", "KPAD_COLS": "constexpr", "T_REPS": "constexpr",
            "BLOCK_M": "constexpr", "BLOCK_K2": "constexpr"},
            dict(RANK = 32, KPAD_COLS = 128, T_REPS = 4, BLOCK_M = 128, BLOCK_K2 = 128)),
    }


def dry_run_one(name: str, arch: int) -> dict:
    """Compile ONE kernel for ONE arch and report. Run in its own process, deliberately."""
    import re
    from triton.backends.compiler import GPUTarget
    fn, sig, cx = _dry_run_jobs()[name]
    src = triton.compiler.ASTSource(fn = fn, signature = sig, constexprs = cx)
    k = triton.compile(src, target = GPUTarget("cuda", arch, 32))
    ptx = k.asm["ptx"]
    mma = sorted(set(re.findall(
        r"^\s*(?:@%\w+\s+)?(tcgen05\.mma[^\s;]*|mma\.sync[^\s;]*|wgmma[^\s;]*)", ptx, re.M)))
    return {"compiled": True, "mma": mma, "shared_bytes": k.metadata.shared,
            "num_warps": k.metadata.num_warps}


def dry_run(args) -> dict:
    """Compile-only check for staging CI. No GPU, no allocation.

    EACH (kernel, arch) RUNS IN ITS OWN SUBPROCESS. This is not defensive style: Triton 3.6/3.7 on
    some targets does not merely raise for an unsupported ``tl.dot_scaled``, it trips an MLIR
    assertion (``DenseElementsAttr::get ... isIntOrIndex()``), which under an assert-enabled build
    aborts the process outright. An in-process loop therefore reports nothing at all for every
    target after the first unsupported one, which is exactly the failure a portability matrix must
    not have. A crashed child is recorded as ``compiled: false`` with its return code and stderr
    tail; ``crashed_process`` distinguishes an abort or signal from an ordinary exception, and it
    is written to be right on Windows too, where an abort surfaces as a large positive status
    rather than as a negative one.

    MEASURED, triton 3.7.1 / torch 2.12.1+cu130, ``outputs/nvfp4_fused_lowrank_kernel/dryrun_current.json``:

      * sm_100, sm_103: everything compiles. The GEMM issues
        ``tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`` and, at rank > 0, a
        second ``tcgen05.mma...kind::f16`` into the same accumulator.
      * sm_120: everything compiles. The GEMM issues
        ``mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::4X`` and, at rank > 0,
        ``mma.sync.aligned.m16n8k16...f32.bf16.bf16.f32``.
      * sm_121: the QUANTISER and the ASSEMBLE pass compile; both GEMM variants FAIL. The failure
        is specific and worth naming, because it is a Triton limitation and not a property of the
        hardware: the pipeline dies in ``TritonGPUAccelerateMatmul`` on the e2m1
        ``tt.dot_scaled``, with the assertion above, before ptxas is ever reached. On this build
        it surfaces as return code 1 rather than as an abort.

    The consequence for a DGX Spark is the useful part: K-augmentation does not need the Triton
    GEMM at all -- it needs a vendor NVFP4 GEMM plus the assemble pass, and the assemble pass is
    one of the two things that DOES compile there. So the kaug path is expected to work on sm_121
    as soon as a vendor NVFP4 GEMM is available for it, and only the (fallback-rung) hand-written
    Triton GEMM is blocked. That is an inference from the compile matrix, not a measurement on
    Spark hardware, which nobody on this campaign has.
    """
    out = {"mode": "dry-run", "env": environment(), "targets": {}}
    if not _HAS_TRITON:
        out["error"] = "triton not importable; nothing to compile"
        return out
    archs = [int(a) for a in args.dry_run_archs.split(",")]
    for name in _dry_run_jobs():
        out["targets"][name] = {}
        for arch in archs:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--dry-run-one", name,
                 "--dry-run-arch", str(arch)],
                capture_output = True, text = True, timeout = 600)
            tag = f"sm_{arch}"
            if proc.returncode == 0 and "G844_DRY_JSON " in proc.stdout:
                out["targets"][name][tag] = json.loads(
                    proc.stdout.split("G844_DRY_JSON ", 1)[1].splitlines()[0])
            else:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                out["targets"][name][tag] = {
                    "compiled": False, "returncode": proc.returncode,
                    "error": " | ".join(err[-4:])[:500] or "no output",
                    "crashed_process": proc.returncode < 0 or proc.returncode > 1}
    return out


# ========================================================================================= bench
def bench_shape(label, M, K, N, rank, args, caps):
    """Every variant for one (M, K, N, rank). Each entry is a time or an explicit skip reason.

    Works with OR WITHOUT flashinfer. Without it the NVFP4 operands come from this file's own
    kernels (the Triton quantiser for activations, the torch reference for the offline weight
    build) and ``mm_fp4`` is skipped with a reason. That is not a nicety: on sm_120 today
    flashinfer refuses to JIT its NVFP4 path unless the toolchain is CUDA >= 12.9, and most
    consumer stacks still ship 12.8, so a benchmark that needs flashinfer produces an empty column
    on exactly the hardware the portability question is about.
    """
    import torch
    import torch.nn.functional as F
    res = {"label": label, "M": M, "K": K, "N": N, "rank": rank,
           "gflop": 2.0 * M * K * N / 1e9, "times": {}, "skipped": {}, "correctness": {},
           "notes": {}}
    warm, iters = (5, 20) if args.quick else (30, 120)
    dev = "cuda"
    g = torch.Generator(device = dev).manual_seed(3407)
    x = (torch.randn(M, K, device = dev, dtype = torch.bfloat16, generator = g) * 0.05).contiguous()
    w = (torch.randn(N, K, device = dev, dtype = torch.bfloat16, generator = g) * 0.02)
    bias = torch.randn(N, device = dev, dtype = torch.bfloat16, generator = g)
    pad = args.kaug_cols

    def skip(name, why):
        res["skipped"][name] = why

    # ---- bf16 and fp8, always available ------------------------------------------------------
    res["times"]["bf16"] = timed(lambda: F.linear(x, w, bias), warm, iters)
    try:
        wrow = (w.abs().amax(1, keepdim = True).float() / FP8_MAX).clamp(min = 1e-8)
        w8 = (w / wrow.to(w.dtype)).to(torch.float8_e4m3fn)
        xs = (x.abs().amax(1, keepdim = True).float() / FP8_MAX).clamp(min = 1e-8)
        x8 = (x / xs.to(x.dtype)).to(torch.float8_e4m3fn)
        res["times"]["fp8_w8a8_gemm"] = timed(
            lambda: torch._scaled_mm(x8, w8.t(), scale_a = xs, scale_b = wrow.t(),
                                     bias = bias, out_dtype = torch.bfloat16), warm, iters)
    except Exception as exc:                                            # noqa: BLE001
        skip("fp8_w8a8_gemm", f"{type(exc).__name__}: {str(exc)[:160]}")

    if not _HAS_TRITON and not caps["flashinfer"]:
        skip("nvfp4_all", "neither triton nor flashinfer is importable")
        return res

    # ---- the quantiser used to BUILD operands (offline in deployment) -------------------------
    fi = None
    if caps["flashinfer"]:
        try:
            import flashinfer as _fi
            _fi.nvfp4_quantize(x[:16].contiguous(), global_scale(x), do_shuffle = False)
            fi = _fi
        except Exception as exc:                                        # noqa: BLE001
            skip("flashinfer_runtime",
                 f"importable but unusable here: {type(exc).__name__}: {str(exc)[:200]}")
    build_q = ((lambda t, gs: fi.nvfp4_quantize(t, gs, do_shuffle = False)) if fi is not None
               else torch_nvfp4_quantize)
    res["notes"]["operand_builder"] = "flashinfer.nvfp4_quantize" if fi else "torch reference"

    ag, wg = global_scale(x), global_scale(w)
    alpha = (1.0 / (ag * wg)).float()
    aq, asf = build_q(x, ag)
    wq, wsf = build_q(w, wg)
    aq = aq.view(torch.uint8)
    wq = wq.view(torch.uint8)
    asf_u = asf.view(torch.uint8).reshape(-1)
    wsf_u = wsf.view(torch.uint8).reshape(-1)
    ob = torch.zeros(M, N, device = dev, dtype = torch.bfloat16)

    # ---- reference: fp32 of the exact dequantised computation --------------------------------
    a_deq = dequant_nvfp4(aq, asf_u, ag, M, K)
    w_deq = dequant_nvfp4(wq, wsf_u, wg, N, K)
    ref_fp4_exact = a_deq @ w_deq.t()

    # SVDQuant error-correction factors of the REAL quantisation error, not random matrices:
    # a random L2 makes the "correction" the size of the main term and every relative error in the
    # table meaningless.
    E = (w.float() - w_deq)
    try:
        torch.manual_seed(3407)
        q_ = min(rank + 8, min(E.shape))
        U, S, V = torch.svd_lowrank(E, q = q_, niter = 4)
        l2 = (U[:, :rank] * S[:rank]).contiguous()
        l1 = V[:, :rank].t().contiguous()
    except Exception as exc:                                            # noqa: BLE001
        skip("nvfp4_all", f"svd_lowrank failed: {type(exc).__name__}: {str(exc)[:160]}")
        return res
    amax_x = float(x.float().abs().amax())
    amax_w = float(w.float().abs().amax())
    t_raw = (x.float() @ l1.t())
    l1p, l2p, s_bal, s_ok = balance_factors(l1.to(torch.bfloat16), l2.to(torch.bfloat16),
                                            amax_x, float(t_raw.abs().amax()), amax_w)
    res["notes"]["balance_feasible"] = s_ok
    l1t = l1p.t().contiguous()                                          # (K, r)
    l2t = l2p.t().contiguous()                                          # (r, N)
    t_ref = (x.float() @ l1t.float())
    exact_corr = t_ref @ l2p.float().t()
    ref32 = ref_fp4_exact + exact_corr
    res["notes"]["balance_s"] = s_bal
    if not s_ok:
        skip("kaug", f"no feasible factor split: s={s_bal:.3g} below the e4m3 floor {S_FLOOR:g}")
    res["notes"]["E_over_W_fro"] = float(E.norm() / w.float().norm())

    def err(y, ref):
        d = (y.float() - ref).abs()
        return {"max_abs": float(d.max()),
                "max_rel": float(d.max() / ref.abs().max().clamp(min = 1e-12)),
                "rms_rel": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())}

    # ---- mm_fp4 and the unfused branch -------------------------------------------------------
    ref_fp4 = None
    if fi is None:
        skip("mm_fp4", "flashinfer unavailable or unusable on this device")
        skip("unfused_branch_full", "needs mm_fp4")
        skip("kaug", "needs mm_fp4 (the whole point of kaug is to use the VENDOR gemm)")
    else:
        def mm(o = None):
            return fi.mm_fp4(aq, wq.T, asf, wsf.T, alpha, torch.bfloat16,
                             out = ob if o is None else o, backend = args.fi_backend)
        try:
            ref_fp4 = mm(torch.zeros_like(ob)).clone()
            res["times"]["mm_fp4"] = timed(mm, warm, iters)
            res["correctness"]["mm_fp4_vs_fp32_dequant"] = err(ref_fp4, ref_fp4_exact)
            res["times"]["quantise_only_flashinfer"] = timed(
                lambda: fi.nvfp4_quantize(x, ag, do_shuffle = False), warm, iters)

            # Preallocated outputs on BOTH sides. Without ``out=`` the up projection allocates a
            # fresh M x N tensor every call, which at M=27280 N=14336 is 782 MB and turns the
            # comparison into a measurement of the caching allocator rather than of the branch.
            ub = torch.empty(M, N, device = dev, dtype = torch.bfloat16)
            td = torch.empty(M, rank, device = dev, dtype = torch.bfloat16)

            def unfused():
                tt = torch.mm(x, l1t, out = td)
                y = mm()
                return torch.addmm(y, tt, l2t, out = ub)
            res["times"]["unfused_branch_full"] = timed(unfused, warm, iters)
            res["times"]["unfused_down_only"] = timed(lambda: torch.mm(x, l1t, out = td),
                                                      warm, iters)
            tt_bf = torch.mm(x, l1t)
            res["times"]["unfused_up_only"] = timed(
                lambda: torch.addmm(ob, tt_bf, l2t, out = ub), warm, iters)
            res["correctness"]["unfused_vs_fp32ref"] = err(
                torch.addmm(mm(torch.zeros_like(ob)), tt_bf, l2t), ref32)
        except Exception as exc:                                        # noqa: BLE001
            skip("mm_fp4", f"{type(exc).__name__}: {str(exc)[:200]}")
            fi = None

    # ---- the K-AUGMENTED fused path ----------------------------------------------------------
    if fi is not None and rank <= pad and _HAS_TRITON and s_ok:
        try:
            wq2, wsf2, wg2, ascale = build_kaug_weight(w, l2p, wg, s_bal, pad, build_q)
            wq2 = wq2.view(torch.uint8)
            same_w = bool(torch.equal(wq2[:, :K // 2], wq))
            alpha2 = (alpha.float() * ascale).float()
            oc = torch.zeros(M, N, device = dev, dtype = torch.bfloat16)

            def kaug_quant():
                return triton_quant_down(x, ag, l1t, kaug_cols = pad)
            aq2, asf2, _t2 = kaug_quant()
            same_a = bool(torch.equal(aq2[:, :K // 2], aq))

            def kaug_gemm():
                return fi.mm_fp4(aq2, wq2.T, asf2, wsf2, alpha2, torch.bfloat16,
                                 out = oc, backend = args.fi_backend) \
                    if False else fi.mm_fp4(aq2, wq2.T, asf2, wsf2.T, alpha2, torch.bfloat16,
                                            out = oc, backend = args.fi_backend)
            y_k = kaug_gemm().clone()
            res["times"]["kaug_gemm_only"] = timed(kaug_gemm, warm, iters)
            res["times"]["kaug_quant_down"] = timed(kaug_quant, warm, iters)
            res["times"]["kaug_full_fusedquant"] = {
                "p50_ms": res["times"]["kaug_gemm_only"]["p50_ms"]
                + res["times"]["kaug_quant_down"]["p50_ms"],
                "min_ms": res["times"]["kaug_gemm_only"]["min_ms"]
                + res["times"]["kaug_quant_down"]["min_ms"],
                "note": "gemm + the fused Triton quantiser (two kernels). Kept for the record; "
                        "the ASSEMBLE variant below is the shipped one and is faster."}
            res["correctness"]["kaug_x_half_bit_identical"] = same_a

            # ---- ASSEMBLE: vendor quantiser + bf16 mm + one re-layout pass -------------------
            # The shipped activation path. Three kernels rather than two, but the expensive one is
            # the vendor's, so it beats the fully fused Triton quantiser on every shape measured.
            td2 = torch.empty(M, rank, device = dev, dtype = torch.bfloat16)

            def kaug_assemble():
                q0, s0 = fi.nvfp4_quantize(x, ag, do_shuffle = False)
                tt = torch.mm(x, l1t, out = td2)
                return triton_assemble_kaug(q0, s0, tt, ag, M, K, kaug_cols = pad)
            aq3, asf3 = kaug_assemble()
            res["correctness"]["kaug_assemble_x_half_bit_identical"] = bool(
                torch.equal(aq3[:, :K // 2], aq))
            res["times"]["kaug_assemble_activation"] = timed(kaug_assemble, warm, iters)
            q0f, s0f = fi.nvfp4_quantize(x, ag, do_shuffle = False)
            res["times"]["kaug_assemble_only"] = timed(
                lambda: triton_assemble_kaug(q0f, s0f, td2, ag, M, K, kaug_cols = pad),
                warm, iters)
            oc3 = torch.zeros(M, N, device = dev, dtype = torch.bfloat16)
            y_k3 = fi.mm_fp4(aq3, wq2.T, asf3, wsf2.T, alpha2, torch.bfloat16,
                             out = oc3, backend = args.fi_backend).clone()
            res["correctness"]["kaug_assemble_total_vs_fp32ref"] = err(y_k3, ref32)
            res["times"]["kaug_full_assemble"] = {
                "p50_ms": res["times"]["kaug_gemm_only"]["p50_ms"]
                + res["times"]["kaug_assemble_activation"]["p50_ms"],
                "min_ms": res["times"]["kaug_gemm_only"]["min_ms"]
                + res["times"]["kaug_assemble_activation"]["min_ms"],
                "note": "augmented gemm + (vendor quantise, bf16 down projection, assemble)"}
            res["correctness"]["kaug_w_half_bit_identical"] = same_w
            res["correctness"]["kaug_total_vs_fp32ref"] = err(y_k, ref32)
            if ref_fp4 is not None:
                res["correctness"]["kaug_correction_vs_exact"] = err(
                    y_k.float() - ref_fp4.float(), exact_corr)
                res["correctness"]["uncorrected_vs_fp32ref"] = err(ref_fp4, ref32)
        except Exception as exc:                                        # noqa: BLE001
            skip("kaug", f"{type(exc).__name__}: {str(exc)[:250]}")

    # ---- the Triton kernels ------------------------------------------------------------------
    if not _HAS_TRITON:
        skip("triton_fused", "triton not importable")
    elif K % 256:
        skip("triton_fused", f"K={K} is not a multiple of BLOCK_K=256")
    else:
        try:
            wl = unswizzle_sf(wsf_u, N, K)
            y_base = triton_mm_lowrank(aq, asf_u, wq, wl, alpha, out = torch.zeros_like(ob))
            if ref_fp4 is not None:
                res["correctness"]["triton_base_bitwise_vs_mm_fp4"] = bool(
                    torch.equal(y_base, ref_fp4))
            res["correctness"]["triton_base_vs_fp32_dequant"] = err(y_base, ref_fp4_exact)
            res["times"]["triton_base"] = timed(
                lambda: triton_mm_lowrank(aq, asf_u, wq, wl, alpha, out = ob), warm, iters)
            t_bf = t_ref.to(torch.bfloat16)
            y_f = triton_mm_lowrank(aq, asf_u, wq, wl, alpha, t = t_bf, l2t = l2t,
                                    out = torch.zeros_like(ob))
            res["times"]["triton_fused"] = timed(
                lambda: triton_mm_lowrank(aq, asf_u, wq, wl, alpha, t = t_bf, l2t = l2t,
                                          out = ob), warm, iters)
            res["correctness"]["triton_fused_vs_fp32ref"] = err(y_f, ref32)
            res["times"]["triton_quant_only"] = timed(
                lambda: triton_quant_down(x, ag, None), warm, iters)
            res["times"]["triton_quant_down"] = timed(
                lambda: triton_quant_down(x, ag, l1t), warm, iters)
            qq, ss, tt2 = triton_quant_down(x, ag, l1t)
            res["correctness"]["triton_quant_bit_identical_to_builder"] = {
                "fp4": bool(torch.equal(qq, aq)),
                "scales": bool(torch.equal(ss[:asf_u.numel()], asf_u))}
            res["correctness"]["triton_t_vs_fp32"] = float(
                (tt2.float() - t_ref).abs().max() / t_ref.abs().max().clamp(min = 1e-12))
        except Exception as exc:                                        # noqa: BLE001
            skip("triton_fused", f"{type(exc).__name__}: {str(exc)[:250]}")

    # ---- the K-AUGMENTED path through the TRITON gemm ------------------------------------------
    # Same operand construction as ``kaug``, but contracted by this file's own Triton kernel
    # instead of the vendor's. Its absolute speed is not the point (Triton's NVFP4 GEMM is well off
    # the vendor kernel on every Blackwell measured); the point is that it gives a fused-branch
    # COST -- kaug_triton against triton_base -- on a device where flashinfer will not build, which
    # is the normal case on sm_120 today. It is also the fallback the shipped layer would use there.
    if _HAS_TRITON and rank <= pad and (K + pad) % 64 == 0 and s_ok:
        try:
            wq2t, wsf2t, wg2t, ascalet = build_kaug_weight(w, l2p, wg, s_bal, pad, build_q)
            wq2t = wq2t.view(torch.uint8)
            wl2 = unswizzle_sf(wsf2t.view(torch.uint8).reshape(-1), N, K + pad)
            alpha2t = (alpha.float() * ascalet).float()
            aq2t, asf2t, _ = triton_quant_down(x, ag, l1t, kaug_cols = pad)
            # K + 64 is a multiple of 64 for every K here but NOT of 128 or 256, so the augmented
            # GEMM has to run with BLOCK_K = 64, the minimum an e2m1 tl.dot_scaled accepts
            # (m16n8k64). That is a worse K tile than the base kernel's 256 and it is why
            # kaug_triton is not simply base + 2%; a masked K tail would fix it and is not
            # implemented here because the Triton GEMM loses to the vendor kernel either way.
            kcfg = dict(BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64, GROUP_M = 8,
                        num_warps = 4, num_stages = 3)
            y_kt = triton_mm_lowrank(aq2t, asf2t, wq2t, wl2, alpha2t,
                                     out = torch.zeros_like(ob), cfg = kcfg)
            res["times"]["kaug_triton_gemm_only"] = timed(
                lambda: triton_mm_lowrank(aq2t, asf2t, wq2t, wl2, alpha2t, out = ob, cfg = kcfg),
                warm, iters)
            res["correctness"]["kaug_triton_total_vs_fp32ref"] = err(y_kt, ref32)
            tb = res["times"].get("triton_base", {}).get("p50_ms")
            if tb:
                res["kaug_triton_branch_cost_pct_of_triton_base"] = (
                    res["times"]["kaug_triton_gemm_only"]["p50_ms"] / tb - 1.0) * 100.0
        except Exception as exc:                                        # noqa: BLE001
            skip("kaug_triton", f"{type(exc).__name__}: {str(exc)[:250]}")
    elif _HAS_TRITON:
        skip("kaug_triton", f"K+pad={K + pad} is not a multiple of BLOCK_K=256")

    # ---- triton_kernels, Triton's own production library --------------------------------------
    if not caps["triton_kernels"]:
        skip("triton_kernels", caps["triton_kernels_why"])
    else:
        try:
            from triton_kernels.matmul import matmul, PrecisionConfig
            al = unswizzle_sf(asf_u, M, K).view(torch.float8_e4m3fn)
            wlT = unswizzle_sf(wsf_u, N, K).view(torch.float8_e4m3fn).T.contiguous()
            pc = PrecisionConfig(a_mx_scale = al, a_microblock_size = 16,
                                 b_mx_scale = wlT, b_microblock_size = 16,
                                 out_dtype = torch.bfloat16)
            pc.a_mx_tensor_scale = alpha.float().expand(M).contiguous()
            wqt = wq.T

            def tk():
                return matmul(aq, wqt, None, precision_config = pc)
            y_tk = tk()
            if ref_fp4 is not None:
                res["correctness"]["triton_kernels_bitwise_vs_mm_fp4"] = bool(
                    torch.equal(y_tk, ref_fp4))
            res["correctness"]["triton_kernels_vs_fp32_dequant"] = err(y_tk, ref_fp4_exact)
            res["times"]["triton_kernels_nvfp4"] = timed(tk, warm, iters)
        except Exception as exc:                                        # noqa: BLE001
            skip("triton_kernels", f"{type(exc).__name__}: {str(exc)[:250]}")

    base = res["times"].get("mm_fp4", {}).get("p50_ms")
    if base:
        res["vs_mm_fp4"] = {k: v["p50_ms"] / base for k, v in res["times"].items()
                            if isinstance(v, dict) and "p50_ms" in v}
        if "kaug_gemm_only" in res["vs_mm_fp4"]:
            res["fused_branch_gemm_only_pct_of_mm_fp4"] = (
                res["vs_mm_fp4"]["kaug_gemm_only"] - 1.0) * 100.0
        if "unfused_branch_full" in res["vs_mm_fp4"]:
            res["unfused_branch_cost_pct_of_mm_fp4"] = (
                res["vs_mm_fp4"]["unfused_branch_full"] - 1.0) * 100.0

        # FULL ACCOUNTING. Both arms pay the vendor quantiser and both pay the bf16 down
        # projection, so those cancel; what does not cancel is the assemble pass on the fused
        # side, the extra K tile in the augmented gemm, and the up projection on the unfused side.
        # Quoting only ``kaug_gemm_only - mm_fp4`` would hide the assemble pass and overstate the
        # win, which is exactly the mistake that made an earlier end-to-end run disagree with the
        # microbenchmark.
        tt_ = res["times"]
        def _p(k):
            v = tt_.get(k)
            return v["p50_ms"] if isinstance(v, dict) and "p50_ms" in v else None
        # ---- INTERNAL CONSISTENCY, permanently on ------------------------------------------
        # Every variant listed here CONTAINS a full mm_fp4 over at least K columns, so none of
        # them can be faster than mm_fp4 on the same row. A violation means the row is comparing
        # quantities that are not the same thing -- which is exactly how a flat, launch-bound
        # assemble timing once made the fused path look cheaper than the GEMM it contains.
        # Recorded rather than raised, so one bad row does not discard the whole sweep, but it is
        # surfaced in the JSON and printed.
        mf_p = _p("mm_fp4")
        viol = []
        if mf_p:
            for k in ("kaug_gemm_only", "kaug_full_assemble", "kaug_full_fusedquant",
                      "unfused_branch_full", "triton_fused"):
                v = _p(k)
                if v is not None and v < mf_p * 0.98:
                    viol.append({"variant": k, "ms": v, "mm_fp4_ms": mf_p,
                                 "ratio": v / mf_p})
        if viol:
            res["CONSISTENCY_VIOLATION"] = {
                "rows": viol,
                "meaning": ("a variant that contains a full mm_fp4 came in below mm_fp4 on the "
                            "same shape. The timing of at least one component is not the timing "
                            "of this row's workload; do not quote a ratio from it.")}
            print(f"      CONSISTENCY VIOLATION on {label} r{rank}: "
                  + ", ".join(f"{v['variant']}={v['ms']:.4f} < mm_fp4={v['mm_fp4_ms']:.4f}"
                              for v in viol), flush = True)

        asm, kg, uu, mf = (_p("kaug_assemble_only"), _p("kaug_gemm_only"),
                           _p("unfused_up_only"), base)
        if None not in (asm, kg, uu):
            res["fused_branch_cost_ms"] = asm + (kg - mf)
            res["fused_branch_cost_pct_of_mm_fp4"] = (asm + (kg - mf)) / mf * 100.0
            res["unfused_branch_cost_ms"] = uu
            res["unfused_branch_up_pct_of_mm_fp4"] = uu / mf * 100.0
            res["fused_vs_unfused_branch_speedup"] = uu / max(asm + (kg - mf), 1e-9)
    return res


def _factorise(w, wdq, ranks, hessian_diag = None):
    """Truncated SVD of the quantisation residual ``E = W - dequant(Q(W))``, in one of two spaces.

    ``hessian_diag = None`` gives plain FORM A: the SVD is taken in weight space, minimising
    ``||E - L2 L1||_F``. Passing a per-input-channel second moment gives FORM A_H, the
    Hessian-whitened fit that this campaign actually ships: with ``S = sqrt(diag(H))`` the SVD is
    taken of ``E S`` and then unwhitened, so the objective becomes ``||(E - L2 L1) S||_F``, which
    is the error the ACTIVATIONS actually see. Whitening moves magnitude between the two halves,
    and that redistribution is exactly what the c window measures. Returns {rank: (l1, l2)}.
    """
    import torch
    e = (w.float() - wdq.float())
    if hessian_diag is None:
        m = e
    else:
        sroot = hessian_diag.float().clamp(min = 1e-12).sqrt()
        m = e * sroot.reshape(1, -1)
    u, sv, vh = torch.linalg.svd(m, full_matrices = False)
    out = {}
    for r in ranks:
        r = min(r, sv.numel())
        l2 = (u[:, :r] * sv[:r])
        l1 = vh[:r]
        if hessian_diag is not None:
            l1 = l1 / sroot.reshape(1, -1)
        out[r] = (l1.to(w.dtype).contiguous(), l2.to(w.dtype).contiguous())
    return out


def window_check(args) -> dict:
    """Is K-augmentation SAFE on this box, before anyone benchmarks how fast it is?

    K-augmentation folds a scalar ``c`` into the factors -- ``L1' = c L1``, ``L2' = L2 / c`` --
    which changes no function but sends the two halves into different quantisers. Each has a
    ceiling, so ``c`` has to satisfy

        amax(L2) / amax(w)  <=  c  <=  amax(x) / amax(t)          (t = x L1^T)

    The upper bound keeps the appended ACTIVATION columns inside the activation's global scale.
    The lower bound keeps the appended WEIGHT columns inside the weight's, and it is the one that
    matters: below it the builder is forced to set ``w_gsf_aug = w_gsf * s`` with ``s < 1``, which
    requantises the FIRST K COLUMNS. The augmented GEMM would then be computing a different base
    product from the shipped one, and it would still run, still be fast, and still look right.

    Whether the window is open depends on the FACTORISATION as much as on the model, so this check
    runs BOTH: plain form A (the SVD of the residual ``E = W - dequant(Q(W))`` in weight space) and
    A_H (the same fit whitened by a diagonal Hessian built from the activation sample, which is
    what the campaign ships). On real diffusion weights form A leaves the window empty on every
    z-image and flux layer while A_H opens it on every layer of every image model at every rank;
    this synthetic reproduction is a self-test of the MACHINERY and of the refusal path, not a
    substitute for that per-model measurement.

    Verified per case: the window bounds, whether ``c`` lands inside, the achieved ``s``, and --
    when ``s == 1`` -- that the first K columns of the augmented operand are bit-identical to the
    shipped one, nibbles and block scales and global scale alike.
    """
    import torch
    out = {"mode": "window-check", "env": environment(), "rows": []}
    if not torch.cuda.is_available():
        out["skipped"] = "no CUDA device"
        return out
    try:
        import flashinfer as fi
    except Exception as exc:                                            # noqa: BLE001
        out["skipped"] = f"flashinfer unavailable: {type(exc).__name__}: {exc}"
        return out
    out["preflight"] = preflight()
    dev = "cuda"
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    shapes = []
    for gname in [g.strip() for g in args.shapes.split(",") if g.strip()]:
        shapes += SHAPES.get(gname, [])
    for label, M, K, N in shapes:
        g = torch.Generator(device = dev).manual_seed(3407)
        # An activation sample with per-channel scale spread, which is what makes a diagonal
        # Hessian non-trivial and therefore makes form A and form A_H differ at all.
        chan = (torch.rand(K, device = dev, generator = g) * 3.0 - 1.5).exp()
        x = (torch.randn(min(M, 4096), K, device = dev, generator = g)
             * 0.05 * chan.reshape(1, -1)).to(torch.bfloat16).contiguous()
        w = (torch.randn(N, K, device = dev, generator = g) * 0.02).to(torch.bfloat16).contiguous()
        ag, wg = global_scale(x), global_scale(w)
        with dev_guard(w):
            wq, wsf = fi.nvfp4_quantize(w, wg, do_shuffle = False)
            wdq = dequant_nvfp4(wq.view(torch.uint8), wsf.view(torch.uint8).reshape(-1),
                                wg, N, K).to(torch.bfloat16)
        hdiag = (x.float() ** 2).mean(0)
        amax_x = float(FP4_MAX * FP8_MAX / float(ag))
        amax_w = float(w.float().abs().amax())
        forms = [("formA", None), ("formA_H", hdiag)]
        for form, hd in forms:
            facs = _factorise(w, wdq, ranks, hessian_diag = hd)
            # Two controls on the guard, using the smallest rank.
            #  "gauge":        L1 / 1e6 and L2 * 1e6 is the SAME function, and the per-column
            #                  window is invariant under it, because G = max_j sqrt(a2_j at_j)
            #                  and the two factors move by reciprocal amounts. The bounds and the
            #                  decision must come out UNCHANGED. If they move, the builder is
            #                  reading a gauge choice as a property of the layer.
            #  "forced_empty": L2 * 1e6 with L1 untouched is a DIFFERENT function with genuinely
            #                  oversized weight-side factors, and must be REFUSED. If it ever
            #                  builds, the guard is not doing its job and every fused number that
            #                  depends on it is suspect.
            if form == "formA":
                l1c, l2c = facs[ranks[0]]
                facs = dict(facs)
                facs["gauge"] = (l1c.float().div(1e6).to(torch.float32),
                                 l2c.float().mul(1e6).to(torch.float32))
                facs["forced_empty"] = (l1c, l2c.float().mul(1e6).to(torch.float32))
            for rank, (l1, l2) in facs.items():
                at = (x.float() @ l1.t().float()).abs().amax(0)
                try:
                    lo, hi, ok, G = _percol_window(amax_x, at, l2.float(), amax_w)
                    l1p, l2p, sc, okb, winfo = balance_windowed_percol(
                        l1, l2, amax_x, at, amax_w)
                except Exception as exc:                                # noqa: BLE001
                    out["rows"].append({"label": label, "form": form, "rank": rank,
                                        "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
                    continue
                rec = {"label": label, "M": M, "K": K, "N": N, "form": form, "rank": rank,
                       "c_lo": lo, "c_hi": hi, "window_feasible": bool(ok),
                       "log10_width": (math.log10(hi / lo) if ok and lo > 0 else None),
                       "s": sc, "reason": winfo.get("reason"),
                       "bit_identity_expected": bool(okb and sc >= 1.0)}
                if okb and sc >= 1.0:
                    pad = max(64, 4 * int(rank if isinstance(rank, int) else l2p.shape[1]))
                    with dev_guard(w):
                        wq2, wsf2, wg2, _a = build_kaug_weight(
                            w, l2p, wg, sc, pad,
                            lambda t, gg: fi.nvfp4_quantize(t, gg, do_shuffle = False),
                            refine = 2)
                    kb, kb2 = K // 16, (K + pad) // 16
                    kpad, kpad2 = (kb + 3) // 4 * 4, (kb2 + 3) // 4 * 4
                    npad = (N + 127) // 128 * 128
                    a = wsf.view(torch.uint8).reshape(-1)[:npad * kpad].reshape(
                        npad // 128, kpad // 4, 512)
                    b = wsf2.view(torch.uint8).reshape(-1).reshape(npad // 128, kpad2 // 4, 512)
                    rec["w_nibbles_identical"] = bool(
                        torch.equal(wq2.view(torch.uint8)[:, :K // 2], wq.view(torch.uint8)))
                    rec["w_scales_identical"] = bool(torch.equal(b[:, :kpad // 4], a))
                    rec["w_gsf_identical"] = bool(torch.equal(wg2, wg))
                else:
                    rec["refused"] = winfo.get("reason", "empty c window")
                out["rows"].append(rec)
                print(json.dumps(rec), flush = True)
        del x, w, wdq
        torch.cuda.empty_cache()
    ctrl = [r for r in out["rows"] if r.get("rank") == "forced_empty"]
    gau = {r["label"]: r for r in out["rows"] if r.get("rank") == "gauge"}
    base = {r["label"]: r for r in out["rows"]
            if r.get("rank") == ranks[0] and r.get("form") == "formA"}
    out["guard_controls"] = {
        "forced_empty": {
            "n_cases": len(ctrl),
            "all_refused": all(("refused" in r) for r in ctrl),
            "reading": ("L2 * 1e6 with L1 untouched has genuinely oversized weight-side factors "
                        "and must be refused. all_refused=false means the builder will silently "
                        "requantise the first K columns.")},
        "gauge_invariance": {
            "n_cases": len(gau),
            "bounds_unchanged": all(
                abs(math.log10(max(gau[k]["c_lo"], 1e-300) / max(base[k]["c_lo"], 1e-300))) < 1e-6
                and gau[k]["window_feasible"] == base[k]["window_feasible"]
                for k in gau if k in base),
            "reading": ("L1 / 1e6 with L2 * 1e6 is the same function; the per-column window must "
                        "return the same bounds and the same decision. This is why the scalar c "
                        "is a gauge and the window is a property of the layer, not of how the "
                        "factors happen to be normalised.")}}
    built = [r for r in out["rows"] if r.get("bit_identity_expected")]
    per_form = {}
    for form in ("formA", "formA_H"):
        rs = [r for r in out["rows"] if r.get("form") == form
              and r.get("rank") not in ("forced_empty", "gauge")]
        wid = [r["log10_width"] for r in rs if r.get("log10_width") is not None]
        wid.sort()
        per_form[form] = {
            "n_cases": len(rs),
            "n_window_open": sum(1 for r in rs if r.get("window_feasible")),
            "n_refused": sum(1 for r in rs if "refused" in r),
            "median_log10_width": (wid[len(wid) // 2] if wid else None),
            "min_s": (min(r["s"] for r in rs) if rs else None)}
    out["summary"] = {
        "per_form": per_form,
        "n_bit_identity_checked": len(built),
        "all_bit_identical": all(r.get("w_nibbles_identical") and r.get("w_scales_identical")
                                 and r.get("w_gsf_identical") for r in built),
        "reading": ("a refusal is the CORRECT outcome for a closed window, not a failure. What "
                    "would be a failure is a case that builds with s < 1, or one that builds "
                    "with s == 1 and is not bit-identical. Compare the two forms: whitening is "
                    "what opens the window on real image models."),
        "caveat": ("the weights here are synthetic Gaussian, whose quantisation residual is far "
                   "better conditioned than a real diffusion layer's. Both forms open the window "
                   "on them. This is a self-test of the machinery and of the refusal path; the "
                   "per-model measurement (form A empty on 96/96 z-image and 114/114 flux "
                   "layers, A_H open on all of them) is in nvfp4_fused_lowrank_kernel.json.")}
    return out


def _percol_window(amax_x, amax_t_cols, l2, amax_w):
    """``(lo, hi, feasible, G)`` for the per-column split, with ``G = max_j sqrt(a2_j at_j)``."""
    import torch
    r = l2.shape[1]
    at = amax_t_cols.float().reshape(r).clamp(min = 1e-30)
    a2 = l2.float().abs().amax(dim = 0).reshape(r).clamp(min = 1e-30)
    g = float((a2 * at).sqrt().max())
    lo = g / max(float(amax_w), 1e-30)
    hi = float(amax_x) / max(g, 1e-30)
    return lo, hi, bool(lo <= hi), g


# ======================================================================================= selftest
_SELFTEST_SHARED_KERNELS = (
    # (name in this file, dev module, name there)
    ("_sf_offsets",          "g844_triton_fused", "_sf_offsets"),
    ("_nvfp4_gemm_lowrank",  "g844_triton_fused", "_nvfp4_gemm_lowrank"),
    ("_e2m1_code",           "g844_quant_down",   "_e2m1_code"),
    ("_quant_down",          "g844_quant_down",   "_quant_down"),
    ("_assemble",            "g844_quant_down",   "_assemble"),
)


def _jit_source(obj):
    """Source text of a Triton ``@triton.jit`` function (or a plain function), dedented.

    The copies in this file live inside ``if _HAS_TRITON:`` and so carry four extra spaces of
    indentation relative to the module-level originals. Dedent before comparing or every kernel
    reports as drifted.
    """
    import inspect
    import textwrap
    fn = getattr(obj, "fn", obj)                    # triton JITFunction wraps the python function
    src = getattr(obj, "src", None)
    if not isinstance(src, str):
        src = inspect.getsource(fn)
    return textwrap.dedent(src).strip()


def _unpack_nibbles(packed):
    """``[M, K//2]`` packed uint8 -> ``[M, K]`` int32 codes, low nibble first."""
    import torch
    p = packed.view(torch.uint8)
    return torch.stack([p & 0xF, (p >> 4) & 0xF], -1).reshape(p.shape[0], -1).int()


def _semantic_signature(src: str) -> str:
    """AST dump of a function with its docstring removed: what it DOES, not how it is described.

    Comparing raw text is the wrong test here and reported three false drifts on its first run.
    This file is meant to be read by someone with no campaign context, so its copies of the
    kernels carry longer docstrings than the development modules do; that is deliberate and is
    not drift. Decorators are dropped too: the copies are wrapped in ``@triton.jit`` from the same
    import either way, and a decorator list difference would only ever be noise here.

    An AST comparison also strips comments and normalises formatting, which means a real change --
    a different constant, a flipped comparison, a reordered accumulation -- is the only thing that
    can still make it fail.
    """
    import ast
    tree = ast.parse(src)
    fn = tree.body[0]
    fn.decorator_list = []
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        fn.body = body[1:] or [ast.Pass()]
    return ast.dump(ast.fix_missing_locations(fn), annotate_fields = True)


def selftest(args) -> dict:
    """Check this file's EMBEDDED copies against their sources of truth, and exit non-zero on drift.

    This file is self-contained by design: it carries its own copies of the Triton kernels and its
    own pure-torch quantiser so it can be dropped onto a 5090, a Spark or a Colab box. Duplication
    that nothing checks is duplication that drifts, and two of the claims this benchmark rests on
    are exactly claims about the copies:

      * the module docstring says the embedded kernels are checked "bit-for-bit against the
        development modules when they happen to be importable";
      * ``torch_nvfp4_quantize``'s docstring calls itself "a verified stand-in and not a second
        opinion" on the strength of being checked against ``flashinfer.nvfp4_quantize`` here.

    Neither statement was true before this function existed: ``--selftest`` was declared in the
    argument parser, advertised in ``--help`` and referenced in two docstrings, but had no
    implementation, so invoking it fell through to the full benchmark and reported nothing. A
    check that is only described is worth less than no check, because it gets quoted as evidence.

    Three groups, each degrading to an explicit SKIP with a reason rather than a silent pass:

      A. source parity, no GPU and no CUDA needed -- the five shared kernels against
         ``scripts/g844/g844_{triton_fused,quant_down}.py``;
      B. layout algebra, no GPU needed -- ``swizzle_sf``/``unswizzle_sf`` round trip and the
         per-column c-window bounds under gauge and under an oversized L2;
      C. numeric, needs CUDA -- the pure-torch quantiser against flashinfer's bit-for-bit, the
         Triton quantiser against the torch reference, the assemble pass' first-K half against the
         vendor operand, and the embedded Triton GEMM at rank 0 against ``mm_fp4``.
    """
    import os
    import sys
    checks = []

    def rec(name, ok, detail = "", skipped = False):
        checks.append({"check": name, "ok": (None if skipped else bool(ok)),
                       "skipped": bool(skipped), "detail": str(detail)[:400]})
        tag = "SKIP" if skipped else ("ok  " if ok else "FAIL")
        print(f"  [{tag}] {name}" + (f": {str(detail)[:200]}" if detail else ""), flush = True)

    # ---- A. source parity against the development modules ------------------------------------
    print("A. embedded kernel source parity", flush = True)
    here = os.path.dirname(os.path.abspath(__file__))
    dev_ok, dev_why = False, ""
    if not _HAS_TRITON:
        rec("kernel_source_parity", None, "triton not importable, embedded kernels not defined",
            skipped = True)
    else:
        if here not in sys.path:
            sys.path.insert(0, here)
        mods = {}
        try:
            import importlib
            for _, modname, _ in _SELFTEST_SHARED_KERNELS:
                if modname not in mods:
                    mods[modname] = importlib.import_module(modname)
            dev_ok = True
        except Exception as exc:                                        # noqa: BLE001
            dev_why = f"{type(exc).__name__}: {str(exc)[:200]}"
        if not dev_ok:
            rec("kernel_source_parity", None,
                f"development modules not importable here (expected off-box): {dev_why}",
                skipped = True)
        else:
            import difflib
            for local, modname, remote in _SELFTEST_SHARED_KERNELS:
                try:
                    a_src = _jit_source(globals()[local])
                    b_src = _jit_source(getattr(mods[modname], remote))
                    same_sem = _semantic_signature(a_src) == _semantic_signature(b_src)
                    same_txt = a_src == b_src
                    if same_sem:
                        rec(f"kernel_source_parity::{local}", True,
                            f"AST-identical to {modname}.{remote}"
                            + ("" if same_txt else " (prose differs, which is intended)"))
                    else:
                        d = "\n".join(list(difflib.unified_diff(
                            b_src.splitlines(), a_src.splitlines(), f"{modname}.{remote}",
                            f"bench.{local}", lineterm = ""))[:40])
                        rec(f"kernel_source_parity::{local}", False, "DRIFTED:\n" + d)
                except Exception as exc:                                # noqa: BLE001
                    rec(f"kernel_source_parity::{local}", False,
                        f"{type(exc).__name__}: {str(exc)[:200]}")

    # ---- B. layout algebra, CPU only ---------------------------------------------------------
    print("B. layout algebra (no GPU needed)", flush = True)
    try:
        import torch
        ok_all, detail = True, []
        for (m, k) in ((128, 256), (200, 3840), (4128, 3840), (64, 640)):
            lin = torch.randint(1, 250, (m, k // 16), dtype = torch.uint8).view(torch.float8_e4m3fn)
            back = unswizzle_sf(swizzle_sf(lin, m, k), m, k)
            same = bool(torch.equal(back.view(torch.uint8), lin.view(torch.uint8)))
            ok_all = ok_all and same
            detail.append(f"{m}x{k}:{'ok' if same else 'MISMATCH'}")
        rec("swizzle_roundtrip", ok_all, " ".join(detail))
    except Exception as exc:                                            # noqa: BLE001
        rec("swizzle_roundtrip", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    try:
        import torch
        torch.manual_seed(3407)
        r, K, N = 32, 512, 256
        l1 = torch.randn(r, K) * 0.1
        l2 = torch.randn(N, r) * 0.3
        x = torch.randn(128, K) * 0.05
        amax_x, amax_w = float(x.abs().amax()), 0.11
        at = (x @ l1.t()).abs().amax(0)
        lo0, hi0, feas0, g0 = _percol_window(amax_x, at, l2, amax_w)
        # GAUGE: L1/a with L2*a is the SAME function, so the per-column window must not move.
        a = 1e3
        at2 = (x @ (l1 / a).t()).abs().amax(0)
        lo1, hi1, feas1, g1 = _percol_window(amax_x, at2, l2 * a, amax_w)
        moved = max(abs(lo1 - lo0) / max(lo0, 1e-30), abs(hi1 - hi0) / max(hi0, 1e-30))
        rec("c_window_gauge_invariance", moved < 1e-5 and feas0 == feas1,
            f"lo {lo0:.6g}->{lo1:.6g} hi {hi0:.6g}->{hi1:.6g} rel move {moved:.2e}")
        # FORCED EMPTY: L2 alone blown up is a genuinely different, oversized factorisation and
        # must be REFUSED. Using the gauge transform here instead -- which is invariant -- is how
        # an earlier version of this control reported a vacuous pass without ever exercising the
        # refusal.
        lo2, hi2, feas2, _ = _percol_window(amax_x, at, l2 * 1e6, amax_w)
        rec("c_window_refuses_oversized_l2", not feas2,
            f"lo {lo2:.6g} > hi {hi2:.6g} -> refused" if not feas2
            else "ACCEPTED an oversized L2; the refusal path is not being exercised")
    except Exception as exc:                                            # noqa: BLE001
        rec("c_window_controls", False, f"{type(exc).__name__}: {str(exc)[:200]}")

    # ---- C. numeric, needs CUDA --------------------------------------------------------------
    print("C. numeric parity (needs CUDA)", flush = True)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:                                                   # noqa: BLE001
        has_cuda = False
    if not has_cuda:
        rec("numeric_parity", None, "no CUDA device on this box", skipped = True)
    else:
        import torch
        dev = "cuda"
        M, K, N, rank = 256, 1024, 512, 32
        gen = torch.Generator(device = dev).manual_seed(3407)
        x = (torch.randn(M, K, device = dev, dtype = torch.bfloat16, generator = gen)
             * 0.05).contiguous()
        w = (torch.randn(N, K, device = dev, dtype = torch.bfloat16, generator = gen)
             * 0.02).contiguous()
        ag, wg = global_scale(x), global_scale(w)

        fi = None
        try:
            import flashinfer as _fi
            with dev_guard(x):
                _fi.nvfp4_quantize(x[:16].contiguous(), ag, do_shuffle = False)
            fi = _fi
        except Exception as exc:                                        # noqa: BLE001
            rec("torch_quantiser_vs_flashinfer", None,
                f"flashinfer unusable here: {type(exc).__name__}: {str(exc)[:180]}",
                skipped = True)
        if fi is not None:
            # C1a. The PURE-TORCH fallback against the vendor. This is deliberately NOT a
            # bit-equality test, because bit-equality is measurably false and demanding it would
            # leave a red check nobody can fix. What is actually true, and is what the fallback
            # needs to be worth using, is: the block scales are exact, the nibble disagreements
            # are rare, confined to elements sitting on an e2m1 tie under fp32 arithmetic, one
            # code step wide, never a sign flip, and cost nothing in reconstruction error.
            try:
                det, ok_all = [], True
                for t_, gs_ in ((x, ag), (w, wg)):
                    with dev_guard(t_):
                        fq, fsf = fi.nvfp4_quantize(t_, gs_, do_shuffle = False)
                    tq, tsf = torch_nvfp4_quantize(t_, gs_)
                    rows, cols = t_.shape
                    # Compare the LOGICAL block scales. The swizzled buffer has padding lanes that
                    # neither producer is required to agree on; comparing those instead is how a
                    # real difference gets hidden behind a fake one, in either direction.
                    fl = unswizzle_sf(fsf.view(torch.uint8).reshape(-1), rows, cols)
                    tls = unswizzle_sf(tsf.view(torch.uint8).reshape(-1), rows, cols)
                    sfe = bool(torch.equal(fl.view(torch.uint8), tls.view(torch.uint8)))
                    fn_, tn_ = _unpack_nibbles(fq), _unpack_nibbles(tq)
                    d = fn_ != tn_
                    nd, tot = int(d.sum()), int(fn_.numel())
                    rate = nd / max(tot, 1)
                    dcode = int(((fn_[d] & 7) - (tn_[d] & 7)).abs().max()) if nd else 0
                    flips = int((((fn_[d] & 8) != (tn_[d] & 8))).sum()) if nd else 0
                    fdq = dequant_nvfp4(fq.view(torch.uint8),
                                        fsf.view(torch.uint8).reshape(-1), gs_, rows, cols)
                    tdq = dequant_nvfp4(tq.view(torch.uint8),
                                        tsf.view(torch.uint8).reshape(-1), gs_, rows, cols)
                    ef = float((fdq - t_.float()).pow(2).mean().sqrt())
                    et = float((tdq - t_.float()).pow(2).mean().sqrt())
                    ratio = et / max(ef, 1e-30)
                    # 5e-3 is a SANITY bound, not a fitted one. Measured disagreement rates on
                    # sm_100 are 0.02% to 0.11% depending on the tensor, and the first version of
                    # this check was set at 0.1%, i.e. at the edge of the four tensors that had
                    # been probed, so the fifth tensor failed it. A bound placed where the data
                    # stops is a bound that will fire on the next machine for no reason; place it
                    # where a REAL regression would be, an order of magnitude out. The properties
                    # that actually matter -- exact block scales, one code step, no sign flip,
                    # equal reconstruction error -- are the tight ones and stay tight.
                    ok = (sfe and rate < 5e-3 and dcode <= 1 and flips == 0
                          and abs(ratio - 1.0) < 1e-4)
                    ok_all = ok_all and ok
                    det.append(f"{tuple(t_.shape)}: scales_exact={sfe} nibble_disagree="
                               f"{nd}/{tot} ({100 * rate:.4f}%) max_code_step={dcode} "
                               f"sign_flips={flips} rms_ratio={ratio:.6f}")
                rec("torch_quantiser_faithful_to_flashinfer", ok_all,
                    "; ".join(det) + " | bounds: scales exact, <0.5% nibbles, 1 code step, "
                    "0 sign flips, rms ratio within 1e-4 of 1")
            except Exception as exc:                                    # noqa: BLE001
                rec("torch_quantiser_faithful_to_flashinfer", False,
                    f"{type(exc).__name__}: {str(exc)[:200]}")

            # C1b. The e2m1 tie RULE itself, on exact midpoints rather than on whatever a random
            # tensor happens to land on. A block whose amax is 6 with gsf 1.0 gives step == 1
            # exactly, so the seven midpoints arrive at the encoder unrounded and the two
            # implementations can be compared on the rule alone. This separates "we disagree about
            # how to break ties" (a bug) from "we disagree about the last ulp of x/step" (noise),
            # and the answer measured here is the second.
            try:
                ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
                rne = [0, 2, 2, 4, 4, 6, 6]         # the even-mantissa (code bit 0 == 0) neighbour
                row = []
                for tv in ties:
                    row += [tv, -tv]
                row += [6.0, -6.0]                  # pins the block amax so that step == 1
                while len(row) % 16:
                    row.append(0.0)
                xt = torch.tensor([row] * 128, device = dev, dtype = torch.bfloat16).contiguous()
                one = torch.tensor([1.0], device = dev)
                with dev_guard(xt):
                    fq, _ = fi.nvfp4_quantize(xt, one, do_shuffle = False)
                tq, _ = torch_nvfp4_quantize(xt, one)
                fc = _unpack_nibbles(fq)[0]
                tc = _unpack_nibbles(tq)[0]
                got_f = [int(fc[2 * i]) for i in range(7)]
                got_t = [int(tc[2 * i]) for i in range(7)]
                rec("e2m1_tie_rule_is_round_half_to_even", got_f == rne and got_t == rne,
                    f"midpoints {ties} -> flashinfer {got_f}, torch reference {got_t}, "
                    f"round-half-to-even requires {rne}")
            except Exception as exc:                                    # noqa: BLE001
                rec("e2m1_tie_rule_is_round_half_to_even", False,
                    f"{type(exc).__name__}: {str(exc)[:200]}")

        if not _HAS_TRITON:
            rec("triton_numeric_parity", None, "triton not importable", skipped = True)
        else:
            # C2. The embedded Triton quantiser against the VENDOR, bit for bit. This is the
            # check that carries weight: the Triton quantiser is on the shipped path, so the
            # campaign's operand-identity claims rest on it, and unlike the torch fallback it
            # really is bit-exact (0 differing bytes on four tensors here; 0 of 6291456 nibbles
            # per case, tie-stress included, in outputs/g844_quant_verify.json). Comparing it
            # against the torch reference instead -- which the first version of this selftest did
            # -- fails for a reason that has nothing to do with the Triton kernel.
            if fi is None:
                rec("triton_quantiser_vs_flashinfer", None,
                    "flashinfer unusable here; the vendor reference is not available to compare "
                    "against, and the torch fallback is not bit-exact so it cannot stand in",
                    skipped = True)
            else:
                try:
                    xq_t, sf_t, _ = triton_quant_down(x, ag)
                    with dev_guard(x):
                        xq_f, sf_f = fi.nvfp4_quantize(x, ag, do_shuffle = False)
                    nib = bool(torch.equal(xq_t.view(torch.uint8), xq_f.view(torch.uint8)))
                    sfe = bool(torch.equal(
                        unswizzle_sf(sf_t.view(torch.uint8).reshape(-1), M, K).view(torch.uint8),
                        unswizzle_sf(sf_f.view(torch.uint8).reshape(-1), M, K).view(torch.uint8)))
                    nd = int((xq_t.view(torch.uint8) != xq_f.view(torch.uint8)).sum())
                    rec("triton_quantiser_vs_flashinfer", nib and sfe,
                        f"nibbles={nib} ({nd} bytes differ) block_scales={sfe}")
                except Exception as exc:                                # noqa: BLE001
                    rec("triton_quantiser_vs_flashinfer", False,
                        f"{type(exc).__name__}: {str(exc)[:200]}")

            # C3. the assemble pass: the first K columns must be the UNaugmented operand exactly.
            # This is the property the whole K-augmentation claim rests on, and here it is checked
            # on THIS box's hardware rather than inherited from the sm_100 run.
            try:
                pad = 64
                if fi is not None:
                    with dev_guard(x):
                        base_q, base_sf = fi.nvfp4_quantize(x, ag, do_shuffle = False)
                else:
                    base_q, base_sf = torch_nvfp4_quantize(x, ag)
                t = (torch.randn(M, rank, device = dev, dtype = torch.bfloat16, generator = gen)
                     * 0.01).contiguous()
                aug_q, aug_sf = triton_assemble_kaug(base_q.view(torch.uint8), base_sf, t, ag,
                                                     M, K, kaug_cols = pad)
                nib = bool(torch.equal(aug_q[:, :K // 2].contiguous().view(torch.uint8),
                                       base_q.view(torch.uint8)[:, :K // 2].contiguous()))
                sf_base = unswizzle_sf(base_sf.view(torch.uint8).reshape(-1), M, K)
                sf_aug = unswizzle_sf(aug_sf.view(torch.uint8).reshape(-1), M, K + pad)
                sfe = bool(torch.equal(sf_aug[:, :K // 16].contiguous().view(torch.uint8),
                                       sf_base.view(torch.uint8)))
                rec("assemble_first_K_bit_identical", nib and sfe,
                    f"nibbles={nib} block_scales={sfe} (pad={pad}, builder="
                    + ("flashinfer)" if fi is not None else "torch reference)"))
            except Exception as exc:                                    # noqa: BLE001
                rec("assemble_first_K_bit_identical", False,
                    f"{type(exc).__name__}: {str(exc)[:200]}")

            # C4. the embedded Triton GEMM at rank 0 against the vendor GEMM.
            if fi is None:
                rec("triton_gemm_vs_mm_fp4", None, "flashinfer unusable here", skipped = True)
            else:
                try:
                    with dev_guard(x):
                        aq, asf = fi.nvfp4_quantize(x, ag, do_shuffle = False)
                        wq, wsf = fi.nvfp4_quantize(w, wg, do_shuffle = False)
                        alpha = (1.0 / (ag * wg)).float()
                        ref = fi.mm_fp4(aq.view(torch.uint8), wq.view(torch.uint8).T,
                                        asf, wsf.T, alpha, torch.bfloat16,
                                        out = torch.zeros(M, N, device = dev,
                                                          dtype = torch.bfloat16),
                                        backend = args.fi_backend)
                    got = triton_mm_lowrank(aq.view(torch.uint8), asf,
                                            wq.view(torch.uint8), wsf, alpha,
                                            a_swizzled = True, b_swizzled = True)
                    same = bool(torch.equal(got, ref))
                    md = float((got.float() - ref.float()).abs().max())
                    rec("triton_gemm_vs_mm_fp4", same, f"bitwise={same} max_abs_diff={md:.3e}")
                except Exception as exc:                                # noqa: BLE001
                    rec("triton_gemm_vs_mm_fp4", False,
                        f"{type(exc).__name__}: {str(exc)[:220]}")

    n_fail = sum(1 for c in checks if c["ok"] is False)
    n_pass = sum(1 for c in checks if c["ok"] is True)
    n_skip = sum(1 for c in checks if c["skipped"])
    out = {"env": environment(), "caps": capabilities(), "checks": checks,
           "summary": {"passed": n_pass, "failed": n_fail, "skipped": n_skip,
                       "verdict": "PASS" if n_fail == 0 else "FAIL"},
           "what_a_skip_means": (
               "a SKIP is a check this box cannot run (no CUDA, no flashinfer, or the g844 "
               "development modules absent because the file was copied off-box), NOT a pass. The "
               "source-parity group is expected to skip everywhere except the development "
               "checkout; the numeric group is the one that matters on new hardware.")}
    print(f"\nselftest: {n_pass} passed, {n_fail} failed, {n_skip} skipped -> "
          f"{out['summary']['verdict']}", flush = True)
    return out


def capabilities() -> dict:
    caps = {"triton": _HAS_TRITON}
    for name in ("flashinfer", "triton_kernels"):
        try:
            __import__(name)
            caps[name] = True
            caps[f"{name}_why"] = ""
        except Exception as exc:                                        # noqa: BLE001
            caps[name] = False
            caps[f"{name}_why"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return caps


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default = "legacy,zimage,wan5b",
                    help = "comma list of groups: " + ",".join(SHAPES) + " (or 'small')")
    ap.add_argument("--ranks", default = "16,32,64")
    ap.add_argument("--kaug-cols", type = int, default = 64,
                    help = "logical K columns appended for the fused branch; must be a multiple "
                           "of 64 so the 128x4 scale swizzle stays aligned")
    ap.add_argument("--fi-backend", default = "cutlass")
    ap.add_argument("--quick", action = "store_true",
                    help = "5 warm / 20 timed instead of 30/120. SMOKE TEST ONLY -- do not quote "
                           "a number from a --quick run. At 20 timed iterations the p50 is noisy "
                           "enough that the consistency assertion fires on shapes where it never "
                           "fires at 120: a --quick pass over the 'small' group on an idle B200 "
                           "flagged 3 of 4 rows for kaug_gemm_only coming in below mm_fp4, which "
                           "is impossible and is the guard correctly refusing to believe the "
                           "timing. Use it to check that the box works, then drop it.")
    ap.add_argument("--dry-run", action = "store_true", help = "compile only, no GPU")
    ap.add_argument("--dry-run-archs", default = "100,103,120,121")
    ap.add_argument("--window-check", action = "store_true",
                    help = "audit the K-augmentation c window and the first-K bit-identity on "
                           "this box, and stop. Safety, not speed: run it before trusting any "
                           "fused number.")
    ap.add_argument("--dry-run-one", default = None,
                    help = "internal: compile one named kernel for --dry-run-arch and exit")
    ap.add_argument("--dry-run-arch", type = int, default = 100)
    ap.add_argument("--selftest", action = "store_true",
                    help = "check the embedded kernels against the g844 development modules")
    ap.add_argument("--out", default = "nvfp4_fused_lowrank_bench.json")
    args = ap.parse_args()

    if args.dry_run_one:
        # Internal single-target child of --dry-run. Prints one JSON line and exits; if the
        # compiler aborts instead, the parent sees the return code.
        print("G844_DRY_JSON " + json.dumps(dry_run_one(args.dry_run_one, args.dry_run_arch)))
        return 0

    if args.selftest:
        out = selftest(args)
        Path(args.out).write_text(json.dumps(out, indent = 2))
        print(f"wrote {args.out}")
        # Non-zero on failure so a launcher's `|| exit` actually gates. A check that always
        # returns 0 is decorative, and a decorative check gets quoted as evidence.
        return 0 if out["summary"]["verdict"] == "PASS" else 4

    if args.window_check:
        out = window_check(args)
        Path(args.out).write_text(json.dumps(out, indent = 2))
        print(json.dumps(out.get("summary", out), indent = 2))
        print(f"wrote {args.out}")
        return 0

    if args.dry_run:
        out = dry_run(args)
        Path(args.out).write_text(json.dumps(out, indent = 2))
        for name, per in out.get("targets", {}).items():
            for a, i in per.items():
                print(f"{name:22s} {a:8s} "
                      + ("ok   " + ", ".join(i["mma"]) if i["compiled"]
                         else "FAIL " + i["error"][:120]))
        print(f"wrote {args.out}")
        return 0

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device; use --dry-run for the compile-only check")
        return 3
    caps = capabilities()
    env = environment()
    print(json.dumps(env, indent = 1))
    # A tiny GUARDED mm_fp4 on every visible device, before any real work. On a multi-GPU box an
    # unguarded flashinfer NVFP4 launch onto a non-current device hangs the card unrecoverably;
    # this both installs the habit and reports, per device, whether the vendor path works at all.
    pf = preflight()
    print("preflight: " + json.dumps(pf, indent = 1), flush = True)
    if any(isinstance(v, dict) and v.get("mm_fp4", "").startswith(("RuntimeError", "Error"))
           for v in pf.values()):
        print("preflight reported a device where mm_fp4 does not work; those columns will be "
              "skipped with a reason rather than silently replaced", flush = True)
    groups = [g.strip() for g in args.shapes.split(",") if g.strip()]
    ranks = [int(r) for r in args.ranks.split(",")]
    shapes = []
    for gname in groups:
        if gname not in SHAPES:
            print(f"unknown shape group {gname!r}; known: {list(SHAPES)}")
            return 2
        shapes += SHAPES[gname]

    rows = []
    for label, M, K, N in shapes:
        for rank in ranks:
            t0 = time.time()
            try:
                r = bench_shape(label, M, K, N, rank, args, caps)
            except torch.cuda.OutOfMemoryError as exc:
                r = {"label": label, "M": M, "K": K, "N": N, "rank": rank,
                     "skipped": {"all": f"OutOfMemory: {str(exc)[:120]}"}}
            r["wall_s"] = time.time() - t0
            rows.append(r)
            tt = r.get("times", {})
            def g(k):
                v = tt.get(k)
                return f"{v['p50_ms']:.4f}" if v else "   -  "
            print(f"{label:24s} r{rank:<3d} mm_fp4 {g('mm_fp4')}  unfused "
                  f"{g('unfused_branch_full')}  kaug {g('kaug_gemm_only')}  triton_fused "
                  f"{g('triton_fused')}  tk {g('triton_kernels_nvfp4')}", flush = True)
            for k, v in r.get("skipped", {}).items():
                print(f"      SKIP {k}: {v}", flush = True)
            torch.cuda.empty_cache()

    out = {"env": env, "caps": caps, "preflight": pf, "args": vars(args), "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent = 2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
