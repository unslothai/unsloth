# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention kernel's CUDA source and its loader.

The kernel is compiled at first use with NVRTC, the runtime compiler the CUDA builds of torch already depend on, and
launched through the CUDA driver on torch's context and current stream. Nothing is built at install time and no
compiler toolchain is needed. ``diffusion_int8_attention`` owns everything around it (quantisers, configs, the
dispatch).

A compile is paid once per machine: NVRTC makes the machine code once per architecture and process (every card of
that architecture loads the same image, so the driver compiles nothing at load), and the image is kept under
Studio's cache root, keyed by the source, the defines, the register cap, the architecture, NVRTC and the driver's
CUDA version, for the next process. Where NVRTC is older than the driver, its PTX is kept instead and the driver
compiles it at load, as before.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import threading
from typing import Any, Optional

KERNEL_NAME = "unsloth_i8attn"

KERNEL_SOURCE = r"""
// INT8 scaled dot-product attention for DiT inference (no mask). One warp owns 16 * NQ query rows.
//
// Q.K^T and P.V both run on the int8 tensor cores (mma.sync m16n8k32, int32 accumulate); the online softmax state is
// fp32 and stays in registers. The block's warps share one K tile and one V tile in shared memory (copied ahead with
// cp.async while the previous tile is computed, one barrier per tile); each warp keeps its Q rows in shared memory
// too. The probabilities go from the Q.K^T accumulators straight into the P.V operand registers.
//
// No int <-> float conversion instruction runs in the tile loop: the int32 accumulators start at the bit pattern of
// MAGIC_F = 2^23 + 2^22, so their bits read as a float are that constant plus the integer sum, and probabilities are
// rounded by adding the same constant. The output accumulators take the P.V sums with the constant still in them
// (one fused multiply-add per element); the constant times the sum of the tile weights comes off every FLUSH tiles,
// so the rounding it adds stays a few units of the last place of MAGIC_F per row whatever the key count, against
// int8 rounding noise of hundreds of units.
//
// Method notes (ideas, not code): int8 Q/K with a softmax-invariant K shift and block scales follows SageAttention
// (arXiv 2410.02367); K scales per accumulator lane group follow SageAttention2 (arXiv 2411.10958); integer
// probabilities with the denominator summed over the rounded values follow INT-FlashAttention (arXiv 2409.16997),
// here rescaled per key tile so each tile uses the full 8-bit range.
//
// sm75 has int8 tensor cores too, with a smaller product (m8n8k16) and no cp.async: there (SM75) one m16n8k32 is
// four m8n8k16 on the same operand registers, so the integer sums are the same, and a tile is copied with plain
// loads and stores.
//
// Compile-time: HD (head dim, 64 or 128), BN (keys per tile, 64 or 128), NQ (16-row query tiles per warp),
// WPC (warps per block), STAGES (1 or 2 shared-memory tile buffers; 2 = the next tile is copied while this one is
// computed), QREG (1 = the Q operand fragments are read once into registers, 2 = straight from global memory, with
// no Q rows in shared memory), FLUSH, FLUSH_TAIL, SM75 (set by the target architecture), and the orderings below
// (QKW, PVW, SPREAD), none of which changes an output bit. SINKS (defined or not) adds the attention-sink outputs
// (rcnt, rml); without it the kernel is the one without the sink path, instruction for instruction.
// Launch: blockDim.x = 32 * WPC, grid.x = n_qpad / (16 * NQ * WPC), grid.y = batch * heads,
// dynamic shared memory = STAGES * STAGE_BYTES + (QREG == 2 ? 0 : WPC * 16 * NQ * KROW) bytes.

typedef unsigned int u32;
typedef int i32;
typedef signed char s8;
typedef long long i64;

#ifndef HD
#define HD 128
#endif
#ifndef BN
#define BN 64
#endif
#ifndef STAGES
#define STAGES 2
#endif
#ifndef NQ
#define NQ 1
#endif
#ifndef WPC
#define WPC 4
#endif
#ifndef QREG
#define QREG 0  // 1 / 2 = the Q operand fragments stay in registers instead of being re-read from shared memory per tile
#endif
#ifndef SM75
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define SM75 1  // m8n8k16 products and plain copies: what sm75 has (the same code runs on every later card)
#else
#define SM75 0
#endif
#endif
#ifndef FLUSH
#define FLUSH 16  // tiles between removals of MAGIC_F from the output accumulators (a power of two; 0 = only at the end)
#endif
#ifndef FLUSH_TAIL
#define FLUSH_TAIL 0  // 1 = the removal is a block of its own after P.V instead of sharing the rebase branch
#endif
// The orderings: the same integer sums and the same float operations per accumulator, issued in another order.
#ifndef QKW
#define QKW 0  // 1 = Q.K^T goes one k step at a time over pairs of key blocks: consecutive products never share an
               // accumulator, so a warp is not held to one product per product latency
#endif
#ifndef PVW
#define PVW 0  // > 0 = the same for P.V over groups of PVW channel blocks (even, a divisor of HD / 8)
#endif
#ifndef SPREAD
#define SPREAD 0  // > 0 = a thread issues its copies of the next tile one chunk per product pair of Q.K^T, from pair
                  // SPREAD - 1 on, instead of all at once after the barrier (needs QKW; two stages only)
#endif
#if STAGES != 2
#undef SPREAD
#define SPREAD 0  // one stage has no next tile to copy during the products
#endif
#if (SPREAD && !QKW) || (PVW && (PVW % 2 || (HD / 8) % PVW))
#error "unsupported combination of QKW / PVW / SPREAD"
#endif
#define NT (32 * WPC)   // threads per block
#define KK (HD / 32)    // k steps of Q.K^T
#define NBK (BN / 8)    // 8-key column blocks per tile
#define NBD (HD / 8)    // 8-channel column blocks of the output
#define KB32 (BN / 32)  // k steps of P.V
// Shared-memory rows carry 16 bytes of padding: the 32 lanes of a fragment load then hit 32 different banks.
#define KROW (HD + 16)
#define VROW (BN + 16)
#define STAGE_BYTES (BN * KROW + HD * VROW)
// clipped keys an RFOLD kernel stages at a time in the free tile buffers (int8 K row, fp32 residual row, K scale)
#define RF ((STAGES * STAGE_BYTES) >= 32 * (5 * HD + 4) ? 32 : 16)
// A tile is copied in 16-byte chunks, thread tid taking chunks tid, tid + NT, ...: with NT a multiple of the chunks
// per row, its destinations are a fixed stride apart.
#define TILE_CHUNKS (BN * HD / 16)
#define COPIES ((TILE_CHUNKS + NT - 1) / NT)
#define KSTEP ((NT / (HD / 16)) * KROW)
#define VSTEP ((NT / (BN / 16)) * VROW)
#define LOG2_255 7.99435343685886f
#define GUARD 64.0f
// bits of MAGIC_F; for an integer |x| < 2^22 the float with bits MAGIC_I + x is MAGIC_F + x exactly, and
// MAGIC_F + v rounds v in [0, 2^22) to the nearest integer (ties to even) in its low mantissa bits.
#define MAGIC_I 0x4b400000
#define MAGIC_F 12582912.0f
#define MASKED_I ((i32)0xff61b1e6)  // bits of -3.0e38f: a padded key, below every score as an int and as a float

__device__ __forceinline__ u32 ld32(const s8* p) { return *reinterpret_cast<const u32*>(p); }

__device__ __forceinline__ u32 smem_addr(const void* p) {
  u32 a;
  asm("{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }" : "=r"(a) : "l"(p));
  return a;
}

// 16 bytes from global to shared memory: asynchronous (cp.async, waited for with cp_async_wait_all) from sm80 on,
// a load and a store on sm75
__device__ __forceinline__ void cp_async16(u32 dst, const s8* src) {
#if SM75
  asm volatile(
      "{ .reg .b32 a, b, c, d; .reg .u64 p; cvt.u64.u32 p, %0; ld.global.v4.b32 {a, b, c, d}, [%1]; "
      "st.shared.v4.b32 [p], {a, b, c, d}; }"
      ::"r"(dst), "l"(src)
      : "memory");
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
#endif
}

// Four 8 x 16-byte matrices from shared memory: lane L passes the address of row L % 8 of matrix L / 8, and thread
// (g, t) receives bytes 4 * t .. 4 * t + 3 of row g of each matrix: the int8 B fragments of two k steps.
__device__ __forceinline__ void ldmatrix4(u32* r, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

#if SM75
__device__ __forceinline__ void cp_async_commit() {}

__device__ __forceinline__ void cp_async_wait_all() {}
#else
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
#endif

// Copy this thread's share of one K tile ([BN][HD], contiguous) and one V tile ([HD][BN], contiguous) into a stage.
// kdst / vdst / ksrc / vsrc: the thread's first chunk; last: whether its last chunk exists.
__device__ __forceinline__ void load_tile(u32 kdst, u32 vdst, const s8* ksrc, const s8* vsrc, bool last) {
#pragma unroll
  for (int i = 0; i < COPIES; ++i) {
    if (i + 1 < COPIES || TILE_CHUNKS % NT == 0 || last) {
      cp_async16(kdst + i * KSTEP, ksrc + i * NT * 16);
      cp_async16(vdst + i * VSTEP, vsrc + i * NT * 16);
    }
  }
  cp_async_commit();
}


__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

#if SM75
// D (8x8 s32: two registers) = A (8x16, row: one register) * B (16x8 s8, col: one register) + C. Register r of an
// m16n8k32 A operand is rows (r & 1 ? g + 8 : g), k half r >> 1, and its B and D registers split the same way.
#define MMA8(sign, d0, d1, a, b, c0, c1)                                                                    \
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32." sign ".s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"     \
               : "=r"(d0), "=r"(d1)                                                                         \
               : "r"(a), "r"(b), "r"(c0), "r"(c1))
#define MMA_PAIR(name, sign)                                                                \
  __device__ __forceinline__ void name(i32* d, const u32* a, u32 b0, u32 b1) {             \
    MMA8(sign, d[0], d[1], a[0], b0, d[0], d[1]);                                           \
    MMA8(sign, d[2], d[3], a[1], b0, d[2], d[3]);                                           \
    MMA8(sign, d[0], d[1], a[2], b1, d[0], d[1]);                                           \
    MMA8(sign, d[2], d[3], a[3], b1, d[2], d[3]);                                           \
  }                                                                                         \
  __device__ __forceinline__ void name##_c(i32* d, const u32* a, u32 b0, u32 b1, i32 c) {  \
    MMA8(sign, d[0], d[1], a[0], b0, c, c);                                                 \
    MMA8(sign, d[2], d[3], a[1], b0, c, c);                                                 \
    MMA8(sign, d[0], d[1], a[2], b1, d[0], d[1]);                                           \
    MMA8(sign, d[2], d[3], a[3], b1, d[2], d[3]);                                           \
  }
// D = A (16x32 s8 or u8, row) * B (32x8 s8, col) + D, and (_c) + c in every element: the first product
MMA_PAIR(mma_s8s8, "s8")
MMA_PAIR(mma_u8s8, "u8")
#else
// D = A (16x32 s8, row) * B (32x8 s8, col) + D
__device__ __forceinline__ void mma_s8s8(i32* d, const u32* a, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}

// D = A (16x32 s8, row) * B (32x8 s8, col) + c in every element: the first product of an accumulator
__device__ __forceinline__ void mma_s8s8_c(i32* d, const u32* a, u32 b0, u32 b1, i32 c) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}

__device__ __forceinline__ void mma_u8s8_c(i32* d, const u32* a, u32 b0, u32 b1, i32 c) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %10, %10, %10};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}

// D = A (16x32 u8, row) * B (32x8 s8, col) + D
__device__ __forceinline__ void mma_u8s8(i32* d, const u32* a, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}

#endif

// A probability before rounding, never above 255. The exponent is rounded in the frame of MAGIC_F * scale, so for
// large scores (sharp attention: late DiT blocks) the largest key of a tile can come out above 255.5; unclamped it
// would round to 256, which pack4 stores as 0 with a carry into the next key.
__device__ __forceinline__ float prob(float x) { return fminf(ex2(x), 255.0f); }

// Four floats MAGIC_F + p (p an integer in [0, 255]) -> the four p packed low byte first.
__device__ __forceinline__ u32 pack4(float a, float b, float c, float d) {
  return (__float_as_uint(a) + (__float_as_uint(b) << 8) + (__float_as_uint(c) << 16) + (__float_as_uint(d) << 24)) -
         (u32)MAGIC_I * 0x01010101u;
}

__device__ __forceinline__ u32 bytesum(u32 w, u32 acc) {
  u32 y;
  asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(y) : "r"(w), "r"(0x01010101u), "r"(acc));
  return y;
}

__device__ __forceinline__ u32 f2bf16(float x) {
  u32 u = __float_as_uint(x);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

__device__ __forceinline__ u32 f2h(float x) {
  unsigned short y;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(y) : "f"(x));
  return (u32)y;
}

#if SPREAD
// A thread's share of the copy of the next tile (what load_tile takes), issued one chunk at a time
struct Copy {
  u32 kdst, vdst;
  const s8 *ksrc, *vsrc;
  bool on, last;
};

// Chunk i of the share (i < COPIES)
__device__ __forceinline__ void copy_chunk(const Copy& cp, int i) {
  if (cp.on && (i + 1 < COPIES || TILE_CHUNKS % NT == 0 || cp.last)) {
    cp_async16(cp.kdst + i * KSTEP, cp.ksrc + i * NT * 16);
    cp_async16(cp.vdst + i * VSTEP, cp.vsrc + i * NT * 16);
  }
}
#endif

#if QKW
// Q.K^T of one 16-row query tile, one k step at a time over pairs of key blocks. qa: the Q operand fragments
// (QREG), else they are read at qaddr; kaddr: this lane's address of the K tile for matrices (keys 0-7, bytes
// 0-15), (0-7, 16-31), (8-15, 0-15), (8-15, 16-31).
__device__ __forceinline__ void qk_pairs(
    i32 (*c)[4], const u32 (*qa)[4], u32 qaddr, u32 kaddr, i32 magic
#if SPREAD
    , const Copy& cp
#endif
) {
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
#if QREG
    const u32* a = qa[kk];
#else
    u32 a[4];
    ldmatrix4(a, qaddr + 32 * kk);
#endif
#pragma unroll
    for (int nb = 0; nb < NBK; nb += 2) {
      u32 b[4];
      ldmatrix4(b, kaddr + nb * 8 * KROW + 32 * kk);
      if (kk == 0) {
        mma_s8s8_c(c[nb], a, b[0], b[1], magic);
        mma_s8s8_c(c[nb + 1], a, b[2], b[3], magic);
      } else {
        mma_s8s8(c[nb], a, b[0], b[1]);
        mma_s8s8(c[nb + 1], a, b[2], b[3]);
      }
#if SPREAD
      {
        const int i = kk * (NBK / 2) + nb / 2 - (SPREAD - 1);
        if (i >= 0 && i < COPIES) copy_chunk(cp, i);
      }
#endif
    }
  }
}
#endif

#if PVW
// P.V of one key tile for the warp's NQ query tiles, one k step at a time over PVW channel blocks. vaddr: this
// lane's address of the V tile for matrices (channels 0-7, bytes 0-15), (0-7, 16-31), (8-15, 0-15), (8-15, 16-31).
__device__ __forceinline__ void pv_groups(
    float (*acc)[NBD][4], const u32 (*pa)[KB32][4], const float (*w)[2], u32 vaddr, i32 magic) {
#pragma unroll
  for (int nb0 = 0; nb0 < NBD; nb0 += PVW) {
    i32 d[NQ][PVW][4];
#pragma unroll
    for (int kb = 0; kb < KB32; ++kb) {
#pragma unroll
      for (int i = 0; i < PVW; i += 2) {
        u32 b[4];
        ldmatrix4(b, vaddr + (nb0 + i) * 8 * VROW + 32 * kb);
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt) {
          if (kb == 0) {
            mma_u8s8_c(d[qt][i], pa[qt][kb], b[0], b[1], magic);
            mma_u8s8_c(d[qt][i + 1], pa[qt][kb], b[2], b[3], magic);
          } else {
            mma_u8s8(d[qt][i], pa[qt][kb], b[0], b[1]);
            mma_u8s8(d[qt][i + 1], pa[qt][kb], b[2], b[3]);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < PVW; ++i) {
#pragma unroll
      for (int qt = 0; qt < NQ; ++qt) {
        acc[qt][nb0 + i][0] = fmaf(__int_as_float(d[qt][i][0]), w[qt][0], acc[qt][nb0 + i][0]);
        acc[qt][nb0 + i][1] = fmaf(__int_as_float(d[qt][i][1]), w[qt][0], acc[qt][nb0 + i][1]);
        acc[qt][nb0 + i][2] = fmaf(__int_as_float(d[qt][i][2]), w[qt][1], acc[qt][nb0 + i][2]);
        acc[qt][nb0 + i][3] = fmaf(__int_as_float(d[qt][i][3]), w[qt][1], acc[qt][nb0 + i][3]);
      }
    }
  }
}
#endif

// q   [BH, n_qpad, HD] int8, rows padded with zeros to a multiple of 16 * NQ * warps
// k   [BH, n_kpad, HD] int8, n_kpad a multiple of BN
// vt  [BH, n_kpad / BN, HD, BN] int8: per key tile, channel-major V; inside every 16 keys, position 4*t + c holds
//     key 8*(c>>1) + 2*t + (c&1)
// qs  [BH, n_qpad] per-row Q scale (softmax scale * log2(e) folded in)
// ks  [BH, n_kpad / 32, 4] K scale of the keys with (key % 8) / 2 == t in a run of 32 keys
// vs  [BH, n_kpad / BN] V scale of a tile; vg / vc [BH, HD] per-channel gain and centre of V
// out [B, n_q, H, HD] (strides in elements); out_kind 0 = fp16, 1 = bf16, 2 = fp32
// rcnt [BH] the keys whose V a head clipped (unsloth_i8quant_kv); rml [BH, n_qpad, 2] = (mref, l) of every row of
// a head with rcnt > 0, for unsloth_i8resid to add those keys back. RFOLD (with SINKS): the kernel adds them back
// itself before the store (ridx [BH, rstride] / rres [BH, rstride, HD] as unsloth_i8resid reads them) and writes no
// rml: no second launch, no second pass over the output and the Q rows
extern "C" __global__ void unsloth_i8attn(
    const s8* __restrict__ q, const s8* __restrict__ k, const s8* __restrict__ vt, const float* __restrict__ qs,
    const float* __restrict__ ks, const float* __restrict__ vs, const float* __restrict__ vg,
    const float* __restrict__ vc, void* __restrict__ out, int n_q, int n_qpad, int n_k, int n_kpad, int n_heads,
#ifndef SINKS
    i64 o_stride_b, i64 o_stride_s, i64 o_stride_h, int out_kind) {
#else
    i64 o_stride_b, i64 o_stride_s, i64 o_stride_h, int out_kind, const int* __restrict__ rcnt,
#ifndef RFOLD
    float* __restrict__ rml) {
#else
    float* __restrict__ rml, const int* __restrict__ ridx, const float* __restrict__ rres, int rstride) {
#endif
#endif
  extern __shared__ __align__(16) s8 smem[];
  const u32 lane = threadIdx.x & 31u;
  const u32 g = lane >> 2, t = lane & 3u;
  const u32 tid = threadIdx.x, warp = tid >> 5;
  const i64 bh = blockIdx.y;
  // first of this warp's 16 * NQ rows; thread (g, t) holds rows g and g + 8 of each of its NQ tiles
  const u32 rowbase = (blockIdx.x * WPC + warp) * (16u * NQ);
  const int n_tiles = n_kpad / BN;
  // this thread's first chunk of tile 0 in K and V, and where its chunks go inside a stage
  const s8* ksrc = k + bh * n_kpad * HD + tid * 16u;
  const s8* vsrc = vt + bh * n_kpad * HD + tid * 16u;
  const u32 kdst = (tid / (HD / 16)) * KROW + (tid % (HD / 16)) * 16u;
  const u32 vdst = BN * KROW + (tid / (BN / 16)) * VROW + (tid % (BN / 16)) * 16u;
  const bool last = tid + (COPIES - 1) * NT < TILE_CHUNKS;
  i32 magic;  // MAGIC_I in one register for the whole loop: the start value of every accumulator
  asm volatile("mov.b32 %0, 0x4b400000;" : "=r"(magic));
  const float* kst = ks + bh * n_tiles * (4 * KB32) + t;
  const float* vst = vs + bh * n_tiles;
  const u32 sbase = smem_addr(smem);

#if QREG != 2
  // this warp's Q rows, padded like the K rows
  const u32 qsm = sbase + STAGES * STAGE_BYTES + warp * (16u * NQ * KROW);
  {
    const s8* qrows = q + (bh * n_qpad + rowbase) * HD;
    for (u32 c = lane; c < 16u * NQ * HD / 16u; c += 32u)
      cp_async16(qsm + (c / (HD / 16)) * KROW + (c % (HD / 16)) * 16, qrows + c * 16);
    cp_async_commit();
  }
  // A fragments of rows (g, g + 8): matrices (rows 0-7, bytes 0-15), (rows 8-15, 0-15), (0-7, 16-31), (8-15, 16-31)
  const u32 qofs = qsm + ((lane & 7u) + ((lane >> 3) & 1u) * 8u) * KROW + (lane >> 4) * 16u;
#endif
  // B fragments: this lane's row / matrix offsets
  const u32 kofs = (lane & 7u) * KROW + 16u * (lane >> 3), vofs = BN * KROW + (lane & 7u) * VROW + 16u * (lane >> 3);
#if QKW || PVW
  // the same for one k step of two blocks: matrices (rows 0-7, bytes 0-15), (0-7, 16-31), (8-15, 0-15), (8-15, 16-31)
  const u32 pair = (lane & 7u) + 8u * (lane >> 4), half = 16u * ((lane >> 3) & 1u);
  const u32 kofs2 = pair * KROW + half, vofs2 = BN * KROW + pair * VROW + half;
#endif

  float sc[NQ][2], mref[NQ][2], l[NQ][2], wsum[NQ][2];
  float acc[NQ][NBD][4];
#pragma unroll
  for (int qt = 0; qt < NQ; ++qt) {
    const float* p = qs + bh * n_qpad + rowbase + 16 * qt + g;
    sc[qt][0] = p[0];
    sc[qt][1] = p[8];
    mref[qt][0] = mref[qt][1] = l[qt][0] = l[qt][1] = wsum[qt][0] = wsum[qt][1] = 0.f;
#pragma unroll
    for (int nb = 0; nb < NBD; ++nb) acc[qt][nb][0] = acc[qt][nb][1] = acc[qt][nb][2] = acc[qt][nb][3] = 0.f;
  }

#if QREG == 2
  // the Q operand fragments for the whole call, each register read where ldmatrix would have found it
  u32 qa[NQ][KK][4];
  const s8* qrows = q + (bh * n_qpad + rowbase) * HD;
#pragma unroll
  for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const s8* p = qrows + (16 * qt + g) * HD + 32 * kk + 4 * t;
      qa[qt][kk][0] = ld32(p);
      qa[qt][kk][1] = ld32(p + 8 * HD);
      qa[qt][kk][2] = ld32(p + 16);
      qa[qt][kk][3] = ld32(p + 8 * HD + 16);
    }
  }
#elif QREG
  // the Q operand fragments for the whole call (this warp's own copies: no barrier needed)
  u32 qa[NQ][KK][4];
  cp_async_wait_all();
#pragma unroll
  for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) ldmatrix4(qa[qt][kk], qofs + qt * 16 * KROW + 32 * kk);
  }
#endif
#if STAGES == 2
  load_tile(sbase + kdst, sbase + vdst, ksrc, vsrc, last);
#endif
  for (int j = 0; j < n_tiles; ++j) {
#if STAGES == 2
    // tile j is in stage j & 1 once every thread's copies landed; the other stage is free for tile j + 1
    cp_async_wait_all();
    __syncthreads();
#if SPREAD
    Copy cpy;
    {
      const u32 next = sbase + ((j + 1) & 1) * STAGE_BYTES;
      const i64 off = (i64)(j + 1) * BN * HD;
      cpy.on = j + 1 < n_tiles;
      cpy.last = last;
      cpy.kdst = next + kdst;
      cpy.vdst = next + vdst;
      cpy.ksrc = ksrc + off;
      cpy.vsrc = vsrc + off;
    }
#else
    if (j + 1 < n_tiles) {
      const u32 next = sbase + ((j + 1) & 1) * STAGE_BYTES;
      const i64 off = (i64)(j + 1) * BN * HD;
      load_tile(next + kdst, next + vdst, ksrc + off, vsrc + off, last);
    }
#endif
    const u32 stage = sbase + (j & 1) * STAGE_BYTES;
#else
    __syncthreads();  // every warp is done with the previous tile
    load_tile(sbase + kdst, sbase + vdst, ksrc + (i64)j * BN * HD, vsrc + (i64)j * BN * HD, last);
    cp_async_wait_all();
    __syncthreads();
    const u32 stage = sbase;
#endif
    float ksc[KB32];
#pragma unroll
    for (int kb = 0; kb < KB32; ++kb) ksc[kb] = kst[4 * (KB32 * j + kb)];
    const float vsc = vst[j];
    const bool tail = (j == n_tiles - 1) && (n_k < n_kpad);  // only the last tile can hold padded keys
    // the u8 A fragments of P.V: register (2 * h + row) of k step kb holds keys
    // 32 * kb + 16 * h + 8 * (c >> 1) + 2 * t + (c & 1), c = 0..3
    u32 pa[NQ][KB32][4];
    float w[NQ][2];
#pragma unroll
    for (int qt = 0; qt < NQ; ++qt) {
      // c[nb] read as floats: MAGIC_F + score of rows (g, g + 8) against keys 8 * nb + 2 * t + {0, 1}
      i32 c[NBK][4];
#if QKW
#if QREG
#define QK_Q qa[qt], 0u
#else
#define QK_Q nullptr, qofs + qt * 16 * KROW
#endif
#if SPREAD
#define QK_COPY , cpy
#else
#define QK_COPY
#endif
      qk_pairs(c, QK_Q, stage + kofs2, magic QK_COPY);
#if SPREAD
      cpy.on = false;  // the copies ride the first query tile
#endif
#else
#pragma unroll
      for (int kk = 0; kk < KK; kk += 2) {
#if QREG
        const u32* a0 = qa[qt][kk];
        const u32* a1 = qa[qt][kk + 1];
#else
        u32 a0[4], a1[4];
        ldmatrix4(a0, qofs + qt * 16 * KROW + 32 * kk);
        ldmatrix4(a1, qofs + qt * 16 * KROW + 32 * kk + 32);
#endif
#pragma unroll
        for (int nb = 0; nb < NBK; ++nb) {
          u32 b[4];
          ldmatrix4(b, stage + kofs + nb * 8 * KROW + 32 * kk);
          if (kk == 0) mma_s8s8_c(c[nb], a0, b[0], b[1], magic);
          else mma_s8s8(c[nb], a0, b[0], b[1]);
          mma_s8s8(c[nb], a1, b[2], b[3]);
        }
      }
#endif
      if (tail) {
        asm volatile("// padded keys");  // keeps this a branch: no selects on the other tiles
#pragma unroll
        for (int nb = 0; nb < NBK; ++nb) {
          const int key = j * BN + 8 * nb + 2 * (int)t;
          if (key >= n_k) c[nb][0] = c[nb][2] = MASKED_I;
          if (key + 1 >= n_k) c[nb][1] = c[nb][3] = MASKED_I;
        }
      }
      // score = fma(bits as float, f, g): one scale per 32 keys. f is never 0: a score scale that underflowed
      // (all-zero q and k rows) would turn the padded keys' -3e38 into 0, the largest score of the tile
      float mt0 = -3.0e38f, mt1 = -3.0e38f;
      float f0[KB32], f1[KB32];
#pragma unroll
      for (int kb = 0; kb < KB32; ++kb) {
        i32 mx0 = max(c[4 * kb][0], c[4 * kb][1]), mx1 = max(c[4 * kb][2], c[4 * kb][3]);
#pragma unroll
        for (int nb = 4 * kb + 1; nb < 4 * kb + 4; ++nb) {
          mx0 = max(mx0, max(c[nb][0], c[nb][1]));
          mx1 = max(mx1, max(c[nb][2], c[nb][3]));
        }
        f0[kb] = fmaxf(sc[qt][0] * ksc[kb], 1e-30f);
        f1[kb] = fmaxf(sc[qt][1] * ksc[kb], 1e-30f);
        mt0 = fmaxf(mt0, fmaf(__int_as_float(mx0), f0[kb], -MAGIC_F * f0[kb]));
        mt1 = fmaxf(mt1, fmaf(__int_as_float(mx1), f1[kb], -MAGIC_F * f1[kb]));
      }
      mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffffu, mt0, 1));
      mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffffu, mt1, 1));
      mt0 = fmaxf(mt0, __shfl_xor_sync(0xffffffffu, mt0, 2));
      mt1 = fmaxf(mt1, __shfl_xor_sync(0xffffffffu, mt1, 2));
      if (j == 0) {
        mref[qt][0] = mt0;
        mref[qt][1] = mt1;
      }
      // The running state stays in the frame of a fixed reference maximum; it is rebased only when a tile maximum
      // rises far above it (rare), so the output accumulators are not rescaled per tile.
      float d0 = mt0 - mref[qt][0], d1 = mt1 - mref[qt][1];
#if FLUSH > 0 && !FLUSH_TAIL
      // Every FLUSH tiles MAGIC_F times the weights since the last time comes off the output accumulators, so they
      // stay near the size of the P.V sums however many keys there are (left in until the end, the constant grows
      // with the key count and the sums lose their low bits to it). It shares the rare branch of the rebase: as a
      // block of its own after P.V (FLUSH_TAIL) it cost 3-5% of the call on sm80 / sm89, and saves 1-2% on sm120.
      const bool flush = (j & (FLUSH - 1)) == FLUSH - 1;
      if (__any_sync(0xffffffffu, (d0 > GUARD) | (d1 > GUARD)) || flush) {
        if (flush) {
#pragma unroll
          for (int nb = 0; nb < NBD; ++nb) {
            acc[qt][nb][0] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][0]);
            acc[qt][nb][1] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][1]);
            acc[qt][nb][2] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][2]);
            acc[qt][nb][3] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][3]);
          }
          wsum[qt][0] = wsum[qt][1] = 0.f;
        }
#else
      if (__any_sync(0xffffffffu, (d0 > GUARD) | (d1 > GUARD))) {
#endif
        if (d0 > GUARD) {
          const float r = ex2(-d0);
          l[qt][0] *= r;
          wsum[qt][0] *= r;
#pragma unroll
          for (int nb = 0; nb < NBD; ++nb) {
            acc[qt][nb][0] *= r;
            acc[qt][nb][1] *= r;
          }
          mref[qt][0] = mt0;
          d0 = 0.f;
        }
        if (d1 > GUARD) {
          const float r = ex2(-d1);
          l[qt][1] *= r;
          wsum[qt][1] *= r;
#pragma unroll
          for (int nb = 0; nb < NBD; ++nb) {
            acc[qt][nb][2] *= r;
            acc[qt][nb][3] *= r;
          }
          mref[qt][1] = mt1;
          d1 = 0.f;
        }
      }
      const float b0 = ex2(d0), b1 = ex2(d1);
      // probabilities min(255, 255 * 2^(score - tile max)), rounded by the MAGIC_F addition
      u32 sum0 = 0u, sum1 = 0u;
#pragma unroll
      for (int kb = 0; kb < KB32; ++kb) {
        const float fa = f0[kb], fb = f1[kb];
        const float oa = -MAGIC_F * fa + (LOG2_255 - mt0), ob = -MAGIC_F * fb + (LOG2_255 - mt1);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int na = 4 * kb + 2 * h, nb = na + 1;
          const u32 p0 = pack4(prob(fmaf(__int_as_float(c[na][0]), fa, oa)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[na][1]), fa, oa)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[nb][0]), fa, oa)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[nb][1]), fa, oa)) + MAGIC_F);
          const u32 p1 = pack4(prob(fmaf(__int_as_float(c[na][2]), fb, ob)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[na][3]), fb, ob)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[nb][2]), fb, ob)) + MAGIC_F,
                                prob(fmaf(__int_as_float(c[nb][3]), fb, ob)) + MAGIC_F);
          pa[qt][kb][2 * h] = p0;
          pa[qt][kb][2 * h + 1] = p1;
          sum0 = bytesum(p0, sum0);
          sum1 = bytesum(p1, sum1);
        }
      }
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
      sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
      sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
      l[qt][0] = fmaf((float)sum0, b0, l[qt][0]);
      l[qt][1] = fmaf((float)sum1, b1, l[qt][1]);
      w[qt][0] = b0 * vsc;
      w[qt][1] = b1 * vsc;
      wsum[qt][0] += w[qt][0];
      wsum[qt][1] += w[qt][1];
    }
    // P.V: every V fragment load serves the warp's NQ query tiles
#if PVW
    pv_groups(acc, pa, w, stage + vofs2, magic);
#else
#pragma unroll
    for (int nb = 0; nb < NBD; ++nb) {
      i32 d[NQ][4];
#pragma unroll
      for (int kb = 0; kb < KB32; kb += 2) {
        u32 b[4];
        ldmatrix4(b, stage + vofs + nb * 8 * VROW + 32 * kb);
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt) {
          if (kb == 0) mma_u8s8_c(d[qt], pa[qt][kb], b[0], b[1], magic);
          else mma_u8s8(d[qt], pa[qt][kb], b[0], b[1]);
        }
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt) mma_u8s8(d[qt], pa[qt][kb + 1], b[2], b[3]);
      }
#pragma unroll
      for (int qt = 0; qt < NQ; ++qt) {
        acc[qt][nb][0] = fmaf(__int_as_float(d[qt][0]), w[qt][0], acc[qt][nb][0]);
        acc[qt][nb][1] = fmaf(__int_as_float(d[qt][1]), w[qt][0], acc[qt][nb][1]);
        acc[qt][nb][2] = fmaf(__int_as_float(d[qt][2]), w[qt][1], acc[qt][nb][2]);
        acc[qt][nb][3] = fmaf(__int_as_float(d[qt][3]), w[qt][1], acc[qt][nb][3]);
      }
    }
#endif
#if FLUSH > 0 && FLUSH_TAIL
    // after tile j = before tile j + 1: the same operations in the same order as the rebase-branch placement, so
    // the output bits do not depend on the placement
    if (((j + 1) & (FLUSH - 1)) == FLUSH - 1) {
#pragma unroll
      for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
        for (int nb = 0; nb < NBD; ++nb) {
          acc[qt][nb][0] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][0]);
          acc[qt][nb][1] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][1]);
          acc[qt][nb][2] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][2]);
          acc[qt][nb][3] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][3]);
        }
        wsum[qt][0] = wsum[qt][1] = 0.f;
      }
    }
#endif
  }

#if defined(SINKS) && defined(RFOLD)
  // A head that clipped V (attention sinks, v_range): each clipped key's exact P (the int8 score unsloth_i8resid
  // takes, in the frame of this row's mref) times the part of its (v - centre) / gain clipped off, added to the
  // accumulators before the normaliser and the gain, so the output is rounded once. Uniform per block.
  // the head, row base and count read again after the tile loop (volatile): addresses shared with the setup above
  // would otherwise be kept in registers through the loop
  i64 hb;
  u32 rb;
  int nres;
  asm volatile("cvt.s64.u32 %0, %%ctaid.y;" : "=l"(hb));
  asm volatile("mov.u32 %0, %1;" : "=r"(rb) : "r"(rowbase));
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(nres) : "l"(rcnt + hb));
  if (nres > 0) {
    // MAGIC_F comes off first: the clipped parts are small next to it
#pragma unroll
    for (int qt = 0; qt < NQ; ++qt) {
#pragma unroll
      for (int nb = 0; nb < NBD; ++nb) {
        acc[qt][nb][0] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][0]);
        acc[qt][nb][1] = fmaf(-MAGIC_F, wsum[qt][0], acc[qt][nb][1]);
        acc[qt][nb][2] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][2]);
        acc[qt][nb][3] = fmaf(-MAGIC_F, wsum[qt][1], acc[qt][nb][3]);
      }
      wsum[qt][0] = wsum[qt][1] = 0.f;
    }
#if !QREG
    // the warp's Q rows stay in shared memory past the tile loop: this thread's words of its rows (g, g + 8 of each
    // tile) are read where the A fragments would hold them, bytes 32 kk + 4 t and 32 kk + 16 + 4 t
    const s8* rqp = smem + STAGES * STAGE_BYTES + warp * (16u * NQ * KROW) + g * KROW + 4 * t;
#endif
    float rqs[NQ][2];
#pragma unroll
    for (int qt = 0; qt < NQ; ++qt) {
      const float* p = qs + hb * n_qpad + rb + 16 * qt + g;
      rqs[qt][0] = p[0];
      rqs[qt][1] = p[8];
    }
    // every warp is past its last tile: the stages hold the clipped keys' int8 K rows, residuals and K scales, RF
    // keys at a time
    cp_async_wait_all();
    __syncthreads();
    i32* rk = reinterpret_cast<i32*>(smem);
    float* rr = reinterpret_cast<float*>(smem + RF * HD);
    float* rks = rr + RF * HD;
    for (int c0 = 0; c0 < nres; c0 += RF) {
      const int n = min(RF, nres - c0);
      if (c0 > 0) __syncthreads();
      for (int x = tid; x < n * (HD / 4); x += NT) {
        const int i = x / (HD / 4), w = x % (HD / 4);
        const int key = ridx[hb * rstride + c0 + i];
        rk[x] = *reinterpret_cast<const i32*>(k + (hb * n_kpad + key) * HD + 4 * w);
        if (w == 0) rks[i] = ks[(hb * (n_kpad / 32) + (key >> 5)) * 4 + ((key & 7) >> 1)];
      }
      for (int x = tid; x < n * HD; x += NT) rr[x] = rres[((i64)hb * rstride + c0) * HD + x];
      __syncthreads();
      for (int i = 0; i < n; ++i) {
        const i32* kw = rk + i * (HD / 4);
        const float ksc = rks[i];
#pragma unroll
        for (int qt = 0; qt < NQ; ++qt) {
          i32 s0 = 0, s1 = 0;
#pragma unroll
          for (int kk = 0; kk < KK; ++kk) {
            const i32 k0 = kw[8 * kk + t], k1 = kw[8 * kk + 4 + t];
#if QREG
            const u32* a = qa[qt][kk];
#else
            const s8* r = rqp + 16 * qt * KROW + 32 * kk;
            const u32 a[4] = {*reinterpret_cast<const u32*>(r), *reinterpret_cast<const u32*>(r + 8 * KROW),
                              *reinterpret_cast<const u32*>(r + 16), *reinterpret_cast<const u32*>(r + 8 * KROW + 16)};
#endif
            s0 = __dp4a((i32)a[0], k0, s0);
            s0 = __dp4a((i32)a[2], k1, s0);
            s1 = __dp4a((i32)a[1], k0, s1);
            s1 = __dp4a((i32)a[3], k1, s1);
          }
          // the four threads of a row hold its four quarters
          s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
          s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
          s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
          s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
          // |score| < 2^22: the int to float conversion by the magic addition, as unsloth_i8resid's; not rounded or
          // packed: the exact P of the clipped key, as unsloth_i8resid takes it
          const float f0 = __int_as_float(s0 + MAGIC_I) - MAGIC_F, f1 = __int_as_float(s1 + MAGIC_I) - MAGIC_F;
          const float e0 = fmaf(f0, fmaxf(rqs[qt][0] * ksc, 1e-30f), -mref[qt][0]);
          const float e1 = fmaf(f1, fmaxf(rqs[qt][1] * ksc, 1e-30f), -mref[qt][1]);
          const float p0 = 255.f * ex2(e0), p1 = 255.f * ex2(e1);
#pragma unroll
          for (int nb = 0; nb < NBD; ++nb) {
            const float2 r = *reinterpret_cast<const float2*>(rr + i * HD + 8 * nb + 2 * t);
            acc[qt][nb][0] = fmaf(p0, r.x, acc[qt][nb][0]);
            acc[qt][nb][1] = fmaf(p0, r.y, acc[qt][nb][1]);
            acc[qt][nb][2] = fmaf(p1, r.x, acc[qt][nb][2]);
            acc[qt][nb][3] = fmaf(p1, r.y, acc[qt][nb][3]);
          }
        }
      }
    }
  }
#endif

  const float* gp = vg + bh * HD + 2 * t;
  const float* cp = vc + bh * HD + 2 * t;
  const i64 base = (bh / n_heads) * o_stride_b + (bh % n_heads) * o_stride_h + 2 * t;
#pragma unroll
  for (int qt = 0; qt < NQ; ++qt) {
    const u32 row0 = rowbase + 16 * qt + g, row1 = row0 + 8u;
    const float i0 = 1.f / l[qt][0], i1 = 1.f / l[qt][1];
    const float w0 = wsum[qt][0], w1 = wsum[qt][1];  // the tiles since the last removal of MAGIC_F
    const bool live0 = row0 < (u32)n_q, live1 = row1 < (u32)n_q;
#pragma unroll
    for (int nb = 0; nb < NBD; ++nb) {
      const float ga = gp[8 * nb], gb = gp[8 * nb + 1], ca = cp[8 * nb], cb = cp[8 * nb + 1];
      const float x00 = fmaf(fmaf(-MAGIC_F, w0, acc[qt][nb][0]) * i0, ga, ca);
      const float x01 = fmaf(fmaf(-MAGIC_F, w0, acc[qt][nb][1]) * i0, gb, cb);
      const float x10 = fmaf(fmaf(-MAGIC_F, w1, acc[qt][nb][2]) * i1, ga, ca);
      const float x11 = fmaf(fmaf(-MAGIC_F, w1, acc[qt][nb][3]) * i1, gb, cb);
      const i64 e0 = base + (i64)row0 * o_stride_s + 8 * nb, e1 = base + (i64)row1 * o_stride_s + 8 * nb;
      if (out_kind == 2) {
        float* o = reinterpret_cast<float*>(out);
        if (live0) {
          o[e0] = x00;
          o[e0 + 1] = x01;
        }
        if (live1) {
          o[e1] = x10;
          o[e1 + 1] = x11;
        }
      } else {
        u32* o = reinterpret_cast<u32*>(out);
        const u32 y0 = out_kind == 1 ? (f2bf16(x00) | (f2bf16(x01) << 16)) : (f2h(x00) | (f2h(x01) << 16));
        const u32 y1 = out_kind == 1 ? (f2bf16(x10) | (f2bf16(x11) << 16)) : (f2h(x10) | (f2h(x11) << 16));
        if (live0) o[e0 >> 1] = y0;
        if (live1) o[e1 >> 1] = y1;
      }
    }
  }
#if defined(SINKS) && !defined(RFOLD)

  // A head that clipped V (attention sinks, v_range): its row maxima and normalisers for unsloth_i8resid, which adds
  // the clipped-off parts back. Uniform per block.
  if (rcnt[bh] > 0 && t == 0) {
#pragma unroll
    for (int qt = 0; qt < NQ; ++qt) {
      float* m = rml + ((i64)bh * n_qpad + rowbase + 16 * qt + g) * 2;
      m[0] = mref[qt][0];
      m[1] = l[qt][0];
      m[16] = mref[qt][1];
      m[17] = l[qt][1];
    }
  }
#endif
}
"""

QUANT_KERNELS = ("unsloth_i8stats_q", "unsloth_i8quant_kv")

QUANT_SOURCE = r"""
// The INT8 attention kernel's operands from [B, S, H, D] q / k / v of any float dtype and any strides on the first
// three dims (the last dim has stride 1 and every row starts 16-byte aligned: the caller checks). Two launches (a
// short call is mostly launches: every one less is time back), each a grid (batch * heads, jobs) whose blocks do
// the job their y index names. Blocks start in y order, so a launch works through its jobs one after the other,
// each across every head, and the order is chosen so that V is read the second time while the first read is still
// in the L2 cache:
//
//   unsloth_i8stats_q   Q, QROWS rows per block (rotated, one scale per row); the K mean over every kstep-th key
//                       (one block per head, a fixed summation order); last the V minimum / maximum per channel
//                       over every vstep-th key, in nvc row chunks, and each chunk's largest |v| with its key
//   unsloth_i8quant_kv  needs the statistics. V, one block per key tile, the last tile first ((v - centre) / gain
//                       per channel, one scale per tile, stored channel-major in the key order of the kernel's
//                       operand registers); then K, KROWS keys per block (centred, rotated, one scale per 32 keys
//                       and accumulator lane group); last the keys a head with an attention sink clipped
//
// and unsloth_i8resid, after the attention kernel: those clipped keys added back to the rows of their head.
//
// The rotation is the same fixed orthogonal map for Q and K (so Q.K^T is unchanged): a sign per channel, then an
// unnormalised Walsh-Hadamard transform over every RG channels (16 inside one thread, the rest by warp shuffles).
// Compile-time: HD (head dim), BN (keys per tile), RGX (log2(RG) - 4: the shuffle stages of the rotation), SINKS
// (defined or not: the attention-sink path, v_range, the residual job and unsloth_i8resid; without it the
// quantisers are the ones without that path, instruction for instruction).

typedef unsigned int u32;
typedef int i32;
typedef signed char s8;
typedef long long i64;
typedef unsigned short u16;

#ifndef HD
#define HD 128
#endif
#ifndef BN
#define BN 64
#endif
#ifndef RGX
#define RGX 0
#endif
#define TPR (HD / 16)   // threads per row: one thread holds 16 channels
#define NTV (BN * HD / 64)  // threads of a V tile: one thread holds 16 channels of 4 keys
#define NTKV (NTV > 32 * TPR ? NTV : 32 * TPR)  // threads per block of unsloth_i8quant_kv: a V tile, or KROWS keys
#define KROWS (NTKV / TPR)  // a multiple of 32: whole scale runs
#define NTS 256         // threads per block of unsloth_i8stats_q
#define SLOTS (NTS / TPR)
#define QROWS SLOTS     // rows of Q per block
#define NWS (NTS / 32)

struct __align__(16) W4 { u32 a, b, c, d; };
struct __align__(16) F4 { float a, b, c, d; };

__device__ __forceinline__ float h2f(u32 bits) {
  float y;
  const u16 h = (u16)bits;
  asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
  return y;
}

// 16 consecutive elements as fp32; kind 0 = fp16, 1 = bf16, 2 = fp32
__device__ __forceinline__ void load16(const void* base, i64 elem, int kind, float* x) {
  if (kind == 2) {
    const F4* p = reinterpret_cast<const F4*>(reinterpret_cast<const float*>(base) + elem);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const F4 f = p[i];
      x[4 * i] = f.a;
      x[4 * i + 1] = f.b;
      x[4 * i + 2] = f.c;
      x[4 * i + 3] = f.d;
    }
  } else {
    const W4* p = reinterpret_cast<const W4*>(reinterpret_cast<const u16*>(base) + elem);
    const W4 lo = p[0], hi = p[1];
    const u32 w[8] = {lo.a, lo.b, lo.c, lo.d, hi.a, hi.b, hi.c, hi.d};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (kind == 1) {
        x[2 * i] = __uint_as_float(w[i] << 16);
        x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
      } else {
        x[2 * i] = h2f(w[i] & 0xffffu);
        x[2 * i + 1] = h2f(w[i] >> 16);
      }
    }
  }
}

__device__ __forceinline__ float load1(const void* base, i64 elem, int kind) {
  if (kind == 2) return reinterpret_cast<const float*>(base)[elem];
  const u32 w = reinterpret_cast<const u16*>(base)[elem];
  return kind == 1 ? __uint_as_float(w << 16) : h2f(w);
}

// bit i: whether the rotation negates channel 16 * g16 + i
constexpr u32 flips(u32 g16) {
  u32 m = 0u;
  for (u32 i = 0; i < 16; ++i) {
    const u32 c = 16u * g16 + i;
    m |= ((((c * 7919u + 13u) * (c + 3u)) >> 7) & 1u) << i;
  }
  return m;
}
static const u32 FLIPS[8] = {flips(0), flips(1), flips(2), flips(3), flips(4), flips(5), flips(6), flips(7)};

// The rotation of the 16 channels 16 * g16 .. 16 * g16 + 15 of a row whose TPR threads sit in adjacent lanes.
__device__ __forceinline__ void rotate(float* x, u32 g16) {
  const u32 flip = FLIPS[g16];
#pragma unroll
  for (u32 i = 0; i < 16; ++i) x[i] = __uint_as_float(__float_as_uint(x[i]) ^ ((flip >> i) << 31));
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (!(j & (1 << m))) {
        const float a = x[j], b = x[j | (1 << m)];
        x[j] = a + b;
        x[j | (1 << m)] = a - b;
      }
    }
  }
#pragma unroll
  for (int s = 0; s < RGX; ++s) {
    const u32 upper = ((g16 >> s) & 1u) << 31;  // the upper lane of a pair holds o - x, as a sign flip and a sum
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float o = __shfl_xor_sync(0xffffffffu, x[i], 1 << s);
      x[i] = __uint_as_float(__float_as_uint(x[i]) ^ upper) + o;
    }
  }
}

__device__ __forceinline__ float amax16(const float* x) {
  float m = fabsf(x[0]);
#pragma unroll
  for (int i = 1; i < 16; ++i) m = fmaxf(m, fabsf(x[i]));
  return m;
}

// the maximum over the TPR lanes of a row
__device__ __forceinline__ float row_max(float m) {
#pragma unroll
  for (int s = 1; s < TPR; s <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
  return m;
}

__device__ __forceinline__ u32 q8(float x, float s) { return (u32)__float2int_rn(x * s) & 255u; }

__device__ __forceinline__ void store16(s8* dst, const float* x, float s) {
  W4 o;
  o.a = q8(x[0], s) | (q8(x[1], s) << 8) | (q8(x[2], s) << 16) | (q8(x[3], s) << 24);
  o.b = q8(x[4], s) | (q8(x[5], s) << 8) | (q8(x[6], s) << 16) | (q8(x[7], s) << 24);
  o.c = q8(x[8], s) | (q8(x[9], s) << 8) | (q8(x[10], s) << 16) | (q8(x[11], s) << 24);
  o.d = q8(x[12], s) | (q8(x[13], s) << 8) | (q8(x[14], s) << 16) | (q8(x[15], s) << 24);
  *reinterpret_cast<W4*>(dst) = o;
}

// kc    [BH, HD] the K mean over the keys 0, kstep, 2 * kstep, ...
// vpart [BH, nvc, 2, HD] per row chunk of vrows keys (a multiple of vstep): the minimum and the maximum of every
//       channel of V over the chunk's keys r0, r0 + vstep, ...
// vsink [BH, nvc, 2] per chunk: the largest |v| of any channel over those keys, and its key (as a float)
// qi    [BH, n_qpad, HD] int8 rotated q (zeros past n_q); qs [BH, n_qpad] the row maximum over 127, times mul
// grid (BH, jobs): job = job0 + y of qblocks + 1 + nvc, qblocks = (n_qpad + QROWS - 1) / QROWS (job0: a grid holds
// 65535 jobs, a longer call takes several launches). Job < qblocks holds QROWS rows of Q (thread: row, 16
// channels), job qblocks is the K mean, the others are the V chunks; NTS threads.
extern "C" __global__ void unsloth_i8stats_q(
    const void* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v, float* __restrict__ kc,
#ifndef SINKS
    float* __restrict__ vpart, s8* __restrict__ qi, float* __restrict__ qs, int n_q, int n_qpad, int n_k,
#else
    float* __restrict__ vpart, float* __restrict__ vsink, s8* __restrict__ qi, float* __restrict__ qs, int n_q,
    int n_qpad, int n_k,
#endif
    int n_heads, int nvc, int vrows, int vstep, int kstep, i64 q_sb, i64 q_ss, i64 q_sh, i64 k_sb, i64 k_ss,
    i64 k_sh, i64 v_sb, i64 v_ss, i64 v_sh, float mul, int q_kind, int k_kind, int v_kind, int job0) {
  __shared__ float sm[NWS][2][HD];
#ifdef SINKS
  __shared__ float smk[NWS][2];
#endif
  const u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const u32 slot = tid / TPR, g16 = tid % TPR;
  const i64 bh = blockIdx.x;
  const i64 b = bh / n_heads, h = bh % n_heads;
  const int qblocks = (n_qpad + QROWS - 1) / QROWS;
  const int job = job0 + (int)blockIdx.y;
  if (job < qblocks) {
    const i64 row = (i64)job * QROWS + slot;
    float x[16];
    if (row < n_q) {
      load16(q, b * q_sb + h * q_sh + row * q_ss + 16 * g16, q_kind, x);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = 0.f;
    }
    rotate(x, g16);
    const float amax = fmaxf(row_max(amax16(x)), 1e-20f);
    if (row < n_qpad) {
      store16(qi + (bh * n_qpad + row) * HD + 16 * g16, x, 127.f / amax);
      if (g16 == 0) qs[bh * n_qpad + row] = row < n_q ? amax * (mul / 127.f) : 0.f;
    }
    return;
  }
  const int chunk = job - qblocks - 1;  // of V; -1: the K mean
  const bool mean = chunk < 0;
  float lo[16], hi[16];  // the K mean block: lo holds the sums
#ifdef SINKS
  float amx = -1.f, akey = 0.f;  // a V chunk: the largest |v| this thread read, and its key
#endif
  if (mean) {
#pragma unroll
    for (int i = 0; i < 16; ++i) lo[i] = hi[i] = 0.f;
    const i64 base = b * k_sb + h * k_sh + 16 * g16;
    for (i64 r = (i64)slot * kstep; r < n_k; r += (i64)SLOTS * kstep) {
      float x[16];
      load16(k, base + r * k_ss, k_kind, x);
#pragma unroll
      for (int i = 0; i < 16; ++i) lo[i] += x[i];
    }
#pragma unroll
    for (int s = TPR; s < 32; s <<= 1) {
#pragma unroll
      for (int i = 0; i < 16; ++i) lo[i] += __shfl_xor_sync(0xffffffffu, lo[i], s);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      lo[i] = 3.0e38f;
      hi[i] = -3.0e38f;
    }
    const i64 r0 = (i64)chunk * vrows;
    const i64 r1 = min(r0 + (i64)vrows, (i64)n_k);
    const i64 base = b * v_sb + h * v_sh + 16 * g16;
    for (i64 r = r0 + (i64)slot * vstep; r < r1; r += (i64)SLOTS * vstep) {
      float x[16];
      load16(v, base + r * v_ss, v_kind, x);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        lo[i] = fminf(lo[i], x[i]);
        hi[i] = fmaxf(hi[i], x[i]);
      }
#ifdef SINKS
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) a = fmaxf(a, fabsf(x[i]));
      if (a > amx) {  // keys rise with r: the first key of the largest value is kept
        amx = a;
        akey = (float)r;
      }
#endif
    }
#pragma unroll
    for (int s = TPR; s < 32; s <<= 1) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        lo[i] = fminf(lo[i], __shfl_xor_sync(0xffffffffu, lo[i], s));
        hi[i] = fmaxf(hi[i], __shfl_xor_sync(0xffffffffu, hi[i], s));
      }
    }
#ifdef SINKS
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
      const float oa = __shfl_xor_sync(0xffffffffu, amx, s), ok = __shfl_xor_sync(0xffffffffu, akey, s);
      if (oa > amx || (oa == amx && ok < akey)) {
        amx = oa;
        akey = ok;
      }
    }
    if (lane == 0) {
      smk[warp][0] = amx;
      smk[warp][1] = akey;
    }
#endif
  }
  if (lane < TPR) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      sm[warp][0][16 * g16 + i] = lo[i];
      sm[warp][1][16 * g16 + i] = hi[i];
    }
  }
  __syncthreads();
  if (tid < HD) {
    float a = sm[0][0][tid], c = sm[0][1][tid];
#pragma unroll
    for (int w = 1; w < NWS; ++w) {
      if (mean) {
        a += sm[w][0][tid];
      } else {
        a = fminf(a, sm[w][0][tid]);
        c = fmaxf(c, sm[w][1][tid]);
      }
    }
    if (mean) {
      kc[bh * HD + tid] = a / (float)((n_k + kstep - 1) / kstep);
    } else {
      float* o = vpart + ((bh * nvc + chunk) * 2) * HD + tid;
      o[0] = a;
      o[HD] = c;
    }
  }
#ifdef SINKS
  if (!mean && tid == 0) {
    float a = smk[0][0], key = smk[0][1];
    for (int w = 1; w < NWS; ++w) {
      if (smk[w][0] > a || (smk[w][0] == a && smk[w][1] < key)) {
        a = smk[w][0];
        key = smk[w][1];
      }
    }
    vsink[(bh * nvc + chunk) * 2] = a;
    vsink[(bh * nvc + chunk) * 2 + 1] = key;
  }
#endif
}

#ifdef SINKS
// The V range a head quantises with, computed alike by every V-tile block and the residual job. A few keys far from
// the rest in most channels (attention sinks: the first text tokens of a joint sequence, 60-85x the median key norm)
// set every channel's range and cost every other key its resolution, while they draw well under 1% of the
// attention. So when the full range of at least half the channels is more than SINK_RATIO times their range without
// the chunk that holds the head's largest |v| (a sink is far out in most channels: 63-95% of them on FLUX.3 and
// Z-Image, against at most 9% for heavy-tailed V with no sink), the head uses that robust range (widened by
// SINK_MARGIN, never past the full range): that chunk's keys are clipped to it in place and the parts clipped off by
// more than SINK_MIN (in units of the gain) are added back exactly after the attention kernel (unsloth_i8quant_kv's
// residual job lists them, unsloth_i8resid adds them); each costs every query row a dot product, so only the sinks
// and the few keys near them go there (FLUX.3 / Z-Image captures: 2-25 keys a head, with a V error 0.4% above that of
// an exact residual for every clipped key; SINK_MIN 1/4 was 16% above it on FLUX.3). Any other head keeps the full
// range, bit for bit. smc / smr get the centre and 1 / gain of channels c < HD; returns whether the head clips.
#define SINK_RATIO 2.0f
#define SINK_MARGIN 2.0f
#define SINK_MIN 0.0625f
__device__ bool v_range(const float* __restrict__ vpart, const float* __restrict__ vsink, i64 bh, int nvc,
                        int vstep, float* smc, float* smr, float (*smx)[HD], float* smw, int* sc_out, u32 tid,
                        u32 nt) {
  int sc = 0;
  float best = -1.f;
  for (int j = 0; j < nvc; ++j) {
    const float a = vsink[(bh * nvc + j) * 2];
    if (a > best) {
      best = a;
      sc = j;
    }
  }
  int far = 0;  // channels whose full range is more than SINK_RATIO times the robust one
  for (u32 c = tid; c < HD; c += nt) {
    float lo = 3.0e38f, hi = -3.0e38f, lor = 3.0e38f, hir = -3.0e38f;
    const float* part = vpart + bh * nvc * 2 * HD + c;
    for (int j = 0; j < nvc; ++j) {
      const float a = part[(2 * j) * HD], b = part[(2 * j + 1) * HD];
      lo = fminf(lo, a);
      hi = fmaxf(hi, b);
      if (j != sc) {
        lor = fminf(lor, a);
        hir = fmaxf(hir, b);
      }
    }
    smc[c] = lo;
    smr[c] = hi;
    smx[0][c] = lor;
    smx[1][c] = hir;
    far += (hi - lo) > SINK_RATIO * (hir - lor);
  }
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) far += __shfl_xor_sync(0xffffffffu, far, s);
  if ((tid & 31u) == 0) smw[tid >> 5] = (float)far;
  __syncthreads();
  float r = 0.f;
  for (u32 w = 0; w < nt / 32; ++w) r += smw[w];
  const bool clip = nvc > 1 && vstep == 1 && 2.f * r >= (float)HD;
  for (u32 c = tid; c < HD; c += nt) {
    const float lo = smc[c], hi = smr[c];
    float centre = 0.5f * (hi + lo), gain = fmaxf(0.5f * (hi - lo), 1e-12f);
    if (clip) {
      centre = 0.5f * (smx[1][c] + smx[0][c]);
      gain = fmaxf(fminf(SINK_MARGIN * 0.5f * (smx[1][c] - smx[0][c]), 0.5f * (hi - lo)), 1e-12f);
    }
    smc[c] = centre;
    smr[c] = 1.f / gain;
    smx[0][c] = gain;
  }
  __syncthreads();
  *sc_out = sc;
  return clip;
}
#endif

// vt [BH, n_kpad / BN, HD, BN] int8 ((v - centre) / gain over the tile scale, zeros past n_k) in the key order of
// the kernel's operand registers; vs [BH, n_kpad / BN] the tile scales; vc / vg [BH, HD] centre and gain, merged from
// vpart by every V block (HD threads) and written by tile 0 (v_range: a head with an attention sink clips one chunk).
// rcnt [BH] / ridx [BH, rstride] / rres [BH, rstride, HD]: the keys a head clipped by more than SINK_MIN, in key
// order, and the parts of their (v - centre) / gain clipped off (the residual job, the last one of the grid).
// ki [BH, n_kpad, HD] int8 rotated (k - kc) (zeros past n_k); ks [BH, n_kpad / 32, 4] = the maximum over the keys
// of a run of 32 with the same (key % 8) / 2, over 127
// grid (BH, jobs): job = job0 + y of n_kpad / BN + n_kpad / KROWS, NTKV threads. Job < n_kpad / BN is a V tile,
// the last one first: thread (16 channels, word kq) reads its four keys row by row (whole 16-byte loads) and
// writes one 32-bit word per channel; adjacent lanes write adjacent words of a channel. The others hold KROWS keys
// of K: thread (key, 16 channels).
extern "C" __global__ void unsloth_i8quant_kv(
    const void* __restrict__ k, const void* __restrict__ v, const float* __restrict__ kc,
#ifndef SINKS
    const float* __restrict__ vpart, s8* __restrict__ ki, float* __restrict__ ks, s8* __restrict__ vt,
    float* __restrict__ vs, float* __restrict__ vc, float* __restrict__ vg, int n_k, int n_kpad, int n_heads,
    int nvc, i64 k_sb, i64 k_ss, i64 k_sh, i64 v_sb, i64 v_ss, i64 v_sh, int k_kind, int v_kind, int job0) {
#else
    const float* __restrict__ vpart, const float* __restrict__ vsink, s8* __restrict__ ki, float* __restrict__ ks,
    s8* __restrict__ vt, float* __restrict__ vs, float* __restrict__ vc, float* __restrict__ vg, int* __restrict__ rcnt,
    int* __restrict__ ridx, float* __restrict__ rres, int n_k, int n_kpad, int n_heads, int nvc, int vrows, int vstep,
    i64 k_sb, i64 k_ss, i64 k_sh, i64 v_sb, i64 v_ss, i64 v_sh, int k_kind, int v_kind, int job0) {
#endif
  __shared__ __align__(16) float smc[HD], smr[HD];
#ifdef SINKS
  __shared__ float smx[2][HD];
  __shared__ float smw[NTKV / 32];
  __shared__ int smf[KROWS + 1];
#endif
  __shared__ float sm[KROWS + KROWS / 8 > NTV / 32 + 1 ? KROWS + KROWS / 8 : NTV / 32 + 1];
  const u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const i64 tiles = n_kpad / BN;
  const i64 bh = blockIdx.x;
  const i64 b = bh / n_heads, h = bh % n_heads;
  const i64 job = (i64)job0 + blockIdx.y;
#ifdef SINKS
  if (job == tiles + n_kpad / KROWS) {
    // the residual job: the keys of the clipped chunk that left the range, compacted in key order
    int sc = 0;
    if (!v_range(vpart, vsink, bh, nvc, vstep, smc, smr, smx, smw, &sc, tid, NTKV)) {
      if (tid == 0) rcnt[bh] = 0;
      return;
    }
    const u32 rin = tid / TPR, g16 = tid % TPR;
    const i64 r0 = (i64)sc * vrows, r1 = min(r0 + (i64)vrows, (i64)n_k);
    int found = 0;
    for (i64 first = r0; first < r1; first += KROWS) {
      const i64 key = first + rin;
      float x[16];
      bool out = false;
      if (key < r1) {
        load16(v, b * v_sb + h * v_sh + key * v_ss + 16 * g16, v_kind, x);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float y = (x[i] - smc[16 * g16 + i]) * smr[16 * g16 + i];
          x[i] = y - fminf(fmaxf(y, -1.f), 1.f);
          out |= fabsf(x[i]) > SINK_MIN;
        }
      }
#pragma unroll
      for (int s = 1; s < TPR; s <<= 1) out |= __shfl_xor_sync(0xffffffffu, (int)out, s) != 0;
      if (g16 == 0) smf[rin] = out;
      __syncthreads();
      if (tid == 0) {
        int n = found;
        for (u32 i = 0; i < KROWS; ++i) {
          const int f = smf[i];
          smf[i] = f ? n : -1;
          n += f;
        }
        smf[KROWS] = n;
      }
      __syncthreads();
      const int at = smf[rin];
      if (at >= 0) {
        float* o = rres + ((i64)bh * vrows + at) * HD + 16 * g16;
#pragma unroll
        for (int i = 0; i < 16; ++i) o[i] = x[i];
        if (g16 == 0) ridx[bh * vrows + at] = (int)key;
      }
      found = smf[KROWS];
      __syncthreads();
    }
    if (tid == 0) rcnt[bh] = found;
    return;
  }
#endif
  if (job >= tiles) {
    const u32 rin = tid / TPR, g16 = tid % TPR;
    const i64 run = job - tiles;
    const i64 key = run * KROWS + rin;
    float x[16];
    if (key < n_k) {
      load16(k, b * k_sb + h * k_sh + key * k_ss + 16 * g16, k_kind, x);
      const F4* c = reinterpret_cast<const F4*>(kc + bh * HD + 16 * g16);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const F4 f = c[i];
        x[4 * i] -= f.a;
        x[4 * i + 1] -= f.b;
        x[4 * i + 2] -= f.c;
        x[4 * i + 3] -= f.d;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) x[i] = 0.f;
    }
    rotate(x, g16);
    const float amax = row_max(amax16(x));
    if (g16 == 0) sm[rin] = amax;
    __syncthreads();
    if (tid < KROWS / 8) {  // scale tid: the keys 32 * (tid / 4) + 8 * a + 2 * (tid % 4) + {0, 1}
      const u32 first = 32u * (tid >> 2) + 2u * (tid & 3u);
      float m = 0.f;
#pragma unroll
      for (int a = 0; a < 4; ++a) m = fmaxf(m, fmaxf(sm[first + 8 * a], sm[first + 8 * a + 1]));
      m = fmaxf(m, 1e-20f);
      sm[KROWS + tid] = m;
      ks[bh * (n_kpad / 8) + run * (KROWS / 8) + tid] = m / 127.f;
    }
    __syncthreads();
    store16(ki + (bh * n_kpad + key) * HD + 16 * g16, x, 127.f / sm[KROWS + 4u * (rin >> 5) + ((rin & 7u) >> 1)]);
    return;
  }
  if (tid >= NTV) return;
  const u32 g16 = tid / (BN / 4), kq = tid % (BN / 4);
  const i64 tile = tiles - 1 - job;  // the keys the statistics read last come first
#ifdef SINKS
  int sc = 0;
  const bool clip = v_range(vpart, vsink, bh, nvc, vstep, smc, smr, smx, smw, &sc, tid, NTV);
  if (tile == 0) {
    for (u32 c = tid; c < HD; c += NTV) {
      vc[bh * HD + c] = smc[c];
      vg[bh * HD + c] = smx[0][c];
    }
  }
#else
  for (u32 c = tid; c < HD; c += NTV) {
    float lo = 3.0e38f, hi = -3.0e38f;
    const float* part = vpart + bh * nvc * 2 * HD + c;
    for (int j = 0; j < nvc; ++j) {
      lo = fminf(lo, part[(2 * j) * HD]);
      hi = fmaxf(hi, part[(2 * j + 1) * HD]);
    }
    const float centre = 0.5f * (hi + lo), gain = fmaxf(0.5f * (hi - lo), 1e-12f);
    smc[c] = centre;
    smr[c] = 1.f / gain;
    if (tile == 0) {
      vc[bh * HD + c] = centre;
      vg[bh * HD + c] = gain;
    }
  }
  __syncthreads();
#endif
  float cen[16], rg[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const F4 c = reinterpret_cast<const F4*>(smc + 16 * g16)[i], r = reinterpret_cast<const F4*>(smr + 16 * g16)[i];
    cen[4 * i] = c.a;
    cen[4 * i + 1] = c.b;
    cen[4 * i + 2] = c.c;
    cen[4 * i + 3] = c.d;
    rg[4 * i] = r.a;
    rg[4 * i + 1] = r.b;
    rg[4 * i + 2] = r.c;
    rg[4 * i + 3] = r.d;
  }
  // byte r of word kq of a tile holds key 16 * (kq / 4) + 2 * (kq % 4) + 8 * (r / 2) + r % 2
  const i64 base = b * v_sb + h * v_sh + 16 * g16;
  float x[4][16];
  float amax = 0.f;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const i64 key = tile * BN + 16 * (kq >> 2) + 2 * (kq & 3u) + 8 * (r >> 1) + (r & 1);
    if (key < n_k) {
      load16(v, base + key * v_ss, v_kind, x[r]);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        x[r][i] = (x[r][i] - cen[i]) * rg[i];
#ifdef SINKS
        if (clip) x[r][i] = fminf(fmaxf(x[r][i], -1.f), 1.f);
#endif
        amax = fmaxf(amax, fabsf(x[r][i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) x[r][i] = 0.f;
    }
  }
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
  if (lane == 0) sm[warp] = amax;
  __syncthreads();
  if (warp == 0) {
    float m = lane < NTV / 32 ? sm[lane] : 0.f;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    if (lane == 0) {
      m = fmaxf(m, 1e-20f);
      sm[NTV / 32] = m;
      vs[bh * tiles + tile] = m / 127.f;
    }
  }
  __syncthreads();
  const float s = 127.f / sm[NTV / 32];
  u32* dst = reinterpret_cast<u32*>(vt + ((bh * tiles + tile) * HD + 16 * g16) * BN + 4 * kq);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    dst[i * (BN / 4)] = q8(x[0][i], s) | (q8(x[1][i], s) << 8) | (q8(x[2][i], s) << 16) | (q8(x[3][i], s) << 24);
}
#ifdef SINKS

#define NTR 256  // threads per block of unsloth_i8resid: a warp per query row
#define RCH 32   // clipped keys staged in shared memory at a time
#define RMAGIC_I 0x4b400000
#define RMAGIC_F 12582912.0f

// The keys a head clipped (attention sinks, v_range), added back after the attention kernel: each key's exact P,
// 255 * 2^(score - mref) from the same int8 score as the kernel's (qi . ki, the q scale times the K scale of its
// lane group) in the frame of the kernel's row maximum mref, times the part of its (v - centre) / gain clipped off,
// over the kernel's l (which holds the rounded P of the clipped key), times the gain. A head with rcnt 0 returns at
// once. rml [BH, n_qpad, 2] = (mref, l) per row from the attention kernel; out as the kernel writes it.
// grid (BH, blocks), NTR threads; warp w of block y takes the rows y * NTR / 32 + w, + gridDim.y * NTR / 32, ...;
// the block stages the keys' int8 K rows, residuals and K scales in shared memory, RCH keys at a time (all of them
// once where there are at most RCH, as on FLUX.3 and Z-Image).
extern "C" __global__ void unsloth_i8resid(
    const s8* __restrict__ qi, const s8* __restrict__ ki, const float* __restrict__ qs, const float* __restrict__ ks,
    const float* __restrict__ vg, const int* __restrict__ rcnt, const int* __restrict__ ridx,
    const float* __restrict__ rres, const float* __restrict__ rml, void* __restrict__ out, int n_q, int n_qpad,
    int n_kpad, int n_heads, int rstride, i64 o_sb, i64 o_ss, i64 o_sh, int out_kind) {
  __shared__ i32 skw[RCH][HD / 4];
  __shared__ float srr[RCH][HD];
  __shared__ float sks[RCH];
  const i64 bh = blockIdx.x;
  const int nres = rcnt[bh];
  if (nres == 0) return;
  const u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  constexpr int CPL = HD / 32;  // channels per lane: lane, lane + 32, ...
  constexpr int NW = NTR / 32;
  float g[CPL];
#pragma unroll
  for (int j = 0; j < CPL; ++j) g[j] = vg[bh * HD + lane + 32 * j];
  const i64 obase = (bh / n_heads) * o_sb + (bh % n_heads) * o_sh + lane;
  // keys c0 .. c0 + RCH - 1 (those below nres) into shared memory
  auto stage = [&](int c0) {
    const int n = min(RCH, nres - c0);
    for (int x = tid; x < n * (HD / 4); x += NTR) {
      const int i = x / (HD / 4), w = x % (HD / 4);
      const int key = ridx[bh * rstride + c0 + i];
      skw[i][w] = *reinterpret_cast<const i32*>(ki + (bh * n_kpad + key) * HD + 4 * w);
      if (w == 0) sks[i] = ks[(bh * (n_kpad / 32) + (key >> 5)) * 4 + ((key & 7) >> 1)];
    }
    for (int x = tid; x < n * HD; x += NTR) srr[x / HD][x % HD] = rres[(bh * rstride + c0) * HD + x];
  };
  const bool once = nres <= RCH;
  if (once) {
    stage(0);
    __syncthreads();
  }
  // a whole number of row groups per block, so every warp meets the barriers of a restage
  for (int r0 = blockIdx.y * NW; r0 < n_q; r0 += gridDim.y * NW) {
    const int row = r0 + warp;
    const bool live = row < n_q;  // uniform per warp
    const i64 r = bh * n_qpad + (live ? row : 0);
    // HD / 4 words of the row; lanes past them hold zeros
    const i32 qw = lane < HD / 4 ? *reinterpret_cast<const i32*>(qi + r * HD + 4 * lane) : 0;
    const float qsc = qs[r], mref = rml[2 * r], il = 1.f / rml[2 * r + 1];
    float a[CPL];
#pragma unroll
    for (int j = 0; j < CPL; ++j) a[j] = 0.f;
    for (int c0 = 0; c0 < nres; c0 += RCH) {
      if (!once) {
        __syncthreads();
        stage(c0);
        __syncthreads();
      }
      const int n = min(RCH, nres - c0);
      for (int i = 0; i < n && live; ++i) {
        i32 sc = __dp4a(qw, lane < HD / 4 ? skw[i][lane] : 0, 0);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, m);
        // |score| < 2^22: the int to float conversion by the magic addition, as the kernel's
        const float f = __int_as_float(sc + RMAGIC_I) - RMAGIC_F;
        const float e = fmaf(f, fmaxf(qsc * sks[i], 1e-30f), -mref);
        float p;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(p) : "f"(e));
        p *= 255.f;
#pragma unroll
        for (int j = 0; j < CPL; ++j) a[j] = fmaf(p, srr[i][lane + 32 * j], a[j]);
      }
    }
    if (!live) continue;
#pragma unroll
    for (int j = 0; j < CPL; ++j) {
      const i64 e = obase + (i64)row * o_ss + 32 * j;
      const float y = fmaf(a[j] * il, g[j], load1(out, e, out_kind));
      if (out_kind == 2) {
        reinterpret_cast<float*>(out)[e] = y;
      } else {
        u16 h;
        if (out_kind == 1) {
          u32 u = __float_as_uint(y);
          u += 0x7fffu + ((u >> 16) & 1u);
          h = (u16)(u >> 16);
        } else {
          asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(y));
        }
        reinterpret_cast<u16*>(out)[e] = h;
      }
    }
  }
}
#endif
"""

_LOCK = threading.Lock()
_NVRTC: Any = None
_DRIVER: Any = None
_LAUNCH: Any = None
_FUNCTIONS: dict = {}
# Keyed by (source digest, arch, defines, max_registers); built once per process, shared.
_BINARIES: dict = {}
_IDENTITY: dict = {}

_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
_CU_FUNC_ATTRIBUTE_NUM_REGS = 4
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_KINDS = ("cubin", "ptx")  # the image kinds, in the order the disk is read

# The compiled kernels are kept on disk, so a process (and every card after the first) loads them instead of running
# NVRTC and the driver's PTX compiler again. ``UNSLOTH_DIFFUSION_INT8_ATTENTION_CACHE``: a directory, or ``0`` for no
# disk cache; unset, ``int8-attention`` under Studio's cache root.
CACHE_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_CACHE"
CACHE_DIR_NAME = "int8-attention"
# Bumped whenever what goes into an image changes in a way the key below does not see (compile options, the loader).
CACHE_FORMAT = 1
CACHE_KEEP = 3  # version directories kept, the most recently used first
CACHE_MAX_FILES = 64  # images per version directory
_MAGIC = b"UI8ATTN1"
_VERSION_DIR = re.compile(r"^v[0-9]+-[0-9a-f]{16}$")
_OFF = ("0", "off", "false", "no")
# cuModuleLoadData results (cuda.h CUresult) meaning the cubin can never load on this driver or GPU: INVALID_IMAGE
# (200), NO_BINARY_FOR_GPU (209), INVALID_PTX (218), JIT_COMPILER_NOT_FOUND (221), UNSUPPORTED_PTX_VERSION (222).
# The PTX is then used for that key (``_PTX_ONLY`` persists it). Other results say nothing about the image and are raised.
_IMAGE_UNUSABLE = frozenset({200, 209, 218, 221, 222})
_PTX_ONLY = ".ptx-only"  # suffix of the marker next to a key whose cubin this host cannot load
# (root, version dir) -> whether images may be read from and written to it (``_usable_dir``), checked once per process
_TRUSTED: dict = {}
_LOGGED: set = set()
_DIGESTS: dict = {}  # (source, ...) -> sha256 hex of the sources joined: the kernel sources are hashed once per process
# where each image came from in this process (read by tests)
CACHE_STATS = {
    "compiled": 0,
    "cubin": 0,
    "ptx": 0,
    "disk_hit": 0,
    "disk_write": 0,
    "disk_bad": 0,
    "memory_hit": 0,
    "load_fallback": 0,
    "disk_refused": 0,
}


def _library_dirs() -> list:
    dirs = [p for p in sys.path if p and os.path.isdir(p)]
    try:
        import torch
        dirs.insert(0, os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:  # noqa: BLE001
        pass
    return dirs


def _nvrtc_candidates() -> list:
    if os.name == "nt":
        patterns = ("nvrtc64_*.dll",)
        subdirs = (
            "",
            os.path.join("nvidia", "cuda_nvrtc", "bin"),
            os.path.join("nvidia", "cu13", "bin"),
            os.path.join("nvidia", "cu12", "bin"),
        )
        fallbacks: tuple = ()
    else:
        patterns = ("libnvrtc.so.*", "libnvrtc.so")
        subdirs = (
            "",
            os.path.join("nvidia", "cu13", "lib"),
            os.path.join("nvidia", "cuda_nvrtc", "lib"),
            os.path.join("nvidia", "cu12", "lib"),
        )
        fallbacks = ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so")
    found = []
    for d in _library_dirs():
        for sub in subdirs:
            for pattern in patterns:
                for path in sorted(glob.glob(os.path.join(d, sub, pattern)), reverse = True):
                    base = os.path.basename(path)
                    # the builtins / alt libraries are loaded by NVRTC itself
                    if "builtins" in base or ".alt." in base:
                        continue
                    found.append(path)
    return found + list(fallbacks)


def nvrtc() -> Any:
    """The NVRTC library, loaded once. Raises RuntimeError when no copy can be loaded."""
    global _NVRTC
    if _NVRTC is not None:
        return _NVRTC
    errors = []
    for name in _nvrtc_candidates():
        try:
            lib = ctypes.CDLL(name)
            lib.nvrtcGetErrorString.restype = ctypes.c_char_p
            _NVRTC = lib
            return lib
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("NVRTC not found: " + "; ".join(errors[:3]))


def nvrtc_version() -> tuple:
    lib = nvrtc()
    major, minor = ctypes.c_int(), ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    return major.value, minor.value


def _nvrtc_outputs(
    source: str, arch: str, defines: Optional[dict], extra: tuple, whats: tuple
) -> tuple:
    """``whats`` ("PTX", "CUBIN") of ``source`` compiled once for ``arch``. Raises RuntimeError with the compiler
    log."""
    lib = nvrtc()
    program = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(
        ctypes.byref(program), source.encode(), b"unsloth_int8_attention.cu", 0, None, None
    )
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram: {lib.nvrtcGetErrorString(rc).decode()}")
    try:
        options = [f"--gpu-architecture={arch}", "--std=c++17", "-default-device"]
        options += [f"-D{k}={v}" for k, v in (defines or {}).items()]
        options += list(extra)
        argv = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
        rc = lib.nvrtcCompileProgram(program, len(options), argv)
        size = ctypes.c_size_t()
        if rc != 0:
            lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(program, log)
            raise RuntimeError(
                f"NVRTC compile failed ({lib.nvrtcGetErrorString(rc).decode()}): "
                + log.value.decode(errors = "replace")[-2000:]
            )
        found = []
        for what in whats:
            get_size, get = (
                getattr(lib, f"nvrtcGet{what}Size", None),
                getattr(lib, f"nvrtcGet{what}", None),
            )
            if get_size is None or get is None:
                raise RuntimeError(f"this NVRTC has no nvrtcGet{what}")
            rc = get_size(program, ctypes.byref(size))
            if rc != 0 or not size.value:
                raise RuntimeError(f"nvrtcGet{what}Size: {lib.nvrtcGetErrorString(rc).decode()}")
            out = ctypes.create_string_buffer(size.value)
            rc = get(program, out)
            if rc != 0:
                raise RuntimeError(f"nvrtcGet{what}: {lib.nvrtcGetErrorString(rc).decode()}")
            found.append(out.raw)
        return tuple(found)
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(program))


def compile_ptx(
    source: str,
    arch: str,
    defines: Optional[dict] = None,
) -> bytes:
    """PTX (NUL-terminated) of ``source`` for ``arch`` such as "sm_89" or "compute_89" (the same PTX; the virtual
    architecture skips the machine-code pass NVRTC runs for a real one). Raises RuntimeError with the compiler log."""
    return _nvrtc_outputs(source, arch, defines, (), ("PTX",))[0]


def compile_cubin(
    source: str,
    arch: str,
    defines: Optional[dict] = None,
    max_registers: Optional[int] = None,
) -> bytes:
    """Machine code of ``source`` for a real ``arch`` ("sm_89") from NVRTC's PTX compiler, the same whole-program
    compile the driver runs on a PTX load. The register cap goes to that compiler alone (``--ptxas-options``): the
    PTX stays the uncapped PTX, as with ``cap_registers``."""
    extra = (f"--ptxas-options=--maxrregcount={int(max_registers)}",) if max_registers else ()
    return _nvrtc_outputs(source, arch, defines, extra, ("CUBIN",))[0]


def cap_registers(ptx: bytes, max_registers: int) -> bytes:
    """``.maxnreg`` on every entry function: registers per thread decide how many blocks an SM holds at once."""
    text = ptx.rstrip(b"\0").decode()
    out, pos = [], 0
    while True:
        i = text.find(".entry ", pos)
        if i < 0:
            break
        j = text.index("{", i)
        out.append(text[pos:j] + f".maxnreg {int(max_registers)}\n")
        pos = j
    out.append(text[pos:])
    return "".join(out).encode() + b"\0"


def driver() -> Any:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    return _DRIVER


def _launcher() -> Any:
    """``cuLaunchKernel`` with its argument types declared: a call then converts plain integers itself."""
    global _LAUNCH
    if _LAUNCH is None:
        fn = driver().cuLaunchKernel
        fn.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        fn.restype = ctypes.c_int
        _LAUNCH = fn
    return _LAUNCH


class CudaError(RuntimeError):
    """A driver call that returned ``rc``."""

    def __init__(self, what: str, rc: int):
        super().__init__(f"{what} failed with CUDA error {rc}")
        self.rc = int(rc)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise CudaError(what, rc)


def _file_identity(path: str) -> list:
    try:
        info = os.stat(path)
        return [os.path.basename(os.path.realpath(path)), int(info.st_size), int(info.st_mtime_ns)]
    except OSError:
        return [os.path.basename(path)]


def driver_version() -> Optional[tuple]:
    """(major, minor) of the CUDA version the driver supports, or None when there is no driver to ask."""
    try:
        version = ctypes.c_int()
        if driver().cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
            return version.value // 1000, (version.value % 1000) // 10
    except Exception:  # noqa: BLE001
        pass
    return None


def toolchain_identity() -> dict:
    """Image depends on NVRTC version and its library file (a patch release keeps the version
    number) and the driver CUDA version (picks kind: binary); read once per process"""
    found = _IDENTITY.get("value")
    if found is None:
        drv = driver_version()
        found = {
            "nvrtc": list(nvrtc_version()),
            "nvrtc_file": _file_identity(getattr(nvrtc(), "_name", "") or ""),
            "driver": list(drv) if drv else None,
        }
        _IDENTITY["value"] = found
    return found


def machine_code_matches_driver() -> bool:
    """True if NVRTC is at least the driver version: same instructions bit for bit (checked on sm75
    sm80 sm89 sm120); a newer driver may schedule them differently, so the driver compiles NVRTC
    PTX itself"""
    identity = toolchain_identity()
    drv = identity.get("driver")
    return not drv or tuple(identity["nvrtc"]) >= tuple(drv)


def cache_root() -> Optional[str]:
    """Where compiled images are kept: ``CACHE_ENV``, else ``int8-attention`` under Studio's cache root; None = no
    disk cache (switched off, or outside the Studio backend)."""
    raw = (os.environ.get(CACHE_ENV) or "").strip()
    if raw.lower() in _OFF:
        return None
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    try:
        from utils.paths.storage_roots import cache_root as studio_cache_root
        return os.path.join(str(studio_cache_root()), CACHE_DIR_NAME)
    except Exception:  # noqa: BLE001 -- outside the Studio backend: memory only
        return None


def version_dir_name() -> str:
    """One directory per code version: a new source or format starts a new one and the old ones are pruned."""
    return f"v{CACHE_FORMAT}-{source_digest()}"


def _digest(*sources: str) -> str:
    found = _DIGESTS.get(sources)
    if found is None:
        found = _DIGESTS[sources] = hashlib.sha256("".join(sources).encode()).hexdigest()
    return found


def cache_key(
    kind: str, source: str, arch: str, defines: dict, max_registers: Optional[int]
) -> str:
    """File name of an image: a hash of everything that went into it."""
    identity = toolchain_identity()
    fields = {
        "format": CACHE_FORMAT,
        "kind": kind,
        "source": _digest(source),
        "arch": arch,
        "defines": sorted((str(k), str(v)) for k, v in defines.items()),
        "max_registers": int(max_registers) if max_registers else None,
        "nvrtc": identity["nvrtc"],
        "nvrtc_file": identity["nvrtc_file"],
        "driver": identity.get("driver"),
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys = True).encode()).hexdigest()[:40]
    return f"{digest}.{kind}"


def _log_once(message: str) -> None:
    if message in _LOGGED:
        return
    _LOGGED.add(message)
    import logging

    logging.getLogger(__name__).warning(message)


def _private(info: os.stat_result) -> bool:
    """Owned by this user and writable by no one else: nothing in it can have been put there by another user."""
    return info.st_uid == os.getuid() and not info.st_mode & 0o022


def _usable_dir(root: str, version: str) -> bool:
    """Images go to disk only in a user-owned dir, not group/world-writable (POSIX; on Windows the
    dirs are made and used as is); else memory."""
    key = (root, version)
    found = _TRUSTED.get(key)
    if found is not None:
        return found
    folder = os.path.join(root, version)
    try:
        os.makedirs(root, mode = 0o700, exist_ok = True)
        os.makedirs(folder, mode = 0o700, exist_ok = True)
        found = os.name == "nt" or all(_private(os.stat(p)) for p in (root, folder))
        if not found:
            _log_once(
                f"INT8 attention: not using the kernel cache at {root}: it must be owned by this user and not "
                "writable by the group or others; compiling in memory"
            )
    except OSError:
        found = False  # read-only or not a directory: memory only, as before
    if not found:
        CACHE_STATS["disk_refused"] += 1
    _TRUSTED[key] = found
    return found


def _cache_read(path: str) -> Optional[bytes]:
    """The image at ``path``, or None when it is missing, unreadable, not this user's own private file (POSIX) or
    does not match its checksum (a torn or corrupt file is then removed and rebuilt)."""
    try:
        with open(path, "rb") as f:
            if os.name != "nt" and not _private(os.fstat(f.fileno())):
                CACHE_STATS["disk_refused"] += 1
                _log_once(f"INT8 attention: not loading {path}: not this user's own private file")
                return None
            data = f.read()
    except OSError:
        return None
    head = len(_MAGIC) + 32
    if (
        len(data) > head
        and data.startswith(_MAGIC)
        and hashlib.sha256(data[head:]).digest() == data[len(_MAGIC) : head]
    ):
        try:
            os.utime(path)  # most recently used, for the pruning
        except OSError:
            pass
        return data[head:]
    CACHE_STATS["disk_bad"] += 1
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def _cache_write(path: str, image: bytes) -> None:
    """Atomic: a temporary file in the same directory renamed over ``path``, so a reader in another process sees the
    old file, the new one, or none. Any failure only costs the next process a compile."""
    folder = os.path.dirname(path)
    tmp = None
    try:
        os.makedirs(folder, mode = 0o700, exist_ok = True)
        fd, tmp = tempfile.mkstemp(prefix = ".tmp-", dir = folder)  # mode 0o600
        with os.fdopen(fd, "wb") as f:
            f.write(_MAGIC + hashlib.sha256(image).digest() + image)
        os.replace(tmp, path)
        tmp = None
        CACHE_STATS["disk_write"] += 1
    except OSError:
        pass  # read-only or full disk; on Windows another process may hold ``path`` open: its copy is the same
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
    _prune(os.path.dirname(folder), os.path.basename(folder))


def _prune(root: str, current: str) -> None:
    """Keeps ``CACHE_KEEP`` version directories (``current`` and the most recently used others) and
    ``CACHE_MAX_FILES`` images in ``current``; drops temporary files left by a killed writer."""
    import shutil
    import time

    try:
        with os.scandir(root) as entries:
            # only names this module makes: a cache root pointed at a shared directory loses nothing else
            versions = [
                (e.stat().st_mtime, e.name)
                for e in entries
                if e.is_dir(follow_symlinks = False)
                and _VERSION_DIR.match(e.name)
                and e.name != current
            ]
        for _, name in sorted(versions, reverse = True)[max(0, CACHE_KEEP - 1) :]:
            shutil.rmtree(os.path.join(root, name), ignore_errors = True)
        folder = os.path.join(root, current)
        os.utime(folder)
        now = time.time()
        with os.scandir(folder) as entries:
            files = [
                (e.stat().st_mtime, e.name) for e in entries if e.is_file(follow_symlinks = False)
            ]
        images = sorted((t, n) for t, n in files if not n.startswith(".tmp-"))
        stale = [n for t, n in files if n.startswith(".tmp-") and now - t > 3600]
        stale += [n for _, n in images[: max(0, len(images) - CACHE_MAX_FILES)]]
        for name in stale:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
    except OSError:
        pass


def _memo(capability: tuple, source: str, defines: dict, max_registers: Optional[int]) -> tuple:
    return (
        _digest(source),
        f"sm_{capability[0]}{capability[1]}",
        tuple(sorted(defines.items())),
        max_registers,
    )


def _image_path(
    kind: str, capability: tuple, source: str, defines: dict, max_registers: Optional[int]
) -> Optional[str]:
    root = cache_root()
    if not root:
        return None
    version = version_dir_name()
    if not _usable_dir(root, version):
        return None
    arch = f"sm_{capability[0]}{capability[1]}"
    return os.path.join(root, version, cache_key(kind, source, arch, defines, max_registers))


def binary(
    capability: tuple,
    source: str,
    defines: dict,
    max_registers: Optional[int] = None,
) -> tuple:
    """(kind, image) to load ``source`` with ``defines`` from on a card of ``capability``: "cubin" (machine code,
    ``compile_cubin``) or, where NVRTC cannot make machine code or is older than the driver
    (``machine_code_matches_driver``), "ptx" (``cap_registers`` on ``compile_ptx``, as before the cache: the driver
    compiles it at load, through its own cache). Made once per process per architecture, and kept on disk
    (``cache_root``) for the next process. Needs no card: every card of that architecture loads the same image."""
    memo = _memo(capability, source, defines, max_registers)
    found = _BINARIES.get(memo)
    if found is not None:
        CACHE_STATS["memory_hit"] += 1
        return found
    paths = {kind: _image_path(kind, capability, source, defines, max_registers) for kind in _KINDS}
    # a process before this one found that this host cannot load the cubin of this key (``_forget``)
    ptx_only = bool(paths["cubin"]) and _cache_read(paths["cubin"] + _PTX_ONLY) is not None
    for kind in ("ptx",) if ptx_only else _KINDS:
        path = paths[kind]
        image = _cache_read(path) if path else None
        if image is not None:
            CACHE_STATS["disk_hit"] += 1
            _BINARIES[memo] = (kind, image)
            return _BINARIES[memo]
    CACHE_STATS["compiled"] += 1
    found = None
    if not ptx_only and machine_code_matches_driver():
        try:
            found = ("cubin", compile_cubin(source, memo[1], defines, max_registers))
        except Exception:  # noqa: BLE001 -- no nvrtcGetCUBIN, no --ptxas-options: fall back to the PTX
            found = None
    if found is None:
        ptx = compile_ptx(source, f"compute_{capability[0]}{capability[1]}", defines)
        found = ("ptx", cap_registers(ptx, max_registers) if max_registers else ptx)
    CACHE_STATS[found[0]] += 1
    path = paths[found[0]]
    if path:
        _cache_write(path, found[1])
    _BINARIES[memo] = found
    return found


def _forget(
    kind: str,
    capability: tuple,
    source: str,
    defines: dict,
    max_registers: Optional[int],
    replacement: tuple,
) -> None:
    """A cubin this driver or GPU cannot load (``_IMAGE_UNUSABLE``): ``replacement`` (the PTX) is used for the rest
    of the process and kept on disk under its own key, with a marker that sends the next processes straight to it;
    the cubin is dropped."""
    path = _image_path(kind, capability, source, defines, max_registers)
    if path:
        _cache_write(path + _PTX_ONLY, b"ptx")
        _cache_write(
            _image_path(replacement[0], capability, source, defines, max_registers), replacement[1]
        )
        try:
            os.remove(path)
        except OSError:
            pass
    _BINARIES[_memo(capability, source, defines, max_registers)] = replacement


def _load_image(image: bytes) -> ctypes.c_void_p:
    module = ctypes.c_void_p()
    _check(
        driver().cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(image)), "cuModuleLoadData"
    )
    return module


class Function:
    """One loaded kernel; ``launch`` runs it."""

    def __init__(self, handle: ctypes.c_void_p, name: str):
        self.handle = handle
        self.name = name
        self._shared_limit = 48 * 1024  # the driver's default cap on dynamic shared memory
        self._local = threading.local()

    def _attribute(self, attribute: int) -> int:
        out = ctypes.c_int()
        _check(
            driver().cuFuncGetAttribute(ctypes.byref(out), attribute, self.handle),
            "cuFuncGetAttribute",
        )
        return out.value

    @property
    def registers(self) -> int:
        """Registers per thread."""
        return self._attribute(_CU_FUNC_ATTRIBUTE_NUM_REGS)

    @property
    def local_bytes(self) -> int:
        """Per-thread stack frame: 0 means nothing was spilled out of registers."""
        return self._attribute(_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)

    def allow_shared(self, nbytes: int) -> None:
        if nbytes > self._shared_limit:
            _check(
                driver().cuFuncSetAttribute(
                    self.handle, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(nbytes)
                ),
                "cuFuncSetAttribute(max dynamic shared memory)",
            )
            self._shared_limit = int(nbytes)

    def launch(self, grid: tuple, block: int, values: tuple, shared: int, stream: int) -> None:
        """Launch with arguments given as plain integers: one 8-byte slot each (a pointer, an int64, or the bits of
        an int32 / fp32 in the low half), kept per thread so a launch builds no ctypes objects."""
        slots = getattr(self._local, "slots", None)
        if slots is None or len(slots) != len(values):
            slots = (ctypes.c_uint64 * len(values))()
            base = ctypes.addressof(slots)
            self._local.slots = slots
            self._local.pointers = (ctypes.c_void_p * len(values))(
                *[base + 8 * i for i in range(len(values))]
            )
        slots[:] = values
        rc = _launcher()(
            self.handle,
            grid[0],
            grid[1],
            1,
            block,
            1,
            1,
            shared,
            stream,
            self._local.pointers,
            None,
        )
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel({self.name}) failed with CUDA error {rc}")


def int_bits(value: int) -> int:
    """An int32 argument for ``Function.launch``."""
    return int(value) & 0xFFFFFFFF


def float_bits(value: float) -> int:
    """An fp32 argument for ``Function.launch``."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def load_module(
    device_index: int,
    capability: tuple,
    source: str,
    names: tuple,
    defines: dict,
    max_registers: Optional[int] = None,
) -> dict:
    """``source`` compiled with ``defines`` and loaded on torch's context of ``device_index``: {name: Function},
    cached per process (the compiled image per architecture, ``binary``; the module per card). Not to be called while
    a CUDA graph is being captured (compiling and loading are not stream operations)."""
    import torch

    key = (int(device_index), names, tuple(sorted(defines.items())), max_registers)
    found = _FUNCTIONS.get(key)
    if found is not None:
        return found
    with _LOCK:
        found = _FUNCTIONS.get(key)
        if found is not None:
            return found
        kind, image = binary(capability, source, defines, max_registers)  # no card needed
        with torch.cuda.device(device_index):
            torch.zeros(
                1, device = f"cuda:{device_index}"
            )  # torch's context exists and is current on this thread
            try:
                module = _load_image(image)
            except CudaError as exc:
                if kind == "ptx" or exc.rc not in _IMAGE_UNUSABLE:
                    raise  # a transient failure (out of memory, ...): the cached cubin stays
                # machine code this driver or GPU cannot load: use the PTX from here on
                CACHE_STATS["load_fallback"] += 1
                ptx = compile_ptx(source, f"compute_{capability[0]}{capability[1]}", defines)
                if max_registers:
                    ptx = cap_registers(ptx, max_registers)
                _forget(kind, capability, source, defines, max_registers, ("ptx", ptx))
                module = _load_image(ptx)
            functions = {}
            for name in names:
                handle = ctypes.c_void_p()
                _check(
                    driver().cuModuleGetFunction(ctypes.byref(handle), module, name.encode()),
                    "cuModuleGetFunction",
                )
                functions[name] = Function(handle, name)
        _FUNCTIONS[key] = functions
        return functions


def load(
    device_index: int,
    capability: tuple,
    defines: dict,
    max_registers: Optional[int] = None,
    shared: int = 0,
) -> Function:
    """The attention kernel compiled with ``defines`` for this device; cached per process."""
    fn = load_module(
        device_index, capability, KERNEL_SOURCE, (KERNEL_NAME,), defines, max_registers
    )[KERNEL_NAME]
    fn.allow_shared(shared)
    return fn


def load_quantisers(device_index: int, capability: tuple, defines: dict) -> dict:
    """The quantiser kernels compiled with ``defines`` (HD, BN, RGX; SINKS adds the attention-sink path and
    unsloth_i8resid) for this device; cached per process."""
    names = QUANT_KERNELS + (("unsloth_i8resid",) if "SINKS" in defines else ())
    return load_module(device_index, capability, QUANT_SOURCE, names, defines)


def source_digest() -> str:
    return _digest(KERNEL_SOURCE, QUANT_SOURCE)[:16]
