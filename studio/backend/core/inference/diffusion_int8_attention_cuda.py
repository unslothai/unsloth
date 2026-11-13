# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention kernel's CUDA source and its loader.

The kernel is compiled at first use with NVRTC, the runtime compiler the CUDA builds of torch already depend on, and
launched through the CUDA driver on torch's context and current stream. Nothing is built at install time and no
compiler toolchain is needed. ``diffusion_int8_attention`` owns everything around it (quantisers, configs, the
dispatch).
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import struct
import sys
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
// Compile-time: HD (head dim, 64 or 128), BN (keys per tile, 64 or 128), NQ (16-row query tiles per warp),
// WPC (warps per block), STAGES (1 or 2 shared-memory tile buffers; 2 = the next tile is copied while this one is
// computed), QREG (1 = the Q operand fragments are read once into registers), FLUSH.
// Launch: blockDim.x = 32 * WPC, grid.x = n_qpad / (16 * NQ * WPC), grid.y = batch * heads,
// dynamic shared memory = STAGES * STAGE_BYTES + WPC * 16 * NQ * KROW bytes.

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
#define QREG 0  // 1 = the Q operand fragments stay in registers instead of being re-read from shared memory per tile
#endif
#ifndef FLUSH
#define FLUSH 16  // tiles between removals of MAGIC_F from the output accumulators (a power of two; 0 = only at the end)
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

__device__ __forceinline__ void cp_async16(u32 dst, const s8* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

// Four 8 x 16-byte matrices from shared memory: lane L passes the address of row L % 8 of matrix L / 8, and thread
// (g, t) receives bytes 4 * t .. 4 * t + 3 of row g of each matrix: the int8 B fragments of two k steps.
__device__ __forceinline__ void ldmatrix4(u32* r, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

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

// q   [BH, n_qpad, HD] int8, rows padded with zeros to a multiple of 16 * NQ * warps
// k   [BH, n_kpad, HD] int8, n_kpad a multiple of BN
// vt  [BH, n_kpad / BN, HD, BN] int8: per key tile, channel-major V; inside every 16 keys, position 4*t + c holds
//     key 8*(c>>1) + 2*t + (c&1)
// qs  [BH, n_qpad] per-row Q scale (softmax scale * log2(e) folded in)
// ks  [BH, n_kpad / 32, 4] K scale of the keys with (key % 8) / 2 == t in a run of 32 keys
// vs  [BH, n_kpad / BN] V scale of a tile; vg / vc [BH, HD] per-channel gain and centre of V
// out [B, n_q, H, HD] (strides in elements); out_kind 0 = fp16, 1 = bf16, 2 = fp32
extern "C" __global__ void unsloth_i8attn(
    const s8* __restrict__ q, const s8* __restrict__ k, const s8* __restrict__ vt, const float* __restrict__ qs,
    const float* __restrict__ ks, const float* __restrict__ vs, const float* __restrict__ vg,
    const float* __restrict__ vc, void* __restrict__ out, int n_q, int n_qpad, int n_k, int n_kpad, int n_heads,
    i64 o_stride_b, i64 o_stride_s, i64 o_stride_h, int out_kind) {
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
  // B fragments: this lane's row / matrix offsets
  const u32 kofs = (lane & 7u) * KROW + 16u * (lane >> 3), vofs = BN * KROW + (lane & 7u) * VROW + 16u * (lane >> 3);

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

#if QREG
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
    if (j + 1 < n_tiles) {
      const u32 next = sbase + ((j + 1) & 1) * STAGE_BYTES;
      const i64 off = (i64)(j + 1) * BN * HD;
      load_tile(next + kdst, next + vdst, ksrc + off, vsrc + off, last);
    }
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
#if FLUSH > 0
      // Every FLUSH tiles MAGIC_F times the weights since the last time comes off the output accumulators, so they
      // stay near the size of the P.V sums however many keys there are (left in until the end, the constant grows
      // with the key count and the sums lose their low bits to it). It shares the rare branch of the rebase: as a
      // block of its own after P.V it cost 3-5% of the call on sm80 / sm89.
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
  }

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
}
"""

QUANT_KERNELS = ("unsloth_i8stats", "unsloth_i8quant_v", "unsloth_i8quant_qk")

QUANT_SOURCE = r"""
// The INT8 attention kernel's operands from [B, S, H, D] q / k / v of any float dtype and any strides on the first
// three dims (the last dim has stride 1 and every row starts 16-byte aligned: the caller checks). Three launches:
//
//   unsloth_i8stats     per (batch, head): the K mean over every kstep-th key (one block, a fixed summation order)
//                       and the V minimum / maximum per channel over every vstep-th key, in nvc row chunks
//   unsloth_i8quant_v   one block per key tile: (v - centre) / gain per channel, one scale per tile, stored
//                       channel-major in the key order of the kernel's operand registers
//   unsloth_i8quant_qk  one block per BN rows of Q (rotated, one scale per row) or per key tile of K (centred,
//                       rotated, one scale per 32 keys and accumulator lane group)
//
// The rotation is the same fixed orthogonal map for Q and K (so Q.K^T is unchanged): a sign per channel, then an
// unnormalised Walsh-Hadamard transform over every RG channels (16 inside one thread, the rest by warp shuffles).
// Compile-time: HD (head dim), BN (keys per tile), RGX (log2(RG) - 4: the shuffle stages of the rotation).

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
#define NTQ (BN * TPR)  // threads per quantiser block (one key tile)
#define NTV (BN * HD / 64)  // threads per V quantiser block: one thread holds 16 channels of 4 keys
#define NTS 256         // threads per statistics block
#define SLOTS (NTS / TPR)
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

// The rotation of the 16 channels 16 * g16 .. 16 * g16 + 15 of a row whose TPR threads sit in adjacent lanes.
__device__ __forceinline__ void rotate(float* x, u32 g16) {
#pragma unroll
  for (u32 i = 0; i < 16; ++i) {
    const u32 c = 16u * g16 + i;
    const u32 flip = (((c * 7919u + 13u) * (c + 3u)) >> 7) & 1u;
    x[i] = __uint_as_float(__float_as_uint(x[i]) ^ (flip << 31));
  }
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
    const bool upper = (g16 >> s) & 1u;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float o = __shfl_xor_sync(0xffffffffu, x[i], 1 << s);
      x[i] = upper ? (o - x[i]) : (x[i] + o);
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
// grid (nvc + 1, BH): block x < nvc is V chunk x, block nvc the K mean; NTS threads.
extern "C" __global__ void unsloth_i8stats(
    const void* __restrict__ k, const void* __restrict__ v, float* __restrict__ kc, float* __restrict__ vpart,
    int n_k, int n_heads, int nvc, int vrows, int vstep, int kstep, i64 k_sb, i64 k_ss, i64 k_sh, i64 v_sb,
    i64 v_ss, i64 v_sh, int k_kind, int v_kind) {
  __shared__ float sm[NWS][2][HD];
  const u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const u32 slot = tid / TPR, g16 = tid % TPR;
  const i64 bh = blockIdx.y;
  const i64 b = bh / n_heads, h = bh % n_heads;
  const bool mean = (int)blockIdx.x == nvc;
  float lo[16], hi[16];  // the K mean block: lo holds the sums
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
    const i64 r0 = (i64)blockIdx.x * vrows;
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
    }
#pragma unroll
    for (int s = TPR; s < 32; s <<= 1) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        lo[i] = fminf(lo[i], __shfl_xor_sync(0xffffffffu, lo[i], s));
        hi[i] = fmaxf(hi[i], __shfl_xor_sync(0xffffffffu, hi[i], s));
      }
    }
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
      float* o = vpart + ((bh * nvc + blockIdx.x) * 2) * HD + tid;
      o[0] = a;
      o[HD] = c;
    }
  }
}

// vt [BH, n_kpad / BN, HD, BN] int8 ((v - centre) / gain over the tile scale, zeros past n_k) in the key order of
// the kernel's operand registers; vs [BH, n_kpad / BN] the tile scales; vc / vg [BH, HD] centre and gain, merged from
// vpart by every block (HD threads) and written by tile 0.
// grid (n_kpad / BN, BH), NTV threads: thread (16 channels, word kq) reads its four keys row by row (whole 16-byte
// loads) and writes one 32-bit word per channel; adjacent lanes write adjacent words of a channel.
extern "C" __global__ void unsloth_i8quant_v(
    const void* __restrict__ v, const float* __restrict__ vpart, s8* __restrict__ vt, float* __restrict__ vs,
    float* __restrict__ vc, float* __restrict__ vg, int n_k, int n_kpad, int n_heads, int nvc, i64 v_sb, i64 v_ss,
    i64 v_sh, int v_kind) {
  __shared__ __align__(16) float smc[HD], smr[HD];
  __shared__ float sm[NTV / 32 + 1];
  const u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
  const u32 g16 = tid / (BN / 4), kq = tid % (BN / 4);
  const i64 bh = (i64)gridDim.y - 1 - blockIdx.y;  // the heads the statistics read last come first
  const i64 b = bh / n_heads, h = bh % n_heads;
  const i64 tile = blockIdx.x, tiles = n_kpad / BN;
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

// ki [BH, n_kpad, HD] int8 rotated (k - kc) (zeros past n_k); ks [BH, n_kpad / 32, 4] = the maximum over the keys
// of a run of 32 with the same (key % 8) / 2, over 127
// grid ((n_qpad + BN - 1) / BN + n_kpad / BN, BH), NTQ threads: thread (row, 16 channels); the first blocks are Q.
extern "C" __global__ void unsloth_i8quant_qk(
    const void* __restrict__ q, const void* __restrict__ k, const float* __restrict__ kc, s8* __restrict__ qi,
    float* __restrict__ qs, s8* __restrict__ ki, float* __restrict__ ks, int n_q, int n_qpad, int n_k, int n_kpad,
    int n_heads, i64 q_sb, i64 q_ss, i64 q_sh, i64 k_sb, i64 k_ss, i64 k_sh, float mul, int q_kind, int k_kind) {
  __shared__ float sm[BN + BN / 8];
  const u32 tid = threadIdx.x;
  const u32 rin = tid / TPR, g16 = tid % TPR;
  const i64 bh = blockIdx.y;
  const i64 b = bh / n_heads, h = bh % n_heads;
  const int qblocks = (n_qpad + BN - 1) / BN;
  float x[16];
  if ((int)blockIdx.x < qblocks) {
    const i64 row = (i64)blockIdx.x * BN + rin;
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
  } else {
    const i64 tile = (i64)blockIdx.x - qblocks, tiles = n_kpad / BN;
    const i64 key = tile * BN + rin;
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
    if (tid < BN / 8) {  // scale tid: the keys 32 * (tid / 4) + 8 * a + 2 * (tid % 4) + {0, 1}
      const u32 first = 32u * (tid >> 2) + 2u * (tid & 3u);
      float m = 0.f;
#pragma unroll
      for (int a = 0; a < 4; ++a) m = fmaxf(m, fmaxf(sm[first + 8 * a], sm[first + 8 * a + 1]));
      m = fmaxf(m, 1e-20f);
      sm[BN + tid] = m;
      ks[(bh * tiles + tile) * (BN / 8) + tid] = m / 127.f;
    }
    __syncthreads();
    store16(ki + (bh * n_kpad + key) * HD + 16 * g16, x, 127.f / sm[BN + 4u * (rin >> 5) + ((rin & 7u) >> 1)]);
  }
}
"""

_LOCK = threading.Lock()
_NVRTC: Any = None
_DRIVER: Any = None
_FUNCTIONS: dict = {}

_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
_CU_FUNC_ATTRIBUTE_NUM_REGS = 4
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


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


def compile_ptx(
    source: str,
    arch: str,
    defines: Optional[dict] = None,
) -> bytes:
    """PTX (NUL-terminated) of ``source`` for ``arch`` such as "sm_89". Raises RuntimeError with the compiler log."""
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
        lib.nvrtcGetPTXSize(program, ctypes.byref(size))
        ptx = ctypes.create_string_buffer(size.value)
        lib.nvrtcGetPTX(program, ptx)
        return ptx.raw
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(program))


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


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with CUDA error {rc}")


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
        rc = driver().cuLaunchKernel(
            self.handle,
            grid[0],
            grid[1],
            1,
            block,
            1,
            1,
            shared,
            ctypes.c_void_p(stream),
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
    cached per process. Not to be called while a CUDA graph is being captured (compiling is not a stream
    operation)."""
    import torch

    key = (int(device_index), names, tuple(sorted(defines.items())), max_registers)
    found = _FUNCTIONS.get(key)
    if found is not None:
        return found
    with _LOCK:
        found = _FUNCTIONS.get(key)
        if found is not None:
            return found
        ptx = compile_ptx(source, f"sm_{capability[0]}{capability[1]}", defines)
        if max_registers:
            ptx = cap_registers(ptx, max_registers)
        with torch.cuda.device(device_index):
            torch.zeros(
                1, device = f"cuda:{device_index}"
            )  # torch's context exists and is current on this thread
            module = ctypes.c_void_p()
            _check(
                driver().cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(ptx)),
                "cuModuleLoadData",
            )
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
    """The three quantiser kernels compiled with ``defines`` (HD, BN, RGX) for this device; cached per process."""
    return load_module(device_index, capability, QUANT_SOURCE, QUANT_KERNELS, defines)


def source_digest() -> str:
    return hashlib.sha256((KERNEL_SOURCE + QUANT_SOURCE).encode()).hexdigest()[:16]
