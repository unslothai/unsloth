// Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <metal_stdlib>
using namespace metal;

// Gemma RMS LayerNorm: Y = X * rsqrt(mean(X^2) + eps) * (1 + W)
// Standard RMSNorm uses mx.fast.rms_norm; this handles Gemma's (1+W) scaling only.

constant uint THREADS_PER_ROW = 256;

kernel void rms_layernorm_gemma_forward(
    device const float* __restrict X [[buffer(0)]],
    device const float* __restrict W [[buffer(1)]],
    device float* __restrict Y [[buffer(2)]],
    device float* __restrict rms_inv_out [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    constant uint& n_cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (row >= n_rows) return;
    
    device const float4* X_row = (device const float4*)(X + row * n_cols);
    device float4* Y_row = (device float4*)(Y + row * n_cols);
    device const float4* W_vec = (device const float4*)W;
    
    const uint n_vec = n_cols >> 2;
    const uint vectors_per_thread = (n_vec + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    
    float acc = 0.0f;
    
    #pragma unroll(4)
    for (uint j = 0; j < vectors_per_thread; ++j) {
        uint i = lid + j * THREADS_PER_ROW;
        if (i < n_vec) {
            float4 v = X_row[i];
            acc = fma(v.x, v.x, acc);
            acc = fma(v.y, v.y, acc);
            acc = fma(v.z, v.z, acc);
            acc = fma(v.w, v.w, acc);
        }
    }
    
    if (lid == 0) {
        #pragma unroll(3)
        for (uint i = n_vec << 2; i < n_cols; ++i) {
            float v = X[row * n_cols + i];
            acc = fma(v, v, acc);
        }
    }
    
    threadgroup float tg_sums[8];
    float simd_total = simd_sum(acc);
    if (simd_lane == 0) tg_sums[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float sum_sq = (lid < (THREADS_PER_ROW / 32)) ? tg_sums[lid] : 0.0f;
    sum_sq = simd_sum(sum_sq);
    
    float rms_inv = precise::rsqrt(sum_sq / float(n_cols) + eps);
    if (lid == 0) rms_inv_out[row] = rms_inv;
    
    #pragma unroll(4)
    for (uint j = 0; j < vectors_per_thread; ++j) {
        uint i = lid + j * THREADS_PER_ROW;
        if (i < n_vec) {
            float4 x = X_row[i];
            float4 w = W_vec[i] + 1.0f;
            Y_row[i] = x * rms_inv * w;
        }
    }
    
    if (lid == 0) {
        #pragma unroll(3)
        for (uint i = n_vec << 2; i < n_cols; ++i) {
            uint idx = row * n_cols + i;
            Y[idx] = X[idx] * rms_inv * (1.0f + W[i]);
        }
    }
}

kernel void rms_layernorm_gemma_forward_f16(
    device const half* __restrict X [[buffer(0)]],
    device const half* __restrict W [[buffer(1)]],
    device half* __restrict Y [[buffer(2)]],
    device float* __restrict rms_inv_out [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    constant uint& n_cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (row >= n_rows) return;
    
    device const half4* X_row = (device const half4*)(X + row * n_cols);
    device half4* Y_row = (device half4*)(Y + row * n_cols);
    device const half4* W_vec = (device const half4*)W;
    
    const uint n_vec = n_cols >> 2;
    const uint vectors_per_thread = (n_vec + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    
    float acc = 0.0f;
    
    #pragma unroll(4)
    for (uint j = 0; j < vectors_per_thread; ++j) {
        uint i = lid + j * THREADS_PER_ROW;
        if (i < n_vec) {
            float4 v = float4(X_row[i]);
            acc = fma(v.x, v.x, acc);
            acc = fma(v.y, v.y, acc);
            acc = fma(v.z, v.z, acc);
            acc = fma(v.w, v.w, acc);
        }
    }
    
    if (lid == 0) {
        #pragma unroll(3)
        for (uint i = n_vec << 2; i < n_cols; ++i) {
            float v = float(X[row * n_cols + i]);
            acc = fma(v, v, acc);
        }
    }
    
    threadgroup float tg_sums[8];
    float simd_total = simd_sum(acc);
    if (simd_lane == 0) tg_sums[simd_id] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float sum_sq = (lid < (THREADS_PER_ROW / 32)) ? tg_sums[lid] : 0.0f;
    sum_sq = simd_sum(sum_sq);
    
    float rms_inv = precise::rsqrt(sum_sq / float(n_cols) + eps);
    if (lid == 0) rms_inv_out[row] = rms_inv;
    
    #pragma unroll(4)
    for (uint j = 0; j < vectors_per_thread; ++j) {
        uint i = lid + j * THREADS_PER_ROW;
        if (i < n_vec) {
            float4 x = float4(X_row[i]);
            float4 w = float4(W_vec[i]) + 1.0f;
            Y_row[i] = half4(x * rms_inv * w);
        }
    }
    
    if (lid == 0) {
        #pragma unroll(3)
        for (uint i = n_vec << 2; i < n_cols; ++i) {
            uint idx = row * n_cols + i;
            Y[idx] = half(float(X[idx]) * rms_inv * (1.0f + float(W[i])));
        }
    }
}
