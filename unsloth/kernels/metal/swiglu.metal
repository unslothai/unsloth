// Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
// Licensed under the Apache License, Version 2.0.

//
// Fused SwiGLU Activation Kernels for Apple Silicon
// ================================================
//
// SwiGLU: h = silu(e) * g = (e * sigmoid(e)) * g
//
// Optimizations:
// - float4/half4 vectorization for 128-bit coalesced memory access
// - Fused read → compute → write (single memory pass)
// - precise::exp for numerical accuracy
// - Unrolled loops for reduced instruction overhead
//

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Forward Kernels: h = silu(e) * g
// =============================================================================

// Float32 forward kernel with float4 vectorization
kernel void swiglu_forward_f32(
    device const float4* __restrict__ e [[buffer(0)]],
    device const float4* __restrict__ g [[buffer(1)]],
    device float4* __restrict__ h [[buffer(2)]],
    constant uint& n_vec [[buffer(3)]],  // Number of float4 vectors
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_vec) return;
    
    // Load e and g as float4 (128-bit coalesced read)
    float4 e4 = e[gid];
    float4 g4 = g[gid];
    
    // Compute sigmoid(e) using precise exp for accuracy
    // sigmoid(x) = 1 / (1 + exp(-x))
    float4 neg_e = -e4;
    float4 exp_neg_e = float4(
        precise::exp(neg_e.x),
        precise::exp(neg_e.y),
        precise::exp(neg_e.z),
        precise::exp(neg_e.w)
    );
    float4 sigmoid_e = 1.0f / (1.0f + exp_neg_e);
    
    // SiLU(e) = e * sigmoid(e)
    float4 silu_e = e4 * sigmoid_e;
    
    // Output: h = silu(e) * g
    h[gid] = silu_e * g4;
}

// Float16 forward kernel with half4 vectorization
kernel void swiglu_forward_f16(
    device const half4* __restrict__ e [[buffer(0)]],
    device const half4* __restrict__ g [[buffer(1)]],
    device half4* __restrict__ h [[buffer(2)]],
    constant uint& n_vec [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_vec) return;
    
    // Load as half4, compute in float32 for accuracy
    float4 e4 = float4(e[gid]);
    float4 g4 = float4(g[gid]);
    
    // Sigmoid with precise exp
    float4 neg_e = -e4;
    float4 exp_neg_e = float4(
        precise::exp(neg_e.x),
        precise::exp(neg_e.y),
        precise::exp(neg_e.z),
        precise::exp(neg_e.w)
    );
    float4 sigmoid_e = 1.0f / (1.0f + exp_neg_e);
    
    // SiLU and gate multiplication
    float4 silu_e = e4 * sigmoid_e;
    float4 result = silu_e * g4;
    
    // Store as half4
    h[gid] = half4(result);
}

// Scalar fallback for handling remainder elements (non-vectorized tail)
kernel void swiglu_forward_f32_scalar(
    device const float* __restrict__ e [[buffer(0)]],
    device const float* __restrict__ g [[buffer(1)]],
    device float* __restrict__ h [[buffer(2)]],
    constant uint& offset [[buffer(3)]],  // Start offset
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = offset + gid;
    if (idx >= n_elements) return;
    
    float e_val = e[idx];
    float g_val = g[idx];
    float sigmoid_e = 1.0f / (1.0f + precise::exp(-e_val));
    h[idx] = (e_val * sigmoid_e) * g_val;
}

kernel void swiglu_forward_f16_scalar(
    device const half* __restrict__ e [[buffer(0)]],
    device const half* __restrict__ g [[buffer(1)]],
    device half* __restrict__ h [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = offset + gid;
    if (idx >= n_elements) return;
    
    float e_val = float(e[idx]);
    float g_val = float(g[idx]);
    float sigmoid_e = 1.0f / (1.0f + precise::exp(-e_val));
    h[idx] = half((e_val * sigmoid_e) * g_val);
}

// =============================================================================
// Backward Kernels: Compute gradients de and dg given upstream gradient dw
// =============================================================================
//
// Given: dw = dL/dh (upstream gradient)
// Forward was: h = silu(e) * g = (e * sigmoid(e)) * g
//
// Backward:
//   dg = dL/dg = dw * silu(e) = dw * (e * sigmoid(e))
//   de = dL/de = dw * g * d(silu(e))/de
//      where d(silu(e))/de = sigmoid(e) * (1 + e * (1 - sigmoid(e)))
//
// In-place storage (matching Triton kernel behavior):
//   DW buffer -> stores h (forward recomputation)
//   e buffer -> stores df = dw * f (where f = silu(e))
//   g buffer -> stores de
//

kernel void swiglu_backward_f32(
    device float4* __restrict__ DW [[buffer(0)]],  // Input: upstream grad, Output: h
    device float4* __restrict__ e [[buffer(1)]],   // Input: e, Output: df
    device float4* __restrict__ g [[buffer(2)]],   // Input: g, Output: de
    constant uint& n_vec [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_vec) return;
    
    // Load inputs
    float4 dw4 = DW[gid];
    float4 e4 = e[gid];
    float4 g4 = g[gid];
    
    // Compute sigmoid(e) using precise exp
    float4 neg_e = -e4;
    float4 exp_neg_e = float4(
        precise::exp(neg_e.x),
        precise::exp(neg_e.y),
        precise::exp(neg_e.z),
        precise::exp(neg_e.w)
    );
    float4 se = 1.0f / (1.0f + exp_neg_e);
    
    // f = silu(e) = e * sigmoid(e)
    float4 f = e4 * se;
    
    // h = f * g (forward recomputation)
    float4 h = f * g4;
    
    // df = dw * f
    float4 df = dw4 * f;
    
    // dg = dw * g
    float4 dg = dw4 * g4;
    
    // de = dg * se * (1 + e * (1 - se))
    // With dg = dw * g, we have:
    // de = dw * g * se * (1 + e * (1 - se))
    float4 one_minus_se = 1.0f - se;
    float4 de = dg * se * (1.0f + e4 * one_minus_se);
    
    // Store results in-place
    DW[gid] = h;   // h = f * g
    e[gid] = df;   // df = dw * f
    g[gid] = de;   // de
}

kernel void swiglu_backward_f16(
    device half4* __restrict__ DW [[buffer(0)]],
    device half4* __restrict__ e [[buffer(1)]],
    device half4* __restrict__ g [[buffer(2)]],
    constant uint& n_vec [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_vec) return;
    
    // Load as half4, compute in float32
    float4 dw4 = float4(DW[gid]);
    float4 e4 = float4(e[gid]);
    float4 g4 = float4(g[gid]);
    
    // Compute sigmoid(e)
    float4 neg_e = -e4;
    float4 exp_neg_e = float4(
        precise::exp(neg_e.x),
        precise::exp(neg_e.y),
        precise::exp(neg_e.z),
        precise::exp(neg_e.w)
    );
    float4 se = 1.0f / (1.0f + exp_neg_e);
    
    // f = silu(e)
    float4 f = e4 * se;
    
    // h = f * g
    float4 h = f * g4;
    
    // df = dw * f
    float4 df = dw4 * f;
    
    // dg = dw * g
    float4 dg = dw4 * g4;
    
    // de = dg * se * (1 + e * (1 - se))
    float4 one_minus_se = 1.0f - se;
    float4 de = dg * se * (1.0f + e4 * one_minus_se);
    
    // Store results in-place as half4
    DW[gid] = half4(h);
    e[gid] = half4(df);
    g[gid] = half4(de);
}

// Scalar backward fallbacks
kernel void swiglu_backward_f32_scalar(
    device float* __restrict__ DW [[buffer(0)]],
    device float* __restrict__ e [[buffer(1)]],
    device float* __restrict__ g [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = offset + gid;
    if (idx >= n_elements) return;
    
    float dw = DW[idx];
    float e_val = e[idx];
    float g_val = g[idx];
    
    float se = 1.0f / (1.0f + precise::exp(-e_val));
    float f = e_val * se;
    float h = f * g_val;
    float df = dw * f;
    float dg = dw * g_val;
    float de = dg * se * (1.0f + e_val * (1.0f - se));
    
    DW[idx] = h;
    e[idx] = df;
    g[idx] = de;
}

kernel void swiglu_backward_f16_scalar(
    device half* __restrict__ DW [[buffer(0)]],
    device half* __restrict__ e [[buffer(1)]],
    device half* __restrict__ g [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& n_elements [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = offset + gid;
    if (idx >= n_elements) return;
    
    float dw = float(DW[idx]);
    float e_val = float(e[idx]);
    float g_val = float(g[idx]);
    
    float se = 1.0f / (1.0f + precise::exp(-e_val));
    float f = e_val * se;
    float h = f * g_val;
    float df = dw * f;
    float dg = dw * g_val;
    float de = dg * se * (1.0f + e_val * (1.0f - se));
    
    DW[idx] = half(h);
    e[idx] = half(df);
    g[idx] = half(de);
}
