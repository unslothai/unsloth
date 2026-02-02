# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx

_GEMV_SOURCE = """
// gemv_row_reduction: y = x @ W.T
// x: [1, K]
// W: [N, K]
// y: [1, N]
// Grid: (N, 1, 1). Each thread computes one element of y (one row of W).

kernel void gemv_row_reduction(
    device const half* x [[ buffer(0) ]],       // Input vector (1, K)
    device const half* W [[ buffer(1) ]],       // Weight matrix (N, K)
    device half* y [[ buffer(2) ]],             // Output vector (1, N)
    constant uint& K [[ buffer(3) ]],           // Input dimension
    constant uint& N [[ buffer(4) ]],           // Output dimension
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= N) return;
    
    float sum = 0.0f;
    uint offset = gid * K;
    
    // float4 vectorized loads
    // We cast to half4* to load 4 halves, then convert to float for math
    device const half4* x4 = (device const half4*)x;
    device const half4* W4 = (device const half4*)(W + offset);
    
    uint K4 = K / 4;
    for (uint k = 0; k < K4; ++k) {
        half4 xv = x4[k];
        half4 wv = W4[k];
        sum += (float)xv.x * (float)wv.x;
        sum += (float)xv.y * (float)wv.y;
        sum += (float)xv.z * (float)wv.z;
        sum += (float)xv.w * (float)wv.w;
    }
    
    // Remainder
    for (uint k = K4 * 4; k < K; ++k) {
        sum += (float)x[k] * (float)W[offset + k];
    }
    
    y[gid] = (half)sum;
}
"""

def fast_gemv(X: mx.array, W: mx.array) -> mx.array:
    """
    Computes y = X @ W.T for Batch=1 case.
    X: (1, K)
    W: (N, K)
    Returns: (1, N)
    """
    if X.ndim == 1:
        X = X[None, :]
    
    K = X.shape[-1]
    N = W.shape[0]
    
    if W.shape[1] != K:
        raise ValueError(f"Shape mismatch: X{X.shape} W{W.shape}")

    kernel = mx.fast.metal_kernel(
        name="gemv_row_reduction",
        input_names=["x", "W", "K", "N"],
        output_names=["y"],
        source=_GEMV_SOURCE,
    )
    
    K_arg = mx.array(K, dtype=mx.uint32)
    N_arg = mx.array(N, dtype=mx.uint32)
    
    outputs = kernel(
        inputs=[X, W, K_arg, N_arg],
        grid=(N, 1, 1),
        threadgroup=(32, 1, 1), # Maximize thread occupancy, though they work independently
        output_shapes=[(1, N)],
        output_dtypes=[X.dtype],
    )
    
    return outputs[0]
