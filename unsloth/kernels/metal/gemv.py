# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import mlx.core as mx

_GEMV_SOURCE = """
    // MLX provides: x, W, y, K, N
    
    // We use a SIMD-group (warp) per row of output to ensure coalesced reads.
    // Each thread in the warp reads a chunk of the row, accumulates, and then we reduce.
    // Grid size must be (N * 32, 1, 1).
    
    uint gid = thread_position_in_grid.x;
    uint lid = thread_index_in_simdgroup; // 0..31
    
    // Each warp handles one row
    uint row = gid / 32;
    uint num_out = N;
    uint num_in = K;
    
    if (row >= num_out) return;
    
    float sum = 0.0f;
    uint row_offset = row * num_in;
    
    // Pointers
    // We assume K is multiple of 4 for half4 loads usually, else we need cleanup.
    // For Unsloth/Llama, K=4096, 11008 etc are multiples of 32/4.
    
    device const half4* x4 = (device const half4*)x;
    device const half4* W4 = (device const half4*)(W + row_offset);
    
    uint K4 = num_in / 4;
    
    // Stride by 32 (warp size)
    for (uint k = lid; k < K4; k += 32) {
        half4 xv = x4[k];
        half4 wv = W4[k];
        
        sum += (float)xv.x * (float)wv.x;
        sum += (float)xv.y * (float)wv.y;
        sum += (float)xv.z * (float)wv.z;
        sum += (float)xv.w * (float)wv.w;
    }
    
    // Simdgroup reduction
    sum = simd_sum(sum);
    
    if (lid == 0) {
        y[row] = (half)sum;
    }
"""


def fast_gemv(X: mx.array, W: mx.array) -> mx.array:
    """
    Computes y = X @ W.T for Batch=1 case using Simdgroup Reduction.
    X: (1, K)
    W: (N, K)
    Returns: (1, N)
    """
    if X.ndim == 1:
        X = X[None, :]

    K = X.shape[-1]
    N = W.shape[0]

    # Kernel assumes K is multiple of 4 (half4) and ideally 128?
    # For now ensuring vec4 alignment check
    if K % 4 != 0:
        # Fallback or strict check
        return X @ W.T

    kernel = mx.fast.metal_kernel(
        name = "gemv_simd_reduction",
        input_names = ["x", "W", "K", "N"],
        output_names = ["y"],
        source = _GEMV_SOURCE,
    )

    K_arg = mx.array(K, dtype = mx.uint32)
    N_arg = mx.array(N, dtype = mx.uint32)

    # Grid: N warps. Each warp is 32 threads.
    grid_size = (N * 32, 1, 1)
    group_size = (32, 1, 1)

    outputs = kernel(
        inputs = [X, W, K_arg, N_arg],
        grid = grid_size,
        threadgroup = group_size,
        output_shapes = [(1, N)],
        output_dtypes = [X.dtype],
    )

    return outputs[0]
