# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import triton
import triton.language as tl

from .autotuning import (
    get_forward_configs,
    prune_kernel_configs_fwd,
)


# PERMUTE_X loads tokens in expert order, PERMUTE_Y stores output in token order: the same
# permutation indices either way.
@triton.jit
def _grouped_gemm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    # Variable depending on routed probs
    m_sizes_ptr,
    gather_indices_ptr,
    topk_weights_ptr,
    # Constant problem shapes
    NUM_EXPERTS: tl.constexpr,
    NUM_TOKENS,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS,
    # Tuning params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    PERMUTE_X: tl.constexpr = False,
    PERMUTE_Y: tl.constexpr = False,
    FUSE_MUL_PRE: tl.constexpr = False,
    FUSE_MUL_POST: tl.constexpr = False,
    USE_FAST_ACCUM: tl.constexpr = False,
    USE_TMA_LOAD_W: tl.constexpr = False,
    USE_TMA_LOAD_X: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    acc_dtype: tl.constexpr = tl.float32,
    FLATTEN: tl.constexpr = True,
) -> None:
    tl.static_assert(K % BLOCK_SIZE_K == 0)

    TOTAL_TOKENS = NUM_TOKENS * TOPK
    SHOULD_PERMUTE: tl.constexpr = PERMUTE_X or PERMUTE_Y
    SHOULD_FUSE_MUL: tl.constexpr = FUSE_MUL_PRE or FUSE_MUL_POST
    SHOULD_PERMUTE_OR_FUSE: tl.constexpr = SHOULD_PERMUTE or SHOULD_FUSE_MUL
    tidx = tl.program_id(0)
    output_dtype: tl.dtype = y_ptr.dtype.element_ty

    # A single global TMA descriptor with one block shape; TMA load never permutes x, so the shape is
    # [TOTAL_TOKENS, K]. Unverified across expert boundaries.
    if USE_TMA_LOAD_X:
        x_desc = tl.make_tensor_descriptor(
            x_ptr,
            shape = [TOTAL_TOKENS, K],
            strides = [K, 1],
            block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

    if USE_TMA_LOAD_W:
        expert_stride = N * K
        w_desc = tl.make_tensor_descriptor(
            w_ptr,
            shape = [NUM_EXPERTS, N, K],
            strides = [expert_stride, K, 1],
            block_shape = [1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    m_end = 0
    processed_tiles = 0
    m_block_range = tl.arange(0, BLOCK_SIZE_M)

    for expert_idx in tl.range(NUM_EXPERTS, flatten = FLATTEN):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size

        if m_size > 0:
            n_start = expert_idx * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles

            # The tma_store is created inside the loop so stores can be predicated on m_size.
            if USE_TMA_STORE:
                y_desc = tl.make_tensor_descriptor(
                    y_ptr,
                    shape = [m_end, N],
                    strides = [N, 1],
                    block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Process tiles for this expert
            while tidx >= processed_tiles and tidx < processed_tiles + num_tiles_per_expert:
                tile_idx = tidx - processed_tiles

                # Check if L2 cache reuse for this order is optimal
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles

                if SHOULD_PERMUTE_OR_FUSE:
                    # These will be used for loading and storing in permuted order
                    gather_offsets = tile_m_idx * BLOCK_SIZE_M + m_block_range
                    indices_to_gather = m_start + tl.max_contiguous(
                        tl.multiple_of(gather_offsets % m_size, BLOCK_SIZE_M),
                        BLOCK_SIZE_M,
                    )
                    expert_token_idx = tl.load(
                        gather_indices_ptr + indices_to_gather,
                        mask = indices_to_gather < TOTAL_TOKENS,
                    )
                    expert_token_offsets = expert_token_idx[:, None]


                    # Masks for permuted load and store
                    row_mask = gather_offsets < m_size
                    row_mask = row_mask[:, None]


                # Only (PERMUTE_X and not PERMUTE_Y) and (not PERMUTE_X and PERMUTE_Y) occur, so load/store offsets
                # are flipped between the two cases, with the strides adjusted.
                if PERMUTE_X:
                    load_idx = (
                        (expert_token_offsets // TOPK) * K
                    )  # Permute on load from token to expert order, dividing by TOPK to index the original token count.
                    store_idx = indices_to_gather[:, None] * N
                else:
                    off_am = tile_m_idx * BLOCK_SIZE_M
                    if not PERMUTE_Y:
                        # These will already be computed if permuting y
                        offs_am = off_am + m_block_range
                        row_mask = offs_am[:, None] < m_size
                        row_idx = m_start + offs_am[:, None]
                        store_idx = row_idx * N
                        if not USE_TMA_LOAD_X:
                            load_idx = row_idx * K

                if PERMUTE_Y:
                    if not USE_TMA_LOAD_X:
                        load_idx = (
                            indices_to_gather[:, None] * K
                        )
                    store_idx = (
                        expert_token_offsets * N
                    )

                # topk weights are always loaded in expert order: pre-multiplication scales hidden states before
                # the first gemm and post-multiplication after the second, both grouped by expert.
                if SHOULD_FUSE_MUL:
                    topk_load_idx = expert_token_offsets

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = acc_dtype)

                offs_k = tl.arange(0, BLOCK_SIZE_K)

                if not USE_TMA_LOAD_X:
                    x_ptrs = x_ptr + load_idx + offs_k[None, :]

                if not USE_TMA_LOAD_W:
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_bn = tl.max_contiguous(
                        tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N
                    )
                    w_ptrs = w_ptr + (n_start + offs_bn[:, None]) * K + offs_k[None, :]

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    if not USE_TMA_LOAD_X:
                        x = tl.load(x_ptrs, mask = row_mask)
                    else:
                        x = x_desc.load([m_start + off_am, k_offset])

                    if FUSE_MUL_PRE:
                        # Check for correct broadcasting
                        topk_weights = tl.load(topk_weights_ptr + topk_load_idx, mask = row_mask)
                        x *= topk_weights.to(x.dtype)

                    if not USE_TMA_LOAD_W:
                        w = tl.load(w_ptrs, mask = offs_bn[:, None] < N)
                    else:
                        w = w_desc.load([expert_idx, tile_n_idx * BLOCK_SIZE_N, k_offset])
                        w = tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K))

                    x = x.to(w.dtype)
                    accumulator += tl.dot(x, w.T)

                    if not USE_TMA_LOAD_X:
                        x_ptrs += BLOCK_SIZE_K

                    if not USE_TMA_LOAD_W:
                        w_ptrs += BLOCK_SIZE_K

                y = accumulator.to(output_dtype)

                # Order of the fused multiplication matters: fusing before the accumulator dtype conversion changes
                # the numerics.
                if FUSE_MUL_POST:
                    topk_weights = tl.load(topk_weights_ptr + topk_load_idx, mask = row_mask)
                    y *= topk_weights.to(output_dtype)

                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                store_mask = row_mask & (offs_bn[None, :] < N)

                if USE_TMA_STORE:
                    offset_m = tile_m_idx * BLOCK_SIZE_M
                    offset_n = tile_n_idx * BLOCK_SIZE_N
                    y_desc.store([m_start + offset_m, offset_n], y)
                else:
                    tl.store(
                        y_ptr + store_idx + offs_bn[None, :],
                        y,
                        mask = store_mask,
                    )
                tidx += NUM_SMS

            processed_tiles += num_tiles_per_expert


_autotuned_grouped_gemm_forward_kernel = triton.autotune(
    configs = get_forward_configs(),
    prune_configs_by = {"early_config_prune": prune_kernel_configs_fwd},
    # NUM_TOKENS is left out of the key to avoid recompiling for every sequence length; the kernel
    # handles variable token counts via m_sizes and tile-based processing.
    key = [
        "NUM_EXPERTS",
        "N",
        "K",
        "PERMUTE_X",
        "PERMUTE_Y",
        "FUSE_MUL_POST",
    ],
)(_grouped_gemm_forward_kernel)
