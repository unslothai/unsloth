# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import torch
import triton
import triton.language as tl

from grouped_gemm.kernels.autotuning import (
    get_forward_configs,
    prune_kernel_configs_fwd,
)


#
# PERMUTE_X -> permute tokens so that they are ordered by expert
# PERMUTE_Y -> permute output so that they are ordered by token
# These are effectively the same thing: the former loads in permuted order, the latter stores in permuted order => we only need to define the permutation indices once
# In the former, we use these row indices when loading X
# For the latter, we use these row indices when storing Y
# FUSE_MUL -> multiply routed outputs by their respective weights
# topk_weights are in token order
# Only account for the case when X is in expert order and we are permuting Y when fusing mul -- this precondition is checked in the interface
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
    NUM_TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
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

    TOTAL_TOKENS: tl.constexpr = NUM_TOKENS * TOPK
    SHOULD_PERMUTE: tl.constexpr = PERMUTE_X or PERMUTE_Y
    SHOULD_FUSE_MUL: tl.constexpr = FUSE_MUL_PRE or FUSE_MUL_POST
    SHOULD_PERMUTE_OR_FUSE: tl.constexpr = SHOULD_PERMUTE or SHOULD_FUSE_MUL
    # tl.static_print("SHOULD_PERMUTE", PERMUTE_X, PERMUTE_Y, FUSE_MUL_PRE, FUSE_MUL_POST, SHOULD_PERMUTE, SHOULD_FUSE, SHOULD_PERMUTE_OR_FUSE)
    tidx = tl.program_id(0)
    output_dtype: tl.dtype = y_ptr.dtype.element_ty

    # Create TMA descriptors for loading sorted tokens
    # When using TMA load, we don't permute_x, so shape should be [TOTAL_TOKENS, K]
    # Also, we are defining a single global descriptor with single block shape
    # Need to check that this does not result in errors when crossing expert boundaries
    if USE_TMA_LOAD_X:
        x_desc = tl._experimental_make_tensor_descriptor(
            x_ptr,
            shape=[TOTAL_TOKENS, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

    if USE_TMA_LOAD_W:
        expert_stride = N * K
        w_desc = tl._experimental_make_tensor_descriptor(
            w_ptr,
            shape=[NUM_EXPERTS, N, K],
            strides=[expert_stride, K, 1],
            block_shape=[1, BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    m_end = 0
    processed_tiles = 0
    m_block_range = tl.arange(0, BLOCK_SIZE_M)

    for expert_idx in tl.range(NUM_EXPERTS, flatten=FLATTEN):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size

        if m_size > 0:
            n_start = expert_idx * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles

            # Need to create tma_store within loop since we need to predicate stores based on m_size
            if USE_TMA_STORE:
                y_desc = tl._experimental_make_tensor_descriptor(
                    y_ptr,  # + m_start * N,
                    shape=[m_end, N],
                    strides=[N, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Process tiles for this expert
            while (
                tidx >= processed_tiles
                and tidx < processed_tiles + num_tiles_per_expert
            ):
                tile_idx = tidx - processed_tiles

                # Check if L2 cache re-use for this order is optimal
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
                        mask=indices_to_gather < TOTAL_TOKENS,
                    )
                    expert_token_offsets = expert_token_idx[:, None]

                    # Masks for permuted load and store

                    row_mask = gather_offsets < m_size
                    row_mask = row_mask[:, None]

                    # row_mask = indices_to_gather < m_end
                    # row_mask = row_mask[:, None]

                # We only take into account the following two cases: (PERMUTE_X and NOT PERMUTE_Y) and (NOT PERMUTE_X and PERMUTE_Y)
                # Hence, we can make the following simplifying assumptions when loading and storing
                # Note the different strides between the two cases: the offsets for loading and storing are flipped and the strides must also be adjusted
                if PERMUTE_X:
                    load_idx = (
                        (expert_token_offsets // TOPK) * K
                    )  # Permute on load from token -> expert order, divide by TOPK to index from original number of tokens
                    store_idx = (
                        indices_to_gather[:, None] * N
                    )  # Store in contiguous order
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
                        )  # Load in contiguous order (no permutation on load)
                    # offs_am = off_am + m_block_range
                    # row_mask = offs_am[:, None] < m_size
                    store_idx = (
                        expert_token_offsets * N
                    )  # Permute on store from expert -> token order

                # We always load topk weights in expert order
                # In the pre-multiplication case, we multiply permuted hidden states by weights before the first gemm
                # In the post-multiplication case, we multiply permuted hidden states by weights after the second gemm
                # In either case, the hidden states are grouped by expert, so we always permute on load of topk weights
                if SHOULD_FUSE_MUL:
                    topk_load_idx = expert_token_offsets

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

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
                        x = tl.load(x_ptrs, mask=row_mask)
                    else:
                        x = x_desc.load([m_start + off_am, k_offset])

                    if FUSE_MUL_PRE:
                        # Check for correct broadcasting
                        topk_weights = tl.load(
                            topk_weights_ptr + topk_load_idx, mask=row_mask
                        )
                        x *= topk_weights.to(x.dtype)

                    if not USE_TMA_LOAD_W:
                        w = tl.load(w_ptrs, mask=offs_bn[:, None] < N)
                    else:
                        w = w_desc.load(
                            [expert_idx, tile_n_idx * BLOCK_SIZE_N, k_offset]
                        )
                        w = tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K))

                    accumulator += tl.dot(x, w.T)

                    if not USE_TMA_LOAD_X:
                        x_ptrs += BLOCK_SIZE_K

                    if not USE_TMA_LOAD_W:
                        w_ptrs += BLOCK_SIZE_K

                y = accumulator.to(output_dtype)

                # NOTE: order of fusing multiplication is important
                # Fusing before accumulator dtype conversion results in numerical diffs
                if FUSE_MUL_POST:
                    # Check for correct broadcasting
                    topk_weights = tl.load(
                        topk_weights_ptr + topk_load_idx, mask=row_mask
                    )
                    y *= topk_weights.to(output_dtype)

                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                store_mask = row_mask & (offs_bn[None, :] < N)

                if USE_TMA_STORE:
                    offset_m = tile_m_idx * BLOCK_SIZE_M  # .to(tl.int32)
                    offset_n = tile_n_idx * BLOCK_SIZE_N  # .to(tl.int32)
                    y_desc.store([m_start + offset_m, offset_n], y)
                else:
                    tl.store(
                        y_ptr + store_idx + offs_bn[None, :],
                        y,
                        mask=store_mask,
                    )
                tidx += NUM_SMS

            processed_tiles += num_tiles_per_expert


_autotuned_grouped_gemm_forward_kernel = triton.autotune(
    configs=get_forward_configs(),
    prune_configs_by={"early_config_prune": prune_kernel_configs_fwd},
    key=[
        "NUM_EXPERTS",
        "NUM_TOKENS",
        "N",
        "K",
        "PERMUTE_X",
        "PERMUTE_Y",
        "FUSE_MUL_POST",
    ],
)(_grouped_gemm_forward_kernel)
