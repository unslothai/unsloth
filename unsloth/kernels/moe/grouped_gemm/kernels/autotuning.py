# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Autotuning utilities for GPU kernel configuration generation and optimization.

This module provides functions to generate and prune kernel configurations
for grouped GEMM operations, including forward pass, backward pass (dX), and
weight gradient (dW) computations.
"""

import logging
from itertools import product
from typing import List

import torch
import triton

logger = logging.getLogger(__name__)

DEFAULT_M_BLOCK_SIZES = [64, 128]
DEFAULT_N_BLOCK_SIZES = [64, 128, 256]
DEFAULT_K_BLOCK_SIZES = [64, 128, 256]
DEFAULT_NUM_CTAS = 1
DEFAULT_NUM_WARPS = [4, 8]
DEFAULT_NUM_STAGES = [3, 4, 5]
BOOLS = [True, False]


def val_to_list(val):
    """
    Convert a single value to a list or return None if input is None.
    
    Args:
        val: Input value that can be None, a list, or a single value
        
    Returns:
        None if input is None, the original list if input is a list,
        or a single-element list containing the input value
    """
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]


def convert_args_to_list(args):
    """
    Convert each argument in a list to a list format using val_to_list.
    
    Args:
        args: List of arguments to convert
        
    Returns:
        List where each element has been processed by val_to_list
    """
    return [val_to_list(arg) for arg in args]


def get_forward_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X=True,
    TMA_LOAD_W=True,
    TMA_STORE=False,  # NOTE: TMA_STORE is disabled for now
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
):
    """
    Generate kernel configurations for forward pass GEMM operations.
    
    Args:
        BLOCK_M: Block sizes for M dimension
        BLOCK_N: Block sizes for N dimension  
        BLOCK_K: Block sizes for K dimension
        TMA_LOAD_X: Whether to use TMA (Tensor Memory Accelerator) for loading X
        TMA_LOAD_W: Whether to use TMA for loading weights
        TMA_STORE: Whether to use TMA for storing results (currently disabled)
        num_warps: Number of warps per thread block
        num_stages: Number of pipeline stages
        num_ctas: Number of cooperative thread arrays
        
    Returns:
        List of triton.Config objects containing all combinations of the input parameters
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_X,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_x,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_X,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dX_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY=True,
    TMA_LOAD_W=True,
    TMA_STORE=False,  # NOTE: TMA_STORE is disabled for now
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
):
    """
    Generate kernel configurations for backward pass dX gradient computation.
    
    Args:
        BLOCK_M: Block sizes for M dimension
        BLOCK_N: Block sizes for N dimension
        BLOCK_K: Block sizes for K dimension
        TMA_LOAD_dY: Whether to use TMA for loading output gradients
        TMA_LOAD_W: Whether to use TMA for loading weights
        TMA_STORE: Whether to use TMA for storing results (currently disabled)
        num_warps: Number of warps per thread block
        num_stages: Number of pipeline stages
        num_ctas: Number of cooperative thread arrays
        
    Returns:
        List of triton.Config objects for dX gradient computation
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_warps,
        num_stages,
        num_ctas,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            TMA_LOAD_dY,
            TMA_LOAD_W,
            TMA_STORE,
            num_warps,
            num_stages,
            num_ctas,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_w,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_W,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_W=tma_load_w,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def get_dW_kernel_configs(
    BLOCK_M=DEFAULT_M_BLOCK_SIZES,
    BLOCK_N=DEFAULT_N_BLOCK_SIZES,
    BLOCK_K=DEFAULT_K_BLOCK_SIZES,
    num_warps=DEFAULT_NUM_WARPS,
    num_stages=DEFAULT_NUM_STAGES,
    num_ctas=DEFAULT_NUM_CTAS,
    TMA_LOAD_dY=True,
    TMA_LOAD_X=True,
    TMA_STORE=False,
):
    """
    Generate kernel configurations for weight gradient (dW) computation.
    
    Args:
        BLOCK_M: Block sizes for M dimension
        BLOCK_N: Block sizes for N dimension
        BLOCK_K: Block sizes for K dimension
        num_warps: Number of warps per thread block
        num_stages: Number of pipeline stages
        num_ctas: Number of cooperative thread arrays
        TMA_LOAD_dY: Whether to use TMA for loading output gradients
        TMA_LOAD_X: Whether to use TMA for loading input data
        TMA_STORE: Whether to use TMA for storing results
        
    Returns:
        List of triton.Config objects for weight gradient computation
    """
    (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        num_ctas,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
    ) = convert_args_to_list(
        [
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps,
            num_stages,
            num_ctas,
            TMA_LOAD_dY,
            TMA_LOAD_X,
            TMA_STORE,
        ]
    )
    kernel_configs = []
    for (
        block_m,
        block_n,
        block_k,
        w,
        s,
        tma_load_dy,
        tma_load_x,
        tma_store,
        num_ctas,
    ) in product(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
        num_stages,
        TMA_LOAD_dY,
        TMA_LOAD_X,
        TMA_STORE,
        num_ctas,
    ):
        kernel_configs.append(
            triton.Config(
                dict(
                    BLOCK_SIZE_M=block_m,
                    BLOCK_SIZE_N=block_n,
                    BLOCK_SIZE_K=block_k,
                    USE_TMA_LOAD_dY=tma_load_dy,
                    USE_TMA_LOAD_X=tma_load_x,
                    USE_TMA_STORE=tma_store,
                ),
                num_warps=w,
                num_stages=s,
                num_ctas=num_ctas,
            )
        )

    return kernel_configs


def estimate_smem_reqs(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
):
    """
    Estimate shared memory requirements for a kernel configuration.
    
    Args:
        num_stages: Number of pipeline stages
        BLOCK_SIZE_M: Block size in M dimension
        BLOCK_SIZE_N: Block size in N dimension
        BLOCK_SIZE_K: Block size in K dimension
        dtype: Data type of the tensors
        
    Returns:
        Estimated shared memory requirement in bytes
    """
    num_bytes = dtype.itemsize
    return (
        num_stages * BLOCK_SIZE_K * (BLOCK_SIZE_M + BLOCK_SIZE_N)
        + BLOCK_SIZE_M * BLOCK_SIZE_N
    ) * num_bytes


def exceeds_smem_capacity(
    num_stages: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    dtype: torch.dtype,
    smem_size: int,
    slack: float = 50000,
):
    """
    Check if a kernel configuration exceeds shared memory capacity.
    
    Args:
        num_stages: Number of pipeline stages
        BLOCK_SIZE_M: Block size in M dimension
        BLOCK_SIZE_N: Block size in N dimension
        BLOCK_SIZE_K: Block size in K dimension
        dtype: Data type of the tensors
        smem_size: Available shared memory size in bytes
        slack: Additional buffer space to account for overhead
        
    Returns:
        True if the configuration exceeds shared memory capacity
    """
    smem_reqs = estimate_smem_reqs(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype
    )
    return smem_reqs > smem_size + slack


def common_prune_criteria(config: triton.Config, kwargs: dict, dtype):
    """
    Apply common pruning criteria to filter out invalid kernel configurations.
    
    Args:
        config: Triton kernel configuration to evaluate
        kwargs: Kernel arguments containing problem dimensions and flags
        dtype: Data type of the tensors
        
    Returns:
        True if the configuration should be pruned (removed)
    """
    from grouped_gemm.interface import supports_tma
    from grouped_gemm.kernels.tuning import get_device_properties

    smem_size = get_device_properties().SIZE_SMEM

    num_stages = config.num_stages
    BLOCK_SIZE_M = config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config.kwargs["BLOCK_SIZE_K"]

    num_tokens = kwargs["NUM_TOKENS"]
    num_experts = kwargs["NUM_EXPERTS"]
    permute_x = kwargs["PERMUTE_X"]
    permute_y = kwargs["PERMUTE_Y"]
    tokens_per_expert = num_tokens // num_experts

    # use_tma = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    MIN_BLOCK_SIZE_M = DEFAULT_M_BLOCK_SIZES[0]
    if exceeds_smem_capacity(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype, smem_size
    ):
        return True
    if BLOCK_SIZE_M > tokens_per_expert * 2 and tokens_per_expert > MIN_BLOCK_SIZE_M:
        return True
    if permute_x and permute_y:
        return True
    # if not supports_tma() and any(use_tma):
    #     return True
    return False


def maybe_disable_tma(config: triton.Config):
    """
    Disable TMA (Tensor Memory Accelerator) features if not supported by the GPU.
    
    Args:
        config: Triton kernel configuration to modify in-place
    """
    from grouped_gemm.interface import supports_tma

    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        logger.info("Disabling TMA")
        for k in tma_keys:
            config.kwargs[k] = False


def prune_kernel_configs_fwd(configs: list[triton.Config], args, **kwargs):
    """
    Prune kernel configurations for forward pass operations.
    
    Args:
        configs: List of kernel configurations to filter
        args: Positional arguments (unused)
        **kwargs: Keyword arguments containing tensor pointers and operation flags
        
    Returns:
        Filtered list of valid kernel configurations
    """
    x = kwargs["x_ptr"]
    dtype = x.dtype

    logger.debug(f"Pruning configs: {len(configs)}")

    pruned_configs = []
    for config in configs:
        # disable TMA if gpu does not support it
        maybe_disable_tma(config)

        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            # Dynamically disable TMA_LOAD_X for permuted X
            config.kwargs["USE_TMA_LOAD_X"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_Y"]:
            continue

        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_dX_configs(configs: List[triton.Config], args, **kwargs):
    """
    Prune kernel configurations for dX gradient computation.
    
    Args:
        configs: List of kernel configurations to filter
        args: Positional arguments (unused)
        **kwargs: Keyword arguments containing tensor pointers and operation flags
        
    Returns:
        Filtered list of valid kernel configurations for dX computation
    """
    dtype = kwargs["w_ptr"].dtype

    logger.debug(f"Pruning configs: {len(configs)}")
    pruned_configs = []

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            # dynamically disable TMA_LOAD_dY for permuted Y
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_STORE"] and kwargs["PERMUTE_X"]:
            continue
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs


def prune_kernel_configs_backward_dW(configs: list[triton.Config], args, **kwargs):
    """
    Prune kernel configurations for weight gradient (dW) computation.
    
    Args:
        configs: List of kernel configurations to filter
        args: Positional arguments (unused)
        **kwargs: Keyword arguments containing tensor pointers and operation flags
        
    Returns:
        Filtered list of valid kernel configurations for dW computation
    """
    dtype = kwargs["x_ptr"].dtype

    pruned_configs = []
    logger.debug(f"Pruning configs: {len(configs)}")

    for config in configs:
        if common_prune_criteria(config, kwargs, dtype):
            continue
        if config.kwargs["USE_TMA_LOAD_dY"] and kwargs["PERMUTE_Y"]:
            config.kwargs["USE_TMA_LOAD_dY"] = False
        if config.kwargs["USE_TMA_LOAD_X"] and kwargs["PERMUTE_X"]:
            config.kwargs["USE_TMA_LOAD_X"] = False
        pruned_configs.append(config)

    logger.debug(f"Pruned configs: {len(pruned_configs)}")
    return pruned_configs
