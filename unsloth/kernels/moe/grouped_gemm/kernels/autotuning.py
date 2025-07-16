# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Autotuning utils
"""

import logging
from itertools import product
from typing import List, Any, Optional, Union

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


def val_to_list(val: Optional[Any]) -> Optional[List[Any]]:
    """
    Converts a value to a list if it is not None or already a list.
    
    Args:
        val (`Any`, *optional*): The value to convert to a list.
    
    Returns:
        `List[Any]` or `None`: A list containing the value if it is not None and not a list,
        the original list if it is already a list, or None if the value is None.
    """
    if val is None:
        return None
    elif isinstance(val, list):
        return val
    else:
        return [val]


def convert_args_to_list(args: List[Any]) -> List[Optional[List[Any]]]:
    """
    Converts a list of values into lists using the `val_to_list` function.
    
    Args:
        args (`List[Any]`): A list of values to convert to lists.
    
    Returns:
        `List[List[Any]]`: A list of lists, where each value from the input list has been
        converted to a list using the `val_to_list` function.
    """
    return [val_to_list(arg) for arg in args]


def get_forward_configs(
    BLOCK_M: Union[int, List[int]]      = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: Union[int, List[int]]      = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: Union[int, List[int]]      = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_X: Union[bool, List[bool]] = True,
    TMA_LOAD_W: Union[bool, List[bool]] = True,
    TMA_STORE: Union[bool, List[bool]]  = False,  # NOTE: TMA_STORE is disabled for now
    num_warps: Union[int, List[int]]    = DEFAULT_NUM_WARPS,
    num_stages: Union[int, List[int]]   = DEFAULT_NUM_STAGES,
    num_ctas: Union[int, List[int]]     = DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
    """
    Generates a list of kernel configurations for forward pass using the provided parameters.
    
    Args:
        BLOCK_M (`Union[int, List[int]]`, defaults to [64, 128]):
            Block size for M dimension.
        BLOCK_N (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for N dimension.
        BLOCK_K (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for K dimension.
        TMA_LOAD_X (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading X.
        TMA_LOAD_W (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading W.
        TMA_STORE (`Union[bool, List[bool]]`, defaults to False):
            Whether to use TMA for storing the result.
        num_warps (`Union[int, List[int]]`, defaults to [4, 8]):
            Number of warps to use.
        num_stages (`Union[int, List[int]]`, defaults to [3, 4, 5]):
            Number of stages to use.
        num_ctas (`Union[int, List[int]]`, defaults to 1):
            Number of CTAs to use.
    
    Returns:
        `List[triton.Config]`: A list of kernel configurations for forward pass.
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
    BLOCK_M: Union[int, List[int]]       = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: Union[int, List[int]]       = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: Union[int, List[int]]       = DEFAULT_K_BLOCK_SIZES,
    TMA_LOAD_dY: Union[bool, List[bool]] = True,
    TMA_LOAD_W: Union[bool, List[bool]]  = True,
    TMA_STORE: Union[bool, List[bool]]   = False,  # NOTE: TMA_STORE is disabled for now
    num_warps: Union[int, List[int]]     = DEFAULT_NUM_WARPS,
    num_stages: Union[int, List[int]]    = DEFAULT_NUM_STAGES,
    num_ctas: Union[int, List[int]]      = DEFAULT_NUM_CTAS,
) -> List[triton.Config]:
    """
    Generates a list of kernel configurations for dX computation using the provided parameters.
    
    Args:
        BLOCK_M (`Union[int, List[int]]`, defaults to [64, 128]):
            Block size for M dimension.
        BLOCK_N (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for N dimension.
        BLOCK_K (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for K dimension.
        TMA_LOAD_dY (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading dY.
        TMA_LOAD_W (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading W.
        TMA_STORE (`Union[bool, List[bool]]`, defaults to False):
            Whether to use TMA for storing the result.
        num_warps (`Union[int, List[int]]`, defaults to [4, 8]):
            Number of warps to use.
        num_stages (`Union[int, List[int]]`, defaults to [3, 4, 5]):
            Number of stages to use.
        num_ctas (`Union[int, List[int]]`, defaults to 1):
            Number of CTAs to use.
    
    Returns:
        `List[triton.Config]`: A list of kernel configurations for dX computation.
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
    BLOCK_M: Union[int, List[int]]       = DEFAULT_M_BLOCK_SIZES,
    BLOCK_N: Union[int, List[int]]       = DEFAULT_N_BLOCK_SIZES,
    BLOCK_K: Union[int, List[int]]       = DEFAULT_K_BLOCK_SIZES,
    num_warps: Union[int, List[int]]     = DEFAULT_NUM_WARPS,
    num_stages: Union[int, List[int]]    = DEFAULT_NUM_STAGES,
    num_ctas: Union[int, List[int]]      = DEFAULT_NUM_CTAS,
    TMA_LOAD_dY: Union[bool, List[bool]] = True,
    TMA_LOAD_X: Union[bool, List[bool]]  = True,
    TMA_STORE: Union[bool, List[bool]]   = False,
) -> List[triton.Config]:
    """
    Generates a list of kernel configurations for dW computation using the provided parameters.
    
    Args:
        BLOCK_M (`Union[int, List[int]]`, defaults to [64, 128]):
            Block size for M dimension.
        BLOCK_N (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for N dimension.
        BLOCK_K (`Union[int, List[int]]`, defaults to [64, 128, 256]):
            Block size for K dimension.
        num_warps (`Union[int, List[int]]`, defaults to [4, 8]):
            Number of warps to use.
        num_stages (`Union[int, List[int]]`, defaults to [3, 4, 5]):
            Number of stages to use.
        num_ctas (`Union[int, List[int]]`, defaults to 1):
            Number of CTAs to use.
        TMA_LOAD_dY (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading dY.
        TMA_LOAD_X (`Union[bool, List[bool]]`, defaults to True):
            Whether to use TMA for loading X.
        TMA_STORE (`Union[bool, List[bool]]`, defaults to False):
            Whether to use TMA for storing the result.
    
    Returns:
        `List[triton.Config]`: A list of kernel configurations for dW computation.
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
) -> int:
    """
    Estimates the shared memory requirements for a kernel configuration.
    
    Args:
        num_stages (`int`):
            Number of stages to use.
        BLOCK_SIZE_M (`int`):
            Block size for M dimension.
        BLOCK_SIZE_N (`int`):
            Block size for N dimension.
        BLOCK_SIZE_K (`int`):
            Block size for K dimension.
        dtype (`torch.dtype`):
            Data type of the tensors.
    
    Returns:
        `int`: Estimated shared memory requirements (in bytes).
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
) -> bool:
    """
    Checks if the shared memory requirements exceed the device's capacity.
    
    Args:
        num_stages (`int`):
            Number of stages to use.
        BLOCK_SIZE_M (`int`):
            Block size for M dimension.
        BLOCK_SIZE_N (`int`):
            Block size for N dimension.
        BLOCK_SIZE_K (`int`):
            Block size for K dimension.
        dtype (`torch.dtype`):
            Data type of the tensors.
        smem_size (`int`):
            Shared memory size of the device.
        slack (`float`, defaults to 50000):
            Slack to add to the shared memory size.
    
    Returns:
        `bool`: True if the shared memory requirements exceed the device's capacity, False otherwise.
    """
    smem_reqs = estimate_smem_reqs(
        num_stages, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype
    )
    return smem_reqs > smem_size + slack


def common_prune_criteria(config: triton.Config, kwargs: dict, dtype: torch.dtype) -> bool:
    """
    Common pruning criteria for kernel configurations.
    
    Determines whether a kernel configuration should be pruned based on shared memory
    constraints and other performance considerations.
    
    Args:
        config (`triton.Config`):
            Kernel configuration to check.
        kwargs (`dict`):
            Additional arguments.
        dtype (`torch.dtype`):
            Data type of the tensors.
    
    Returns:
        `bool`: True if the kernel configuration should be pruned based on:
            - Shared memory requirements exceeding device capacity
            - Block size being too large relative to tokens per expert
            - Both permute_x and permute_y being enabled simultaneously
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


def maybe_disable_tma(config: triton.Config) -> None:
    """
    Disables TMA in the kernel configuration if the device does not support it.
    
    Args:
        config (`triton.Config`):
            Kernel configuration to modify.
    
    Returns:
        None
    """
    from grouped_gemm.interface import supports_tma

    tma_keys = [k for k in config.kwargs.keys() if k.startswith("USE_TMA_")]
    if not supports_tma():
        logger.info("Disabling TMA")
        for k in tma_keys:
            config.kwargs[k] = False


def prune_kernel_configs_fwd(configs: list[triton.Config], args: List[triton.Config], **kwargs) -> List[triton.Config]:
    """
    Prunes kernel configurations for forward pass based on common criteria and TMA usage.
    
    Args:
        configs (`List[triton.Config]`):
            List of kernel configurations to prune.
        args (`List[triton.Config]`):
            Additional arguments.
        kwargs (`dict`):
            Additional keyword arguments.
    
    Returns:
        `List[triton.Config]`: A list of pruned kernel configurations for forward pass.
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


def prune_dX_configs(configs: List[triton.Config], args: List[triton.Config], **kwargs) -> List[triton.Config]:
    """
    Prunes kernel configurations for dX computation based on common criteria and TMA usage.
    
    Args:
        configs (`List[triton.Config]`):
            List of kernel configurations to prune.
        args (`List[triton.Config]`):
            Additional arguments.
        kwargs (`dict`):
            Additional keyword arguments.
    
    Returns:
        `List[triton.Config]`: A list of pruned kernel configurations for dX computation.
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


def prune_kernel_configs_backward_dW(configs: list[triton.Config], args: List[triton.Config], **kwargs) -> List[triton.Config]:
    """
    Prunes kernel configurations for dW computation based on common criteria and TMA usage.
    
    Args:
        configs (`List[triton.Config]`):
            List of kernel configurations to prune.
        args (`List[triton.Config]`):
            Additional arguments.
        kwargs (`dict`):
            Additional keyword arguments.
    
    Returns:
        `List[triton.Config]`: A list of pruned kernel configurations for dW computation.
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