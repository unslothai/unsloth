import argparse
import datetime
import json
import logging
import math
import os
from itertools import product

import pandas as pd
import torch
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    KernelResult,
)

SEED = 42


def create_merged_results(
    df: pd.DataFrame, mode: str, seqlen: int, dtype: torch.dtype, autotune: bool
):
    kernel_result_cols = df.columns.to_list()
    test_config_dict = {"mode": mode, "seqlen": seqlen, "dtype": dtype, "autotune": autotune}
    test_config_cols = list(test_config_dict.keys())
    for col in test_config_cols:
        df[col] = test_config_dict[col]
    # Reorder columns so that test config cols are first
    df = df[test_config_cols + kernel_result_cols]
    return df


def post_process_results(
    results: list[KernelResult],
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
):
    df = KernelResult.to_dataframe(results, sort_by="speedup")
    df = create_merged_results(df, mode, seqlen, dtype, autotune)
    return df


def save_results(
    df: pd.DataFrame,
    results_dir: str,
    mode: str,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
):
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{results_dir}/{mode}"
    save_path = f"{save_dir}/{dt}_{seqlen}_{str(dtype).split('.')[-1]}.csv"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving results to {save_path}")
    df.to_csv(save_path, index=False)


def create_kernel_configs(args: argparse.Namespace, permute_x: bool, permute_y: bool):
    block_m_range = power_of_two_range(args.BLOCK_SIZE_M[0], args.BLOCK_SIZE_M[1])
    block_n_range = power_of_two_range(args.BLOCK_SIZE_N[0], args.BLOCK_SIZE_N[1])
    block_k_range = power_of_two_range(args.BLOCK_SIZE_K[0], args.BLOCK_SIZE_K[1])
    num_warps_range = multiples_of_range(args.num_warps[0], args.num_warps[1], step=2)
    num_stages_range = multiples_of_range(args.num_stages[0], args.num_stages[1], step=1)

    mode = args.mode
    kernel_configs = []
    for block_m, block_n, block_k, num_warps, num_stages, tma_load_a, tma_load_b in product(
        block_m_range, block_n_range, block_k_range, num_warps_range, num_stages_range, [True, False], [True, False]
    ):
        if mode == "forward":
            kernel_config = KernelConfigForward(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_w=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dW":
            kernel_config = KernelConfigBackward_dW(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_x=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        elif mode == "dX":
            kernel_config = KernelConfigBackward_dX(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tma_load_dy=tma_load_a,
                use_tma_load_w=tma_load_b,
                permute_x=permute_x,
                permute_y=permute_y,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        kernel_configs.append(kernel_config)

    logging.info(f"Pruning {len(kernel_configs)} kernel configs")

    pruned_configs = []
    for config in kernel_configs:
        if mode == "forward":
            if permute_x and config.use_tma_load_x:
                continue
        elif mode == "dW":
            if permute_x and config.use_tma_load_x:
                continue
            if permute_y and config.use_tma_load_dy:
                continue
        elif mode == "dX":
            if permute_y and config.use_tma_load_dy:
                continue
        pruned_configs.append(config)
    logging.info(f"After pruning, {len(pruned_configs)} kernel configs")

    return pruned_configs


def power_of_two_range(start, end):
    start = math.log2(start)
    end = math.log2(end)
    return [2**i for i in range(int(start), int(end) + 1)]


def multiples_of_range(start, end, step=1):
    return list(range(start, end + step, step))
