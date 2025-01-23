# Adapted from https://github.com/triton-lang/kernels/blob/eeeebdd8be7d13629de22d600621e6234057eed3/kernels/matmul_perf_model.py
# https://github.com/triton-lang/kernels is licensed under the MIT License.

import functools
import heapq

import torch

from triton import cdiv
from triton.runtime import driver
from triton.testing import (
    get_dram_gbps,
    get_max_simd_tflops,
    get_max_tensorcore_tflops,
    nvsmi,
)


@functools.lru_cache
def get_clock_rate_in_khz():
    try:
        return nvsmi(["clocks.max.sm"])[0] * 1e3
    except FileNotFoundError:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps) / num_subcores * get_max_simd_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_tflops(device, num_ctas, num_warps, dtype):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(device, num_ctas, num_warps, dtype)


def estimate_matmul_time(
    # backend, device,
    num_warps,
    num_stages,  #
    A,
    B,
    C,  #
    M,
    N,
    K,  #
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    SPLIT_K,  #
    debug=False,
    **kwargs,  #
):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    device = torch.cuda.current_device()
    dtype = A.dtype
    dtsize = A.element_size()

    num_cta_m = cdiv(M, BLOCK_M)
    num_cta_n = cdiv(N, BLOCK_N)
    num_cta_k = SPLIT_K
    num_ctas = num_cta_m * num_cta_n * num_cta_k

    # If the input is smaller than the block size
    M, N = max(M, BLOCK_M), max(N, BLOCK_N)

    # time to compute
    total_ops = 2 * M * N * K / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # time to load data
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)  # 32 active ctas are enough to saturate
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)  # 32-108, remaining 5%
    dram_bw = get_dram_gbps(device) * (active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05)  # in GB/s
    l2_bw = dram_bw * 4  # rough estimation (should be 4.7 for A100?)
    # assume 80% of (following) loads are in L2 cache
    load_a_dram = M * K * dtsize * (1 + 0.2 * (num_cta_n - 1))
    load_a_l2 = M * K * dtsize * 0.8 * (num_cta_n - 1)
    load_b_dram = N * K * dtsize * (1 + 0.2 * (num_cta_m - 1))
    load_b_l2 = N * K * dtsize * 0.8 * (num_cta_m - 1)
    # total
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)  # MB
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    # loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # estimate storing time
    store_bw = dram_bw * 0.6  # :o
    store_c_dram = M * N * dtsize * SPLIT_K / (1024 * 1024)  # MB
    if SPLIT_K == 1:
        store_ms = store_c_dram / store_bw
    else:
        reduce_bw = store_bw
        store_ms = store_c_dram / reduce_bw
        # c.zero_()
        zero_ms = M * N * 2 / (1024 * 1024) / store_bw
        store_ms += zero_ms

    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        print(
            f"Total time: {total_time_ms}ms, compute time: {compute_ms}ms, "
            f"loading time: {load_ms}ms, store time: {store_ms}ms, "
            f"Activate CTAs: {active_cta_ratio*100}%"
        )
    return total_time_ms


def early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    dtsize = named_args["A"].element_size()
    dtype = named_args["A"].dtype

    # 1. make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
        )

        max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs

    # Some dtypes do not allow atomic_add
    if dtype not in [torch.float16, torch.float32]:
        configs = [config for config in configs if config.kwargs["SPLIT_K"] == 1]

    # group configs by (BLOCK_M,_N,_K, SPLIT_K, num_warps)
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            kw["SPLIT_K"],
            config.num_warps,
            config.num_stages,
        )

        key = (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps)
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]

    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps = k
        if capability[0] >= 8:
            # compute cycles (only works for ampere GPUs)
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8

            ldgsts_latency = 300  # Does this matter?
            optimal_num_stages = ldgsts_latency / mma_cycles

            # nearest stages, prefer large #stages
            nearest = heapq.nsmallest(
                2,
                v,
                key=lambda x: (
                    10 + abs(x[1] - optimal_num_stages)
                    if (x[1] - optimal_num_stages) < 0
                    else x[1] - optimal_num_stages
                ),
            )

            for n in nearest:
                pruned_configs.append(n[0])
        else:  # Volta & Turing only supports num_stages <= 2
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs

class PerfOps:
    def __init__(self): return
    @staticmethod
    def early_config_prune(*args, **kwargs):
        return _early_config_prune(*args, **kwargs)
    @staticmethod
    def estimate_matmul_time(*args, **kwargs):
        return _estimate_matmul_time(*args, **kwargs)
pass

class TritonOps:
    __slots__ = "matmul_perf_model",
    def __init__(self): self.matmul_perf_model = PerfOps()
pass
