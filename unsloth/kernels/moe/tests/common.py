from typing import Any, Optional
import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from grouped_gemm.kernels.tuning import (
    KernelConfig,
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    prune_kernel_configs_backward_dW,
    prune_kernel_configs_backward_dX,
    prune_kernel_configs_fwd,
)


def print_delimiter(char: str="-", length: int=80) -> None:
    """
    Prints a delimiter line of specified length and character.
    """
    print(char * length)


@contextmanager
def delimiter_context() -> None:
    """
    Context manager that prints a delimiter line before and after the execution of a block of code.
    """
    print_delimiter()
    yield
    print_delimiter()


def make_inputs(M: int, N: int, K: int, E: int, topk: int, dtype: torch.dtype, requires_grad: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates random input tensors for testing purposes.
    """
    X1 = (
        torch.randn((M, K), device="cuda", dtype=dtype, requires_grad=requires_grad)
        / 10
    )
    X2 = (
        torch.randn(
            (M * topk, N), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        / 10
    )
    W1 = (
        torch.randn(
            (E, 2 * N, K), device="cuda", dtype=dtype, requires_grad=requires_grad
        )
        / 10
    )
    W2 = (
        torch.randn((E, K, N), device="cuda", dtype=dtype, requires_grad=requires_grad)
        / 10
    )
    score = torch.randn((M, E), device="cuda", dtype=dtype, requires_grad=requires_grad)
    if requires_grad:
        X1.retain_grad()
        X2.retain_grad()
        W1.retain_grad()
        W2.retain_grad()
        score.retain_grad()
    return X1, X2, W1, W2, score


@dataclass(kw_only=True)
class DataConfig:
    """
    Data configuration class containing sequence length, data type, and device information.
    """
    seq_len: int
    dtype: torch.dtype
    device: str = "cuda"

    bs: int     = 1



@dataclass(kw_only=True)
class ModelConfig:
    """
    Model configuration class containing model parameters such as hidden size, number of experts, and activation functions.
    """
    hidden_size: int
    intermediate_size: int
    num_experts: int
    topk: int
    use_sigmoid: bool
    renormalize: bool
    pre_mul: bool  = False

    post_mul: bool = field(init=False)


    def __post_init__(self):
        """
        Post-initialization method to set additional model configuration parameters.
        """
        self.post_mul = not self.pre_mul


@dataclass(kw_only=True)
class GroupedGEMMTestConfig:
    """
    Test configuration class combining data and model configurations for grouped GEMM operations.
    """
    name: str = "test"

    data_config: DataConfig
    model_config: ModelConfig


TOLERANCE = {
    torch.bfloat16: (1e-3, 1e-3),
    torch.float16: (1e-4, 1e-4),
    torch.float32: (1e-5, 1e-5),
}


# from https://github.com/triton-lang/triton/blob/main/bench/triton_bench/testing.py
def assert_equal(ref, tri) -> None:
    """
    Asserts that two values or tensors are equal.
    """
    if isinstance(ref, torch.Tensor):
        assert torch.all(ref == tri), f"tensors not equal {ref} != {tri}"
    else:
        assert ref == tri, f"ref not equal to tri {ref} != {tri}"


def assert_close(ref: torch.Tensor, tri: torch.Tensor, maxtol: Optional[float]=None, rmstol: Optional[float]=None, description: str="--", verbose: bool=True) -> None:
    """
    Asserts that two tensors are close within a specified tolerance.
    """
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert ref.shape == tri.shape, (
        f"Tensors must have same size {ref.shape=} {tri.shape=}"
    )

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(inf_mask_ref, inf_mask_tri), (
        "Tensor must have same infinite elements"
    )
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(
            "%s maximum relative error = %s (threshold = %s)"
            % (description, max_err, maxtol)
        )
        print(
            "%s RMS relative error = %s (threshold = %s)"
            % (description, rms_err, rmstol)
        )

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print(
            "%d / %d mismatched elements (shape = %s) at coords %s"
            % (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist())
        )

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[*bad_idxs].cpu())
        print("tri values: ", tri[*bad_idxs].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


def assert_indx_equal(ref: torch.Tensor, tri: torch.Tensor) -> None:
    """
    Asserts that two tensors are equal up to the length of the first tensor.
    """
    assert_equal(ref, tri[: len(ref)])
    assert torch.all(tri[len(ref) :] == -1)


def get_kernel_test_configs(
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int    = 4,
    num_stages: int   = 2,
) -> list[KernelConfig]:
    """
    Generates a list of kernel configurations for testing.
    """
    configs_fwd = []
    configs_bwd_dX = []
    configs_bwd_dW = []

    for permute_x in [False, True]:
        for permute_y in [False, True]:
            for use_tma_load_w in [True, False]:
                for use_tma_load_x in [True, False]:
                    for use_tma_store in [True, False]:
                        configs_fwd.append(
                            KernelConfigForward(
                                BLOCK_SIZE_M=BLOCK_SIZE_M,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                BLOCK_SIZE_K=BLOCK_SIZE_K,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                use_tma_load_w=use_tma_load_w,
                                use_tma_load_x=use_tma_load_x,
                                use_tma_store=use_tma_store,
                                permute_x=permute_x,
                                permute_y=permute_y,
                            )
                        )
                        configs_bwd_dX.append(
                            KernelConfigBackward_dX(
                                BLOCK_SIZE_M=BLOCK_SIZE_M,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                BLOCK_SIZE_K=BLOCK_SIZE_K,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                use_tma_load_dy=use_tma_load_x,
                                use_tma_load_w=use_tma_load_w,
                                permute_x=permute_x,
                                permute_y=permute_y,
                                use_tma_store=use_tma_store,
                            )
                        )
                        configs_bwd_dW.append(
                            KernelConfigBackward_dW(
                                BLOCK_SIZE_M=BLOCK_SIZE_M,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                BLOCK_SIZE_K=BLOCK_SIZE_K,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                use_tma_load_dy=use_tma_load_w,
                                use_tma_load_x=use_tma_load_x,
                                permute_x=permute_x,
                                permute_y=permute_y,
                                use_tma_store=use_tma_store,
                            )
                        )
    configs_fwd = prune_kernel_configs_fwd(configs_fwd)
    configs_bwd_dX = prune_kernel_configs_backward_dX(configs_bwd_dX)
    configs_bwd_dW = prune_kernel_configs_backward_dW(configs_bwd_dW)
    return configs_fwd, configs_bwd_dX, configs_bwd_dW


def remove_feature_flags(
    kernel_configs: list[KernelConfig],
    permute_x: bool = True,
    permute_y: bool = True,
    tma_loads: bool = True,
    tma_store: bool = True,
) -> None:
    """
    Removes specific feature flags from a list of kernel configurations.
    """
    pruned_configs = []
    for config in kernel_configs:
        # Remove permute flags first:
        if permute_x and config.permute_x:
            continue
        if permute_y and config.permute_y:
            continue
        if tma_loads:
            if isinstance(config, KernelConfigForward):
                if config.use_tma_load_w or config.use_tma_load_x:
                    continue
            if isinstance(config, KernelConfigBackward_dX):
                if config.use_tma_load_dy or config.use_tma_load_w:
                    continue
            if isinstance(config, KernelConfigBackward_dW):
                if config.use_tma_load_dy or config.use_tma_load_x:
                    continue
        if tma_store:
            if config.use_tma_store:
                continue
        pruned_configs.append(config)
    return pruned_configs


# Test Configs

TOPK = [1, 4]
NUM_EXPERTS = [4, 16]

TEST_MODEL_SIZES = [
    (32, 32),  # Debug
    (128, 128),  # Small
    (512, 512),  # Medium
]

SMALL_MODEL_CONFIGS = [
    ModelConfig(
        topk=topk,
        num_experts=num_experts,
        hidden_size=model_size[0],
        intermediate_size=model_size[1],
        use_sigmoid=False,
        renormalize=False,
    )
    for topk, num_experts, model_size in itertools.product(
        TOPK, NUM_EXPERTS, TEST_MODEL_SIZES
    )
]
LLAMA_MODEL_CONFIG = ModelConfig(
    topk=1,
    num_experts=16,
    hidden_size=5120,
    intermediate_size=8192,
    use_sigmoid=True,
    renormalize=False,
)
QWEN_MODEL_CONFIG = ModelConfig(
    topk=8,
    num_experts=128,
    hidden_size=2048,
    intermediate_size=768,
    use_sigmoid=False,
    renormalize=False,
)

SEQLENS = [128, 1024]
DTYPE = [torch.bfloat16]

DATA_CONFIGS = [
    DataConfig(seq_len=seq_len, dtype=dtype)
    for seq_len, dtype in itertools.product(SEQLENS, DTYPE)
]
KERNEL_CONFIGS_FWD, KERNEL_CONFIGS_BWD_dX, KERNEL_CONFIGS_BWD_dW = (
    get_kernel_test_configs()
)

if __name__ == "__main__":
    print(
        KERNEL_CONFIGS_BWD_dX[0].to_string(
            include_tuning_params=False, include_tma=False
        )
    )