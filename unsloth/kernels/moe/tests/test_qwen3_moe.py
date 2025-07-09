# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import argparse
from contextlib import contextmanager

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.layers.qwen3_moe import Qwen3MoeGroupedGEMMBlock

from .moe_utils import (
    Qwen3MoeFusedGroupedGEMMBlock,
    check_fwd,
    check_grads,
    check_grouped_gemm_results,
    run_backward,
    run_forward,
)

"""
Qwen3 MoE tests

NOTE: Test this as a module and NOT with pytest as running with pytest results in random numerical errors: python -m tests.test_qwen3_moe --permute_x --permute_y --autotune NOT pytest -sv tests/test_qwen3_moe.py
More specifically, all tests pass when run individually, but some will fail randomly (even with the same seed) when the entire test is run as a parametrized test suite using pytest, likely due to how pytest interacts with triton / autotuning.

See tests/run_qwen3_moe_tests.sh for a script that runs all the tests

The tests run the following:
Huggingface's Qwen3 MoE block (Qwen3MoeSparseMoeBlock)
Torch-native grouped gemm version of MoE block (Qwen3MoeGroupedGEMMBlock), which is the HF block with the expert computation replaced with a torch-native grouped gemm
Triton kernel grouped gemm version of MoE block (Qwen3MoeFusedGroupedGEMMBlock), which is the HF block with the expert computation replaced with the fused triton grouped gemm kernel

The tests check the following:
- HF MoE block vs torch grouped gemm MoE block (sanity check)
- torch grouped gemm MoE block vs fused grouped gemm MoE block -- this allows us to test each of the intermediate results for easier debugging
- HF MoE block vs fused grouped gemm MoE block -- this is the actual test

Both forward and backward passes are tests:
- forward: output of the moe block
- backwards:
    - X: gradient of the input to the moe block
    - gate.weight: gradient of the gate weights (router weights)
    - gate_proj: gradient of concatenated gate projections
    - up_proj: gradient of the concatenated up projections
    - down_proj: gradient of the concatenated down projections

Additionally, for the torch grouped gemm and triton grouped gemm versions, the intermediate outputs of the forward pass are checked:
- first_gemm: output of the first grouped gemm (X @ fused_gate_proj)
- intermediate: output of silu_mul(first_gemm)
- second_gemm: output of the second grouped gemm (intermediate @ down_proj)
- hidden_states_unpermute: output of the second_gemm after unpermuting back to token order (from expert grouped order); in the case where the permutation is fused in the triton kernel, this is the same as second_gemm
- hidden_states: output with the topk_weights applied
"""

TOLERANCES = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float: (1e-5, 1e-5),
}


@pytest.fixture(scope="module")
def model_id():
    return "Qwen/Qwen3-30B-A3B"


@pytest.fixture(scope="module")
def config(model_id: str):
    return AutoConfig.from_pretrained(model_id)


@contextmanager
def annotated_context(prelude, epilogue="Passed!", char="-", num_chars=80):
    print(char * num_chars)
    print(prelude)
    yield
    print(epilogue)
    print(char * num_chars)


SEED = 42
SEQ_LENS = [1024]
DTYPES = [torch.bfloat16]

# Reduce the number of autotuning configs to prevent excessive runtime
NUM_AUTOTUNE_CONFIGS = 50


@pytest.mark.parametrize(
    "permute_y", [True], ids=lambda x: "permute_y" if x else "no_permute_y"
)
@pytest.mark.parametrize(
    "permute_x", [True], ids=lambda x: "permute_x" if x else "no_permute_x"
)
@pytest.mark.parametrize(
    "autotune", [True], ids=lambda x: "autotune" if x else "manual"
)
@pytest.mark.parametrize("seqlen", SEQ_LENS, ids=lambda x: f"seqlen={x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_qwen3_moe(
    config: Qwen3MoeConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
):
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size
    bs = 1
    atol, rtol = TOLERANCES[dtype]
    # Reference op -- HF
    moe_block = Qwen3MoeSparseMoeBlock(config).to(device, dtype)

    # Torch-native grouped gemm version of MoE Block -- for sanity checking
    grouped_gemm_block = Qwen3MoeGroupedGEMMBlock.from_hf(moe_block).to(device, dtype)
    grouped_gemm_block.check_weights(moe_block)

    if not autotune:
        kernel_config_fwd = KernelConfigForward()
        kernel_config_bwd_dW = KernelConfigBackward_dW()
        kernel_config_bwd_dX = KernelConfigBackward_dX()
    else:
        from grouped_gemm.kernels.backward import (
            _autotuned_grouped_gemm_dW_kernel,
            _autotuned_grouped_gemm_dX_kernel,
        )
        from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

        # Hack to reduce number of autotuning configs
        _autotuned_grouped_gemm_forward_kernel.configs = (
            _autotuned_grouped_gemm_forward_kernel.configs[:NUM_AUTOTUNE_CONFIGS]
        )
        _autotuned_grouped_gemm_dW_kernel.configs = (
            _autotuned_grouped_gemm_dW_kernel.configs[:NUM_AUTOTUNE_CONFIGS]
        )
        _autotuned_grouped_gemm_dX_kernel.configs = (
            _autotuned_grouped_gemm_dX_kernel.configs[:NUM_AUTOTUNE_CONFIGS]
        )

        kernel_config_fwd = None
        kernel_config_bwd_dW = None
        kernel_config_bwd_dX = None

    # Triton kernel grouped gemm version of MoE Block -- this is what we're testing
    fused_gemm_block = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
        moe_block,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
    ).to(device, dtype)
    fused_gemm_block.check_weights(moe_block)

    X = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True
    )

    # Forward
    ref_result = run_forward(moe_block, X, is_grouped_gemm=False)
    grouped_result = run_forward(grouped_gemm_block, X, is_grouped_gemm=True)
    fused_result = run_forward(fused_gemm_block, X, is_grouped_gemm=True)

    with annotated_context(
        "Testing forward pass",
        epilogue="Passed forward tests!",
        char="=",
        num_chars=100,
    ):
        # Sanity checks

        with annotated_context(
            "Checking HF vs torch grouped gemm MoE forward outputs..."
        ):
            check_fwd(ref_result, grouped_result, atol, rtol, verbose=False)

        with annotated_context(
            "Checking torch grouped gemm MoE vs fused grouped gemm MoE forward outputs..."
        ):
            # We implement a custom check for grouped gemm results to test each of the intermediate results for easier debugging
            check_grouped_gemm_results(
                grouped_result.grouped_gemm_result,
                fused_result.grouped_gemm_result,
                permute_y=permute_y,
                atol=atol,
                rtol=rtol,
                verbose=False,
            )
        # Actual test
        with annotated_context(
            "Checking HF vs fused grouped gemm MoE forward outputs..."
        ):
            check_fwd(ref_result, fused_result, atol, rtol, verbose=True)

    # Backward
    grad_output = torch.randn_like(ref_result.output)
    ref_backward_result = run_backward(
        moe_block, grad_output, output=ref_result.output, X=ref_result.X
    )
    grouped_backward_result = run_backward(
        grouped_gemm_block,
        grad_output,
        output=grouped_result.output,
        X=grouped_result.X,
    )
    fused_backward_result = run_backward(
        fused_gemm_block, grad_output, output=fused_result.output, X=fused_result.X
    )

    with annotated_context(
        "Testing backward pass",
        epilogue="Passed backward tests!",
        char="=",
        num_chars=100,
    ):
        # Sanity checks
        with annotated_context("Checking HF vs torch grouped gemm MoE grads..."):
            check_grads(
                ref_backward_result, grouped_backward_result, atol, rtol, verbose=False
            )
        with annotated_context(
            "Checking torch grouped gemm MoE vs fused grouped gemm MoE grads..."
        ):
            check_grads(
                grouped_backward_result,
                fused_backward_result,
                atol,
                rtol,
                verbose=False,
            )

        # Actual test
        with annotated_context("Checking HF vs fused grouped gemm MoE grads..."):
            check_grads(
                ref_backward_result, fused_backward_result, atol, rtol, verbose=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--permute_x", action="store_true")
    parser.add_argument("--permute_y", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args_dict = vars(args)

    model_id = "Qwen/Qwen3-30B-A3B"
    config = AutoConfig.from_pretrained(model_id)
    atol, rtol = TOLERANCES[args.dtype]

    print(
        f"Testing {model_id} with seqlen={args.seqlen}, dtype={args.dtype}, permute_x={args.permute_x}, permute_y={args.permute_y}, autotune={args.autotune}, atol={atol}, rtol={rtol}"
    )
    test_qwen3_moe(config, **args_dict)
