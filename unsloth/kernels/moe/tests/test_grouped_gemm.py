# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

from dataclasses import asdict

import pytest
import torch

from grouped_gemm.interface import (
    grouped_gemm,
    grouped_gemm_dW,
    grouped_gemm_dX,
    grouped_gemm_forward,
)
from grouped_gemm.kernels.tuning import (
    KernelConfig,
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.moe_ops import (
    calculate_topk,
    get_routing_indices,
    permute,
    torch_grouped_gemm,
    unpermute,
)

from .common import (
    DATA_CONFIGS,
    KERNEL_CONFIGS_FWD,
    LLAMA_MODEL_CONFIG,
    QWEN_MODEL_CONFIG,
    SMALL_MODEL_CONFIGS,
    TOLERANCE,
    DataConfig,
    KERNEL_CONFIGS_BWD_dW,
    KERNEL_CONFIGS_BWD_dX,
    ModelConfig,
    make_inputs,
)

SEED = 0


# Only certain combinations of permute_x, permute_y, use_W1 are valid.
# use_W1 => first grouped GEMM in a fused MoE MLP
# use_W2 => second grouped GEMM in a fused MoE MLP
# permute_x => permute the input to the grouped GEMM, only done for the first grouped GEMM
# permute_y => permute the output of the grouped GEMM, only done for the second grouped GEMM
# fuse_mul_post => fuse the multiplication of topk weights in the epilogue of the second grouped GEMM; only used for inference, not currently tested
def check_valid_config(
    permute_x, permute_y, use_W1, fuse_mul_post=False, is_backward=False, verbose=False
):
    use_W2 = not use_W1

    if permute_x and permute_y:
        if verbose:
            print(f"Skipping test: {permute_x=} {permute_y=}")
        return False
    if use_W2 and permute_x:
        if verbose:
            print(f"Skipping test: {permute_x=} {use_W2=}")
        return False
    if use_W1 and permute_y:
        if verbose:
            print(f"Skipping test: {permute_y=} {use_W1=}")
        return False
    if fuse_mul_post and use_W1:
        if verbose:
            print(f"Skipping test: {fuse_mul_post=} {use_W1=}")
        return False
    if is_backward and fuse_mul_post:
        if verbose:
            print(f"Skipping test: {fuse_mul_post=} {is_backward=}")
        return False

    return True


"""
grouped_gemm_forward

permute_x: typically in a fused MoE MLP, we can fuse the permutation of hidden states (X) from token order to expert grouped order needed for grouped GEMM by directly loading X in permuted order rather than launching a separate permutation kernel.
permute_y: We can also fuse the unpermutation of tokens after the second grouped GEMM to restore to original token order.  This is fused into the second grouped GEMM by directly storing the output in unpermuted order.
fuse_mul: We can also fuse the multiplication of topk weights in the epilogue of the second grouped GEMM.  Note that this is only supported for inference and not training, although this may change in the future.
use_W1 test the shapes for the first grouped GEMM in a fused MoE MLP
use_W2 = `not use_W1` tests the shapes for the second grouped GEMM in a fused MoE MLP

Given the above, only certain combinations are valid:
- use_W1 is always False when permute_y is True since we only permute the second grouped GEMM
- use_W2 is always False when permute_x is True since we only permute the first grouped GEMM
- only one of permute_x and permute_y can be True
- fuse_mul is only True if permute_y is also True

See `check_valid_config` for more details.
"""


def _test_grouped_gemm_forward(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,  # W1 -> first grouped GEMM in a fused MoE MLP, not W1 -> second grouped GEMM in a fused MoE MLP
    fuse_mul_post: bool = False,
    flatten: bool = True,
    # Manually tuned parameters
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    BLOCK_SIZE_M: int = None,
    BLOCK_SIZE_N: int = None,
    BLOCK_SIZE_K: int = None,
    num_warps: int = None,
    num_stages: int = None,
    # Autotuning parameters
    autotune: bool = False,
    num_autotune_configs: int = None,
    # Flag to manually enable TMA store
    allow_tma_store: bool = False,
    use_autograd: bool = False,
):
    if not check_valid_config(
        permute_x, permute_y, use_W1=use_W1, fuse_mul_post=fuse_mul_post
    ):
        pytest.skip(
            f"Skipping test due to invalid config: {permute_x=} {permute_y=} {use_W1=} {fuse_mul_post=}"
        )

    if use_tma_store and not allow_tma_store:
        pytest.skip("TMA store needs to be debugged due to non-deterministic behavior")

    X1, X2, W1, W2, gating_output = make_inputs(
        M=data_config.bs * data_config.seq_len,
        N=model_config.intermediate_size,
        K=model_config.hidden_size,
        E=model_config.num_experts,
        topk=model_config.topk,
        dtype=data_config.dtype,
    )
    topk = model_config.topk
    use_sigmoid = model_config.use_sigmoid
    renormalize = model_config.renormalize

    X = X1 if use_W1 else X2
    num_tokens = data_config.bs * data_config.seq_len
    E, K, N = W2.shape  # E = num_experts, K = hidden_size, N = intermediate_size
    assert W1.shape == (E, 2 * N, K)
    W = W1 if use_W1 else W2

    if use_W1:
        assert X.shape == (num_tokens, K), (
            f"X.shape: {X.shape}, num_tokens: {num_tokens}, K: {K}"
        )
    else:
        assert X.shape == (num_tokens * topk, N), (
            f"X.shape: {X.shape}, num_tokens: {num_tokens}, topk: {topk}, N: {N}"
        )

    total_tokens = num_tokens * topk
    output_shape = (total_tokens, 2 * N) if use_W1 else (total_tokens, K)

    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, use_sigmoid=use_sigmoid, renormalize=renormalize
    )
    topk_weights = topk_weights.view(-1)  # num_tokens * topk
    topk_ids = topk_ids.view(-1)  # num_tokens * topk

    expert_token_counts, gather_indices = get_routing_indices(topk_ids, num_experts=E)
    assert len(gather_indices) == total_tokens
    assert len(expert_token_counts) == E

    atol, rtol = TOLERANCE[X.dtype]

    Xperm = permute(X, gather_indices, topk)

    Xref = Xperm

    assert Xperm.shape == (total_tokens, K) if use_W1 else (total_tokens, N), (
        f"Xperm.shape: {Xperm.shape}, total_tokens: {total_tokens}, K: {K}"
    )

    ref_output = torch_grouped_gemm(X=Xref, W=W, m_sizes=expert_token_counts)

    if permute_x:
        X_test = X
    else:
        X_test = Xperm

    # No need to run all configs for tests, otherwise takes too long
    if autotune:
        from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

        if num_autotune_configs is not None:
            _autotuned_grouped_gemm_forward_kernel.configs = (
                _autotuned_grouped_gemm_forward_kernel.configs[:num_autotune_configs]
            )

    # Use autograd.Function interface
    if use_autograd:
        from grouped_gemm.interface import grouped_gemm

        kernel_config_fwd = KernelConfigForward(
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=num_warps,
            num_stages=num_stages,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
            use_tma_load_w=use_tma_load_w,
            use_tma_load_x=use_tma_load_x,
            use_tma_store=use_tma_store,
        )

        test_output = grouped_gemm(
            X=X_test,
            W=W,
            topk=topk,
            m_sizes=expert_token_counts,
            gather_indices=gather_indices,
            topk_weights=topk_weights if fuse_mul_post else None,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
            kernel_config_fwd=kernel_config_fwd,
            autotune=autotune,
            is_first_gemm=use_W1,
        )
    # Use manual interface
    else:
        test_output = grouped_gemm_forward(
            X=X_test,
            W=W,
            topk=topk,
            m_sizes=expert_token_counts,
            gather_indices=gather_indices,
            topk_weights=topk_weights if fuse_mul_post else None,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
            use_tma_load_w=use_tma_load_w,
            use_tma_load_x=use_tma_load_x,
            use_tma_store=use_tma_store,
            autotune=autotune,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=num_warps,
            num_stages=num_stages,
            flatten=flatten,
        )
    assert ref_output.shape == output_shape
    assert test_output.shape == output_shape

    if permute_y:
        ref_output = unpermute(ref_output, gather_indices)
    if fuse_mul_post:
        # if we don't permute_y, then test output is permuted with topk weights applied
        # the ref output needs to be unpermuted before multiplying by topk weights since topk weights are in token order
        if not permute_y:
            ref_output = unpermute(ref_output, gather_indices)
            test_output = unpermute(test_output, gather_indices)
        ref_output = ref_output * topk_weights[:, None]

    assert torch.allclose(ref_output, test_output, atol=atol, rtol=rtol), (
        f"Grouped gemm forward failed: {(ref_output - test_output).abs().max().item():.6f}"
    )


# NOTE: Fuse multiplication of topk weights is only supported for inference and not training, although this may change in the future; not currently tested.
@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_FWD,
    ids=lambda x: x.to_string(include_tuning_params=True, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS + [QWEN_MODEL_CONFIG, LLAMA_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_forward_manual(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfigForward,
    use_W1: bool,
):
    _test_grouped_gemm_forward(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_FWD,
    ids=lambda x: x.to_string(include_tuning_params=True, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS + [QWEN_MODEL_CONFIG, LLAMA_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_forward_manual_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfigForward,
    use_W1: bool,
):
    _test_grouped_gemm_forward(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        use_autograd=True,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [10], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_forward_autotune(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    _test_grouped_gemm_forward(
        data_config=data_config,
        model_config=model_config,
        permute_x=permute_x,
        permute_y=permute_y,
        use_W1=use_W1,
        num_autotune_configs=num_autotune_configs,
        autotune=True,
        use_autograd=False,
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [10], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_forward_autotune_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    _test_grouped_gemm_forward(
        data_config=data_config,
        model_config=model_config,
        permute_x=permute_x,
        permute_y=permute_y,
        use_W1=use_W1,
        num_autotune_configs=num_autotune_configs,
        autotune=True,
        use_autograd=True,
    )


"""
grouped_gemm_backward_dX

use_W1 test the shapes for the first grouped GEMM in a fused MoE MLP
use_W2 = `not use_W1` tests the shapes for the second grouped GEMM in a fused MoE MLP

Only certain combinations of permute_x, permute_y, and fuse_mul are supported.

Typically in a fused MoE MLP, we can fuse the permutation of hidden states (X) from token order to expert grouped order needed for grouped GEMM by directly loading X in permuted order rather than launching a separate permutation kernel.
We can also fuse the unpermutation of tokens after the second grouped GEMM to restore to original token order.  This is fused into the second grouped GEMM by directly storing the output in unpermuted order.

Hence the following conditions:
- If use_W1 there are two cases:
    - permute_x is False and topk > 1:
    - dX_test is still in permuted order and has shape (total_tokens, K)
    - it needs to be unpermuted and summed across topk before comparing to ref_grad
- permute_x is True:
    - dX_test is already unpermuted and summed across topk with shape (num_tokens, K)
    - no further processing is needed
- permute_x is False and topk == 1:
    - dX_test needs to be permuted, no need to sum since topk == 1

- If use_W2:
    - permute_x is always False
    - if permute_y:
        - grad_output needs to be unpermuted before passing to grouped_gemm_dX
        - dX_test is permuted and has shape (total_tokens, N)
        - it needs to be unpermuted before comparing to ref_grad or can be compared directly to Xperm.grad
    - if not permute_y:
        - dX_test is not permuted and has shape (total_tokens, N)
        - no further processing is needed
"""


def _test_grouped_gemm_backward_dX(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_load_w: bool = False,
    use_tma_store: bool = False,
    use_W1: bool = True,
    autotune: bool = False,
    num_autotune_configs: int = None,
    BLOCK_SIZE_M: int = None,
    BLOCK_SIZE_N: int = None,
    BLOCK_SIZE_K: int = None,
    num_warps: int = None,
    num_stages: int = None,
    flatten: bool = True,
    allow_tma_store: bool = False,
    use_autograd: bool = False,
    fuse_mul_post: bool = False,
):
    if not check_valid_config(permute_x, permute_y, use_W1=use_W1, is_backward=True):
        pytest.skip(
            f"Skipping test due to invalid config: {permute_x=} {permute_y=} {use_W1=}"
        )

    if use_tma_store and not allow_tma_store:
        pytest.skip("TMA store needs to be debugged due to non-deterministic behavior")

    if (
        autotune
        and model_config.intermediate_size <= 128
        and model_config.hidden_size <= 128
    ):
        pytest.skip("Skipping autotuning for small model configs")

    # Prevent OOM for large intermediate sizes
    if model_config.intermediate_size > 2048:
        model_config.intermediate_size = 1024
    if model_config.hidden_size > 2048:
        model_config.hidden_size = 1024

    use_W2 = not use_W1
    X1, X2, W1, W2, gating_output = make_inputs(
        M=data_config.bs * data_config.seq_len,
        N=model_config.intermediate_size,
        K=model_config.hidden_size,
        E=model_config.num_experts,
        topk=model_config.topk,
        dtype=data_config.dtype,
        requires_grad=True,
    )
    topk = model_config.topk
    num_experts = model_config.num_experts
    use_sigmoid = model_config.use_sigmoid
    renormalize = model_config.renormalize

    X = X1 if use_W1 else X2
    num_tokens = data_config.bs * data_config.seq_len
    total_tokens = num_tokens * topk

    E, K, N = W2.shape  # E = num_experts, K = hidden_size, N = intermediate_size
    assert W1.shape == (E, 2 * N, K)
    W = W1 if use_W1 else W2

    if use_W1:
        assert X.shape == (num_tokens, K), (
            f"X.shape: {X.shape}, num_tokens: {num_tokens}, K: {K}"
        )
    else:
        assert X.shape == (total_tokens, N), (
            f"X.shape: {X.shape}, total_tokens: {total_tokens}, N: {N}"
        )

    W_test = W.detach().clone().requires_grad_(True)

    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, use_sigmoid=use_sigmoid, renormalize=renormalize
    )
    topk_weights = topk_weights.view(-1)  # num_tokens * topk
    topk_ids = topk_ids.view(-1)  # num_tokens * topk

    expert_token_counts, gather_indices = get_routing_indices(topk_ids, num_experts=E)
    assert len(gather_indices) == total_tokens
    assert len(expert_token_counts) == num_experts

    atol, rtol = TOLERANCE[X.dtype]
    Xperm = permute(X, gather_indices, topk)

    # Need to retain grad otherwise grad is not propagated
    X.retain_grad()
    W.retain_grad()
    Xperm.retain_grad()

    assert Xperm.shape == (total_tokens, K) if use_W1 else (total_tokens, N)

    output_shape = (total_tokens, 2 * N) if use_W1 else (total_tokens, K)
    ref_output = torch_grouped_gemm(X=Xperm, W=W, m_sizes=expert_token_counts)
    assert ref_output.shape == output_shape, (
        f"ref_output.shape: {ref_output.shape}, output_shape: {output_shape}"
    )

    if permute_y:
        ref_output = unpermute(ref_output, gather_indices)

    grad_output = torch.randn_like(ref_output)
    ref_output.backward(grad_output)

    assert X.grad is not None
    assert W.grad is not None

    ref_grad = Xperm.grad

    if autotune:
        # No need to run all configs for autotuning
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dX_kernel

        if num_autotune_configs is not None:
            _autotuned_grouped_gemm_dX_kernel.configs = (
                _autotuned_grouped_gemm_dX_kernel.configs[:num_autotune_configs]
            )

    if use_autograd:
        from grouped_gemm.interface import grouped_gemm

        if not autotune:
            kernel_config_fwd = KernelConfigForward()
            kernel_config_bwd_dX = KernelConfigBackward_dX(
                use_tma_load_dy=use_tma_load_dy,
                use_tma_load_w=use_tma_load_w,
                use_tma_store=use_tma_store,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            kernel_config_bwd_dW = KernelConfigBackward_dW()
        else:
            from grouped_gemm.kernels.backward import (
                _autotuned_grouped_gemm_dW_kernel,
                _autotuned_grouped_gemm_dX_kernel,
            )
            from grouped_gemm.kernels.forward import (
                _autotuned_grouped_gemm_forward_kernel,
            )

            if num_autotune_configs is not None:
                _autotuned_grouped_gemm_dX_kernel.configs = (
                    _autotuned_grouped_gemm_dX_kernel.configs[:num_autotune_configs]
                )
                _autotuned_grouped_gemm_forward_kernel.configs = (
                    _autotuned_grouped_gemm_forward_kernel.configs[
                        :num_autotune_configs
                    ]
                )

            kernel_config_fwd = None
            kernel_config_bwd_dX = None
        X_ = (
            X.detach().clone().requires_grad_(True)
            if permute_x
            else Xperm.detach().clone().requires_grad_(True)
        )
        test_output = grouped_gemm(
            X=X_,
            W=W_test,
            m_sizes=expert_token_counts,
            gather_indices=gather_indices,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            is_first_gemm=use_W1,
            dX_only=True,
        )
        assert test_output.shape == ref_output.shape, (
            f"test_output.shape: {test_output.shape}, ref_output.shape: {ref_output.shape}"
        )
        assert torch.allclose(test_output, ref_output, atol=atol, rtol=rtol), (
            f"Grouped gemm backward_dX forward outputs mismatch: {(test_output - ref_output).abs().max().item():.6f}"
        )
        test_output.backward(grad_output)
        assert X_.grad is not None

        # NOTE:need to handle grad differenlty in this case due to errors arising to do how torch autograd handles unpermute and sum reduction
        # the grad of Xperm unpermuted and reduced across topk should match X_.grad
        # However, both will have a numerical difference with that of ref_grad
        # This is due to the fact that torch autograd handles unpermute and sum reduction differently see: https://discuss.pytorch.org/t/permute-unpermute-gradient/219557    else:
        if permute_x and use_W1:
            X_grad_unperm = unpermute(Xperm.grad, gather_indices)
            manual_grad_check = X_grad_unperm.view(num_tokens, topk, K).sum(dim=1)
            assert manual_grad_check.shape == X_.grad.shape, (
                f"manual_grad_check.shape: {manual_grad_check.shape}, X_.grad.shape: {X_.grad.shape}"
            )
            assert torch.allclose(manual_grad_check, X_.grad, atol=atol, rtol=rtol), (
                f"Grouped gemm backward_dX forward outputs mismatch: {(manual_grad_check - X_.grad).abs().max().item():.6f}"
            )
            manual_diff = (X_.grad - manual_grad_check).abs().max().item()
            autograd_diff = (X_.grad - X.grad).abs().max().item()
            print(f"manual_diff: {manual_diff:.6f}, autograd_diff: {autograd_diff:.6f}")
        else:
            assert torch.allclose(X_.grad, ref_grad, atol=atol, rtol=rtol), (
                f"Grouped gemm backward_dX forward outputs mismatch: {(X_.grad - ref_grad).abs().max().item():.6f}"
            )
        return
    else:
        dX_test = grouped_gemm_dX(
            dY=grad_output,
            W=W_test,
            gather_indices=gather_indices,
            m_sizes=expert_token_counts,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
            use_tma_load_w=use_tma_load_w,
            use_tma_load_dy=use_tma_load_dy,
            use_tma_store=use_tma_store,
            autotune=autotune,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=num_warps,
            num_stages=num_stages,
            flatten=flatten,
            # debug=True,
        )

    # if permute_x and use_W1 (first grouped GEMM) then the kernel should have unpermuted the dX
    # therefore we need to unpermute the ref_grad to compare to the output of the kernel
    if permute_x and use_W1:
        ref_grad = unpermute(ref_grad, gather_indices)

    assert ref_grad.shape == dX_test.shape, (
        f"Grouped gemm manual backward_dX outputs mismatch: ref_grad: {ref_grad.shape}, dX_test: {dX_test.shape}"
    )
    diff = (ref_grad - dX_test).abs().max().item()

    assert torch.allclose(ref_grad, dX_test, atol=atol, rtol=rtol), (
        f"Grouped gemm manual backward_dX outputs mismatch: {diff:.6f}"
    )

    if permute_x and use_W1:
        # Show that reduction results in diffs
        # First calculate X.grad manually by backpropping through unpermuted ref_grad
        dX_ref_check = ref_grad.view(num_tokens, topk, K).sum(dim=1)
        # Do the same for the actual output of the kernel
        dX_test_check = dX_test.view(num_tokens, topk, K).sum(dim=1)
        # Show diffs for each combination
        diff_ref_check = (X.grad - dX_ref_check).abs().max().item()
        diff_test_check = (X.grad - dX_test_check).abs().max().item()
        diff_check_test = (dX_ref_check - dX_test_check).abs().max().item()
        print(
            f"diff_ref_check: {diff_ref_check:.6f}, diff_test_check: {diff_test_check:.6f}, diff_check_test: {diff_check_test:.6f}"
        )


# NOTE: We reduce the size of the Llama4 model configs to prevent OOM
# Important to note that for the full model size (5120, 8192), the tests do result in diffs on the order of 1e-2.
@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_BWD_dX,
    ids=lambda x: x.to_string(include_tuning_params=True, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS[:1] + [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dX_manual(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfigBackward_dX,
    use_W1: bool,
):
    _test_grouped_gemm_backward_dX(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        use_autograd=False,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_BWD_dX,
    ids=lambda x: x.to_string(include_tuning_params=True, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS[:1] + [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dX_manual_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfigBackward_dX,
    use_W1: bool,
):
    _test_grouped_gemm_backward_dX(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        use_autograd=True,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [20], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dX_autotune(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    # TMA loads / stores will be autotuned
    _test_grouped_gemm_backward_dX(
        data_config=data_config,
        model_config=model_config,
        permute_x=permute_x,
        permute_y=permute_y,
        use_W1=use_W1,
        autotune=True,
        use_autograd=False,
        num_autotune_configs=num_autotune_configs,
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [20], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dX_autotune_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    # TMA loads / stores will be autotuned
    _test_grouped_gemm_backward_dX(
        data_config=data_config,
        model_config=model_config,
        permute_x=permute_x,
        permute_y=permute_y,
        use_W1=use_W1,
        autotune=True,
        use_autograd=True,
        num_autotune_configs=num_autotune_configs,
    )


def _test_grouped_gemm_backward_dW(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    use_tma_load_dy: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    BLOCK_SIZE_M: int = None,
    BLOCK_SIZE_N: int = None,
    BLOCK_SIZE_K: int = None,
    num_warps: int = None,
    num_stages: int = None,
    flatten: bool = True,
    autotune: bool = False,
    num_autotune_configs: int = None,
    allow_tma_store: bool = False,
    debug: bool = False,
    fuse_mul_post: bool = False,  # Unused for backward_dW
    use_autograd: bool = False,
):
    if not check_valid_config(
        permute_x,
        permute_y,
        fuse_mul_post=fuse_mul_post,
        use_W1=use_W1,
        is_backward=True,
    ):
        pytest.skip(
            f"Skipping test due to invalid config: {permute_x=} {permute_y=} {use_W1=}"
        )

    if use_tma_store and not allow_tma_store:
        pytest.skip("TMA store needs to be debugged due to non-deterministic behavior")

    X1, X2, W1, W2, gating_output = make_inputs(
        M=data_config.bs * data_config.seq_len,
        N=model_config.intermediate_size,
        K=model_config.hidden_size,
        E=model_config.num_experts,
        topk=model_config.topk,
        dtype=data_config.dtype,
        requires_grad=True,
    )
    topk = model_config.topk
    num_experts = model_config.num_experts
    use_sigmoid = model_config.use_sigmoid
    renormalize = model_config.renormalize

    X = X1 if use_W1 else X2
    num_tokens = data_config.bs * data_config.seq_len
    E, K, N = W2.shape  # E = num_experts, K = hidden_size, N = intermediate_size
    assert W1.shape == (E, 2 * N, K)
    W = W1 if use_W1 else W2

    if use_W1:
        assert X.shape == (num_tokens, K), (
            f"X.shape: {X.shape}, num_tokens: {num_tokens}, K: {K}"
        )
    else:
        assert X.shape == (num_tokens * topk, N), (
            f"X.shape: {X.shape}, num_tokens: {num_tokens}, topk: {topk}, N: {N}"
        )

    total_tokens = num_tokens * topk
    output_shape = (total_tokens, 2 * N) if use_W1 else (total_tokens, K)

    X_test = X.detach().clone().requires_grad_(True)
    W_test = W.detach().clone().requires_grad_(True)

    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, use_sigmoid=use_sigmoid, renormalize=renormalize
    )
    topk_weights = topk_weights.view(-1)  # num_tokens * topk
    topk_ids = topk_ids.view(-1)  # num_tokens * topk

    expert_token_counts, gather_indices = get_routing_indices(topk_ids, num_experts=E)
    assert len(gather_indices) == total_tokens
    assert len(expert_token_counts) == num_experts

    atol, rtol = TOLERANCE[X.dtype]
    Xperm = permute(X, gather_indices, topk)
    Xperm_test = Xperm.detach().clone().requires_grad_(True)

    # Need to retain grad otherwise grad is not propagated
    X.retain_grad()
    W.retain_grad()
    Xperm.retain_grad()
    assert Xperm.shape == (total_tokens, K) if use_W1 else (total_tokens, N)

    output_shape = (total_tokens, 2 * N) if use_W1 else (total_tokens, K)

    ref_output = torch_grouped_gemm(X=Xperm, W=W, m_sizes=expert_token_counts)
    assert ref_output.shape == output_shape

    # if permute_y then the assumption is that the output of grouped_gemm was unpermuted on store
    # Therefore we have to unpermute before backpropping to ensure proper alignment
    if permute_y:
        ref_output = unpermute(ref_output, gather_indices)

    grad_output = torch.randn_like(ref_output)
    ref_output.backward(grad_output)
    assert X.grad is not None
    assert W.grad is not None

    # Test backward kernel directly
    X_ = X_test if permute_x else Xperm_test

    if debug:
        torch.set_printoptions(precision=4)
        for i in range(num_experts):
            print(f"Expert {i} weight grad:\n{W.grad[i, :5, :5]}")

    if autotune:
        from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dW_kernel

        if num_autotune_configs is not None:
            _autotuned_grouped_gemm_dW_kernel.configs = (
                _autotuned_grouped_gemm_dW_kernel.configs[:num_autotune_configs]
            )

    if use_autograd:
        from grouped_gemm.interface import grouped_gemm

        if not autotune:
            kernel_config_fwd = KernelConfigForward(
                # Only care about backward_dW config
                use_tma_load_w=False,
                use_tma_load_x=False,
                use_tma_store=False,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            kernel_config_bwd_dW = KernelConfigBackward_dW(
                use_tma_load_dy=use_tma_load_dy,
                use_tma_load_x=use_tma_load_x,
                use_tma_store=use_tma_store,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            from grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dW_kernel
            from grouped_gemm.kernels.forward import (
                _autotuned_grouped_gemm_forward_kernel,
            )

            if num_autotune_configs is not None:
                _autotuned_grouped_gemm_forward_kernel.configs = (
                    _autotuned_grouped_gemm_forward_kernel.configs[
                        :num_autotune_configs
                    ]
                )
                _autotuned_grouped_gemm_dW_kernel.configs = (
                    _autotuned_grouped_gemm_dW_kernel.configs[:num_autotune_configs]
                )
            kernel_config_fwd = None
            kernel_config_bwd_dW = None

        test_output = grouped_gemm(
            X=X_,
            W=W_test,
            m_sizes=expert_token_counts,
            gather_indices=gather_indices,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            autotune=autotune,
            is_first_gemm=use_W1,
            dW_only=True,
        )
        assert test_output.shape == ref_output.shape, (
            f"Grouped gemm autograd backward_dW outputs mismatch: {test_output.shape} != {ref_output.shape}"
        )
        assert torch.allclose(test_output, ref_output, atol=atol, rtol=rtol), (
            f"Grouped gemm autograd backward_dW forward outputs mismatch: {test_output.shape} != {ref_output.shape}"
        )
        test_output.backward(grad_output)
        assert W_test.grad is not None
        dW_test = W_test.grad
    else:
        dW_test = grouped_gemm_dW(
            dY=grad_output,
            X=X_,
            m_sizes=expert_token_counts,
            gather_indices=gather_indices,
            topk=topk,
            permute_x=permute_x,
            permute_y=permute_y,
            use_tma_load_dy=use_tma_load_dy,
            use_tma_load_x=use_tma_load_x,
            use_tma_store=use_tma_store,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            num_warps=num_warps,
            num_stages=num_stages,
            flatten=flatten,
            autotune=autotune,
            debug=debug,
        )
    assert W.grad.shape == dW_test.shape, (
        f"Grouped gemm manual backward_dW outputs mismatch: W.grad: {W.grad.shape}, dW_test: {dW_test.shape}"
    )

    if debug:
        with torch.no_grad():
            if not torch.allclose(W.grad, dW_test, atol=atol, rtol=rtol):
                print(f"Ref Wgrad sum: {W.grad.sum().item():.4f}")
            print(f"Test Wgrad sum: {dW_test.sum().item():.4f}")

            for i in range(num_experts):
                print(f"Expert {i} weight grad:\n{W.grad[i, :5, :5]}")
                print(f"Expert {i} dW_test:\n{dW_test[i, :5, :5]}")
                expert_diff = (W.grad[i, :, :] - dW_test[i, :, :]).abs().max().item()
                print(f"Expert {i} diff: {expert_diff:.6f}")

            diff = (W.grad - dW_test).abs().max().item()
            assert False, (
                f"Grouped gemm manual backward_dW outputs mismatch: {diff:.6f}"
            )
    else:
        diff = (W.grad - dW_test).abs().max().item()
        assert torch.allclose(W.grad, dW_test, atol=atol, rtol=rtol), (
            f"Grouped gemm manual backward_dW outputs mismatch: {diff:.6f}"
        )


@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_BWD_dW,
    ids=lambda x: x.to_string(include_tuning_params=False, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS + [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dW_manual(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfig,
    use_W1: bool,
    debug: bool = False,
):
    _test_grouped_gemm_backward_dW(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        use_autograd=False,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "kernel_config",
    KERNEL_CONFIGS_BWD_dW,
    ids=lambda x: x.to_string(include_tuning_params=False, include_tma=True),
)
@pytest.mark.parametrize(
    "model_config",
    SMALL_MODEL_CONFIGS + [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dW_manual_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    kernel_config: KernelConfig,
    use_W1: bool,
    debug: bool = False,
):
    _test_grouped_gemm_backward_dW(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        use_autograd=True,
        **asdict(kernel_config),
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [20], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dW_autotune(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    _test_grouped_gemm_backward_dW(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=True,
        use_autograd=False,
        num_autotune_configs=num_autotune_configs,
    )


@pytest.mark.parametrize(
    "num_autotune_configs", [20], ids=lambda x: f"num_autotune_configs={x}"
)
@pytest.mark.parametrize(
    "permute_x", [True, False], ids=lambda x: "permute_x" if x else ""
)
@pytest.mark.parametrize(
    "permute_y", [True, False], ids=lambda x: "permute_y" if x else ""
)
@pytest.mark.parametrize(
    "model_config",
    [LLAMA_MODEL_CONFIG, QWEN_MODEL_CONFIG],
    ids=lambda x: f"topk={x.topk} num_experts={x.num_experts} hidden_size={x.hidden_size} intermediate_size={x.intermediate_size}",
)
@pytest.mark.parametrize(
    "data_config", DATA_CONFIGS, ids=lambda x: f"seq_len={x.seq_len} dtype={x.dtype}"
)
@pytest.mark.parametrize("use_W1", [True, False], ids=lambda x: f"use_W1={x}")
def test_grouped_gemm_backward_dW_autotune_autograd(
    data_config: DataConfig,
    model_config: ModelConfig,
    permute_x: bool,
    permute_y: bool,
    use_W1: bool,
    num_autotune_configs: int,
):
    _test_grouped_gemm_backward_dW(
        data_config=data_config,
        model_config=model_config,
        use_W1=use_W1,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=True,
        use_autograd=True,
        num_autotune_configs=num_autotune_configs,
    )
