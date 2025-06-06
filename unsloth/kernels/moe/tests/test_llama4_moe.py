from typing import Optional
import argparse
import sys
from contextlib import contextmanager
from functools import partial

import pytest
import torch
from transformers import AutoConfig
from transformers.models.llama4 import Llama4Config, Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)
from grouped_gemm.reference.layers.llama4_moe import (
    Llama4GroupedGemmTextMoe,
    Llama4TritonTextMoe,
)

TOLERANCES = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float: (1e-5, 1e-5),
}

LLAMA4_SCOUT_ID = "meta-llama/Llama-4-Scout-17B-16E"
SEED = 42
SEQ_LENS = [1024]
DTYPES = [torch.bfloat16]
# Reduce the number of autotuning configs to prevent excessive runtime
NUM_AUTOTUNE_CONFIGS = 50


@contextmanager
def annotated_context(prelude: str, epilogue: str="Passed!", char: str="-", num_chars: int=80) -> None:
    """
    Context manager that prints a prelude and epilogue around a code block.
    
    Args:
        prelude (`str`): Text to print before entering the context.
        epilogue (`str`, optional): Text to print after exiting the context. Defaults to "Passed!".
        char (`str`, optional): Character to use for the border. Defaults to "-".
        num_chars (`int`, optional): Number of characters to use for the border. Defaults to 80.
    
    Yields:
        None
    """
    print(char * num_chars)
    print(prelude)
    yield
    print(epilogue)
    print(char * num_chars)


def get_text_config(model_id: str) -> Llama4TextConfig:
    """
    Retrieves the text configuration from a Llama4 model.
    
    Args:
        model_id (`str`): Identifier of the Llama4 model.
    
    Returns:
        `Llama4TextConfig`: The text configuration of the specified Llama4 model.
    """
    config: Llama4Config = AutoConfig.from_pretrained(model_id)
    return config.text_config


def prep_triton_kernel_traits(autotune: bool) -> tuple[Optional[KernelConfigForward], Optional[KernelConfigBackward_dW], Optional[KernelConfigBackward_dX]]:
    """
    Prepares the kernel configurations for Triton-based operations.
    
    Args:
        autotune (`bool`): Whether to use autotuning for the kernels.
    
    Returns:
        tuple[Optional[KernelConfigForward], Optional[KernelConfigBackward_dW], Optional[KernelConfigBackward_dX]]: A tuple containing the kernel configurations for forward, backward dW, and backward dX operations.
    """
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

    return kernel_config_fwd, kernel_config_bwd_dW, kernel_config_bwd_dX


def sparse_to_dense(t: torch.Tensor) -> None:
    """
    Converts a sparse tensor to a dense tensor by summing over the first dimension and flattening the result.
    
    Args:
        t (`torch.Tensor`): The sparse tensor to convert.
    
    Returns:
        `torch.Tensor`: The converted dense tensor.
    """
    t = t.sum(dim=0).view(-1)
    return t


@torch.no_grad()
def _check_diff(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float,
    rtol: float,
    precision: str = ".6f",
    verbose: bool  = False,
    msg: str       = "",
) -> None:
    """
    Checks the difference between two tensors and asserts if they are within the specified tolerances.
    
    Args:
        t1 (`torch.Tensor`): First tensor for comparison.
        t2 (`torch.Tensor`): Second tensor for comparison.
        atol (`float`): Absolute tolerance.
        rtol (`float`): Relative tolerance.
        precision (`str`, optional): String format for printing precision. Defaults to ".6f".
        verbose (`bool`, optional): Whether to print detailed output. Defaults to False.
        msg (`str`, optional): Message to print with the result. Defaults to "".
    """
    t2 = t2.view_as(t1)
    diff = t1.sub(t2).abs().max().item()
    if verbose:
        if msg == "":
            msg = "diff"
        print(f"{msg}: {diff:{precision}}")
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol)


def run_backwards(y: torch.Tensor, grad_output: torch.Tensor, module: torch.nn.Module) -> None:
    """
    Performs a backward pass on the given output tensor with the provided gradient output.
    
    Args:
        y (`torch.Tensor`): Output tensor from a forward pass.
        grad_output (`torch.Tensor`): Gradient to backpropagate.
        module (`torch.nn.Module`): The module used in the forward pass.
    """
    y.backward(grad_output)
    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} missing grad!"


def _check_grads(
    m1: torch.nn.Module,
    m2: torch.nn.Module,
    atol: float,
    rtol: float,
    precision: str = ".6f",
    verbose: bool  = False,
    msg: str       = "",
) -> None:
    """
    Checks the gradients of two modules to ensure they are within the specified tolerances.
    
    Args:
        m1 (`torch.nn.Module`): First module with gradients to compare.
        m2 (`torch.nn.Module`): Second module with gradients to compare.
        atol (`float`): Absolute tolerance.
        rtol (`float`): Relative tolerance.
        precision (`str`, optional): String format for printing precision. Defaults to ".6f".
        verbose (`bool`, optional): Whether to print detailed output. Defaults to False.
        msg (`str`, optional): Message to print with the result. Defaults to "".
    """
    for name, param in m1.named_parameters():
        _check_diff(
            param.grad,
            m2.get_parameter(name).grad,
            atol=atol,
            rtol=rtol,
            precision=precision,
            verbose=verbose,
            msg=f"{msg}:{name}.grad",
        )


@pytest.fixture
def model_config() -> None:
    """
    Retrieves the model configuration for testing purposes.
    
    Args:
        None
    
    Returns:
        None
    """
    return AutoConfig.from_pretrained(LLAMA4_SCOUT_ID).text_config


@pytest.mark.parametrize(
    "overlap_router_shared",
    [False, True],
    ids=lambda x: "overlap_router_shared" if x else "no_overlap",
)
@pytest.mark.parametrize(
    "permute_y", [False, True], ids=lambda x: "permute_y" if x else "no_permute_y"
)
@pytest.mark.parametrize(
    "permute_x", [False], ids=lambda x: "permute_x" if x else "no_permute_x"
)  # Llama4 does not support permute_x
@pytest.mark.parametrize(
    "autotune", [True], ids=lambda x: "autotune" if x else "manual"
)
@pytest.mark.parametrize("seqlen", SEQ_LENS, ids=lambda x: f"seqlen={x}")
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_llama4_ref(
    dtype: torch.dtype,
    seqlen: int,
    autotune: bool,
    permute_x: bool,
    permute_y: bool,
    overlap_router_shared: bool,
    model_config: Llama4TextConfig,  # test fixture
    bs: int        = 1,
    device: str    = "cuda",
    precision: str = ".6f",
    verbose: bool  = False,
) -> None:
    """
    Test function to validate the correctness of the Llama4 reference model against different implementations (Torch grouped GEMM and Triton).
    
    Args:
        dtype (`torch.dtype`): Data type for the test.
        seqlen (`int`): Sequence length for the input data.
        autotune (`bool`): Whether to use autotuning for the kernels.
        permute_x (`bool`): Whether to permute the input tensor x.
        permute_y (`bool`): Whether to permute the output tensor y.
        overlap_router_shared (`bool`): Whether the router shares weights with the experts.
        model_config (`Llama4TextConfig`): Configuration for the Llama4 model.
        bs (`int`, optional): Batch size. Defaults to 1.
        device (`str`, optional): Device to run the test on. Defaults to "cuda".
        precision (`str`, optional): String format for printing precision. Defaults to ".6f".
        verbose (`bool`, optional): Whether to print detailed output. Defaults to False.
    
    Returns:
        None
    """
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_dim = model_config.hidden_size
    atol, rtol = TOLERANCES[dtype]
    check_diff = partial(
        _check_diff, atol=atol, rtol=rtol, precision=precision, verbose=verbose
    )
    check_grads = partial(
        _check_grads, atol=atol, rtol=rtol, precision=precision, verbose=verbose
    )

    # Reference op -- HF
    llama4_ref = Llama4TextMoe(model_config).to(dtype=dtype, device=device)

    # Torch grouped gemm impl
    llama4_gg_ref = Llama4GroupedGemmTextMoe(
        model_config, overlap_router_shared=overlap_router_shared
    ).to(dtype=dtype, device=device)
    llama4_gg_ref.copy_weights(llama4_ref)
    llama4_gg_ref.check_weights(llama4_ref)

    x_ref = torch.randn(
        bs, seqlen, hidden_dim, dtype=dtype, device=device, requires_grad=True
    )
    x_torch_gg = x_ref.detach().clone().requires_grad_()
    x_triton = x_ref.detach().clone().requires_grad_()

    y_ref, routing_ref = llama4_ref(x_ref)
    y_torch_gg, routing_torch_gg = llama4_gg_ref(x_torch_gg)
    assert y_ref.shape == y_torch_gg.shape, f"{y_ref.shape} != {y_torch_gg.shape}"
    with annotated_context("Testing torch grouped gemm Llama4TextMoe"):
        check_diff(y_ref, y_torch_gg, msg="y_torch_gg")
        check_diff(
            sparse_to_dense(routing_ref), routing_torch_gg, msg="routing_torch_gg"
        )

    kernel_config_fwd, kernel_config_bwd_dW, kernel_config_bwd_dX = (
        prep_triton_kernel_traits(autotune)
    )

    llama4_triton = Llama4TritonTextMoe(
        model_config,
        overlap_router_shared=overlap_router_shared,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
    ).to(device=device, dtype=dtype)
    llama4_triton.copy_weights(llama4_ref)
    llama4_triton.check_weights(llama4_ref)

    y_triton, routing_triton = llama4_triton(x_triton)
    with annotated_context("Testing triton grouped gemm Llama4TextMoe forward"):
        check_diff(y_ref, y_triton, msg="y_triton")
        check_diff(sparse_to_dense(routing_ref), routing_triton, msg="routing_triton")

    ref_grad = torch.randn_like(y_ref)
    run_backwards(y_ref, ref_grad, llama4_ref)
    run_backwards(y_torch_gg, ref_grad, llama4_gg_ref)
    with annotated_context("Testing torch group gemm Llama4TextMoe backward"):
        check_grads(llama4_ref, llama4_gg_ref, msg="torch_gg")

    run_backwards(y_triton, ref_grad, llama4_triton)
    with annotated_context("Testing triton group gemm Llama4TextMoe backward"):
        check_grads(llama4_ref, llama4_triton, msg="triton")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16"
    )
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args_dict = vars(args)

    model_id = LLAMA4_SCOUT_ID

    text_config: Llama4TextConfig = get_text_config(model_id)
    for overlap in [False, True]:
        test_llama4_ref(
            seqlen=args.seqlen,
            model_config=text_config,
            dtype=args.dtype,
            autotune=True,
            permute_x=False,
            permute_y=True,
            overlap_router_shared=overlap,
            verbose=True,
        )
