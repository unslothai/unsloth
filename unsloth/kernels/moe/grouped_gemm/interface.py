from typing import Any, Optional, Callable
import logging
import warnings
from dataclasses import asdict

import torch
import triton

from grouped_gemm.kernels.backward import (
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dX_kernel,
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dX_kernel,
)
from grouped_gemm.kernels.forward import (
    _autotuned_grouped_gemm_forward_kernel,
    _grouped_gemm_forward_kernel,
)
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)

logger = logging.getLogger(__name__)
# Set formatter to include timestamp, pathname and lineno
formatter = logging.Formatter(
    "%(asctime)s::%(levelname)s,%(pathname)s:%(lineno)d:: %(message)s"
)

# Add console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

_FUSED_MUL_WARN = False
_SUPPORTS_TMA = None


def supports_tma() -> bool:
    """
    Checks if the current device supports Tensor Memory Acceleration (TMA).
    
    Returns:
        `bool`: True if TMA is supported (device capability >= 9), False otherwise.
    """
    global _SUPPORTS_TMA
    if _SUPPORTS_TMA is None:
        _SUPPORTS_TMA = torch.cuda.get_device_capability()[0] >= 9
    return _SUPPORTS_TMA


_per_device_alloc_fns = {}


def get_per_device_per_stream_alloc_fn(device: torch.device) -> Callable[[int, int, int], torch.Tensor]:
    """
    Creates a memory allocation function for a specific device that manages per-stream memory buffers.
    
    Args:
        device (`torch.device`): The device for which to create the allocator.
    
    Returns:
        `Callable[[int, int, int], torch.Tensor]`: A function that allocates memory buffers with specified size, alignment, and stream.
    """
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}

        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if (
                stream not in _per_stream_tensors
                or _per_stream_tensors[stream].numel() < size
            ):
                _per_stream_tensors[stream] = torch.empty(
                    size, device=device, dtype=torch.int8
                )
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]


def log_kernel_info(
    compiled_kernel: triton.compiler.CompiledKernel, best_config: triton.Config = None
) -> None:
    """
    Logs information about a compiled Triton kernel including register usage, spill counts, and metadata.
    
    Args:
        compiled_kernel (`triton.compiler.CompiledKernel`): The compiled Triton kernel to log information about.
        best_config (`triton.Config`, *optional*): The best configuration found during autotuning, if available.
    """
    kernel_name = compiled_kernel.name
    nregs = compiled_kernel.n_regs
    nspills = compiled_kernel.n_spills
    metadata = compiled_kernel.metadata
    logger.debug(
        f"{kernel_name}: n_regs={nregs} n_spills={nspills} metadata={metadata}"
    )
    if best_config is not None:
        logger.debug(f"{kernel_name} autotuned best_config: {best_config}")


def grouped_gemm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    topk: int,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor = None,
    topk_weights: torch.Tensor   = None,
    # Fusions
    permute_x: bool              = False,
    permute_y: bool              = False,
    fuse_mul_post: bool          = False,
    # Autotuning - manual kernel params will be ignored if autotune is True
    autotune: bool               = False,
    # Kernel tuning params if not autotuning -- NOTE: these params need to be tuned, otherwise performance will be poor
    BLOCK_SIZE_M: int            = 32,
    BLOCK_SIZE_N: int            = 32,
    BLOCK_SIZE_K: int            = 32,
    num_warps: int               = 4,
    num_stages: int              = 2,
    use_tma_load_w: bool         = False,
    use_tma_load_x: bool         = False,
    use_tma_store: bool          = False,
    # software pipelining -- set to True for now, won't impact until loop is re-written
    flatten: bool                = True,
    # debugging
    debug: bool                  = False,
) -> torch.Tensor:
    """
    Grouped GEMM forward pass for MoE MLPs.

    The implementation offers a number of fusions specific to MoE:
    - `permute_x`: fuse the permutation of hidden states from token order (original order) to grouped expert order, typically only needed for the first grouped GEMM in an MoE MLP.
        - When `permute_x` is True, `X` is expected to be of shape (num_tokens, K).
        - When `permute_x` is False, `X` is expected to be of shape (total_tokens, K) where `total_tokens = num_tokens * topk` AND already permuted to grouped expert order, i.e., hidden states are sorted such that tokens assigned to each expert are contiguous.
    - `permute_y`: fused the permutation of the output from expert grouped order back to original token order, typically only needed for the second grouped GEMM in an MoE MLP.
    - `fuse_mul_pre`: fuse the multiplication of the routed input with topk_weights, only done in the first grouped GEMM in an MoE MLP as for Llama4.  Do not use, since results in performance regression as it interrupts the GEMM mainloop.
    - `fuse_mul_post`: fuse the multiplication of the routed output with topk_weights, used only when `permute_y` is True. NOTE: this should only be used when using this kernel for inference, not for training.

    Args:
        X (`torch.Tensor`):
            (M, K) hidden states where M is the num_tokens if `permute_x` is True,
            otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
        W (`torch.Tensor`):
            (E, N, K) expert weights, where E is number of experts, N in the 
            intermediate (output) dim, and K is the reduction dim
        topk (`int`):
            Number of experts selected per token.
        m_sizes (`torch.Tensor`):
            tokens assigned to each expert which correspond to the size of M
            in the respective GEMMs in the grouped GEMM.
        gather_indices (`torch.Tensor`, *optional*):
            (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing
            `gather_indices` by cumsum of `m_sizes` gives the indices of tokens assigned
            to each expert.
        topk_weights (`torch.Tensor`, *optional*):
            (total_tokens,) weights to multiply routed output by in expert MLP calculation,
            used only when `fuse_mul_post` is True (see note on `fuse_mul_post`).
        permute_x (`bool`, *optional*):
            If True, permute input from token order to expert-grouped order.
            Only used for first grouped GEMM in MoE MLP. Defaults to False.
        permute_y (`bool`, *optional*):
            If True, permute output from expert-grouped order to token order.
            Only used for second grouped GEMM in MoE MLP. Defaults to False.
        fuse_mul_post (`bool`, *optional*):
            If True, fuse multiplication with routing weights after GEMM.
            Should only be used for inference, not training. Defaults to False.
        autotune (`bool`, *optional*):
            If True, use autotuning to find optimal kernel configuration.
            Manual kernel parameters will be ignored. Defaults to False.
        BLOCK_SIZE_M (`int`, *optional*):
            Block size for M dimension tiling. Defaults to 32.
        BLOCK_SIZE_N (`int`, *optional*):
            Block size for N dimension tiling. Defaults to 32.
        BLOCK_SIZE_K (`int`, *optional*):
            Block size for K dimension tiling. Defaults to 32.
        num_warps (`int`, *optional*):
            Number of warps per thread block. Defaults to 4.
        num_stages (`int`, *optional*):
            Number of pipeline stages. Defaults to 2.
        use_fast_accum (`bool`, *optional*):
            TODO: Currently unused; trade off faster accumulation dtype in GEMM
            for less precision.
        use_tma_load_w (`bool`, *optional*):
            If True, use TMA for loading weights. If TMA supported, this
            should always be enabled as it is faster than global memory load.
            Defaults to False.
        use_tma_load_x (`bool`, *optional*):
            If True, use TMA for loading activations, incompatible with `permute_x`.
            TODO: add TMA gather / scatter support for Blackwell+.
            Defaults to False.
        use_tma_store (`bool`, *optional*):
            If True, use TMA for storing output, incompatible with `permute_y`.
            TODO: add TMA scatter support for Blackwell+.
            Defaults to False.
        flatten (`bool`, *optional*):
            If True, flatten loops for better performance. Defaults to True.
        debug (`bool`, *optional*):
            If True, print debug information. Defaults to False.

    Returns:
        `torch.Tensor`: Output tensor with shape (total_tokens, N) containing the
            grouped GEMM result.

    Notes:
        - permute_x and permute_y are mutually exclusive
        - use_tma_load_x is incompatible with permute_x
        - use_tma_store is incompatible with permute_y
        - fuse_mul_post should only be used for inference
        - Manual kernel parameters need proper tuning for good performance
    """

    assert X.device.type == "cuda", "X and W must be on CUDA"
    assert m_sizes.device.type == "cuda", "m_sizes must be on CUDA"

    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (permute_y and use_tma_store), "Cannot use both TMA store and permute_y"

    if use_tma_load_x:
        # TMA load for activations, TMA gather only supported on Blackwell+
        assert not permute_x, "Cannot use both use_tma_load_x and permute_x"

    use_tma = use_tma_load_w or use_tma_load_x or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_x = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    X = X.view(-1, X.shape[-1])
    W = W.view(-1, W.shape[-1])

    if permute_x or permute_y:
        assert gather_indices is not None, (
            "gather_indices must be provided when permute_x or permute_y is True"
        )
        assert gather_indices.is_contiguous()
        assert gather_indices.device.type == "cuda"
        assert gather_indices.ndim == 1
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
        if permute_x:
            assert X.shape[0] == num_tokens, (
                f"X.shape[0] ({X.shape[0]}) must match num_tokens ({num_tokens})"
            )
        else:
            assert X.shape[0] == total_tokens, (
                f"X.shape[0] ({X.shape[0]}) must match total_tokens ({total_tokens})"
            )
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk

    num_experts = m_sizes.shape[0]
    _, K = X.shape
    N = W.shape[0] // num_experts
    assert K == W.shape[1], f"K ({K}) must match W.shape[1] ({W.shape[1]})"

    if fuse_mul_post:
        global _FUSED_MUL_WARN
        if not _FUSED_MUL_WARN:
            warnings.warn(
                "fused_mul should only be used for inference, not for training"
            )
            _FUSED_MUL_WARN = True
        assert permute_y, "FUSE_MUL requires PERMUTE_Y"
        assert topk_weights is not None
        assert topk_weights.numel() == total_tokens
        assert topk_weights.device.type == "cuda"
        assert topk_weights.is_contiguous()
        topk_weights = topk_weights.view(-1)
        if debug:
            print(
                f"DEBUG::GROUPED_GEMM {topk_weights.tolist()} {gather_indices.tolist()}"
            )

    y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)
    if total_tokens == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)
        BLOCK_SIZE_N = min(N, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(total_tokens, BLOCK_SIZE_M)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens=} {topk=} {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {permute_x=}"
        )
        print(
            f"DEBUG::GROUPED_GEMM {m_sizes.tolist()} {(gather_indices // topk).tolist()}"
        )

    kernel_args = {
        # Inputs
        "x_ptr": X,
        "w_ptr": W,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        "topk_weights_ptr": topk_weights,
        # Output
        "y_ptr": y,
        # Problem shapes
        "NUM_TOKENS": num_tokens,
        "NUM_EXPERTS": num_experts,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        # TopK weight merging
        "FUSE_MUL_POST": fuse_mul_post,
        # Loop pipelining
        "FLATTEN": flatten,
    }
    if not autotune:
        kernel_args.update(
            {
                "USE_TMA_LOAD_W": use_tma_load_w,
                "USE_TMA_LOAD_X": use_tma_load_x,
                "USE_TMA_STORE": use_tma_store,
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    kernel = (
        _autotuned_grouped_gemm_forward_kernel
        if autotune
        else _grouped_gemm_forward_kernel
    )
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)

    return y


def grouped_gemm_dX(
    dY: torch.Tensor,
    W: torch.Tensor,
    gather_indices: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int     = 32,
    BLOCK_SIZE_N: int     = 32,
    BLOCK_SIZE_K: int     = 32,
    debug: bool           = False,
    permute_x: bool       = False,
    permute_y: bool       = False,
    use_tma_load_w: bool  = False,
    use_tma_load_dy: bool = False,
    use_tma_store: bool   = False,
    num_warps: int        = 4,
    num_stages: int       = 2,
    flatten: bool         = True,
    fuse_mul_pre: bool    = False,
    fuse_mul_post: bool   = False,
    autotune: bool        = False,
) -> torch.Tensor:
    """
    Computes gradient with respect to input X in grouped GEMM backward pass.
    
    This function computes dX = dY @ W^T for the backward pass of grouped GEMM,
    handling multiple experts in parallel with support for permutation and TMA
    optimizations.
    
    For permute_x handling:
    - If X was permuted on load in forward pass, we permute on store in backward
    - Output shape is [NUM_TOKENS * TOPK, K] for gradient accumulation
    
    For permute_y handling:
    - If Y was permuted on store in forward pass, we permute dY on load in backward
    - Output dX is stored in contiguous order
    
    Args:
        dY (`torch.Tensor`):
            Gradient from next layer with shape (M, N).
        W (`torch.Tensor`):
            Expert weight matrices with shape (E, N, K).
        gather_indices (`torch.Tensor`):
            Indices of tokens assigned to each expert. Shape: (total_tokens,).
        m_sizes (`torch.Tensor`):
            Number of tokens assigned to each expert.
        topk (`int`):
            Number of experts selected per token.
        BLOCK_SIZE_M (`int`, *optional*):
            Block size for M dimension tiling. Defaults to 32.
        BLOCK_SIZE_N (`int`, *optional*):
            Block size for N dimension tiling. Defaults to 32.
        BLOCK_SIZE_K (`int`, *optional*):
            Block size for K dimension tiling. Defaults to 32.
        debug (`bool`, *optional*):
            If True, print debug information. Defaults to False.
        permute_x (`bool`, *optional*):
            Whether X was permuted in forward pass. Defaults to False.
        permute_y (`bool`, *optional*):
            Whether Y was permuted in forward pass. Defaults to False.
        use_tma_load_w (`bool`, *optional*):
            If True, use TMA for loading weights. Defaults to False.
        use_tma_load_dy (`bool`, *optional*):
            If True, use TMA for loading dY. Incompatible with permute_y.
            Defaults to False.
        use_tma_store (`bool`, *optional*):
            If True, use TMA for storing dX. Incompatible with permute_x.
            Defaults to False.
        num_warps (`int`, *optional*):
            Number of warps per thread block. Defaults to 4.
        num_stages (`int`, *optional*):
            Number of pipeline stages. Defaults to 2.
        flatten (`bool`, *optional*):
            If True, flatten loops for better performance. Defaults to True.
        fuse_mul_pre (`bool`, *optional*):
            Must be False for training. Defaults to False.
        fuse_mul_post (`bool`, *optional*):
            Must be False for training. Defaults to False.
        autotune (`bool`, *optional*):
            If True, use autotuning for kernel configuration. Defaults to False.
    
    Returns:
        `torch.Tensor`: Gradient with respect to input X. Shape: (total_tokens, K).
    
    Notes:
        - Output shape is [NUM_TOKENS * TOPK, K] requiring post-processing reduction
        - fuse_mul_pre and fuse_mul_post must be False for training
        - use_tma_load_dy is incompatible with permute_y
        - use_tma_store is incompatible with permute_x
    """
    assert not fuse_mul_pre, (
        "fuse_mul_pre should only be used for inference, not for training"
    )
    assert not fuse_mul_post, (
        "fuse_mul_post should only be used for inference, not for training"
    )
    assert dY.is_contiguous()
    assert W.is_contiguous()
    assert m_sizes.is_contiguous()
    assert m_sizes.ndim == 1

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    # Note that this is flipped from the forward pass
    # If we permuted y in the forward, we need to permute on load in the backward
    assert not (permute_y and use_tma_load_dy), "Cannot use both TMA load and permute_y"
    assert not (permute_x and use_tma_store), "Cannot use both TMA store and permute_x"

    use_tma = use_tma_load_dy or use_tma_load_w or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_dy = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            # print(f"DEBUG::GROUPED_GEMM alloc_fn {size=} {alignment=} {stream=}")
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    num_experts = m_sizes.shape[0]
    dY = dY.view(-1, dY.shape[-1])
    W = W.view(-1, W.shape[-1])

    M_total, N_grad = dY.shape
    N_total, K = W.shape
    N = N_total // num_experts
    assert N_grad == N, f"Grad_output N ({N_grad}) must match weight N ({N})"

    assert M_total % topk == 0, (
        f"M_total ({M_total}) must be divisible by topk ({topk})"
    )
    num_tokens = M_total // topk

    total_tokens = gather_indices.shape[0]
    assert total_tokens == M_total, (
        f"Total tokens ({total_tokens}) must match M_total ({M_total})"
    )

    # Note that the output shape is [NUM_TOKENS * TOPK, K] even when `permute_x` is True since we need to accumulate gradients across all experts chosen by the token.
    # This will be done in a post-processing step reduction step.
    output_shape = (total_tokens, K)
    dX = torch.zeros(output_shape, device=dY.device, dtype=dY.dtype)

    NUM_SMS = torch.cuda.get_device_properties(
        "cuda"
    ).multi_processor_count  # if not debug else 1

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_M = min(M_total, BLOCK_SIZE_M)
        BLOCK_SIZE_N = min(N_grad, BLOCK_SIZE_N)
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens=} {topk=} {output_shape=} {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {NUM_SMS=}"
        )
        print(f"DEBUG::GROUPED_GEMM {m_sizes.tolist()}")

    kernel_args = {
        # Inputs
        "dY_ptr": dY,
        "w_ptr": W,
        "gather_indices_ptr": gather_indices,
        "m_sizes_ptr": m_sizes,
        # Output
        "dX_ptr": dX,
        # Problem sizes
        "NUM_EXPERTS": num_experts,
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        "FLATTEN": flatten,
    }
    if not autotune:
        kernel_args.update(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "USE_TMA_LOAD_dY": use_tma_load_dy,
                "USE_TMA_LOAD_W": use_tma_load_w,
                "USE_TMA_STORE": use_tma_store,
            }
        )
    kernel = _autotuned_grouped_gemm_dX_kernel if autotune else _grouped_gemm_dX_kernel
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)
    return dX


def grouped_gemm_dW(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int     = 32,
    BLOCK_SIZE_N: int     = 32,
    BLOCK_SIZE_K: int     = 32,
    permute_x: bool       = False,
    permute_y: bool       = False,
    use_tma_load_dy: bool = False,
    use_tma_load_x: bool  = False,
    use_tma_store: bool   = False,
    fuse_mul_pre: bool    = False,
    fuse_mul_post: bool   = False,
    num_warps: int        = 4,
    num_stages: int       = 2,
    flatten: bool         = True,
    autotune: bool        = False,
    debug: bool           = False,
) -> torch.Tensor:
    """
    X: (M, K) hidden states where M is the num_tokens if `permute_x` is True, otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
    dY: (M, N)
    topk: number of experts to choose per token.
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    gather_indices: (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert.
    permute_x: whether X was permuted on load in the forward pass, typically only used for the first grouped GEMM in an MoE MLP to group tokens by expert.
    - for the first grouped GEMM, we permuted on load -> X was [num_tokens, K] and stored y in expert grouped order [num_tokens * topk, K]
    - in the backwards pass, we need to permute on load of X while loading dy in contiguous (expert grouped) order
    - since we are writing out dW, there is no need to permute on store
    permute_y: whether the output was permuted on store in the forward pass, typically only used for the second grouped GEMM in an MoE MLP to restore to the original token order.
    - for the second grouped GEMM, we permuted on store -> y was permuted from expert grouped order to token order while X was loaded in expert grouped order since it was the output of the first grouped GEMM
    - in the backwards pass, we need to permute on load of dy to get from token order to expert grouped order to match the order of X
    - since we are writing out dW, there is no need to permute on store
    use_tma_load_dy: use TMA for loading dy. use_tma_load_dy is incompatible with permute_y.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_y and use_tma_load_dy.
    use_tma_load_x: use TMA for loading x. use_tma_load_x is incompatible with permute_x.  TODO: add TMA gather / scatter support for Blackwell+ which will enable permute_x and use_tma_load_x.
    use_tma_store: use TMA for storing dW.  If TMA supported, this should always be enabled as it is faster than global memory store.
    """
    assert not fuse_mul_pre, "fuse_mul_pre not supported"
    assert not fuse_mul_post, "fuse_mul_post not supported"
    NUM_SMS = (
        torch.cuda.get_device_properties("cuda").multi_processor_count
        if not debug
        else 1
    )
    X = X.view(-1, X.shape[-1]).contiguous()
    dY = dY.contiguous()
    m_sizes = m_sizes.contiguous()

    # Preconditions
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (permute_y and use_tma_load_dy), "Cannot use both TMA load and permute_y"
    assert not (permute_x and use_tma_load_x), "Cannot use both TMA load and permute_x"

    use_tma = use_tma_load_dy or use_tma_load_x or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_x = False
        use_tma_load_dy = False
        use_tma_store = False

    if use_tma or autotune:

        def alloc_fn(size: int, alignment: int, stream: int):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    if permute_x or permute_y:
        assert gather_indices is not None
        assert gather_indices.is_contiguous()
        assert gather_indices.device.type == "cuda"
        assert gather_indices.ndim == 1
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
        if permute_x:
            assert X.shape[0] == num_tokens
        else:
            assert X.shape[0] == total_tokens
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk

    num_experts = m_sizes.shape[0]
    # Get dimensions
    _, K = X.shape
    M_grad, N = dY.shape

    assert M_grad == total_tokens, f"dY M ({M_grad}) != total_tokens ({total_tokens})"

    dW = torch.zeros((num_experts, N, K), device=X.device, dtype=X.dtype)

    if not autotune:
        BLOCK_SIZE_M = min(total_tokens, BLOCK_SIZE_M)
        BLOCK_SIZE_N = min(N, BLOCK_SIZE_N)
        BLOCK_SIZE_K = min(K, BLOCK_SIZE_K)

    def grid(META):
        return (NUM_SMS,)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM_DW_TMA {num_experts=} {N=} {K=} {BLOCK_SIZE_M=} {BLOCK_SIZE_N=} {BLOCK_SIZE_K=} {NUM_SMS=}"
        )

        print(f"DEBUG::GROUPED_GEMM_DW_TMA {m_sizes.tolist()=}")
        print(f"DEBUG::GROUPED_GEMM_DW_TMA {gather_indices.tolist()=}")
        m_start = 0
        for i in range(num_experts):
            expert_token_idx = gather_indices[m_start : m_start + m_sizes[i]]
            t_start = 0
            while t_start < m_sizes[i]:
                token_idx = expert_token_idx[t_start : t_start + BLOCK_SIZE_M]
                if permute_x:
                    token_idx = token_idx // topk
                print(
                    f"DEBUG::GROUPED_GEMM_DW_TMA Token expert {i} indices: {token_idx.tolist()}"
                )
                t_start += BLOCK_SIZE_M

            m_start += m_sizes[i]

    kernel_args = {
        # Inputs
        "x_ptr": X,
        "dY_ptr": dY,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        # Output
        "dW_ptr": dW,
        # Problem sizes
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "NUM_EXPERTS": num_experts,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        # Gather / Scatter
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        # Loop pipelining
        "FLATTEN": flatten,
    }

    if not autotune:
        kernel_args.update(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "USE_TMA_LOAD_dY": use_tma_load_dy,
                "USE_TMA_LOAD_X": use_tma_load_x,
                "USE_TMA_STORE": use_tma_store,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    kernel = _autotuned_grouped_gemm_dW_kernel if autotune else _grouped_gemm_dW_kernel
    compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

    if autotune:
        log_kernel_info(compiled_kernel, kernel.best_config)
    else:
        log_kernel_info(compiled_kernel)

    return dW


class GroupedGemm(torch.autograd.Function):
    """
    Implements a grouped GEMM operation with multiple optimizations for Mixture-of-Experts (MoE) architectures.
    
    The class provides:
    - Forward and backward passes for grouped matrix multiplication
    - Support for permutation operations to handle token-to-expert grouping
    - Memory optimizations through TMA (Tensor Memory Acceleration)
    - Autotuning capabilities for performance optimization
    
    Key Features:
    - `permute_x`: Fuses permutation of hidden states from token order to expert order
    - `permute_y`: Fuses permutation of outputs back to token order
    - TMA support for faster memory operations on compatible hardware
    - Autotuning for optimal kernel configuration
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
        m_sizes: torch.Tensor,
        topk: int,
        gather_indices: torch.Tensor,
        permute_x: bool,
        permute_y: bool,
        topk_weights: Optional[torch.Tensor],
        fuse_mul_post: bool,
        kernel_config_fwd: Optional[KernelConfigForward],
        kernel_config_bwd_dX: Optional[KernelConfigBackward_dX],
        kernel_config_bwd_dW: Optional[KernelConfigBackward_dW],
        autotune: bool,
        dX_only: bool,
        dW_only: bool,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the grouped GEMM operation.
        
        Args:
            ctx: Context object for saving state during forward pass.
            X (`torch.Tensor`): Input tensor of shape (M, K) containing hidden states.
            W (`torch.Tensor`): Weight tensor of shape (E, N, K) containing expert weights.
            m_sizes (`torch.Tensor`): Tensor indicating the number of tokens assigned to each expert.
            topk (`int`): Number of experts selected per token.
            gather_indices (`torch.Tensor`): Indices for gathering tokens assigned to each expert.
            permute_x (`bool`): Whether to permute input X from token order to expert order.
            permute_y (`bool`): Whether to permute output Y from expert order to token order.
            topk_weights (`Optional[torch.Tensor]`): Weights for top-k selection.
            fuse_mul_post (`bool`): Whether to fuse multiplication with topk weights in the forward pass.
            kernel_config_fwd (`Optional[KernelConfigForward]`): Configuration for forward kernel if not autotuning.
            kernel_config_bwd_dX (`Optional[KernelConfigBackward_dX]`): Configuration for backward pass of dX.
            kernel_config_bwd_dW (`Optional[KernelConfigBackward_dW]`): Configuration for backward pass of dW.
            autotune (`bool`): Whether to use autotuning for kernel configuration.
            dX_only (`bool`): If True, only compute dX in backward pass.
            dW_only (`bool`): If True, only compute dW in backward pass.
        
        Returns:
            `torch.Tensor`: Output tensor of shape (total_tokens, N) containing the result of the grouped GEMM operation.
        """
        ctx.topk = topk
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.fuse_mul_post = fuse_mul_post
        ctx.kernel_config_fwd = kernel_config_fwd
        ctx.kernel_config_bwd_dX = kernel_config_bwd_dX
        ctx.kernel_config_bwd_dW = kernel_config_bwd_dW
        ctx.autotune = autotune
        ctx.dX_only = dX_only
        ctx.dW_only = dW_only

        # NOTE: we don't save topk_weights for backward since we do not support training with fused_mul
        ctx.save_for_backward(X, W, m_sizes, gather_indices)

        fwd_config = {}
        if kernel_config_fwd is not None:
            fwd_config["BLOCK_SIZE_M"] = kernel_config_fwd.BLOCK_SIZE_M
            fwd_config["BLOCK_SIZE_N"] = kernel_config_fwd.BLOCK_SIZE_N
            fwd_config["BLOCK_SIZE_K"] = kernel_config_fwd.BLOCK_SIZE_K
            fwd_config["num_warps"] = kernel_config_fwd.num_warps
            fwd_config["num_stages"] = kernel_config_fwd.num_stages
            fwd_config["use_tma_load_x"] = kernel_config_fwd.use_tma_load_x
            fwd_config["use_tma_load_w"] = kernel_config_fwd.use_tma_load_w
            fwd_config["use_tma_store"] = kernel_config_fwd.use_tma_store

        return grouped_gemm_forward(
            X=X,
            W=W,
            topk=topk,
            m_sizes=m_sizes,
            gather_indices=gather_indices,
            topk_weights=topk_weights,
            permute_x=permute_x,
            permute_y=permute_y,
            fuse_mul_post=fuse_mul_post,
            # Autotune -- this will override the manual kernel config if true
            autotune=autotune,
            # Manual kernel config
            **fwd_config,
        )

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None, None, None, None, None, None, None, None, None]:
        """
        Performs the backward pass of the grouped GEMM operation to compute gradients.
        
        Args:
            ctx: Context object containing saved state from forward pass.
            dY (`torch.Tensor`): Gradient of the output tensor of shape (M, N).
        
        Returns:
            `tuple[Optional[torch.Tensor], ...]`: Gradients with respect to input tensors:
            - dX: Gradient of input X
            - dW: Gradient of weights W
            - None for non-differentiable inputs (m_sizes, gather_indices, etc.)
            - None for other non-differentiable parameters
        """
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        permute_x = ctx.permute_x
        permute_y = ctx.permute_y
        fuse_mul_post = ctx.fuse_mul_post
        kernel_config_bwd_dX = ctx.kernel_config_bwd_dX
        kernel_config_bwd_dW = ctx.kernel_config_bwd_dW
        autotune = ctx.autotune
        dX_only = ctx.dX_only
        dW_only = ctx.dW_only

        if not autotune:
            if not dW_only:
                assert kernel_config_bwd_dX is not None, (
                    "kernel_config_bwd_dX must be provided if autotune is False"
                )
            if not dX_only:
                assert kernel_config_bwd_dW is not None, (
                    "kernel_config_bwd_dW must be provided if autotune is False"
                )

        assert not fuse_mul_post, (
            "fused_mul should only be used for inference, not for training"
        )

        if not dX_only:
            bwd_dW_config = {}

            if kernel_config_bwd_dW is not None:
                bwd_dW_config["use_tma_load_dy"] = kernel_config_bwd_dW.use_tma_load_dy
                bwd_dW_config["use_tma_load_x"] = kernel_config_bwd_dW.use_tma_load_x
                bwd_dW_config["use_tma_store"] = kernel_config_bwd_dW.use_tma_store
                bwd_dW_config["BLOCK_SIZE_M"] = kernel_config_bwd_dW.BLOCK_SIZE_M
                bwd_dW_config["BLOCK_SIZE_N"] = kernel_config_bwd_dW.BLOCK_SIZE_N
                bwd_dW_config["BLOCK_SIZE_K"] = kernel_config_bwd_dW.BLOCK_SIZE_K
                bwd_dW_config["num_warps"] = kernel_config_bwd_dW.num_warps
                bwd_dW_config["num_stages"] = kernel_config_bwd_dW.num_stages

            dW = grouped_gemm_dW(
                X=X,
                dY=dY,
                m_sizes=m_sizes,
                gather_indices=gather_indices,
                topk=topk,
                permute_x=permute_x,
                permute_y=permute_y,
                # Autotune -- this will override the manual kernel config if true
                autotune=autotune,
                # Manual kernel config
                **bwd_dW_config,
            )
        else:
            dW = None

        if not dW_only:
            bwd_dX_config = {}
            if kernel_config_bwd_dX is not None:
                bwd_dX_config["use_tma_load_dy"] = kernel_config_bwd_dX.use_tma_load_dy
                bwd_dX_config["use_tma_load_w"] = kernel_config_bwd_dX.use_tma_load_w
                bwd_dX_config["use_tma_store"] = kernel_config_bwd_dX.use_tma_store
                bwd_dX_config["BLOCK_SIZE_M"] = kernel_config_bwd_dX.BLOCK_SIZE_M
                bwd_dX_config["BLOCK_SIZE_N"] = kernel_config_bwd_dX.BLOCK_SIZE_N
                bwd_dX_config["BLOCK_SIZE_K"] = kernel_config_bwd_dX.BLOCK_SIZE_K
                bwd_dX_config["num_warps"] = kernel_config_bwd_dX.num_warps
                bwd_dX_config["num_stages"] = kernel_config_bwd_dX.num_stages

            dX = grouped_gemm_dX(
                dY=dY,
                W=W,
                m_sizes=m_sizes,
                gather_indices=gather_indices,
                topk=topk,
                permute_x=permute_x,
                permute_y=permute_y,
                # Autotune -- this will override the manual kernel config if true
                autotune=autotune,
                # Manual kernel config
                **bwd_dX_config,
            )

            if topk > 1 and permute_x:
                dX = dX.view(X.shape[0], topk, -1).sum(dim=1)
        else:
            dX = None

        return (
            dX,
            dW,
            None,  # m_sizes
            None,  # gather_indices
            None,  # topk
            None,  # permute_x
            None,  # permute_y
            None,  # topk_weights
            None,  # fuse_mul_post
            None,  # kernel_config_fwd
            None,  # kernel_config_bwd_dX
            None,  # kernel_config_bwd_dW
            None,  # autotune
            None,  # dX_only
            None,  # dW_only
        )


def check_valid_config_fwd(
    permute_x: bool,
    permute_y: bool,
    use_tma_load_x: bool,
    use_tma_load_w: bool,
    use_tma_store: bool,
    fuse_mul_post: bool,
    is_first_gemm: bool,
) -> None:
    """
    Check if the configuration is valid for the forward pass.
    """
    is_second_gemm = not is_first_gemm

    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (is_second_gemm and permute_x), (
        "Cannot permute X for the second grouped GEMM"
    )
    assert not (is_first_gemm and permute_y), (
        "Cannot permute Y for the first grouped GEMM"
    )
    assert not (fuse_mul_post and is_first_gemm), (
        "Cannot fuse mul for the first grouped GEMM"
    )
    assert not (use_tma_load_x and permute_x), (
        "Cannot use TMA load and permute X unless on sm100+ (Blackwell+)"
    )
    assert not (use_tma_store and permute_y and is_second_gemm), (
        "Cannot use TMA store and permute Y for the second grouped GEMM unless on sm100+ (Blackwell+)"
    )


def check_valid_config_bwd_dW(
    permute_x: bool,
    permute_y: bool,
    use_tma_load_dY: bool,
    use_tma_load_x: bool,
    use_tma_store: bool,
    fuse_mul_post: bool,
    is_first_gemm: bool,
) -> None:
    """
    Check if the configuration is valid for the backward pass of dW.
    """
    is_second_gemm = not is_first_gemm
    if fuse_mul_post:
        assert False, "Cannot fuse_mul is not supported for backward pass"
    if is_second_gemm and permute_y and use_tma_load_dY:
        assert False, "Cannot use TMA load and permute Y for the second grouped GEMM"
    if is_first_gemm and permute_x and use_tma_load_x:
        assert False, "Cannot use TMA load and permute X for the first grouped GEMM"


def check_valid_config_bwd_dX(
    permute_x: bool,
    permute_y: bool,
    use_tma_load_dY: bool,
    use_tma_load_w: bool,
    use_tma_store: bool,
    fuse_mul_post: bool,
    is_first_gemm: bool,
) -> None:
    """
    Check if the configuration is valid for the backward pass of dW.
    """
    is_second_gemm = not is_first_gemm
    if fuse_mul_post:
        assert False, "Cannot fuse_mul is not supported for backward pass"
    if is_second_gemm and permute_y and use_tma_load_dY:
        assert False, "Cannot use TMA load and permute Y for the second grouped GEMM"
    if use_tma_store and permute_x and is_first_gemm:
        assert False, "Cannot use TMA store and permute X for the first grouped GEMM"


def grouped_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor                  = None,
    permute_x: bool                               = False,
    permute_y: bool                               = False,
    topk_weights: Optional[torch.Tensor]          = None,
    fuse_mul_post: bool                           = False,
    kernel_config_fwd: KernelConfigForward        = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    autotune: bool                                = False,
    is_first_gemm: bool                           = True,
    # Only for debugging
    dX_only: bool                                 = False,
    dW_only: bool                                 = False,
) -> torch.Tensor:
    """
    Grouped GEMM for MoE MLPs.

    The implementation offers a number of fusions specific to MoE:
    - `permute_x`: fuse the permutation of hidden states from token order (original order) to grouped expert order, typically only needed for the first grouped GEMM in an MoE MLP.
        - When `permute_x` is True, `X` is expected to be of shape (num_tokens, K).
        - When `permute_x` is False, `X` is expected to be of shape (total_tokens, K) where `total_tokens = num_tokens * topk` AND already permuted to grouped expert order, i.e., hidden states are sorted such that tokens assigned to each expert are contiguous.
    - `permute_y`: fused the permutation of the output from expert grouped order back to original token order, typically only needed for the second grouped GEMM in an MoE MLP.
    - `fuse_mul`: fuse the multiplication of the routed output with topk_weights, used only when `permute_y` is True. NOTE: this should only be used when using this kernel for inference, not for training.

    X: (M, K) hidden states where M is the num_tokens if `permute_x` is True, otherwise `total_tokens` where `total_tokens = num_tokens * topk`.
    W: (E, N, K) expert weights, where E is number of experts, N in the intermediate (output) dim, and K is the reduction dim
    m_sizes: tokens assigned to each expert which correspond to the size of M in the respective GEMMs in the grouped GEMM.
    gather_indices: (total_tokens,) indices of tokens assigned to each expert.  E.g., slicing gather_indices by cumsum of m_sizes gives the indices of tokens assigned to each expert. Needed when either `permute_x` or `permute_y` is True.
    topk_weights: (total_tokens,) weights to multiply routed output by in expert MLP calculation, used only when `fuse_mul` is True (see note on `fuse_mul`).
    kernel_config_fwd: KernelConfigForward for forward pass.
    kernel_config_bwd_dX: KernelConfigBackward_dX for backward pass of dX.
    kernel_config_bwd_dW: KernelConfigBackward_dW for backward pass of dW.
    autotune: whether to autotune the kernel, if yes, kernel_config_fwd, kernel_config_bwd_dX, and kernel_config_bwd_dW will be ignored.
    is_first_gemm: whether this is the first grouped GEMM in an MoE MLP.  This is needed to check whether kernel configs are valid.  `permute_x` should only be used for first gemm; `permute_y` should only be used for second gemm.
    This will impact whether TMA can be used for loading and storing.

    """
    if not autotune:
        assert kernel_config_fwd is not None, (
            "kernel_config_fwd must be provided if autotune is False"
        )

        check_valid_config_fwd(
            permute_x,
            permute_y,
            use_tma_load_x=kernel_config_fwd.use_tma_load_x,
            use_tma_load_w=kernel_config_fwd.use_tma_load_w,
            use_tma_store=kernel_config_fwd.use_tma_store,
            fuse_mul_post=fuse_mul_post,
            is_first_gemm=is_first_gemm,
        )
        if kernel_config_bwd_dW is not None and not dX_only:
            check_valid_config_bwd_dW(
                permute_x,
                permute_y,
                use_tma_load_dY=kernel_config_bwd_dW.use_tma_load_dy,
                use_tma_load_x=kernel_config_bwd_dW.use_tma_load_x,
                use_tma_store=kernel_config_bwd_dW.use_tma_store,
                fuse_mul_post=fuse_mul_post,
                is_first_gemm=is_first_gemm,
            )
        if kernel_config_bwd_dX is not None and not dW_only:
            check_valid_config_bwd_dX(
                permute_x,
                permute_y,
                use_tma_load_dY=kernel_config_bwd_dX.use_tma_load_dy,
                use_tma_load_w=kernel_config_bwd_dX.use_tma_load_w,
                use_tma_store=kernel_config_bwd_dX.use_tma_store,
                fuse_mul_post=fuse_mul_post,
                is_first_gemm=is_first_gemm,
            )

    if permute_x or permute_y:
        assert gather_indices is not None, (
            "gather_indices is required when either permute_x or permute_y is True"
        )

    if fuse_mul_post:
        assert topk_weights is not None, (
            "topk_weights is required when fuse_mul_post is True"
        )

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    gather_indices = gather_indices.view(-1)

    return GroupedGemm.apply(
        X,
        W,
        m_sizes,
        topk,
        gather_indices,
        permute_x,
        permute_y,
        topk_weights,
        fuse_mul_post,
        kernel_config_fwd,
        kernel_config_bwd_dX,
        kernel_config_bwd_dW,
        autotune,
        dX_only,
        dW_only,
    )