# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import logging
import warnings
from dataclasses import asdict
from unsloth import DEVICE_TYPE

import torch
import triton

from .kernels.backward import (
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dX_kernel,
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dX_kernel,
)
from .kernels.forward import (
    _autotuned_grouped_gemm_forward_kernel,
    _grouped_gemm_forward_kernel,
)
from .kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s::%(levelname)s,%(pathname)s:%(lineno)d:: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


# Precompute TMA support (NVIDIA capability >= 9 plus a triton carrying make_tensor_descriptor) to avoid graph breaks.
def _check_tma_support():
    if DEVICE_TYPE in ("xpu", "hip"):
        return False
    import triton.language as tl

    gpu_supports_tma = torch.cuda.get_device_capability()[0] >= 9
    triton_has_tma_api = hasattr(tl, "make_tensor_descriptor") or hasattr(
        tl, "_experimental_make_tensor_descriptor"
    )
    return gpu_supports_tma and triton_has_tma_api


_SUPPORTS_TMA = _check_tma_support()

# triton.set_allocator is Triton 3.0+.
_HAS_SET_ALLOCATOR = hasattr(triton, "set_allocator")


def supports_tma():
    return _SUPPORTS_TMA


try:
    from torch.compiler import allow_in_graph
except ImportError:
    from torch._dynamo import allow_in_graph


def _is_tracing(*tensors):
    """True if tensors are fake tensors used during torch.compile tracing (Triton cannot run). Not torch.compiler.is_compiling(): that is True during both tracing AND execution, and only tracing must skip the kernels."""
    for t in tensors:
        name = type(t).__name__
        if name in ("FakeTensor", "FunctionalTensor", "FunctionalTensorWrapper"):
            return True
    return False


_per_device_alloc_fns = {}


def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}

        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if stream not in _per_stream_tensors or _per_stream_tensors[stream].numel() < size:
                _per_stream_tensors[stream] = torch.empty(size, device = device, dtype = torch.int8)
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]


def log_kernel_info(
    compiled_kernel: triton.compiler.CompiledKernel, best_config: triton.Config = None
):
    kernel_name = compiled_kernel.name
    nregs = compiled_kernel.n_regs
    nspills = compiled_kernel.n_spills
    metadata = compiled_kernel.metadata
    logger.debug(f"{kernel_name}: n_regs={nregs} n_spills={nspills} metadata={metadata}")
    if best_config is not None:
        logger.debug(f"{kernel_name} autotuned best_config: {best_config}")


@allow_in_graph
def grouped_gemm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    topk: int,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    fuse_mul_post: bool = False,
    # Autotuning -- overrides manual kernel params when True
    autotune: bool = False,
    # Kernel tuning params: must be tuned, else poor performance.
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    # Software pipelining; no effect until the loop is re-written.
    flatten: bool = True,
    debug: bool = False,
) -> torch.Tensor:
    """Grouped GEMM forward pass for MoE MLPs.

    X is (num_tokens, K) when permute_x, else (num_tokens * topk, K) already sorted into expert-grouped order; W is (E, N, K); m_sizes is the token count per expert; gather_indices is (total_tokens,) token indices per expert; topk_weights is (total_tokens,); returns y (total_tokens, N).

    permute_x fuses the token-order to expert-order permutation (first GEMM of an MoE MLP); permute_y fuses the reverse (second GEMM).
    fuse_mul_pre is a performance regression since it interrupts the GEMM mainloop: do not use it.
    fuse_mul_post requires permute_y and is inference-only, never training.
    use_tma_load_x is incompatible with permute_x and use_tma_store with permute_y (no TMA gather / scatter before Blackwell+); use_tma_load_w should always be on where TMA is supported, being faster than a global memory load.
    use_fast_accum is currently unused.
    """

    assert X.device.type == "cuda", "X and W must be on CUDA"
    assert m_sizes.device.type == "cuda", "m_sizes must be on CUDA"

    X = X.contiguous()
    W = W.contiguous()
    m_sizes = m_sizes.contiguous()

    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (permute_y and use_tma_store), "Cannot use both TMA store and permute_y"

    if use_tma_load_x:
        # TMA load for activations; TMA gather is Blackwell+ only.
        assert not permute_x, "Cannot use both use_tma_load_x and permute_x"

    use_tma = use_tma_load_w or use_tma_load_x or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_x = False
        use_tma_store = False

    if use_tma or autotune:
        if _HAS_SET_ALLOCATOR and not getattr(triton, "_unsloth_allocator_set", False):

            def alloc_fn(size: int, alignment: int, stream: int):
                return torch.empty(size, device = "cuda", dtype = torch.int8)

            triton.set_allocator(alloc_fn)

    if W.ndim == 3:
        num_experts = W.shape[0]
        N = W.shape[1]
    else:
        num_experts = m_sizes.shape[0]
        N = W.shape[0] // num_experts

    X = X.view(-1, X.shape[-1])
    W = W.view(-1, W.shape[-1])

    if permute_x or permute_y:
        assert (
            gather_indices is not None
        ), "gather_indices must be provided when permute_x or permute_y is True"
        assert gather_indices.is_contiguous()
        assert gather_indices.device.type == "cuda"
        assert gather_indices.ndim == 1
        total_tokens = gather_indices.shape[0]
        num_tokens = total_tokens // topk
        if permute_x:
            assert (
                X.shape[0] == num_tokens
            ), f"X.shape[0] ({X.shape[0]}) must match num_tokens ({num_tokens})"
        else:
            assert (
                X.shape[0] == total_tokens
            ), f"X.shape[0] ({X.shape[0]}) must match total_tokens ({total_tokens})"
    else:
        total_tokens = X.shape[0]
        num_tokens = total_tokens // topk

    _, K = X.shape
    assert K == W.shape[1], f"K ({K}) must match W.shape[1] ({W.shape[1]})"

    if fuse_mul_post:
        global _FUSED_MUL_WARN
        if not _FUSED_MUL_WARN:
            warnings.warn("fused_mul should only be used for inference, not for training")
            _FUSED_MUL_WARN = True
        assert permute_y, "FUSE_MUL requires PERMUTE_Y"
        assert topk_weights is not None
        assert topk_weights.numel() == total_tokens
        assert topk_weights.device.type == "cuda"
        assert topk_weights.is_contiguous()
        topk_weights = topk_weights.view(-1)
        if debug:
            print(f"DEBUG::GROUPED_GEMM {topk_weights.tolist()} {gather_indices.tolist()}")

    y = torch.empty((total_tokens, N), device = X.device, dtype = X.dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        pass

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens = } {topk = } {num_experts = } {N = } {K = } {BLOCK_SIZE_M = } {BLOCK_SIZE_N = } {BLOCK_SIZE_K = } {permute_x = }"
        )
        print(f"DEBUG::GROUPED_GEMM {m_sizes.tolist()} {(gather_indices // topk).tolist()}")

    kernel_args = {
        "x_ptr": X,
        "w_ptr": W,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        "topk_weights_ptr": topk_weights,
        "y_ptr": y,
        "NUM_TOKENS": num_tokens,
        "NUM_EXPERTS": num_experts,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
        "PERMUTE_X": permute_x,
        "PERMUTE_Y": permute_y,
        "FUSE_MUL_POST": fuse_mul_post,
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

    kernel = _autotuned_grouped_gemm_forward_kernel if autotune else _grouped_gemm_forward_kernel

    is_fake = _is_tracing(X, W)
    if not is_fake:
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)
        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)

    return y


@allow_in_graph
def grouped_gemm_dX(
    dY: torch.Tensor,
    W: torch.Tensor,
    gather_indices: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    debug: bool = False,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_w: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_store: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    flatten: bool = True,
    fuse_mul_pre: bool = False,
    fuse_mul_post: bool = False,
    autotune: bool = False,
) -> torch.Tensor:
    """Backward dX kernel. grad_output is (M, N); m_sizes is the token count per expert; gather_indices (total_tokens,) holds the token index per expert slot and may be None unless permute_x or permute_y.

    permute_x and permute_y describe what the forward pass did, and the backward is its mirror: a forward permute on load becomes a permute on store, a forward permute on store becomes a permute on load. dX stays [NUM_TOKENS * TOPK, K] because the gradients of every expert a token chose are accumulated in a later step.
    fuse_mul_pre and fuse_mul_post must stay False here, being inference-only.
    use_tma_load_dy is incompatible with permute_y and use_tma_store with permute_x (no TMA gather / scatter before Blackwell+); use_tma_load_w should always be on where TMA is supported.
    """
    assert not fuse_mul_pre, "fuse_mul_pre should only be used for inference, not for training"
    assert not fuse_mul_post, "fuse_mul_post should only be used for inference, not for training"
    assert dY.is_contiguous()
    assert W.is_contiguous()
    assert m_sizes.is_contiguous()
    assert m_sizes.ndim == 1

    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    # Flipped from the forward pass: a y permuted in the forward must be permuted on load in the backward.
    assert not (permute_y and use_tma_load_dy), "Cannot use both TMA load and permute_y"
    assert not (permute_x and use_tma_store), "Cannot use both TMA store and permute_x"

    use_tma = use_tma_load_dy or use_tma_load_w or use_tma_store
    if not supports_tma() and use_tma:
        warnings.warn("TMA not supported, tma_load will be set to False")
        use_tma_load_w = False
        use_tma_load_dy = False
        use_tma_store = False

    if use_tma or autotune:
        if _HAS_SET_ALLOCATOR and not getattr(triton, "_unsloth_allocator_set", False):

            def alloc_fn(size: int, alignment: int, stream: int):
                return torch.empty(size, device = "cuda", dtype = torch.int8)

            triton.set_allocator(alloc_fn)

    if W.ndim == 3:
        num_experts = W.shape[0]
        N = W.shape[1]
    else:
        num_experts = m_sizes.shape[0]
        N = W.shape[0] // num_experts

    dY = dY.view(-1, dY.shape[-1])
    W = W.view(-1, W.shape[-1])

    M_total, N_grad = dY.shape
    N_total, K = W.shape
    assert N_grad == N, f"Grad_output N ({N_grad}) must match weight N ({N})"

    assert M_total % topk == 0, f"M_total ({M_total}) must be divisible by topk ({topk})"
    num_tokens = M_total // topk

    # The kernel only reads gather_indices under permute_x / permute_y, so it stays optional otherwise.
    if permute_x or permute_y:
        assert (
            gather_indices is not None
        ), "gather_indices must be provided when permute_x or permute_y is True"
    total_tokens = gather_indices.shape[0] if gather_indices is not None else M_total
    assert total_tokens == M_total, f"Total tokens ({total_tokens}) must match M_total ({M_total})"

    # The output stays [NUM_TOKENS * TOPK, K] even under permute_x: gradients accumulate across every expert a token chose, reduced in a post-processing step.
    output_shape = (total_tokens, K)
    dX = torch.zeros(output_shape, device = dY.device, dtype = dY.dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        pass

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM {num_tokens = } {topk = } {output_shape = } {num_experts = } {N = } {K = } {BLOCK_SIZE_M = } {BLOCK_SIZE_N = } {BLOCK_SIZE_K = } {NUM_SMS = }"
        )
        print(f"DEBUG::GROUPED_GEMM {m_sizes.tolist()}")

    kernel_args = {
        "dY_ptr": dY,
        "w_ptr": W,
        "gather_indices_ptr": gather_indices,
        "m_sizes_ptr": m_sizes,
        "dX_ptr": dX,
        "NUM_EXPERTS": num_experts,
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
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

    is_fake = _is_tracing(dY, W)
    if not is_fake:
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)
    return dX


@allow_in_graph
def grouped_gemm_dW(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    fuse_mul_pre: bool = False,
    fuse_mul_post: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    flatten: bool = True,
    autotune: bool = False,
    debug: bool = False,
) -> torch.Tensor:
    """Backward dW kernel. X is (num_tokens, K) when permute_x, else (num_tokens * topk, K); dY is (M, N); m_sizes is the token count per expert; gather_indices is (total_tokens,) token indices per expert.

    permute_x and permute_y describe what the forward pass did; the backward permutes on LOAD to bring X and dy into the same expert-grouped order, and never on store since it writes dW.
    use_tma_load_dy is incompatible with permute_y and use_tma_load_x with permute_x (no TMA gather / scatter before Blackwell+); use_tma_store should always be on where TMA is supported, being faster than a global memory store.
    """
    assert not fuse_mul_pre, "fuse_mul_pre not supported"
    assert not fuse_mul_post, "fuse_mul_post not supported"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count if not debug else 1
    X = X.view(-1, X.shape[-1]).contiguous()
    dY = dY.contiguous()
    m_sizes = m_sizes.contiguous()

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
        if _HAS_SET_ALLOCATOR and not getattr(triton, "_unsloth_allocator_set", False):

            def alloc_fn(size: int, alignment: int, stream: int):
                return torch.empty(size, device = "cuda", dtype = torch.int8)

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
    _, K = X.shape
    M_grad, N = dY.shape

    assert M_grad == total_tokens, f"dY M ({M_grad}) != total_tokens ({total_tokens})"

    dW = torch.zeros((num_experts, N, K), device = X.device, dtype = X.dtype)

    if not autotune:
        pass

    def grid(META):
        return (NUM_SMS,)

    if debug:
        print(
            f"DEBUG::GROUPED_GEMM_DW_TMA {num_experts = } {N = } {K = } {BLOCK_SIZE_M = } {BLOCK_SIZE_N = } {BLOCK_SIZE_K = } {NUM_SMS = }"
        )

        print(f"DEBUG::GROUPED_GEMM_DW_TMA {m_sizes.tolist() = }")
        print(f"DEBUG::GROUPED_GEMM_DW_TMA {gather_indices.tolist() = }")
        m_start = 0
        for i in range(num_experts):
            expert_token_idx = gather_indices[m_start : m_start + m_sizes[i]]
            t_start = 0
            while t_start < m_sizes[i]:
                token_idx = expert_token_idx[t_start : t_start + BLOCK_SIZE_M]
                if permute_x:
                    token_idx = token_idx // topk
                print(f"DEBUG::GROUPED_GEMM_DW_TMA Token expert {i} indices: {token_idx.tolist()}")
                t_start += BLOCK_SIZE_M

            m_start += m_sizes[i]

    kernel_args = {
        "x_ptr": X,
        "dY_ptr": dY,
        "m_sizes_ptr": m_sizes,
        "gather_indices_ptr": gather_indices,
        "dW_ptr": dW,
        "NUM_TOKENS": num_tokens,
        "TOPK": topk,
        "NUM_EXPERTS": num_experts,
        "N": N,
        "K": K,
        "NUM_SMS": NUM_SMS,
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
                "USE_TMA_LOAD_dY": use_tma_load_dy,
                "USE_TMA_LOAD_X": use_tma_load_x,
                "USE_TMA_STORE": use_tma_store,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )

    kernel = _autotuned_grouped_gemm_dW_kernel if autotune else _grouped_gemm_dW_kernel

    is_fake = _is_tracing(X, dY)
    if not is_fake:
        compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](**kernel_args)

        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)

    return dW


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
    ):
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

        # topk_weights is not saved for backward: training with fused_mul is unsupported.
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
            X = X,
            W = W,
            topk = topk,
            m_sizes = m_sizes,
            gather_indices = gather_indices,
            topk_weights = topk_weights,
            permute_x = permute_x,
            permute_y = permute_y,
            fuse_mul_post = fuse_mul_post,
            autotune = autotune,
            **fwd_config,
        )

    @staticmethod
    def backward(ctx, dY):
        dY = dY.contiguous()
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
                assert (
                    kernel_config_bwd_dX is not None
                ), "kernel_config_bwd_dX must be provided if autotune is False"
            if not dX_only:
                assert (
                    kernel_config_bwd_dW is not None
                ), "kernel_config_bwd_dW must be provided if autotune is False"

        assert not fuse_mul_post, "fused_mul should only be used for inference, not for training"

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
                X = X,
                dY = dY,
                m_sizes = m_sizes,
                gather_indices = gather_indices,
                topk = topk,
                permute_x = permute_x,
                permute_y = permute_y,
                autotune = autotune,
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
                dY = dY,
                W = W,
                m_sizes = m_sizes,
                gather_indices = gather_indices,
                topk = topk,
                permute_x = permute_x,
                permute_y = permute_y,
                autotune = autotune,
                **bwd_dX_config,
            )

            if topk > 1 and permute_x:
                dX = dX.view(X.shape[0], topk, -1).sum(dim = 1)
        else:
            dX = None

        return (
            dX,
            dW,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def check_valid_config_fwd(
    permute_x,
    permute_y,
    use_tma_load_x,
    use_tma_load_w,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """Check if the configuration is valid for the forward pass."""
    is_second_gemm = not is_first_gemm

    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not (is_second_gemm and permute_x), "Cannot permute X for the second grouped GEMM"
    assert not (is_first_gemm and permute_y), "Cannot permute Y for the first grouped GEMM"
    assert not (fuse_mul_post and is_first_gemm), "Cannot fuse mul for the first grouped GEMM"
    assert not (
        use_tma_load_x and permute_x
    ), "Cannot use TMA load and permute X unless on sm100+ (Blackwell+)"
    assert not (
        use_tma_store and permute_y and is_second_gemm
    ), "Cannot use TMA store and permute Y for the second grouped GEMM unless on sm100+ (Blackwell+)"


def check_valid_config_bwd_dW(
    permute_x,
    permute_y,
    use_tma_load_dY,
    use_tma_load_x,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """Check if the configuration is valid for the backward pass of dW."""
    is_second_gemm = not is_first_gemm
    if fuse_mul_post:
        assert False, "Cannot fuse_mul is not supported for backward pass"
    if is_second_gemm and permute_y and use_tma_load_dY:
        assert False, "Cannot use TMA load and permute Y for the second grouped GEMM"
    if is_first_gemm and permute_x and use_tma_load_x:
        assert False, "Cannot use TMA load and permute X for the first grouped GEMM"


def check_valid_config_bwd_dX(
    permute_x,
    permute_y,
    use_tma_load_dY,
    use_tma_load_w,
    use_tma_store,
    fuse_mul_post,
    is_first_gemm,
):
    """Check if the configuration is valid for the backward pass of dW."""
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
    gather_indices: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights = None,
    fuse_mul_post = False,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    autotune: bool = False,
    is_first_gemm: bool = True,
    dX_only: bool = False,
    dW_only: bool = False,
):
    """Grouped GEMM for MoE MLPs.

    X is (num_tokens, K) when permute_x, else (num_tokens * topk, K) already sorted into expert-grouped order; W is (E, N, K); m_sizes is the token count per expert; gather_indices (total_tokens,) is required when either permutation is on; topk_weights is used only with fuse_mul.

    permute_x fuses the token-order to expert-order permutation (first GEMM of an MoE MLP); permute_y fuses the reverse (second GEMM).
    fuse_mul requires permute_y and is inference-only, never training.
    autotune ignores kernel_config_fwd, kernel_config_bwd_dX and kernel_config_bwd_dW.
    is_first_gemm gates config validation: permute_x belongs to the first GEMM and permute_y to the second, which also decides where TMA load and store may be used.
    """
    if not autotune:
        assert (
            kernel_config_fwd is not None
        ), "kernel_config_fwd must be provided if autotune is False"

        check_valid_config_fwd(
            permute_x,
            permute_y,
            use_tma_load_x = kernel_config_fwd.use_tma_load_x,
            use_tma_load_w = kernel_config_fwd.use_tma_load_w,
            use_tma_store = kernel_config_fwd.use_tma_store,
            fuse_mul_post = fuse_mul_post,
            is_first_gemm = is_first_gemm,
        )
        if kernel_config_bwd_dW is not None and not dX_only:
            check_valid_config_bwd_dW(
                permute_x,
                permute_y,
                use_tma_load_dY = kernel_config_bwd_dW.use_tma_load_dy,
                use_tma_load_x = kernel_config_bwd_dW.use_tma_load_x,
                use_tma_store = kernel_config_bwd_dW.use_tma_store,
                fuse_mul_post = fuse_mul_post,
                is_first_gemm = is_first_gemm,
            )
        if kernel_config_bwd_dX is not None and not dW_only:
            check_valid_config_bwd_dX(
                permute_x,
                permute_y,
                use_tma_load_dY = kernel_config_bwd_dX.use_tma_load_dy,
                use_tma_load_w = kernel_config_bwd_dX.use_tma_load_w,
                use_tma_store = kernel_config_bwd_dX.use_tma_store,
                fuse_mul_post = fuse_mul_post,
                is_first_gemm = is_first_gemm,
            )

    if permute_x or permute_y:
        assert (
            gather_indices is not None
        ), "gather_indices is required when either permute_x or permute_y is True"

    if fuse_mul_post:
        assert topk_weights is not None, "topk_weights is required when fuse_mul_post is True"

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    if gather_indices is not None:
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
