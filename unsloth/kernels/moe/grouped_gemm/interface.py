# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

import copy
import functools
import inspect
import logging
import os
import types
import warnings
from typing import Optional
from dataclasses import asdict
from unsloth import DEVICE_TYPE
from unsloth.device_type import cuda_visible_capabilities
from unsloth.kernels.utils import torch_gpu_device

import torch
import triton

try:
    from triton.runtime.errors import OutOfResources
except ImportError:  # older Triton
    from triton.runtime import OutOfResources

from .kernels.backward import (
    _autotuned_grouped_gemm_dW_kernel,
    _autotuned_grouped_gemm_dW_per_expert_kernel,
    _autotuned_grouped_gemm_dX_kernel,
    _grouped_gemm_dW_kernel,
    _grouped_gemm_dW_per_expert_kernel,
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


# Process-wide, so every visible card must be sm90+ (experts may sit on any); an unreadable card means False.
def _check_tma_support():
    if DEVICE_TYPE in ("xpu", "hip"):
        return False
    import triton.language as tl

    caps = cuda_visible_capabilities()
    gpu_supports_tma = (
        bool(caps) and len(caps) == torch.cuda.device_count() and min(c[0] for c in caps) >= 9
    )
    triton_has_tma_api = hasattr(tl, "make_tensor_descriptor") or hasattr(
        tl, "_experimental_make_tensor_descriptor"
    )
    return gpu_supports_tma and triton_has_tma_api


_SUPPORTS_TMA = _check_tma_support()

_DEVICE_AUTOTUNERS = {}
_MODEL_AUTOTUNE_CACHES = {}
_DEVICE_MODELS = {}


def _device_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_model(device):
    index = _device_index(device)
    model = _DEVICE_MODELS.get(index)
    if model is None:
        try:
            p = torch.cuda.get_device_properties(index)
            model = (
                getattr(p, "name", None),
                getattr(p, "major", None),
                getattr(p, "minor", None),
                getattr(p, "multi_processor_count", None),
            )
        except Exception:
            model = index
        _DEVICE_MODELS[index] = model
    return model


_DEVICE_SM_COUNTS = {}


def _num_sms(device):
    """multi_processor_count of ``device``, read once per card (it cannot change in a process)."""
    index = _device_index(device)
    count = _DEVICE_SM_COUNTS.get(index)
    if count is None:
        count = _DEVICE_SM_COUNTS[index] = torch.cuda.get_device_properties(
            index
        ).multi_processor_count
    return count


_OPERAND_TILE_FITS = {}


def _autotune_cache():
    from .. import autotune_cache
    return autotune_cache


def _fit_tiles_to_operand_device(
    kind, operand, weight_dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages
):
    """Caller-supplied tiles were fitted to the current card's shared memory when the caller passed no device
    (unsloth_zoo <= 2026.10.7). If the operands sit on a different GPU model, shrink a copy to fit their card.
    One GPU, or the same model as the current card: returned unchanged."""
    tiles = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages)
    device = operand.device
    try:
        if device.type != "cuda" or DEVICE_TYPE != "cuda" or torch.cuda.device_count() <= 1:
            return tiles
        current = torch.cuda.current_device()
        index = _device_index(device)
        if index == current or _device_model(device) == _device_model(
            torch.device("cuda", current)
        ):
            return tiles
    except Exception:
        return tiles
    key = (kind, index, operand.dtype, weight_dtype, tiles)
    fitted = _OPERAND_TILE_FITS.get(key)
    if fitted is None:
        fitted = tiles
        try:
            from types import SimpleNamespace

            autotune_cache = _autotune_cache()

            smem = autotune_cache._device_smem_per_block(device)
            if smem is not None:
                config = SimpleNamespace(
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    BLOCK_SIZE_N = BLOCK_SIZE_N,
                    BLOCK_SIZE_K = BLOCK_SIZE_K,
                    num_stages = num_stages,
                )
                autotune_cache._fit_config_to_smem(
                    kind,
                    config,
                    operand.dtype.itemsize,
                    weight_dtype.itemsize,
                    tuple(torch.cuda.get_device_capability(device)),
                    smem,
                )
                fitted = (
                    config.BLOCK_SIZE_M,
                    config.BLOCK_SIZE_N,
                    config.BLOCK_SIZE_K,
                    config.num_stages,
                )
        except Exception:
            fitted = tiles
        _OPERAND_TILE_FITS[key] = fitted
    return fitted


def _autotuner_for(kernel, device):
    """Triton's autotune key omits the device, so on several GPUs each card runs its own copy. Copies on
    the same GPU model share one results cache, so identical cards reuse each other's tuning; a run's
    scratch state (nargs) stays per card, since backward runs one autograd thread per device. No lock:
    a cached launch never waits on another card's tuning."""
    if torch.cuda.device_count() <= 1 or not hasattr(kernel, "cache"):
        return kernel
    key = (id(kernel), _device_index(device))
    tuner = _DEVICE_AUTOTUNERS.get(key)
    if tuner is None:
        tuner = copy.copy(kernel)
        model_key = (id(kernel), _device_model(device))
        tuner.cache = _MODEL_AUTOTUNE_CACHES.setdefault(model_key, {})
        _DEVICE_AUTOTUNERS[key] = tuner
    # Callers reassign kernel.configs (benchmarks, tests): follow it like the shared tuner would.
    configs = getattr(kernel, "configs", None)
    if configs is not None and getattr(tuner, "configs", None) is not configs:
        tuner.configs = configs
    return tuner


_HAS_SET_ALLOCATOR = hasattr(triton, "set_allocator")


def supports_tma():
    return _SUPPORTS_TMA


def _is_tracing(*tensors):
    """True if tensors are fake tensors used during torch.compile tracing (Triton cannot run). Not torch.compiler.is_compiling(): that is True during both tracing AND execution, and only tracing must skip the kernels."""
    for t in tensors:
        name = type(t).__name__
        if name in ("FakeTensor", "FunctionalTensor", "FunctionalTensorWrapper"):
            return True
    return False


_per_device_alloc_fns = {}
_FUSED_MUL_WARN = False


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
    # Runs after every launch: formatting the metadata costs several us per call even when nothing logs.
    if not logger.isEnabledFor(logging.DEBUG):
        return
    kernel_name = compiled_kernel.name
    nregs = compiled_kernel.n_regs
    nspills = compiled_kernel.n_spills
    metadata = compiled_kernel.metadata
    logger.debug(f"{kernel_name}: n_regs={nregs} n_spills={nspills} metadata={metadata}")
    if best_config is not None:
        logger.debug(f"{kernel_name} autotuned best_config: {best_config}")


def _grouped_gemm_forward_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    topk: int,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    fuse_mul_post: bool = False,
    autotune: bool = False,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    use_tma_load_w: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
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

    NUM_SMS = _num_sms(X.device)

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages = _fit_tiles_to_operand_device(
            "fwd", X, W.dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages
        )

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

    kernel = (
        _autotuner_for(_autotuned_grouped_gemm_forward_kernel, X.device)
        if autotune
        else _grouped_gemm_forward_kernel
    )

    is_fake = _is_tracing(X, W)
    if not is_fake:
        with torch_gpu_device(X.device):
            compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](
                **kernel_args, enable_fp_fusion = True
            )
        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)

    return y


def _grouped_gemm_dX_impl(
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

    if permute_x or permute_y:
        assert (
            gather_indices is not None
        ), "gather_indices must be provided when permute_x or permute_y is True"
    total_tokens = gather_indices.shape[0] if gather_indices is not None else M_total
    assert total_tokens == M_total, f"Total tokens ({total_tokens}) must match M_total ({M_total})"

    output_shape = (total_tokens, K)
    dX = torch.zeros(output_shape, device = dY.device, dtype = dY.dtype)

    NUM_SMS = _num_sms(dY.device)

    def grid(META):
        return (NUM_SMS,)

    if not autotune:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages = _fit_tiles_to_operand_device(
            "dX", dY, W.dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages
        )

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
    kernel = (
        _autotuner_for(_autotuned_grouped_gemm_dX_kernel, dY.device)
        if autotune
        else _grouped_gemm_dX_kernel
    )

    is_fake = _is_tracing(dY, W)
    if not is_fake:
        with torch_gpu_device(dY.device):
            compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](
                **kernel_args, enable_fp_fusion = True
            )

        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)
    return dX


def _use_per_expert_dW_kernel():
    """Per-expert dW grid (one program per (expert, n, k) tile); UNSLOTH_MOE_DW_KERNEL=legacy restores the
    kernel that walks every expert inside each (n, k) program. Measured faster on every shape, dtype and GPU
    tried (A100, L4, RTX PRO 6000, B200), so there is no shape routing."""
    return os.environ.get("UNSLOTH_MOE_DW_KERNEL", "").strip().lower() != "legacy"


_PER_EXPERT_DW_FITS = {}
_DW_CARD_FITS = {}


def _card_smem(index):
    """(shared_memory_per_block_optin, (major, minor)) of cuda:index, else None (ROCm, no CUDA)."""
    if getattr(torch.version, "hip", None):
        return None
    props = torch.cuda.get_device_properties(index)
    return int(props.shared_memory_per_block_optin), (props.major, props.minor)


def _per_expert_dW_stages(
    device,
    act_bytes,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    pipelined = False,
):
    """Largest num_stages <= the requested one at which the per-expert dW kernel fits ``device`` shared memory, or
    None (use the legacy kernel). Unpermuted operands (``pipelined``) multi-buffer X per stage, so 4 stages of the
    heuristic 64x64x256 tile need 106,496 B: only the stage count drops, the tile (and so the bits) stay."""
    index = _device_index(device)
    key = (index, act_bytes, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, pipelined)
    if key in _PER_EXPERT_DW_FITS:
        return _PER_EXPERT_DW_FITS[key]
    stages = num_stages
    try:
        card = _card_smem(index)
        if card is not None:
            from ..autotune_cache import _estimate_smem

            smem, capability = card
            stages = None
            for candidate in range(max(int(num_stages), 1), 0, -1):
                tile = types.SimpleNamespace(
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    BLOCK_SIZE_N = BLOCK_SIZE_N,
                    BLOCK_SIZE_K = BLOCK_SIZE_K,
                    num_stages = candidate,
                )
                need = _estimate_smem("dW", tile, act_bytes, act_bytes, capability, pipelined)
                if need <= smem:
                    stages = candidate
                    break
                if not pipelined:
                    break  # stage count does not change a gathered launch's footprint
            if stages is None:
                logger.info(
                    f"Unsloth: MoE dW tile (M, N, K, stages) {key[2:6]} needs ~{need} B shared "
                    f"memory per-expert, over {smem} B on cuda:{index}; using the legacy dW kernel."
                )
            elif stages != num_stages:
                logger.info(
                    f"Unsloth: MoE dW tile (M, N, K) {key[2:5]} runs {stages} stages instead of "
                    f"{num_stages} to fit {smem} B shared memory on cuda:{index}."
                )
    except Exception:
        stages = num_stages
    _PER_EXPERT_DW_FITS[key] = stages
    return stages


def _per_expert_dW_fits(
    device,
    act_bytes,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    pipelined = False,
):
    """False when this tile overflows ``device`` shared memory in the per-expert dW kernel at every stage count.

    A caller-supplied tile the legacy kernel launched can raise OutOfResources there (gathered loads double buffer)."""
    return (
        _per_expert_dW_stages(
            device, act_bytes, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, pipelined
        )
        is not None
    )


def _fit_dW_tiles_to_card(
    device, act_bytes, pipelined, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages
):
    """Shrunk copy of a dW tile that raised OutOfResources on ``device`` (an explicit tile sized for a bigger card,
    e.g. 128x256x256 on a 64 KiB T4, where the legacy kernel overflows too). Only called after a launch failed, so a
    tile the estimate over-counts but that launches is never touched."""
    tiles = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages)
    key = (_device_index(device), act_bytes, pipelined, tiles)
    fitted = _DW_CARD_FITS.get(key)
    if fitted is None:
        fitted = tiles
        try:
            card = _card_smem(key[0])
            if card is not None:
                config = types.SimpleNamespace(
                    BLOCK_SIZE_M = BLOCK_SIZE_M,
                    BLOCK_SIZE_N = BLOCK_SIZE_N,
                    BLOCK_SIZE_K = BLOCK_SIZE_K,
                    num_stages = num_stages,
                )
                _autotune_cache()._fit_config_to_smem(
                    "dW", config, act_bytes, act_bytes, card[1], card[0], pipelined
                )
                fitted = (
                    config.BLOCK_SIZE_M,
                    config.BLOCK_SIZE_N,
                    config.BLOCK_SIZE_K,
                    config.num_stages,
                )
        except Exception:
            fitted = tiles
        _DW_CARD_FITS[key] = fitted
    return fitted


def _launch_grouped_gemm_dW(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    topk: int,
    dW_out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
    dW_dtype: Optional[torch.dtype] = None,
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
    NUM_SMS = _num_sms(X.device) if not debug else 1
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
    per_expert = _use_per_expert_dW_kernel()
    if not autotune:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages = _fit_tiles_to_operand_device(
            "dW", X, X.dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages
        )
        legacy_stages = num_stages
        per_expert_stages = (
            _per_expert_dW_stages(
                X.device,
                X.element_size(),
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                num_stages,
                pipelined = not (permute_x or permute_y),
            )
            if per_expert
            else None
        )
        if per_expert_stages is None:
            per_expert = False
        else:
            num_stages = per_expert_stages

    if accumulate:
        # Read-modify-write into an existing gradient: one program owns each (expert, n, k) tile.
        assert not autotune, "autotune benchmarks re-run the kernel and would add dW several times"
        assert dW_out is not None
        assert dW_out.numel() == num_experts * N * K, f"{dW_out.shape} != {(num_experts, N, K)}"
        assert dW_out.is_contiguous() and dW_out.device == X.device
        dW = dW_out
    else:
        # The per-expert kernel writes every tile (zeros for an empty expert); the legacy one skips them.
        dW = (torch.empty if per_expert else torch.zeros)(
            (num_experts, N, K), device = X.device, dtype = X.dtype if dW_dtype is None else dW_dtype
        )
    if use_tma_store and (dW.element_size() > X.element_size() or dW.data_ptr() % 16 != 0):
        # TMA needs a 16 byte aligned base (a DDP bucket view may not be), and the tuned tile
        # sizes assume an output no wider than X.
        use_tma_store = False

    def grid(META):
        if per_expert:
            return (
                num_experts
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
                * triton.cdiv(K, META["BLOCK_SIZE_K"]),
            )
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
    }
    if accumulate:
        kernel_args["ACCUMULATE"] = True
    if per_expert:
        kernel_args["EXPERTS_POW2"] = triton.next_power_of_2(num_experts)
        if X.element_size() >= 4:
            # fp32 operands (fp32 RMSNorm outputs in a bf16 full finetune): the tf32 dot reads both tiles
            # M-major, and a multi-stage pipeline doubles shared memory and halves occupancy; one stage is
            # faster for every tile measured. The legacy kernel never pipelines this loop either.
            num_stages = 1

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

    if per_expert:
        kernel = (
            _autotuner_for(_autotuned_grouped_gemm_dW_per_expert_kernel, X.device)
            if autotune
            else _grouped_gemm_dW_per_expert_kernel
        )
    else:
        kernel = (
            _autotuner_for(_autotuned_grouped_gemm_dW_kernel, X.device)
            if autotune
            else _grouped_gemm_dW_kernel
        )

    is_fake = _is_tracing(X, dY)
    if not is_fake:
        with torch_gpu_device(X.device):
            fit_args = None
            if not autotune:
                tiles = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, kernel_args["num_stages"])
                pipelined = per_expert and not (permute_x or permute_y)
                fit_args = (X.device, X.element_size(), pipelined, *tiles)
                if (_device_index(X.device), X.element_size(), pipelined, tiles) in _DW_CARD_FITS:
                    _set_dW_tiles(kernel_args, _fit_dW_tiles_to_card(*fit_args))  # failed before
            try:
                compiled_kernel: triton.compiler.CompiledKernel = kernel[grid](
                    **kernel_args, enable_fp_fusion = True
                )
            except OutOfResources:
                # Raised before the kernel runs (dW untouched): retry once with a tile fitted to this card.
                if fit_args is None:
                    raise
                tiles = fit_args[3:]
                fitted = _fit_dW_tiles_to_card(*fit_args)
                if fitted == tiles:
                    if not per_expert:
                        raise
                    # No smaller tile known for this card (ROCm has no shared memory model here: the
                    # heuristic 64x64x256 4-stage tile needs ~80 KiB of LDS on gfx1151's 64 KiB): run the
                    # legacy kernel, which launched this tile before the per-expert kernel, and remember.
                    logger.info(
                        f"Unsloth: MoE per-expert dW tile (M, N, K, stages) {tiles} is over the shared "
                        f"memory of {X.device}; using the legacy dW kernel."
                    )
                    _PER_EXPERT_DW_FITS[
                        (_device_index(X.device), X.element_size(), *tiles[:3], legacy_stages,
                         not (permute_x or permute_y))
                    ] = None  # fmt: skip
                    per_expert = False  # read by grid()
                    kernel = _grouped_gemm_dW_kernel
                    kernel_args.pop("EXPERTS_POW2", None)
                    kernel_args["num_stages"] = legacy_stages
                    if not accumulate:
                        dW.zero_()  # the legacy kernel skips empty experts
                else:
                    logger.info(
                        f"Unsloth: MoE dW tile (M, N, K, stages) {tiles} is over the shared memory of "
                        f"cuda:{_device_index(X.device)}; using {fitted}."
                    )
                    _set_dW_tiles(kernel_args, fitted)
                compiled_kernel = kernel[grid](**kernel_args, enable_fp_fusion = True)

        if autotune:
            log_kernel_info(compiled_kernel, kernel.best_config)
        else:
            log_kernel_info(compiled_kernel)

    return dW


def _set_dW_tiles(kernel_args, tiles):
    (
        kernel_args["BLOCK_SIZE_M"],
        kernel_args["BLOCK_SIZE_N"],
        kernel_args["BLOCK_SIZE_K"],
        kernel_args["num_stages"],
    ) = tiles


def _grouped_gemm_dW_impl(
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
    dW_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """dW into a fresh zeroed (E, N, K) buffer of dW_dtype (default X.dtype)."""
    kwargs = dict(locals())
    return _launch_grouped_gemm_dW(dW_out = None, accumulate = False, **kwargs)


def _grouped_gemm_dW_accumulate_impl(
    X: torch.Tensor,
    dY: torch.Tensor,
    m_sizes: torch.Tensor,
    gather_indices: torch.Tensor,
    dW: torch.Tensor,
    topk: int,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    permute_x: bool = False,
    permute_y: bool = False,
    use_tma_load_dy: bool = False,
    use_tma_load_x: bool = False,
    use_tma_store: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """dW += X^T dY per expert, in place, added in fp32 and rounded once to dW.dtype."""
    kwargs = dict(locals())
    dW_out = kwargs.pop("dW")
    _launch_grouped_gemm_dW(dW_out = dW_out, accumulate = True, **kwargs)


# Opaque ops: allow_in_graph let AOT trace the body on fake tensors, keeping torch.empty and dropping the launch.
# Not triton_op: torch.compile rejects prune_configs_by on triton.autotune.
def _fwd_fake(X, W, topk, m_sizes, gather_indices, topk_weights, permute_x, permute_y, *args):
    N = W.shape[1] if W.ndim == 3 else W.shape[0] // m_sizes.shape[0]
    total_tokens = gather_indices.shape[0] if (permute_x or permute_y) else X.numel() // X.shape[-1]
    return X.new_empty((total_tokens, N))


def _dX_fake(dY, W, gather_indices, m_sizes, topk, *args):
    total_tokens = (
        gather_indices.shape[0] if gather_indices is not None else dY.numel() // dY.shape[-1]
    )
    return dY.new_empty((total_tokens, W.shape[-1]))


def _dW_fake(X, dY, m_sizes, *args):
    dW_dtype = dict(zip(_DW_PARAM_NAMES[3:], args)).get("dW_dtype")
    return X.new_empty(
        (m_sizes.shape[0], dY.shape[-1], X.shape[-1]),
        dtype = X.dtype if dW_dtype is None else dW_dtype,
    )


def _dW_accumulate_fake(X, dY, m_sizes, gather_indices, dW, *args):
    return None


_DW_PARAM_NAMES = tuple(inspect.signature(_grouped_gemm_dW_impl).parameters)
_OPTIONAL_TENSORS = ("gather_indices", "topk_weights")


def _make_op(
    name,
    impl,
    fake,
    mutates_args = (),
    returns = torch.Tensor,
):
    """torch.library.custom_op (torch >= 2.4) over impl; torch.compiler.disable where it is missing."""
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return torch._dynamo.disable(impl)
    existing = getattr(getattr(torch.ops, "unsloth", None), name, None)
    if existing is not None:
        return existing
    params = [
        p.replace(
            default = inspect.Parameter.empty,
            annotation = Optional[torch.Tensor] if p.name in _OPTIONAL_TENSORS else p.annotation,
        )
        for p in inspect.signature(impl).parameters.values()
    ]
    names = [p.name for p in params]

    def op(*args):
        return impl(**dict(zip(names, args)))

    op.__signature__ = inspect.Signature(params, return_annotation = returns)
    op.__name__ = name
    op = custom_op(f"unsloth::{name}", op, mutates_args = tuple(mutates_args))
    op.register_fake(fake)
    return op


_grouped_gemm_forward_op = _make_op("grouped_gemm_forward", _grouped_gemm_forward_impl, _fwd_fake)
_grouped_gemm_dX_op = _make_op("grouped_gemm_dX", _grouped_gemm_dX_impl, _dX_fake)
_grouped_gemm_dW_op = _make_op("grouped_gemm_dW", _grouped_gemm_dW_impl, _dW_fake)
_grouped_gemm_dW_accumulate_op = _make_op(
    "grouped_gemm_dW_accumulate",
    _grouped_gemm_dW_accumulate_impl,
    _dW_accumulate_fake,
    mutates_args = ("dW",),
    returns = None,
)


def _signature(impl):
    params = inspect.signature(impl).parameters.values()
    return tuple(p.name for p in params), {
        p.name: p.default for p in params if p.default is not p.empty
    }


_SIGNATURES = {
    impl: _signature(impl)
    for impl in (
        _grouped_gemm_forward_impl,
        _grouped_gemm_dX_impl,
        _grouped_gemm_dW_impl,
        _grouped_gemm_dW_accumulate_impl,
    )
}


def _dispatch(impl, op, args, kwargs):
    names, defaults = _SIGNATURES[impl]
    bound = dict(defaults)
    bound.update(zip(names, args))
    bound.update(kwargs)
    if torch.compiler.is_compiling() or _is_tracing(bound[names[0]], bound[names[1]]):
        return op(*[bound[n] for n in names])
    return impl(**bound)


@functools.wraps(_grouped_gemm_forward_impl)
def grouped_gemm_forward(*args, **kwargs) -> torch.Tensor:
    return _dispatch(_grouped_gemm_forward_impl, _grouped_gemm_forward_op, args, kwargs)


@functools.wraps(_grouped_gemm_dX_impl)
def grouped_gemm_dX(*args, **kwargs) -> torch.Tensor:
    return _dispatch(_grouped_gemm_dX_impl, _grouped_gemm_dX_op, args, kwargs)


@functools.wraps(_grouped_gemm_dW_impl)
def grouped_gemm_dW(*args, **kwargs) -> torch.Tensor:
    return _dispatch(_grouped_gemm_dW_impl, _grouped_gemm_dW_op, args, kwargs)


@functools.wraps(_grouped_gemm_dW_accumulate_impl)
def grouped_gemm_dW_accumulate(*args, **kwargs) -> None:
    return _dispatch(_grouped_gemm_dW_accumulate_impl, _grouped_gemm_dW_accumulate_op, args, kwargs)


DW_ACCUMULATE_STATS = {"inplace": 0, "fallback": 0}


_DW_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _dW_accumulate_enabled():
    return os.environ.get("UNSLOTH_MOE_DW_ACCUMULATE", "1") != "0"


def _dW_accumulate_target(ctx, X, autotune):
    """The leaf parameter whose existing .grad the dW kernel may add into, else None.

    Any case where AccumulateGrad would do more than `param.grad += dW` falls back to returning dW,
    including Node hooks that can rewrite dW: prehooks on W's AccumulateGrad and post hooks on this
    GroupedGemmBackward. Post hooks on AccumulateGrad only observe (they see None for this dW)."""
    if torch.compiler.is_compiling() or _is_tracing(X) or torch.is_grad_enabled():
        return None
    # Anomaly mode checks the returned gradients for NaN; an in-place add returns None for W.
    if torch.is_anomaly_enabled():
        return None
    if autotune or not _dW_accumulate_enabled():
        return None
    try:
        acc = ctx.next_functions[1][0]
    except (AttributeError, IndexError, TypeError):
        return None
    if acc is None or type(acc).__name__ != "AccumulateGrad":
        return None
    param = acc.variable
    if type(param) not in (torch.Tensor, torch.nn.Parameter):
        return None  # tensor subclasses (DTensor, quantized) own their gradient layout
    grad = param.grad
    if grad is None or grad.layout != torch.strided or type(grad) is not torch.Tensor:
        return None
    if (
        grad.dtype not in _DW_DTYPES
        or grad.dtype != param.dtype
        or grad.device != X.device
        or grad.shape != param.shape
        or not grad.is_contiguous()
    ):
        return None
    if getattr(param, "_backward_hooks", None) or getattr(
        param, "_post_accumulate_grad_hooks", None
    ):
        return None
    if getattr(param, "_unsloth_dw_side_stream", False):
        return None
    if torch.cuda.current_stream(X.device) != torch.cuda.default_stream(X.device):
        return None
    try:
        if not torch._C._will_engine_execute_node(acc):
            return None
    except RuntimeError:
        return None  # autograd.grad(): W's gradient is captured, never accumulated
    # Last: the only checks that cost Python work, so they run only for an otherwise eligible weight.
    if _node_has_python_hooks(acc.register_prehook) or _node_has_python_hooks(ctx.register_hook):
        return None
    return param


def _node_hook_probe(*args):
    return None


def _node_has_python_hooks(register):
    """True if the Node behind `register` (register_hook / register_prehook) already holds a Python hook.

    Nodes cannot list their hooks, but every Python hook of one kind shares a single dict, and the
    handle of a new registration points at it: register a no-op, count, remove it again."""
    try:
        handle = register(_node_hook_probe)
    except Exception:
        return True
    try:
        hooks = handle.hooks_dict_ref()
        return hooks is None or len(hooks) > 1
    except Exception:
        return True
    finally:
        handle.remove()


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

        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        if W.requires_grad and X.is_cuda and not torch.compiler.is_compiling():
            if torch.cuda.current_stream(X.device) != torch.cuda.default_stream(X.device):
                # A use on a side stream may run its backward concurrently with another use of W:
                # in-place accumulation would race, so this weight keeps the returned-gradient path.
                W._unsloth_dw_side_stream = True

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
        kernel_config_bwd_dX = ctx.kernel_config_bwd_dX
        kernel_config_bwd_dW = ctx.kernel_config_bwd_dW
        autotune = ctx.autotune
        assert (
            not ctx.fuse_mul_post
        ), "fused_mul should only be used for inference, not for training"
        dX_only = ctx.dX_only or not ctx.needs_input_grad[1]
        dW_only = ctx.dW_only or not ctx.needs_input_grad[0]
        if dX_only and dW_only:
            return (None,) * 15

        if not autotune:
            if not dW_only:
                assert (
                    kernel_config_bwd_dX is not None
                ), "kernel_config_bwd_dX must be provided if autotune is False"
            if not dX_only:
                assert (
                    kernel_config_bwd_dW is not None
                ), "kernel_config_bwd_dW must be provided if autotune is False"

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

            target = _dW_accumulate_target(ctx, X, autotune)
            if target is not None:
                grouped_gemm_dW_accumulate(
                    X = X,
                    dY = dY,
                    m_sizes = m_sizes,
                    gather_indices = gather_indices,
                    dW = target.grad,
                    topk = topk,
                    permute_x = permute_x,
                    permute_y = permute_y,
                    **bwd_dW_config,
                )
                # Already in param.grad: returning it as well would add it twice.
                dW = None
                DW_ACCUMULATE_STATS["inplace"] += 1
            else:
                if W.dtype != X.dtype and W.dtype in _DW_DTYPES and _dW_accumulate_enabled():
                    bwd_dW_config["dW_dtype"] = W.dtype
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
                if not torch.compiler.is_compiling():
                    DW_ACCUMULATE_STATS["fallback"] += 1
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
