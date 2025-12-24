# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Auto-tuning cache system for MoE kernels to ensure tuning runs only once at training start.
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import triton

logger = logging.getLogger(__name__)

# Global cache for kernel configurations
_kernel_config_cache: Dict[str, Any] = {}
_autotune_completed: Dict[str, bool] = {}

def _get_cache_key(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    device_capability: Tuple[int, int],
    seq_len: int = 8192,  # Default sequence length for tuning
) -> str:
    """Generate a unique cache key based on model configuration."""
    key_data = {
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "top_k": top_k,
        "dtype": str(dtype),
        "device_capability": device_capability,
        "seq_len": seq_len,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cache_file_path(cache_key: str) -> str:
    """Get the file path for the cache file."""
    cache_dir = os.path.expanduser("~/.cache/unsloth/moe_autotune")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.json")

def load_cached_config(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached kernel configuration from disk."""
    cache_file = _get_cache_file_path(cache_key)
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)

        # Verify cache is still valid (same device, etc.)
        current_device_capability = torch.cuda.get_device_capability()
        if cached_data.get("device_capability") != current_device_capability:
            logger.info("Device capability changed, invalidating cache")
            os.remove(cache_file)
            return None

        logger.info(f"Loaded cached MoE kernel config: {cache_key}")
        return cached_data
    except Exception as e:
        logger.warning(f"Failed to load cache file {cache_file}: {e}")
        try:
            os.remove(cache_file)
        except:
            pass
        return None

def save_cached_config(
    cache_key: str,
    config_fwd: Any,
    config_bwd_dx: Any,
    config_bwd_dw: Any,
    metadata: Dict[str, Any] = None
) -> None:
    """Save kernel configuration to disk cache."""
    cache_file = _get_cache_file_path(cache_key)

    cache_data = {
        "timestamp": time.time(),
        "device_capability": torch.cuda.get_device_capability(),
        "config_fwd": config_fwd.__dict__ if hasattr(config_fwd, '__dict__') else str(config_fwd),
        "config_bwd_dx": config_bwd_dx.__dict__ if hasattr(config_bwd_dx, '__dict__') else str(config_bwd_dx),
        "config_bwd_dw": config_bwd_dw.__dict__ if hasattr(config_bwd_dw, '__dict__') else str(config_bwd_dw),
        "metadata": metadata or {},
    }

    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Saved MoE kernel config cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save cache file {cache_file}: {e}")

def get_or_autotune_moe_kernels(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    force_autotune: bool = False,
    seq_len: int = 8192,
) -> Tuple[Any, Any, Any]:
    """
    Get cached kernel configurations or run auto-tuning.

    Args:
        num_experts: Number of experts in the MoE layer
        hidden_dim: Hidden dimension of the model
        intermediate_dim: Intermediate dimension for MoE MLP
        top_k: Number of experts to route to
        dtype: Data type for computation
        force_autotune: Force re-running autotuning even if cache exists
        seq_len: Sequence length to use for tuning benchmarks

    Returns:
        Tuple of (config_fwd, config_bwd_dx, config_bwd_dw)
    """
    device_capability = torch.cuda.get_device_capability()
    cache_key = _get_cache_key(
        num_experts, hidden_dim, intermediate_dim, top_k, dtype, device_capability, seq_len
    )

    # Check if we already have cached configs
    if not force_autotune and cache_key in _kernel_config_cache:
        logger.info(f"Using in-memory cached MoE kernel configs: {cache_key}")
        return _kernel_config_cache[cache_key]

    # Try to load from disk
    if not force_autotune:
        cached_data = load_cached_config(cache_key)
        if cached_data is not None:
            # Reconstruct config objects from cached data
            try:
                from grouped_gemm.kernels.tuning import (
                    KernelConfigForward,
                    KernelConfigBackward_dX,
                    KernelConfigBackward_dW,
                )

                config_fwd = KernelConfigForward(**cached_data["config_fwd"])
                config_bwd_dx = KernelConfigBackward_dX(**cached_data["config_bwd_dx"])
                config_bwd_dw = KernelConfigBackward_dW(**cached_data["config_bwd_dw"])

                configs = (config_fwd, config_bwd_dx, config_bwd_dw)
                _kernel_config_cache[cache_key] = configs
                return configs
            except Exception as e:
                logger.warning(f"Failed to reconstruct cached configs: {e}")

    # Run autotuning
    if cache_key in _autotune_completed and not force_autotune:
        logger.info(f"Autotuning already completed for: {cache_key}")
        return _kernel_config_cache[cache_key]

    logger.info(f"Running MoE kernel auto-tuning for: {cache_key}")
    logger.info(f"Configuration: {num_experts} experts, {hidden_dim} hidden, {intermediate_dim} intermediate, top_k={top_k}")

    try:
        configs = _run_moe_autotuning(
            num_experts, hidden_dim, intermediate_dim, top_k, dtype, seq_len
        )

        # Cache the results
        _kernel_config_cache[cache_key] = configs
        _autotune_completed[cache_key] = True

        # Save to disk
        config_fwd, config_bwd_dx, config_bwd_dw = configs
        save_cached_config(
            cache_key,
            config_fwd,
            config_bwd_dx,
            config_bwd_dw,
            {"num_experts": num_experts, "hidden_dim": hidden_dim, "intermediate_dim": intermediate_dim}
        )

        logger.info(f"MoE kernel auto-tuning completed: {cache_key}")
        return configs

    except Exception as e:
        logger.error(f"MoE kernel auto-tuning failed: {e}")
        if "AttributeError" in str(e) and "_experimental_make_tensor_descriptor" in str(e):
             logger.warning("Unsloth: Your Triton version might be incompatible with TMA features. Falling back to default configs.")
        logger.info("Falling back to default kernel configurations")
        return _get_default_configs()

def _run_moe_autotuning(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    seq_len: int,
) -> Tuple[Any, Any, Any]:
    """Run the actual auto-tuning for MoE kernels."""

    # Create dummy inputs for tuning
    device = "cuda"
    # Use a fixed, safe number of tokens for autotuning to avoid OOMs and dependency on seq_len
    # 4096 is standard for finding good kernels without consuming 10GB+ VRAM
    # We ignore the passed seq_len for the actual allocation to satisfy user request
    num_tokens = 4096
    total_tokens = num_tokens * top_k

    # Create dummy tensors
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)

    # Create dummy weights
    gate_up_weights = torch.randn(
        num_experts, 2 * intermediate_dim, hidden_dim, device=device, dtype=dtype
    )
    down_weights = torch.randn(
        num_experts, hidden_dim, intermediate_dim, device=device, dtype=dtype
    )

    # Create dummy routing data
    m_sizes = torch.randint(1, total_tokens // num_experts + 1, (num_experts,), device=device)
    m_sizes = m_sizes * (total_tokens // m_sizes.sum().item())
    # Adjust to ensure exact total
    diff = total_tokens - m_sizes.sum().item()
    if diff != 0:
        m_sizes[0] += diff

    gather_indices = torch.arange(total_tokens, device=device)
    torch.randperm(total_tokens, out=gather_indices)

    # Autotune forward kernel - use the interface function with autotune=True
    # This properly invokes the kernel and lets triton handle the autotuning
    from grouped_gemm.interface import (
        grouped_gemm_forward,
        grouped_gemm_dX,
        grouped_gemm_dW,
    )
    from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel
    from grouped_gemm.kernels.backward import (
        _autotuned_grouped_gemm_dX_kernel,
        _autotuned_grouped_gemm_dW_kernel,
    )
    from grouped_gemm.kernels.tuning import (
        KernelConfigForward,
        KernelConfigBackward_dX,
        KernelConfigBackward_dW,
    )

    logger.info("Autotuning forward kernel (first GEMM)...")
    # Run with autotune=True to trigger autotuning
    _ = grouped_gemm_forward(
        X=hidden_states,
        W=gate_up_weights,
        topk=top_k,
        m_sizes=m_sizes,
        gather_indices=gather_indices,
        permute_x=True,
        permute_y=False,
        autotune=True,
    )
    triton_config_fwd = _autotuned_grouped_gemm_forward_kernel.best_config

    # Convert triton.Config to KernelConfigForward
    config_fwd = KernelConfigForward(
        BLOCK_SIZE_M=triton_config_fwd.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=triton_config_fwd.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=triton_config_fwd.kwargs["BLOCK_SIZE_K"],
        num_warps=triton_config_fwd.num_warps,
        num_stages=triton_config_fwd.num_stages,
        use_tma_load_x=triton_config_fwd.kwargs.get("USE_TMA_LOAD_X", False),
        use_tma_load_w=triton_config_fwd.kwargs.get("USE_TMA_LOAD_W", False),
        use_tma_store=triton_config_fwd.kwargs.get("USE_TMA_STORE", False),
    )

    # Autotune backward dX kernel
    logger.info("Autotuning backward dX kernel...")
    dummy_grad = torch.randn(total_tokens, 2 * intermediate_dim, device=device, dtype=dtype)
    _ = grouped_gemm_dX(
        dY=dummy_grad,
        W=gate_up_weights,
        gather_indices=gather_indices,
        m_sizes=m_sizes,
        topk=top_k,
        permute_x=True,
        permute_y=False,
        autotune=True,
    )
    triton_config_bwd_dx = _autotuned_grouped_gemm_dX_kernel.best_config

    # Convert triton.Config to KernelConfigBackward_dX
    config_bwd_dx = KernelConfigBackward_dX(
        BLOCK_SIZE_M=triton_config_bwd_dx.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=triton_config_bwd_dx.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=triton_config_bwd_dx.kwargs["BLOCK_SIZE_K"],
        num_warps=triton_config_bwd_dx.num_warps,
        num_stages=triton_config_bwd_dx.num_stages,
        use_tma_load_dy=triton_config_bwd_dx.kwargs.get("USE_TMA_LOAD_dY", False),
        use_tma_load_w=triton_config_bwd_dx.kwargs.get("USE_TMA_LOAD_W", False),
        use_tma_store=triton_config_bwd_dx.kwargs.get("USE_TMA_STORE", False),
    )

    # Autotune backward dW kernel
    logger.info("Autotuning backward dW kernel...")
    _ = grouped_gemm_dW(
        X=hidden_states,
        dY=dummy_grad,
        m_sizes=m_sizes,
        gather_indices=gather_indices,
        topk=top_k,
        permute_x=True,
        permute_y=False,
        autotune=True,
    )
    triton_config_bwd_dw = _autotuned_grouped_gemm_dW_kernel.best_config

    # Convert triton.Config to KernelConfigBackward_dW
    config_bwd_dw = KernelConfigBackward_dW(
        BLOCK_SIZE_M=triton_config_bwd_dw.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=triton_config_bwd_dw.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=triton_config_bwd_dw.kwargs["BLOCK_SIZE_K"],
        num_warps=triton_config_bwd_dw.num_warps,
        num_stages=triton_config_bwd_dw.num_stages,
        use_tma_load_dy=triton_config_bwd_dw.kwargs.get("USE_TMA_LOAD_dY", False),
        use_tma_load_x=triton_config_bwd_dw.kwargs.get("USE_TMA_LOAD_X", False),
        use_tma_store=triton_config_bwd_dw.kwargs.get("USE_TMA_STORE", False),
    )

    return config_fwd, config_bwd_dx, config_bwd_dw



def _get_default_configs() -> Tuple[Any, Any, Any]:
    """Get default kernel configurations as fallback."""
    from grouped_gemm.kernels.tuning import (
        KernelConfigForward,
        KernelConfigBackward_dX,
        KernelConfigBackward_dW,
    )

    logger.warning("Using default MoE kernel configurations (not optimal)")

    config_fwd = KernelConfigForward(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_warps=8,
        num_stages=3,
        use_tma_load_x=False,
        use_tma_load_w=False,
        use_tma_store=False,
    )

    config_bwd_dx = KernelConfigBackward_dX(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_warps=8,
        num_stages=3,
        use_tma_load_dy=False,
        use_tma_load_w=False,
        use_tma_store=False,
    )

    config_bwd_dw = KernelConfigBackward_dW(
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_warps=8,
        num_stages=3,
        use_tma_load_dy=False,
        use_tma_load_x=False,
        use_tma_store=False,
    )

    return config_fwd, config_bwd_dx, config_bwd_dw

def clear_cache() -> None:
    """Clear all cached kernel configurations."""
    global _kernel_config_cache, _autotune_completed
    _kernel_config_cache.clear()
    _autotune_completed.clear()
    logger.info("Cleared MoE kernel cache")

def is_autotuning_completed(cache_key: str) -> bool:
    """Check if autotuning has been completed for a given cache key."""
    return cache_key in _autotune_completed
