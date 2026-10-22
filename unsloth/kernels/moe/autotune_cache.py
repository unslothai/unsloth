# Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

_kernel_config_cache: Dict[str, Any] = {}
_autotune_completed: Dict[str, bool] = {}

# Bump when a kernel change alters which tiles fit; other revisions / Triton versions are misses.
_CACHE_SCHEMA_VERSION = 2


def _get_cache_key(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    device_capability: Tuple[int, int],
    seq_len: int = 8192,  # Default sequence length for tuning
    weight_dtype: Optional[torch.dtype] = None,
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
        "schema_version": _CACHE_SCHEMA_VERSION,
        "triton_version": getattr(triton, "__version__", None),
    }
    if weight_dtype is not None and weight_dtype != dtype:
        key_data["weight_dtype"] = str(weight_dtype)
    key_str = json.dumps(key_data, sort_keys = True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_file_path(cache_key: str) -> str:
    """Get the file path for the cache file."""
    cache_dir = os.path.expanduser("~/.cache/unsloth/moe_autotune")
    os.makedirs(cache_dir, exist_ok = True)
    return os.path.join(cache_dir, f"{cache_key}.json")


def load_cached_config(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load cached kernel configuration from disk."""
    cache_file = _get_cache_file_path(cache_key)
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "r", encoding = "utf-8") as f:
            cached_data = json.load(f)

        # Invalidate if device capability changed. JSON stores the tuple as a list, so compare as tuples.
        current_device_capability = tuple(torch.cuda.get_device_capability())
        cached_device_capability = cached_data.get("device_capability")
        if not isinstance(cached_device_capability, (list, tuple)) or (
            tuple(cached_device_capability) != current_device_capability
        ):
            logger.info("Device capability changed, invalidating cache")
            os.remove(cache_file)
            return None

        if cached_data.get("schema_version") != _CACHE_SCHEMA_VERSION or cached_data.get(
            "triton_version"
        ) != getattr(triton, "__version__", None):
            logger.info("MoE kernel cache schema or Triton version changed, ignoring cache")
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
    metadata: Dict[str, Any] = None,
) -> None:
    """Save kernel configuration to disk cache."""
    cache_file = _get_cache_file_path(cache_key)

    cache_data = {
        "timestamp": time.time(),
        "device_capability": torch.cuda.get_device_capability(),
        "schema_version": _CACHE_SCHEMA_VERSION,
        "triton_version": getattr(triton, "__version__", None),
        "config_fwd": config_fwd.__dict__ if hasattr(config_fwd, "__dict__") else str(config_fwd),
        "config_bwd_dx": config_bwd_dx.__dict__
        if hasattr(config_bwd_dx, "__dict__")
        else str(config_bwd_dx),
        "config_bwd_dw": config_bwd_dw.__dict__
        if hasattr(config_bwd_dw, "__dict__")
        else str(config_bwd_dw),
        "metadata": metadata or {},
    }

    try:
        with open(cache_file, "w", encoding = "utf-8") as f:
            json.dump(cache_data, f, indent = 2)
        logger.info(f"Saved MoE kernel config cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save cache file {cache_file}: {e}")


# Always eager: unsloth_zoo calls this from the MoE forward inside compiled decoder blocks, and a
# Dynamo trace reads `.best_config` before any kernel ran, so every tuning silently fell back.
@torch.compiler.disable
def get_or_autotune_moe_kernels(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    force_autotune: bool = False,
    seq_len: int = 8192,
    weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, Any]:
    """
    Get cached kernel configurations or run auto-tuning.

    Args:
        num_experts: Number of experts in the MoE layer
        hidden_dim: Hidden dimension of the model
        intermediate_dim: Intermediate dimension for MoE MLP
        top_k: Number of experts to route to
        dtype: Data type of the activations (X and dY)
        weight_dtype: Data type of the expert weights, if different from dtype
        force_autotune: Force re-running autotuning even if cache exists
        seq_len: Sequence length to use for tuning benchmarks

    Returns:
        Tuple of (config_fwd, config_bwd_dx, config_bwd_dw)
    """
    device_capability = torch.cuda.get_device_capability()
    cache_key = _get_cache_key(
        num_experts,
        hidden_dim,
        intermediate_dim,
        top_k,
        dtype,
        device_capability,
        seq_len,
        weight_dtype,
    )

    # Env override to disable autotuning
    if os.environ.get("UNSLOTH_MOE_DISABLE_AUTOTUNE", "0") == "1":
        logger.info(
            f"UNSLOTH_MOE_DISABLE_AUTOTUNE=1: Using Heuristic (Safe) MoE kernel configs for SM{device_capability[0]}{device_capability[1]}"
        )
        return _get_heuristic_configs(
            n_dim = 2 * intermediate_dim, k_dim = hidden_dim, dtype = dtype, weight_dtype = weight_dtype
        )
    if not force_autotune and cache_key in _kernel_config_cache:
        logger.info(f"Using in-memory cached MoE kernel configs: {cache_key}")
        return _kernel_config_cache[cache_key]

    if not force_autotune:
        cached_data = load_cached_config(cache_key)
        if cached_data is not None:
            try:
                from .grouped_gemm.kernels.tuning import (
                    KernelConfigForward,
                    KernelConfigBackward_dX,
                    KernelConfigBackward_dW,
                )

                config_fwd = KernelConfigForward(**cached_data["config_fwd"])
                config_bwd_dx = KernelConfigBackward_dX(**cached_data["config_bwd_dx"])
                config_bwd_dw = KernelConfigBackward_dW(**cached_data["config_bwd_dw"])

                # A disk entry may predate a kernel change: refit to dims and SMEM.
                configs = _fit_configs_to_dims(
                    (config_fwd, config_bwd_dx, config_bwd_dw), 2 * intermediate_dim, hidden_dim
                )
                configs = _fit_configs_to_smem(configs, dtype, weight_dtype)
                _kernel_config_cache[cache_key] = configs
                return configs
            except Exception as e:
                logger.warning(f"Failed to reconstruct cached configs: {e}")

    if cache_key in _autotune_completed and not force_autotune:
        logger.info(f"Autotuning already completed for: {cache_key}")
        return _kernel_config_cache[cache_key]

    logger.info(f"Running MoE kernel auto-tuning for: {cache_key}")
    logger.info(
        f"Configuration: {num_experts} experts, {hidden_dim} hidden, {intermediate_dim} intermediate, top_k={top_k}"
    )

    try:
        configs = _run_moe_autotuning(
            num_experts, hidden_dim, intermediate_dim, top_k, dtype, seq_len, weight_dtype
        )

        _kernel_config_cache[cache_key] = configs
        _autotune_completed[cache_key] = True

        config_fwd, config_bwd_dx, config_bwd_dw = configs
        save_cached_config(
            cache_key,
            config_fwd,
            config_bwd_dx,
            config_bwd_dw,
            {
                "num_experts": num_experts,
                "hidden_dim": hidden_dim,
                "intermediate_dim": intermediate_dim,
            },
        )

        logger.info(f"MoE kernel auto-tuning completed: {cache_key}")
        return configs

    except Exception as e:
        logger.error(f"MoE kernel auto-tuning failed: {e}")
        if "AttributeError" in str(e) and "_experimental_make_tensor_descriptor" in str(e):
            logger.warning(
                "Unsloth: Your Triton version might be incompatible with TMA features. Falling back to default configs."
            )
        logger.info("Falling back to default kernel configurations")
        return _get_default_configs(
            n_dim = 2 * intermediate_dim, k_dim = hidden_dim, dtype = dtype, weight_dtype = weight_dtype
        )


def _run_moe_autotuning(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    top_k: int,
    dtype: torch.dtype,
    seq_len: int,
    weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, Any]:
    """Run the actual auto-tuning for MoE kernels."""

    device = "cuda"
    # Fixed token count avoids OOMs and a seq_len dependency, so the passed seq_len is ignored.
    num_tokens = 4096
    total_tokens = num_tokens * top_k

    hidden_states = torch.randn(num_tokens, hidden_dim, device = device, dtype = dtype)

    gate_up_weights = torch.randn(
        num_experts,
        2 * intermediate_dim,
        hidden_dim,
        device = device,
        dtype = dtype if weight_dtype is None else weight_dtype,
    )

    # Dummy routing data
    m_sizes = torch.randint(1, total_tokens // num_experts + 1, (num_experts,), device = device)
    m_sizes = m_sizes * (total_tokens // m_sizes.sum().item())
    # Adjust to exact total
    diff = total_tokens - m_sizes.sum().item()
    if diff != 0:
        m_sizes[0] += diff

    gather_indices = torch.arange(total_tokens, device = device)
    torch.randperm(total_tokens, out = gather_indices)

    # Autotune via the interface function with autotune=True (lets triton tune)
    from .grouped_gemm.interface import (
        _autotuner_for,
        grouped_gemm_forward,
        grouped_gemm_dX,
    )
    from .grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel
    from .grouped_gemm.kernels.backward import _autotuned_grouped_gemm_dX_kernel
    from .grouped_gemm.kernels.tuning import (
        KernelConfigForward,
        KernelConfigBackward_dX,
    )

    logger.info("Autotuning forward kernel (first GEMM)...")
    _ = grouped_gemm_forward(
        X = hidden_states,
        W = gate_up_weights,
        topk = top_k,
        m_sizes = m_sizes,
        gather_indices = gather_indices,
        permute_x = True,
        permute_y = False,
        autotune = True,
    )
    triton_config_fwd = _autotuner_for(
        _autotuned_grouped_gemm_forward_kernel, hidden_states.device
    ).best_config

    config_fwd = KernelConfigForward(
        BLOCK_SIZE_M = triton_config_fwd.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N = triton_config_fwd.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K = triton_config_fwd.kwargs["BLOCK_SIZE_K"],
        num_warps = triton_config_fwd.num_warps,
        num_stages = triton_config_fwd.num_stages,
        use_tma_load_x = triton_config_fwd.kwargs.get("USE_TMA_LOAD_X", False),
        use_tma_load_w = triton_config_fwd.kwargs.get("USE_TMA_LOAD_W", False),
        use_tma_store = triton_config_fwd.kwargs.get("USE_TMA_STORE", False),
    )

    logger.info("Autotuning backward dX kernel...")
    dummy_grad = torch.randn(total_tokens, 2 * intermediate_dim, device = device, dtype = dtype)
    _ = grouped_gemm_dX(
        dY = dummy_grad,
        W = gate_up_weights,
        gather_indices = gather_indices,
        m_sizes = m_sizes,
        topk = top_k,
        permute_x = True,
        permute_y = False,
        autotune = True,
    )
    triton_config_bwd_dx = _autotuner_for(
        _autotuned_grouped_gemm_dX_kernel, dummy_grad.device
    ).best_config

    # Safe Backward Configs: 64x64x256
    config_bwd_dx = KernelConfigBackward_dX(
        BLOCK_SIZE_M = triton_config_bwd_dx.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N = triton_config_bwd_dx.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K = triton_config_bwd_dx.kwargs["BLOCK_SIZE_K"],
        num_warps = triton_config_bwd_dx.num_warps,
        num_stages = triton_config_bwd_dx.num_stages,
        use_tma_load_dy = triton_config_bwd_dx.kwargs.get("USE_TMA_LOAD_dY", False),
        use_tma_load_w = triton_config_bwd_dx.kwargs.get("USE_TMA_LOAD_W", False),
        use_tma_store = triton_config_bwd_dx.kwargs.get("USE_TMA_STORE", False),
    )

    config_bwd_dw = _autotune_dw_config(
        hidden_states, dummy_grad, m_sizes, gather_indices, top_k, num_experts
    )

    return config_fwd, config_bwd_dx, config_bwd_dw


def _autotune_dw_config(hidden_states, dummy_grad, m_sizes, gather_indices, top_k, num_experts):
    """Autotune the dW kernel the launcher dispatches to and read the config from that kernel's tuner."""
    from .grouped_gemm.interface import _autotuner_for, _use_per_expert_dW_kernel, grouped_gemm_dW
    from .grouped_gemm.kernels.backward import (
        _autotuned_grouped_gemm_dW_kernel,
        _autotuned_grouped_gemm_dW_per_expert_kernel,
    )
    from .grouped_gemm.kernels.tuning import KernelConfigBackward_dW

    per_expert = _use_per_expert_dW_kernel()
    logger.info("Autotuning backward dW kernel...")
    _ = grouped_gemm_dW(
        X = hidden_states,
        dY = dummy_grad,
        m_sizes = m_sizes,
        gather_indices = gather_indices,
        topk = top_k,
        permute_x = True,
        permute_y = False,
        autotune = True,
    )
    if per_expert:
        triton_config_bwd_dw = _autotuner_for(
            _autotuned_grouped_gemm_dW_per_expert_kernel, hidden_states.device
        ).best_config
    else:
        triton_config_bwd_dw = _autotuner_for(
            _autotuned_grouped_gemm_dW_kernel, hidden_states.device
        ).best_config

    return KernelConfigBackward_dW(
        BLOCK_SIZE_M = triton_config_bwd_dw.kwargs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N = triton_config_bwd_dw.kwargs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K = triton_config_bwd_dw.kwargs["BLOCK_SIZE_K"],
        num_warps = triton_config_bwd_dw.num_warps,
        num_stages = triton_config_bwd_dw.num_stages,
        use_tma_load_dy = triton_config_bwd_dw.kwargs.get("USE_TMA_LOAD_dY", False),
        use_tma_load_x = triton_config_bwd_dw.kwargs.get("USE_TMA_LOAD_X", False),
        use_tma_store = triton_config_bwd_dw.kwargs.get("USE_TMA_STORE", False),
    )


_MIN_FALLBACK_BLOCK_SIZE = 16


def _fit_block_size(block_size: int, dim: Optional[int]) -> int:
    # Largest power of two <= block_size and >= 16 dividing dim; else unchanged (the kernel asserts still fire).
    if dim is None or dim <= 0 or dim % block_size == 0:
        return block_size
    candidate = 1 << (block_size.bit_length() - 1)
    while candidate >= _MIN_FALLBACK_BLOCK_SIZE:
        if dim % candidate == 0:
            return candidate
        candidate >>= 1
    return block_size


def _fit_configs_to_dims(
    configs: Tuple[Any, Any, Any], n_dim: Optional[int], k_dim: Optional[int]
) -> Tuple[Any, Any, Any]:
    # The kernels static_assert K % BLOCK_SIZE_K == 0 and N % BLOCK_SIZE_N == 0; W is [E, N, K] for all three.
    # BLOCK_SIZE_M tiles the masked per-expert token count, so it stays.
    for config in configs:
        config.BLOCK_SIZE_N = _fit_block_size(config.BLOCK_SIZE_N, n_dim)
        config.BLOCK_SIZE_K = _fit_block_size(config.BLOCK_SIZE_K, k_dim)
    return configs


def _device_smem_per_block() -> Optional[int]:
    # Read per current device: the launching card on a mixed host.
    try:
        index = torch.cuda.current_device()
    except Exception:
        return None
    try:
        return int(torch.cuda.get_device_properties(index).shared_memory_per_block_optin)
    except Exception:
        try:
            return int(
                triton.runtime.driver.active.utils.get_device_properties(index)["max_shared_mem"]
            )
        except Exception:
            return None


def _itemsize(dtype: Optional[torch.dtype], default: int = 2) -> int:
    return default if dtype is None else dtype.itemsize


def _estimate_smem(
    kind: str, config: Any, act_bytes: int, weight_bytes: int, capability: Tuple[int, int]
) -> int:
    """Shared memory (B) per launch; matches compiled metadata.shared (Triton 3.6-3.8, sm_75..120) or exceeds it by <= 1 KiB."""
    M, N, K = config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K
    stages = max(int(config.num_stages), 1)
    if kind == "fwd":
        a, b, a_cast = M * K * act_bytes, N * K * weight_bytes, act_bytes != weight_bytes
    elif kind == "dX":
        a, b, a_cast = M * N * act_bytes, N * K * weight_bytes, act_bytes != weight_bytes
    else:
        # The launcher runs float32 X single stage.
        a, b, a_cast = M * N * act_bytes, M * K * act_bytes, False
        if act_bytes >= 4:
            stages = 1
    major = capability[0]
    if major < 8:
        # No cp.async multi-buffering; 16-bit dot operands add a 2x larger-tile scratch.
        dot_bytes = act_bytes if kind == "dW" else weight_bytes
        if dot_bytes >= 4:
            return a + b
        return max(a + b, 2 * max(a * dot_bytes // act_bytes, b))
    hopper_like = major in (9, 10, 11)
    if kind == "dW":
        if stages > 4:
            buffers = stages if hopper_like else stages - 1
        else:
            buffers = min(stages, 2) if hopper_like else 1
        return buffers * (a + b) + (2048 if hopper_like else 1024)
    if hopper_like:
        # A cast operand keeps num_stages - 1 buffers plus one copy in the dot dtype.
        if a_cast:
            return max(stages - 1, 1) * a + stages * b + a * weight_bytes // act_bytes + 64
        return stages * (a + b) + 64
    return max(stages - 1, 1) * (a + b)


_LOOP_DIM = {"fwd": "BLOCK_SIZE_K", "dX": "BLOCK_SIZE_N", "dW": "BLOCK_SIZE_M"}
_MIN_SMEM_BLOCK = 16


def _fit_config_to_smem(
    kind: str,
    config: Any,
    act_bytes: int,
    weight_bytes: int,
    capability: Tuple[int, int],
    smem: int,
) -> Any:
    # Halving a power-of-two tile keeps it dividing N / K.
    def fits():
        return _estimate_smem(kind, config, act_bytes, weight_bytes, capability) <= smem

    before = (config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K, config.num_stages)
    loop_dim = _LOOP_DIM[kind]
    while not fits():
        if config.num_stages > 3:
            config.num_stages -= 1
        elif getattr(config, loop_dim) > 32:
            setattr(config, loop_dim, getattr(config, loop_dim) // 2)
        elif config.num_stages > 2:
            config.num_stages -= 1
        else:
            name = max(
                ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"), key = lambda n: getattr(config, n)
            )
            if getattr(config, name) > _MIN_SMEM_BLOCK:
                setattr(config, name, getattr(config, name) // 2)
            elif config.num_stages > 1:
                config.num_stages = 1
            else:
                break
    after = (config.BLOCK_SIZE_M, config.BLOCK_SIZE_N, config.BLOCK_SIZE_K, config.num_stages)
    if after != before:
        logger.info(
            f"MoE {kind} config (M, N, K, stages) {before} -> {after} to fit {smem} B shared memory"
        )
    return config


def _fit_configs_to_smem(
    configs: Tuple[Any, Any, Any],
    dtype: Optional[torch.dtype] = None,
    weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, Any]:
    try:
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            return configs
        capability = tuple(torch.cuda.get_device_capability())
    except Exception:
        return configs
    smem = _device_smem_per_block()
    if smem is None:
        return configs
    act_bytes = _itemsize(dtype)
    weight_bytes = _itemsize(weight_dtype, act_bytes)
    for kind, config in zip(("fwd", "dX", "dW"), configs):
        _fit_config_to_smem(kind, config, act_bytes, weight_bytes, capability, smem)
    return configs


def _get_heuristic_configs(
    n_dim: Optional[int] = None,
    k_dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, Any]:
    """
    Get 'Safe Heuristic' kernel configurations.
    These are verified to be safe on A100 (SM80) and provide ~9x speedup on H100/B200.
    """
    from .grouped_gemm.kernels.tuning import (
        KernelConfigForward,
        KernelConfigBackward_dX,
        KernelConfigBackward_dW,
    )

    # Safe forward config 64x128x128 fits A100 SMEM; the backward pair below is 64x64x256.
    config_fwd = KernelConfigForward(
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 128,
        num_warps = 8,
        num_stages = 3,
        permute_x = True,
        permute_y = True,
        use_tma_load_x = False,
        use_tma_load_w = False,  # TMA loads might need alignment checks, safer to disable for heuristic
        use_tma_store = False,
    )

    config_bwd_dx = KernelConfigBackward_dX(
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        num_warps = 8,
        num_stages = 4,
        permute_x = True,
        permute_y = True,
        use_tma_load_dy = False,
        use_tma_load_w = False,
        use_tma_store = False,
    )

    config_bwd_dw = KernelConfigBackward_dW(
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_K = 256,
        num_warps = 8,
        num_stages = 4,
        permute_x = True,
        permute_y = True,
        use_tma_load_dy = False,
        use_tma_load_x = False,
        use_tma_store = False,
    )

    configs = _fit_configs_to_dims((config_fwd, config_bwd_dx, config_bwd_dw), n_dim, k_dim)
    return _fit_configs_to_smem(configs, dtype, weight_dtype)


def _get_default_configs(
    n_dim: Optional[int] = None,
    k_dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    weight_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, Any]:
    """Get default kernel configurations as fallback."""
    from .grouped_gemm.kernels.tuning import (
        KernelConfigForward,
        KernelConfigBackward_dX,
        KernelConfigBackward_dW,
    )

    logger.warning("Using default MoE kernel configurations (not optimal)")

    config_fwd = KernelConfigForward(
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        num_warps = 8,
        num_stages = 3,
        use_tma_load_x = False,
        use_tma_load_w = False,
        use_tma_store = False,
    )

    config_bwd_dx = KernelConfigBackward_dX(
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        num_warps = 8,
        num_stages = 3,
        use_tma_load_dy = False,
        use_tma_load_w = False,
        use_tma_store = False,
    )

    config_bwd_dw = KernelConfigBackward_dW(
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 64,
        num_warps = 8,
        num_stages = 3,
        use_tma_load_dy = False,
        use_tma_load_x = False,
        use_tma_store = False,
    )

    configs = _fit_configs_to_dims((config_fwd, config_bwd_dx, config_bwd_dw), n_dim, k_dim)
    return _fit_configs_to_smem(configs, dtype, weight_dtype)


def clear_cache() -> None:
    """Clear all cached kernel configurations."""
    global _kernel_config_cache, _autotune_completed
    _kernel_config_cache.clear()
    _autotune_completed.clear()
    logger.info("Cleared MoE kernel cache")


def is_autotuning_completed(cache_key: str) -> bool:
    """Check if autotuning has been completed for a given cache key."""
    return cache_key in _autotune_completed
