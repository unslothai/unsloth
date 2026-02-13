# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib.metadata
import functools
import gc
from packaging.version import Version, parse
import torch
from .quantization import quantize_4bit
from tqdm import tqdm

_MLX_AVAILABLE = False
_MLX_VERSION = None

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
    try:
        _MLX_VERSION = importlib.metadata.version("mlx")
    except Exception:
        pass
except (Exception, ImportError):
    pass


class UnslothMLXError(ImportError):
    """Exception raised when MLX is required but not installed/functional."""


def is_mlx_available(min_version: str = "0.0.0") -> bool:
    """Checks if MLX is available and meets the minimum version requirement."""
    if not _MLX_AVAILABLE:
        return False

    if min_version == "0.0.0":
        return True

    if _MLX_VERSION is None:
        # Cannot determine version, assume compatible if available?
        # Or fail safe? Let's assume compatible for now but log warning?
        return True

    return parse(_MLX_VERSION) >= parse(min_version)


def get_mlx_version() -> str:
    """Returns the installed MLX version string."""
    return _MLX_VERSION


def require_mlx(min_version: str = "0.0.0"):
    """Decorator to require MLX for a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_mlx_available(min_version):
                raise UnslothMLXError(
                    f"MLX >= {min_version} is required for this function.\n"
                    "Please install it via `pip install mlx`"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Quantization Utilities
# =============================================================================


def fast_quantize(model):
    """
    Iterates over the model and pre-quantizes Linear layers to 4-bit MLX format.
    The quantized weights are stored in the `_mlx_cache` attribute of the PyTorch weight.

    This allows the Unsloth/MLX bridge to transparently use the optimized 4-bit kernel
    without any changes to the model structure itself.
    """
    print("Unsloth: Fast 4-bit quantization for Apple Silicon...")

    # Identify layers to quantize.
    # We focus on the heavy lifters: MLP and Attention projections.
    # usually: layers.N.mlp.gate_proj, up_proj, down_proj
    #          layers.N.self_attn.q_proj, k_proj, v_proj, o_proj

    count = 0
    total_layers = 0

    # First pass to count for progress bar
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # rudimentary filter to avoid quantizing classifier head or embeddings if they appear as linear
            if "head" in name or "embed" in name:
                continue
            total_layers += 1

    pbar = tqdm(total=total_layers, desc="Quantizing layers", unit="layer")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "head" in name or "embed" in name:
                continue

            # Apply quantization
            # We store it directly on the weight tensor
            # The 'fast_lora' kernels check hasattr(weight, '_mlx_cache')
            if module.weight.device.type == "meta":
                print(
                    f"Unsloth: Warning - Skipping {name} as it is on the meta device (offloaded). "
                    "Quantization requires the weight to be in memory."
                )
                continue

            if hasattr(module.weight, "_mlx_cache"):
                count += 1
                pbar.update(1)
                continue

            module.weight._mlx_cache = quantize_4bit(module.weight)
            count += 1
            pbar.update(1)

            # Periodic memory cleanup to prevent offloading/OOM
            if count % 10 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    pbar.close()
    print(f"Unsloth: Quantized {count} layers to 4-bit MLX.")
