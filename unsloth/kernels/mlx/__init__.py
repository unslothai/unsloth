# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLX Integration Layer for Unsloth.

This module provides MLX-accelerated kernels for Apple Silicon Macs.
MLX offers superior performance to PyTorch MPS for certain operations
by leveraging Apple's unified memory architecture more efficiently.

Usage:
    from unsloth.kernels.mlx import is_mlx_available, UnslothMLXError

    if is_mlx_available():
        # Use MLX-accelerated operations
        pass
    else:
        # Fall back to MPS or CPU
        pass

Note:
    MLX is optional and only works on Apple Silicon Macs.
    Install with: pip install 'unsloth[apple]'
"""

from .utils import (
    is_mlx_available,
    get_mlx_version,
    UnslothMLXError,
    require_mlx,
)

from .bridge import (
    torch_to_mlx,
    mlx_to_torch,
    mlx_context,
    with_mlx_context,
    synchronize_mps,
    synchronize_mlx,
)

from .fast_ops import (
    is_mlx_fast_available,
    USE_MLX_FAST,
    mlx_layer_norm,
    mlx_rms_norm,
    mlx_rope,
    mlx_rope_qk,
    mlx_scaled_dot_product_attention,
)

from .merge_lora import (
    mlx_merge_lora,
    mlx_merge_lora_layer,
)

__all__ = [
    # Availability checks
    "is_mlx_available",
    "get_mlx_version",
    "is_mlx_fast_available",
    "USE_MLX_FAST",
    # Error handling
    "UnslothMLXError",
    "require_mlx",
    # Tensor conversion bridge
    "torch_to_mlx",
    "mlx_to_torch",
    "mlx_context",
    "with_mlx_context",
    "synchronize_mps",
    "synchronize_mlx",
    # Fast operations
    "mlx_layer_norm",
    "mlx_rms_norm",
    "mlx_rope",
    "mlx_rope_qk",
    "mlx_scaled_dot_product_attention",
    # LoRA merge
    "mlx_merge_lora",
    "mlx_merge_lora_layer",
]
