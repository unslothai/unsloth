# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Low-precision training policy: one recipe per format (per-GEMM dtypes and scaling, high-precision
layers, GradScaler kind, per-GPU gates), refusals that run before the model is touched, structural
high-precision layer selection and engagement checks. Data and checks only: the kernels stay in torchao and
Unsloth's kernels, reached through the existing entry points (`float8_training`, `qat_scheme`, `load_in_fp8`)."""

from .recipes import (
    FLOAT8_TRAINING_NAMES,
    RECIPES,
    Gate,
    GemmPolicy,
    HPLayerPolicy,
    Operand,
    Recipe,
    get_recipe,
    measured_faster,
    named_recipes,
    recipes_of_kind,
)
from .hardware import (
    PrecisionRefused,
    check_recipe,
    current_gpu_class,
    dtype_tag,
    gpu_class,
    grad_scaler_kind,
)
from .selectors import (
    decoder_layers,
    find_embeddings,
    find_output_heads,
    find_routers,
    high_precision_layers,
    hp_filter,
    num_experts_of,
)
from .engage import EngagementError, count_lowp_gemms, engaged, record, require, strict_engage

__all__ = [
    "FLOAT8_TRAINING_NAMES",
    "RECIPES",
    "Gate",
    "GemmPolicy",
    "HPLayerPolicy",
    "Operand",
    "Recipe",
    "get_recipe",
    "measured_faster",
    "named_recipes",
    "recipes_of_kind",
    "PrecisionRefused",
    "check_recipe",
    "current_gpu_class",
    "dtype_tag",
    "gpu_class",
    "grad_scaler_kind",
    "decoder_layers",
    "find_embeddings",
    "find_output_heads",
    "find_routers",
    "high_precision_layers",
    "hp_filter",
    "num_experts_of",
    "EngagementError",
    "count_lowp_gemms",
    "engaged",
    "record",
    "require",
    "strict_engage",
]
