# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
MPS (Apple Silicon) backend for Grouped GEMM operations.

This module provides PyTorch-native fallback implementations of the grouped GEMM
operations used in Mixture of Experts (MoE) models. These implementations work
on MPS devices where Triton kernels are not available.

The fallback uses iterative GEMM over experts, which is slower than a fused kernel
but maintains correctness and enables MoE models to run on Apple Silicon.
"""

from .fallback import (
    grouped_gemm_mps_forward,
    grouped_gemm_mps_dX,
    grouped_gemm_mps_dW,
    GroupedGemmMPS,
    grouped_gemm_mps,
)

__all__ = [
    "grouped_gemm_mps_forward",
    "grouped_gemm_mps_dX",
    "grouped_gemm_mps_dW",
    "GroupedGemmMPS",
    "grouped_gemm_mps",
]
