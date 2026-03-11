# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Training submodule - Training backends and trainer classes
"""

from .training import TrainingBackend, TrainingProgress, get_training_backend

__all__ = [
    "TrainingProgress",
    "TrainingBackend",
    "get_training_backend",
]
