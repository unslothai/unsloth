# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Centralized training hyperparameter defaults.

All backend modules should import from here instead of hardcoding values.
The frontend mirrors these in studio/frontend/src/config/training.ts.
"""

DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_LEARNING_RATE = 2e-4  # LoRA / QLoRA
DEFAULT_LEARNING_RATE_FULL = 2e-5  # Full fine-tuning
DEFAULT_LEARNING_RATE_STR = "2e-4"  # String form used by the API model / config dicts
DEFAULT_LEARNING_RATE_FULL_STR = "2e-5"  # String form for full fine-tuning
