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
CGGR (Confidence-Gated Gradient Routing) integration for Unsloth.

This module provides selective backpropagation via label masking, enabling
~1.5-2x speedup in backward pass by only computing gradients for "hard" tokens.

Usage:
    from unsloth.cggr import CGGRUnslothBridge

    trainer = SFTTrainer(...)
    CGGRUnslothBridge.patch_trainer(trainer, min_tokens_ratio=0.25)
    trainer.train()

Requires: pip install cggr
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "CGGRUnslothBridge",
    "patch_trainer_for_cggr",
    "create_truncated_router",
    "CGGR_AVAILABLE",
]

# Check if CGGR package is available
try:
    import cggr

    CGGR_AVAILABLE = True
except ImportError:
    CGGR_AVAILABLE = False
    logger.debug("CGGR package not installed. Install with: pip install cggr")

# Conditional imports
if CGGR_AVAILABLE:
    from .router import create_truncated_router, TruncatedRouter
    from .bridge import CGGRUnslothBridge, patch_trainer_for_cggr
else:
    # Provide stub implementations that raise helpful errors
    def _cggr_not_available(*args, **kwargs):
        raise ImportError(
            "CGGR is not installed. Install with: pip install cggr\n"
            "For CUDA acceleration: pip install cggr[cuda]"
        )

    create_truncated_router = _cggr_not_available
    CGGRUnslothBridge = type(
        "CGGRUnslothBridge",
        (),
        {
            "patch_trainer": staticmethod(_cggr_not_available),
        },
    )
    patch_trainer_for_cggr = _cggr_not_available

    class TruncatedRouter:
        def __init__(self, *args, **kwargs):
            _cggr_not_available()
