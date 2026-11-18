# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Trimmed to the shape of the pinned q4nx commit's q4nx/model_converter.py.
from .constants import ModelArch

_QWEN35_GEOMETRY = {
    (24, 1024): ModelArch.QWEN35_08B,
    (32, 4096): ModelArch.QWEN35_9B,
}
if hasattr(ModelArch, "QWEN35_27B"):
    _QWEN35_GEOMETRY[(64, 5120)] = ModelArch.QWEN35_27B
