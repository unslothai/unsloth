# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generic numeric downsampling utility."""


def downsample(values: list[float], target_count: int) -> list[float]:
    """Reduce a list to target_count points via evenly-spaced index sampling."""
    if len(values) <= target_count:
        return list(values)
    if target_count <= 0:
        return []
    if target_count == 1:
        return [values[-1]]
    indices = [
        round(i * (len(values) - 1) / (target_count - 1)) for i in range(target_count)
    ]
    return [values[i] for i in indices]
