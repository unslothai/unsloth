# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import math
from typing import Any, Optional


EVAL_SPLIT_CANDIDATES = ("eval", "validation", "valid", "val", "test")
MIN_EVAL_ROWS = 16
MIN_TOTAL_ROWS_FOR_EVAL = MIN_EVAL_ROWS * 2


def evaluation_enabled(value: Any) -> bool:
    """Return whether a configured eval interval is finite and positive."""
    if isinstance(value, bool):
        return False
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(interval) and interval > 0


def split_dataset_for_evaluation(dataset: Any, *, seed: int = 3407) -> Optional[tuple[Any, Any]]:
    """Create the bounded deterministic train/eval split used by both training backends."""
    total_rows = len(dataset)
    if total_rows < MIN_TOTAL_ROWS_FOR_EVAL:
        return None

    eval_rows = max(MIN_EVAL_ROWS, min(128, int(0.05 * total_rows)))
    eval_rows = min(eval_rows, total_rows // 2)
    split = dataset.train_test_split(test_size = eval_rows, seed = seed)
    return split["train"], split["test"]
