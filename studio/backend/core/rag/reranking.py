# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

_retry_after = 0.0
_QUESTION = {
    "type": "noul",
    "instructions": "Does the passage contain information that helps answer the question?",
}


def enabled() -> bool:
    from utils import systemone_settings
    return systemone_settings.get_enabled() and systemone_settings.get_rag_rerank()


def score(query: str, passages: list[str]) -> list[float] | None:
    global _retry_after
    if not passages or time.monotonic() < _retry_after or not enabled():
        return None
    from core.systemone import laya_runtime

    states = [f"Question:\n{query}\n\nPassage:\n{passage}" for passage in passages]
    try:
        scores = laya_runtime.score_noul(_QUESTION, states)
    except ValueError:
        logger.info("Laya reranking skipped: input exceeds the model context window")
        return None
    except Exception as exc:
        _retry_after = time.monotonic() + 60.0
        logger.warning("Laya reranking unavailable (%s); using retrieval order", type(exc).__name__)
        return None
    if scores is None:
        return None
    if len(scores) != len(passages) or not all(
        isinstance(value, float) and math.isfinite(value) and 0.0 <= value <= 1.0
        for value in scores
    ):
        _retry_after = time.monotonic() + 60.0
        logger.warning("Laya reranking returned invalid scores; using retrieval order")
        return None
    return scores
