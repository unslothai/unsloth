# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference.chat_template_helpers import normalize_reasoning_snapshots
from core.inference.live_token_progress import (
    inherit_live_token_progress,
    live_token_progress_text,
)


def test_buffered_text_uses_generated_token_ids_for_live_tps():
    text = live_token_progress_text(
        "one buffered fragment",
        generated_tokens = 6,
        elapsed_seconds = 0.5,
    )

    assert text == "one buffered fragment"
    assert text.generated_tokens == 6
    assert text.tok_per_sec == pytest.approx(10.0)


def test_transformed_snapshot_keeps_request_local_progress():
    source = live_token_progress_text(
        "native markers",
        generated_tokens = 7,
        tok_per_sec = 14.0,
    )
    transformed = inherit_live_token_progress("<think>normalized</think>", source)

    assert transformed.generated_tokens == 7
    assert transformed.tok_per_sec == 14.0


def test_reasoning_normalization_keeps_live_progress():
    source = live_token_progress_text(
        "<reasoning>work</reasoning>answer",
        generated_tokens = 8,
        tok_per_sec = 16.0,
    )
    normalized = list(
        normalize_reasoning_snapshots(
            [source],
            markers = ("<reasoning>", "</reasoning>"),
        )
    )

    assert normalized[-1].generated_tokens == 8
    assert normalized[-1].tok_per_sec == 16.0
