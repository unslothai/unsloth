# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-local token progress attached to streamed text snapshots."""

import math
from typing import Optional


class LiveTokenProgressText(str):
    def __new__(
        cls,
        text: str,
        *,
        generated_tokens: Optional[int] = None,
        tok_per_sec: Optional[float] = None,
    ):
        value = super().__new__(cls, text)
        value.generated_tokens = generated_tokens
        value.tok_per_sec = tok_per_sec
        return value


def live_token_progress_text(
    text: str,
    *,
    generated_tokens,
    elapsed_seconds = None,
    tok_per_sec = None,
) -> LiveTokenProgressText:
    try:
        generated_tokens = int(generated_tokens)
    except (TypeError, ValueError, OverflowError):
        generated_tokens = None
    if generated_tokens is not None and generated_tokens < 0:
        generated_tokens = None

    try:
        rate = float(tok_per_sec) if tok_per_sec is not None else None
    except (TypeError, ValueError, OverflowError):
        rate = None
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        rate = None
    if rate is None and generated_tokens is not None and generated_tokens > 1:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError, OverflowError):
            elapsed = 0.0
        if math.isfinite(elapsed) and elapsed > 0:
            rate = (generated_tokens - 1) / elapsed

    return LiveTokenProgressText(
        text,
        generated_tokens = generated_tokens,
        tok_per_sec = rate,
    )


def inherit_live_token_progress(text: str, source: str) -> LiveTokenProgressText:
    return LiveTokenProgressText(
        text,
        generated_tokens = getattr(source, "generated_tokens", None),
        tok_per_sec = getattr(source, "tok_per_sec", None),
    )
