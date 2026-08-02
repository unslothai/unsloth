# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation shared by every reasoning-budget-message persistence boundary."""

from __future__ import annotations


# Keep ample headroom below Windows' total command-line limit and Linux's
# per-argument MAX_ARG_STRLEN. Budget-exhaustion messages should be short prose.
MAX_REASONING_BUDGET_MESSAGE_BYTES = 8_192


def validate_reasoning_budget_message(value: str) -> str:
    """Return ``value`` unchanged when it is safe to pass as one argv token."""
    if "\0" in value:
        raise ValueError("llama-server --reasoning-budget-message cannot contain NUL characters.")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(
            "llama-server --reasoning-budget-message contains invalid Unicode."
        ) from exc
    if size > MAX_REASONING_BUDGET_MESSAGE_BYTES:
        raise ValueError(
            "llama-server --reasoning-budget-message exceeds the "
            f"{MAX_REASONING_BUDGET_MESSAGE_BYTES}-byte UTF-8 limit."
        )
    return value
