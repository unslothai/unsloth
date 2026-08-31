# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime context length helpers shared by inference backends."""

from __future__ import annotations

from typing import Any, Optional

#: Longest context a load may ask for, and the ceiling a resolved window is held to.
#: LoadRequest.max_seq_length bounds requests by this; a backend that reads a wider
#: window from the model reports it as native but does not serve past it.
MAX_REQUESTABLE_CONTEXT = 1048576


def runtime_context_length(model: Any, fallback: Optional[int] = None) -> Optional[int]:
    """Return the effective context length Unsloth attached to a loaded model."""
    for value in (getattr(model, "max_seq_length", None), fallback):
        if isinstance(value, bool):
            continue
        try:
            value_int = int(value)
        except (TypeError, ValueError):
            continue
        if value_int > 0:
            return value_int
    return None
