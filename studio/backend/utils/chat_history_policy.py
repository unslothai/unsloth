# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup-only host policy for durable Studio chat content."""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException


ENV_VAR = "UNSLOTH_NO_CHAT_HISTORY"
DISABLED_DETAIL = "Chat history is disabled by the server operator."
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


def _parse(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    if normalized in _TRUE_VALUES:
        return True
    logging.getLogger(__name__).warning(
        "%s has an unrecognized non-empty value; disabling chat history to fail closed.",
        ENV_VAR,
    )
    return True


# Deliberately fixed at import/startup. A shared Studio user cannot change host policy.
NO_CHAT_HISTORY = _parse(os.environ.get(ENV_VAR))


def disabled() -> bool:
    return NO_CHAT_HISTORY


def require_enabled() -> None:
    if disabled():
        raise HTTPException(status_code = 403, detail = DISABLED_DETAIL)
