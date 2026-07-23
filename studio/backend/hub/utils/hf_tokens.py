# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hugging Face token boundary helpers."""

from __future__ import annotations

from typing import Literal, Optional

HfTokenArg = str | Literal[False]


def hf_token_arg(hf_token: Optional[str]) -> HfTokenArg:
    """Return an explicit token or disable Hugging Face's ambient credentials."""
    token = (hf_token or "").strip()
    return token or False
