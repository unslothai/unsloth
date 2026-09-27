# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared FastAPI dependencies for Hub routes."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header

from auth.authentication import allow_ambient_hf_token
from hub.utils.hf_tokens import HfTokenArg, hf_token_arg

HUB_HF_TOKEN_HEADER = "X-Unsloth-HF-Token"
HUB_HF_TOKEN_MAX_LENGTH = 512


def get_hf_token(
    hf_token: Optional[str] = Header(
        None,
        alias = HUB_HF_TOKEN_HEADER,
        max_length = HUB_HF_TOKEN_MAX_LENGTH,
    ),
) -> Optional[str]:
    token = (hf_token or "").strip()
    return token or None


def get_request_hf_token(
    hf_token: Optional[str] = Depends(get_hf_token),
    allow_ambient_token: bool = Depends(allow_ambient_hf_token),
) -> HfTokenArg:
    """Resolve the Hub token under the caller boundary established by authentication."""
    return hf_token_arg(
        hf_token,
        allow_ambient_token = allow_ambient_token,
    )
