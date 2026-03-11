# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""Pydantic models for authentication tokens.

This module defines the Token response model used by auth routes.
"""

from pydantic import BaseModel, Field


class Token(BaseModel):
    """Authentication token model with access and refresh tokens."""

    access_token: str = Field(..., description = "JWT access token (60 min expiry)")
    refresh_token: str = Field(..., description = "Opaque refresh token (7 day expiry)")
    token_type: str = Field(..., description = "Token type, always 'bearer'")
