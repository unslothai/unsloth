# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic models for authentication tokens.

This module defines the Token response model used by auth routes.
"""

from pydantic import BaseModel, Field


class Token(BaseModel):
    """Authentication response model for session credentials."""

    access_token: str = Field(
        ..., description = "Session access credential used for authenticated API requests"
    )
    refresh_token: str = Field(
        ...,
        description = "Session refresh credential used to renew an expired access credential",
    )
    token_type: str = Field(
        ..., description = "Credential type for the Authorization header, always 'bearer'"
    )
    must_change_password: bool = Field(
        ..., description = "True when the user must change the seeded default password"
    )
